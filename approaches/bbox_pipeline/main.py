"""
Bounding Box-Based Vehicle Speed Estimation Pipeline

This pipeline uses the bottom-center of vehicle bounding boxes
to estimate ground contact points and calculate vehicle speeds.

Approach:
1. Detect vehicles using YOLOv8 pose model (best.pt)
2. Track vehicles across frames
3. Calculate ground position from bounding box bottom-center
4. Transform to real-world coordinates using homography
5. Calculate speed from position changes over time

Usage:
    python main.py
"""

import cv2
import sys
import os

# Add project root to path for imports
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
approaches_dir = os.path.dirname(pipeline_dir)
project_root = os.path.dirname(approaches_dir)
sys.path.insert(0, project_root)

from ultralytics import YOLO
from utils import (
    preprocess_frame, load_calibration_data, rescale_coordinates,
    CoordinateTransformer, calculate_real_world_coordinates, calculate_real_box_width,
    SpeedTracker, draw_annotations, CSVExporter
)
from config import (
    VIDEO_PATH, MODEL_PATH, CALIBRATION_FILE, MAPPING_FILE,
    RECOGNITION_SIZE, DISPLAY_SIZE, OUTPUT_TRACKING_CSV, OUTPUT_WORLD_COORDS_CSV
)


def main():
    print("=" * 60)
    print("BOUNDING BOX SPEED ESTIMATION PIPELINE")
    print("=" * 60)

    # Load the YOLOv8 model
    print(f"\nLoading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Set device
    try:
        model.to("cuda")
        print("Using device: CUDA")
    except:
        print("Using device: CPU")

    # Load calibration data
    print(f"\nLoading calibration: {CALIBRATION_FILE}")
    K, D, DIM = load_calibration_data(CALIBRATION_FILE)
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Initialize coordinate transformer and speed tracker
    print(f"Loading coordinate mapping: {MAPPING_FILE}")
    transformer = CoordinateTransformer(MAPPING_FILE)
    speed_tracker = SpeedTracker()

    # Open the video file
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        print(f"Warning: Invalid FPS ({fps}), defaulting to 30")
        fps = 30.0

    print(f"Video: {total_frames} frames @ {fps} FPS")

    # Setup CSV exporters
    tracking_header = ['frame', 'id', 'x', 'y', 'width', 'real_width']
    for i in range(10):  # 10 keypoints
        tracking_header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
    tracking_exporter = CSVExporter(OUTPUT_TRACKING_CSV, tracking_header)

    world_coord_header = ['frame', 'id', 'world_x', 'world_y', 'speed_kmh']
    world_coord_exporter = CSVExporter(OUTPUT_WORLD_COORDS_CSV, world_coord_header)

    frame_count = 0
    print("\nProcessing... Press 'q' to quit\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Progress update
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        # Preprocess the frame
        recognition_frame, display_frame = preprocess_frame(
            frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        # Run YOLOv8 tracking
        results = model.track(recognition_frame, persist=True, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []

            # Rescale to display size
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            scaled_keypoints = []
            if len(keypoints) > 0:
                scaled_keypoints = [
                    [
                        rescale_coordinates(kp[:2], RECOGNITION_SIZE, DISPLAY_SIZE) + [kp[2]]
                        if len(kp) == 3 and kp[2] > 0 else [0, 0, 0]
                        for kp in obj_kps
                    ]
                    for obj_kps in keypoints
                ]

            # Calculate real-world coordinates (bottom-center of bbox)
            real_world_coords = calculate_real_world_coordinates(scaled_boxes, transformer)

            # Calculate speeds
            speeds = speed_tracker.get_speeds(track_ids, real_world_coords, frame_count, fps)

            # Export data
            for idx, (box, track_id, world_coord, speed) in enumerate(
                zip(scaled_boxes, track_ids, real_world_coords, speeds)
            ):
                x, y, w, h = box
                real_width = calculate_real_box_width(box, transformer)

                # Write tracking data
                row = [frame_count, track_id, x, y, w, real_width]
                if idx < len(scaled_keypoints):
                    for kp in scaled_keypoints[idx]:
                        row.extend(kp)
                tracking_exporter.write_row(row)

                # Write world coordinates
                world_coord_exporter.write_row([
                    frame_count, track_id,
                    world_coord[0], world_coord[1], speed
                ])

            # Draw annotations
            if len(scaled_keypoints) == 0:
                scaled_keypoints = [[[0, 0, 0]] * 10] * len(scaled_boxes)
            annotated_frame = draw_annotations(
                display_frame.copy(), scaled_boxes, scaled_keypoints, track_ids, speeds
            )
        else:
            annotated_frame = display_frame

        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Method: BBox Bottom-Center",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display
        cv2.imshow("BBox Speed Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracking_exporter.close()
    world_coord_exporter.close()

    print(f"\nProcessing complete!")
    print(f"Tracking data saved to: {OUTPUT_TRACKING_CSV}")
    print(f"World coordinates saved to: {OUTPUT_WORLD_COORDS_CSV}")


if __name__ == "__main__":
    main()
