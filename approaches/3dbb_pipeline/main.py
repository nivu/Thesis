"""
3D Bounding Box Vehicle Localization Pipeline

This pipeline integrates monocular depth estimation with 2D detection
to produce 3D bounding boxes for improved vehicle localization.

Approach:
1. Detect vehicles using YOLOv8 pose model (existing)
2. Estimate depth using Depth Anything v2
3. Create 3D bounding boxes from depth + 2D detections
4. Fuse with homography-based 2D estimates
5. Calculate speed from fused positions

Usage:
    python main.py [--no-display] [--max-frames N]
"""

import cv2
import numpy as np
import sys
import os
import argparse
import time

# Add project root to path for imports
pipeline_dir = os.path.dirname(os.path.abspath(__file__))
approaches_dir = os.path.dirname(pipeline_dir)
project_root = os.path.dirname(approaches_dir)
sys.path.insert(0, project_root)

from ultralytics import YOLO
from utils import (
    preprocess_frame, load_calibration_data, rescale_coordinates,
    CoordinateTransformer, SpeedTracker, CSVExporter
)
# Import from local 3dbb_pipeline modules
sys.path.insert(0, pipeline_dir)
from config import (
    VIDEO_PATH, MODEL_2D_PATH, CALIBRATION_FILE, MAPPING_FILE,
    RECOGNITION_SIZE, DISPLAY_SIZE, FUSION_CONFIG,
    OUTPUT_WORLD_COORDS_CSV, OUTPUT_COMPARISON_CSV
)
from detector_3d import Detector3D, Box3D, load_or_calibrate_depth_scale
from fusion import CoordinateFusion, FusedResult


def main():
    parser = argparse.ArgumentParser(description='3D Bounding Box Vehicle Localization')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (overrides config)')
    args = parser.parse_args()

    print("=" * 60)
    print("3D BOUNDING BOX LOCALIZATION PIPELINE")
    print("=" * 60)

    # Use video from args or config
    video_path = args.video if args.video else VIDEO_PATH

    # Check if video path exists, if not try default
    if not os.path.exists(video_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_dir, "traffic_analyis_data/Uni_west_1/GOPR0574.MP4")
        print(f"Using default video: {video_path}")

    # Load calibration data
    print(f"\nLoading calibration: {CALIBRATION_FILE}")
    K, D, DIM = load_calibration_data(CALIBRATION_FILE)
    if K is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Initialize coordinate transformer
    print(f"Loading coordinate mapping: {MAPPING_FILE}")
    transformer = CoordinateTransformer(MAPPING_FILE)

    # Load 2D detection model
    print(f"\nLoading 2D model: {MODEL_2D_PATH}")
    model_2d = YOLO(MODEL_2D_PATH)

    # Set device
    try:
        model_2d.to("cuda")
        device = "CUDA"
    except:
        device = "CPU"
    print(f"Using device: {device}")

    # Load depth scale
    print("\nLoading depth calibration...")
    depth_scale = load_or_calibrate_depth_scale()

    # Initialize 3D detector
    print("\nInitializing 3D detector...")
    detector_3d = Detector3D(K, depth_scale)
    detector_3d.load_models()

    # Initialize fusion module
    fusion = CoordinateFusion(
        transformer=transformer,
        weight_3d=FUSION_CONFIG['weight_3d'],
        weight_2d=FUSION_CONFIG['weight_2d'],
        max_discrepancy=FUSION_CONFIG['max_discrepancy_m']
    )

    # Initialize speed tracker
    speed_tracker = SpeedTracker()

    # Open video
    print(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    if args.max_frames:
        total_frames = min(total_frames, args.max_frames)

    print(f"Video: {total_frames} frames @ {fps} FPS")

    # Setup CSV exporter
    header = [
        'frame', 'track_id', 'world_x', 'world_y', 'yaw_deg',
        'speed_kmh', 'method', 'confidence', 'discrepancy_m',
        'pos_3d_x', 'pos_3d_y', 'pos_2d_x', 'pos_2d_y'
    ]
    exporter = CSVExporter(OUTPUT_WORLD_COORDS_CSV, header)

    frame_count = 0
    total_time = 0
    print("\nProcessing... Press 'q' to quit\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        if args.max_frames and frame_count > args.max_frames:
            break

        start_time = time.time()

        # Progress update
        if frame_count % 50 == 0:
            avg_fps = frame_count / total_time if total_time > 0 else 0
            print(f"Processing frame {frame_count}/{total_frames} ({avg_fps:.1f} FPS)")

        # Preprocess frame
        recognition_frame, display_frame = preprocess_frame(
            frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        # Step 1: Run 2D detection + tracking
        results = model_2d.track(recognition_frame, persist=True, verbose=False)

        fused_results = []
        track_ids = []
        boxes_3d = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().numpy()

            # Scale boxes to display size
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            # Step 2: Get depth map
            depth_map = detector_3d.get_depth_map(display_frame)

            # Step 3: Create 3D boxes from 2D detections
            boxes_2d_with_conf = [
                (box[0], box[1], box[2], box[3], conf)
                for box, conf in zip(scaled_boxes, confs)
            ]
            boxes_3d = detector_3d.detect_from_boxes(
                display_frame, boxes_2d_with_conf, depth_map, track_ids
            )

            # Step 4: Fuse 3D and 2D estimates
            for idx, (box_2d, box_3d) in enumerate(zip(scaled_boxes, boxes_3d)):
                x, y, w, h = box_2d
                # Contact point: bottom center of bbox
                contact_2d = (x, y + h / 2)

                # Fuse estimates (pass camera matrix for 3D corner transformation)
                fused = fusion.fuse(box_3d, contact_2d, confidence_2d=confs[idx], camera_matrix=K)
                fused_results.append(fused)

            # Step 5: Calculate speeds
            positions = [f.position for f in fused_results]
            speeds = speed_tracker.get_speeds(track_ids, positions, frame_count, fps)

            # Export results
            for idx, (track_id, fused, speed) in enumerate(
                zip(track_ids, fused_results, speeds)
            ):
                yaw_deg = np.degrees(fused.orientation) if fused.orientation else 0

                row = [
                    frame_count, track_id,
                    fused.position[0], fused.position[1], yaw_deg,
                    speed, fused.method, fused.confidence, fused.discrepancy,
                    fused.position_3d[0] if fused.position_3d else '',
                    fused.position_3d[1] if fused.position_3d else '',
                    fused.position_2d[0], fused.position_2d[1]
                ]
                exporter.write_row(row)

        # Visualization
        if not args.no_display:
            annotated_frame = draw_3d_annotations(
                display_frame, boxes_3d, fused_results, track_ids,
                speeds if track_ids else [], K, transformer
            )

            # Add frame info
            elapsed = time.time() - start_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0
            cv2.putText(annotated_frame,
                       f"Frame: {frame_count} | FPS: {current_fps:.1f} | Method: 3D+2D Fusion",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("3D BBox Localization", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        total_time += time.time() - start_time

    # Cleanup
    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    exporter.close()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
    print(f"Results saved to: {OUTPUT_WORLD_COORDS_CSV}")


def draw_3d_annotations(frame, boxes_3d, fused_results, track_ids, speeds,
                        camera_matrix, transformer):
    """Draw 3D bounding box annotations on frame."""
    annotated = frame.copy()

    for idx, (box_3d, fused, track_id) in enumerate(
        zip(boxes_3d, fused_results, track_ids)
    ):
        if box_3d.bbox_2d is None:
            continue

        cx, cy, w, h = box_3d.bbox_2d

        # Draw 2D bounding box
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)

        # Color based on method
        if fused.method == 'fused':
            color = (0, 255, 0)  # Green for fused
        elif fused.method == '3d_only':
            color = (255, 0, 0)  # Blue for 3D only
        else:
            color = (0, 165, 255)  # Orange for 2D only

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw contact point (bottom center)
        contact_pt = (int(cx), int(cy + h/2))
        cv2.circle(annotated, contact_pt, 5, (0, 255, 255), -1)

        # Draw label
        speed = speeds[idx] if idx < len(speeds) else 0
        yaw_deg = np.degrees(box_3d.yaw) if box_3d.yaw else 0

        label = f"ID:{track_id} {speed:.1f}km/h"
        cv2.putText(annotated, label, (x1, y1 - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw world position
        world_label = f"({fused.position[0]:.1f}, {fused.position[1]:.1f})m"
        cv2.putText(annotated, world_label, (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw yaw indicator (line showing vehicle orientation)
        yaw_length = 30
        yaw_end_x = int(cx + yaw_length * np.cos(box_3d.yaw))
        yaw_end_y = int(cy + yaw_length * np.sin(box_3d.yaw))
        cv2.arrowedLine(annotated, (int(cx), int(cy)),
                       (yaw_end_x, yaw_end_y), (0, 0, 255), 2)

        # Draw discrepancy warning if high
        if fused.discrepancy > 0.5:
            cv2.putText(annotated, f"DISCREPANCY: {fused.discrepancy:.2f}m",
                       (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw legend
    legend_y = frame.shape[0] - 80
    cv2.putText(annotated, "Legend:", (10, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(annotated, "Green=Fused  Blue=3D  Orange=2D", (10, legend_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(annotated, "Yellow dot=Contact point  Red arrow=Yaw", (10, legend_y + 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return annotated


if __name__ == "__main__":
    main()
