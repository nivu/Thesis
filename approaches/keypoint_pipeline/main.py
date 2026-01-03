"""
Keypoint-Based Vehicle Speed Estimation Pipeline

This pipeline uses vehicle wheel keypoints detected by the YOLOv8 pose model
to estimate ground contact points for vehicle localization and speed calculation.

Approach:
1. Detect vehicles using YOLOv8 pose model (best.pt with 10 keypoints)
2. Track vehicles across frames
3. Extract wheel keypoints (indices 0-3: FL, FR, RL, RR wheels)
4. Use bottom-most visible wheel keypoint as ground contact point
5. Transform to real-world coordinates using homography
6. Calculate speed from position changes over time

Keypoint Schema (10 keypoints):
0-3: Four wheel centers (FL, FR, RL, RR)
4-9: Six vehicle reference points

Usage:
    python main.py
"""

import cv2
import numpy as np
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
    CoordinateTransformer, SpeedTracker, CSVExporter
)

# Import local config
sys.path.insert(0, pipeline_dir)
from config import (
    VIDEO_PATH, MODEL_PATH, CALIBRATION_FILE, MAPPING_FILE,
    RECOGNITION_SIZE, DISPLAY_SIZE, KEYPOINT_CONFIDENCE_THRESHOLD,
    OUTPUT_TRACKING_CSV, OUTPUT_WORLD_COORDS_CSV
)


def get_wheel_contact_point(keypoints, recognition_size, display_size, confidence_threshold=0.3):
    """
    Extract the ground contact point from wheel keypoints.

    Uses the bottom-most visible wheel keypoint (indices 0-3) as the contact point.
    Falls back to the rear wheel average if both rear wheels are visible.

    Args:
        keypoints: Array of 10 keypoints [x, y, confidence]
        recognition_size: Size of recognition frame
        display_size: Size of display frame
        confidence_threshold: Minimum confidence for keypoint

    Returns:
        (x, y) contact point in display coordinates, or None if no valid keypoints
    """
    if keypoints is None or len(keypoints) < 4:
        return None

    scale_x = display_size[0] / recognition_size[0]
    scale_y = display_size[1] / recognition_size[1]

    # Extract wheel keypoints (indices 0-3)
    wheel_keypoints = []
    for i in range(4):
        kp = keypoints[i]
        x, y, conf = kp[0], kp[1], kp[2]
        if conf >= confidence_threshold:
            display_x = int(x * scale_x)
            display_y = int(y * scale_y)
            wheel_keypoints.append((display_x, display_y, conf, i))

    if not wheel_keypoints:
        return None

    # Strategy: Use rear wheels (indices 2, 3) if both visible, else use bottom-most wheel
    rear_wheels = [wp for wp in wheel_keypoints if wp[3] in [2, 3]]

    if len(rear_wheels) == 2:
        # Average of both rear wheels
        avg_x = (rear_wheels[0][0] + rear_wheels[1][0]) / 2
        avg_y = (rear_wheels[0][1] + rear_wheels[1][1]) / 2
        return (int(avg_x), int(avg_y))
    elif len(rear_wheels) == 1:
        # Single rear wheel
        return (rear_wheels[0][0], rear_wheels[0][1])
    else:
        # No rear wheels visible, use bottom-most front wheel
        bottom_wheel = max(wheel_keypoints, key=lambda wp: wp[1])
        return (bottom_wheel[0], bottom_wheel[1])


def calculate_real_world_from_keypoints(keypoints_list, transformer, recognition_size, display_size, conf_threshold=0.3):
    """
    Calculate real-world coordinates from keypoints for each vehicle.

    Returns list of (world_x, world_y) tuples and list of contact points used.
    """
    real_world_coords = []
    contact_points_used = []
    methods_used = []

    for kps in keypoints_list:
        contact_pt = get_wheel_contact_point(kps, recognition_size, display_size, conf_threshold)

        if contact_pt:
            world_coord = transformer.pixel_to_world(contact_pt[0], contact_pt[1])
            real_world_coords.append(world_coord if world_coord else (0, 0))
            contact_points_used.append(contact_pt)
            methods_used.append('keypoint')
        else:
            # This should not happen if keypoints are detected, but handle gracefully
            real_world_coords.append((0, 0))
            contact_points_used.append(None)
            methods_used.append('none')

    return real_world_coords, contact_points_used, methods_used


def draw_annotations(image, boxes, keypoints_list, track_ids, speeds, contact_points, recognition_size, display_size):
    """Draw vehicle annotations with keypoints and contact points."""
    annotated = image.copy()

    scale_x = display_size[0] / recognition_size[0]
    scale_y = display_size[1] / recognition_size[1]

    for idx, (box, track_id, speed) in enumerate(zip(boxes, track_ids, speeds)):
        x, y, w, h = box

        # Draw bounding box
        cv2.rectangle(annotated,
                     (int(x - w/2), int(y - h/2)),
                     (int(x + w/2), int(y + h/2)),
                     (0, 255, 0), 2)

        # Draw keypoints
        if idx < len(keypoints_list):
            kps = keypoints_list[idx]
            for kp_idx, kp in enumerate(kps):
                kp_x, kp_y, conf = kp[0], kp[1], kp[2]
                if conf > 0.3:
                    display_x = int(kp_x * scale_x)
                    display_y = int(kp_y * scale_y)

                    # Color: wheels (0-3) in yellow, others in blue
                    if kp_idx < 4:
                        color = (0, 255, 255)  # Yellow for wheels
                        radius = 6
                    else:
                        color = (255, 0, 0)  # Blue for reference points
                        radius = 4

                    cv2.circle(annotated, (display_x, display_y), radius, color, -1)

        # Draw contact point (if available)
        if idx < len(contact_points) and contact_points[idx]:
            cp = contact_points[idx]
            cv2.circle(annotated, cp, 10, (0, 0, 255), 2)
            cv2.circle(annotated, cp, 4, (0, 255, 255), -1)

        # Draw label
        label = f"ID:{track_id} {speed:.1f}km/h"
        cv2.putText(annotated, label,
                   (int(x - w/2), int(y - h/2 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return annotated


def main():
    print("=" * 60)
    print("KEYPOINT-BASED SPEED ESTIMATION PIPELINE")
    print("=" * 60)

    # Load the YOLOv8 pose model
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
    tracking_header = ['frame', 'id', 'x', 'y', 'width', 'height']
    for i in range(10):  # 10 keypoints
        tracking_header.extend([f'kp{i}_x', f'kp{i}_y', f'kp{i}_conf'])
    tracking_exporter = CSVExporter(OUTPUT_TRACKING_CSV, tracking_header)

    world_coord_header = ['frame', 'id', 'world_x', 'world_y', 'speed_kmh',
                         'contact_x', 'contact_y', 'method', 'num_visible_wheels']
    world_coord_exporter = CSVExporter(OUTPUT_WORLD_COORDS_CSV, world_coord_header)

    frame_count = 0
    total_keypoint_detections = 0
    total_fallback_detections = 0

    print("\nProcessing... Press 'q' to quit")
    print("Legend: Yellow=Wheel keypoints | Blue=Reference points | Red circle=Contact point\n")

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

            # Rescale boxes to display size
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            # Calculate real-world coordinates from keypoints
            if len(keypoints) > 0:
                real_world_coords, contact_points, methods = calculate_real_world_from_keypoints(
                    keypoints, transformer, RECOGNITION_SIZE, DISPLAY_SIZE,
                    KEYPOINT_CONFIDENCE_THRESHOLD
                )
            else:
                # No keypoints, use bounding box fallback
                real_world_coords = []
                contact_points = []
                methods = []
                for box in scaled_boxes:
                    x, y, w, h = box
                    world_coord = transformer.pixel_to_world(x, y + h/2)
                    real_world_coords.append(world_coord if world_coord else (0, 0))
                    contact_points.append((int(x), int(y + h/2)))
                    methods.append('bbox_fallback')

            # Count detection methods
            for m in methods:
                if m == 'keypoint':
                    total_keypoint_detections += 1
                else:
                    total_fallback_detections += 1

            # Calculate speeds
            speeds = speed_tracker.get_speeds(track_ids, real_world_coords, frame_count, fps)

            # Export data
            for idx, (box, track_id, world_coord, speed, contact_pt, method) in enumerate(
                zip(scaled_boxes, track_ids, real_world_coords, speeds, contact_points, methods)
            ):
                x, y, w, h = box

                # Count visible wheel keypoints
                num_visible_wheels = 0
                if idx < len(keypoints):
                    for i in range(4):
                        if keypoints[idx][i][2] >= KEYPOINT_CONFIDENCE_THRESHOLD:
                            num_visible_wheels += 1

                # Write tracking data
                row = [frame_count, track_id, x, y, w, h]
                if idx < len(keypoints):
                    for kp in keypoints[idx]:
                        row.extend([kp[0], kp[1], kp[2]])
                else:
                    row.extend([0, 0, 0] * 10)
                tracking_exporter.write_row(row)

                # Write world coordinates
                world_coord_exporter.write_row([
                    frame_count, track_id,
                    world_coord[0], world_coord[1], speed,
                    contact_pt[0] if contact_pt else 0,
                    contact_pt[1] if contact_pt else 0,
                    method, num_visible_wheels
                ])

            # Draw annotations
            annotated_frame = draw_annotations(
                display_frame.copy(), scaled_boxes, keypoints,
                track_ids, speeds, contact_points, RECOGNITION_SIZE, DISPLAY_SIZE
            )
        else:
            annotated_frame = display_frame

        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {frame_count} | Method: Keypoint Detection",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Yellow=Wheels | Blue=Ref | Red circle=Contact",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display
        cv2.imshow("Keypoint Speed Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    tracking_exporter.close()
    world_coord_exporter.close()

    # Summary
    total_detections = total_keypoint_detections + total_fallback_detections
    keypoint_rate = (total_keypoint_detections / total_detections * 100) if total_detections > 0 else 0

    print(f"\nProcessing complete!")
    print(f"Total detections: {total_detections}")
    print(f"  - Keypoint-based: {total_keypoint_detections} ({keypoint_rate:.1f}%)")
    print(f"  - Fallback: {total_fallback_detections} ({100-keypoint_rate:.1f}%)")
    print(f"\nTracking data saved to: {OUTPUT_TRACKING_CSV}")
    print(f"World coordinates saved to: {OUTPUT_WORLD_COORDS_CSV}")


if __name__ == "__main__":
    main()
