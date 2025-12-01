"""
Segmentation & Keypoint-Based Vehicle Speed Estimation Pipeline

This pipeline uses multiple methods to detect tire-ground contact points
for more accurate vehicle localization and speed calculation.

Methods (in priority order):
1. Wheel Segmentation: Extract contact points from segmentation masks
2. Keypoint Detection: Use wheel keypoints from pose model
3. Bounding Box Fallback: Use bottom-center of vehicle bounding box

Usage:
    python main.py
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from utils import (
    preprocess_frame, load_calibration_data, rescale_coordinates,
    CoordinateTransformer, SpeedTracker, CSVExporter
)
from config import (
    VIDEO_PATH, VEHICLE_MODEL_PATH, WHEEL_SEG_MODEL_PATH,
    CALIBRATION_FILE, MAPPING_FILE, RECOGNITION_SIZE, DISPLAY_SIZE,
    KEYPOINT_CONFIDENCE_THRESHOLD, OUTPUT_CSV
)


class KeypointContactPointExtractor:
    """Extracts tire-ground contact points from vehicle keypoints."""

    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold

    def get_contact_points(self, keypoints, recognition_size, display_size):
        """Extract valid contact points from keypoints."""
        if keypoints is None or len(keypoints) == 0:
            return []

        scale_x = display_size[0] / recognition_size[0]
        scale_y = display_size[1] / recognition_size[1]

        contact_points = []
        for kp in keypoints:
            x, y, conf = kp[0], kp[1], kp[2]
            if conf >= self.confidence_threshold:
                display_x = int(x * scale_x)
                display_y = int(y * scale_y)
                contact_points.append((display_x, display_y, conf))

        return contact_points

    def get_ground_contact_point(self, keypoints, recognition_size, display_size):
        """Get the bottom-most high-confidence keypoint."""
        contact_points = self.get_contact_points(keypoints, recognition_size, display_size)

        if not contact_points:
            return None

        # Return bottom-most point (highest y value)
        bottom_point = max(contact_points, key=lambda p: p[1])
        return (bottom_point[0], bottom_point[1])


class WheelContactPointExtractor:
    """Extracts tire-ground contact points from wheel segmentation masks."""

    def get_contact_point(self, mask):
        """Extract the tire-ground contact point from a segmentation mask."""
        if mask is None or mask.sum() == 0:
            return None

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 10:
            return None

        points = largest_contour.reshape(-1, 2)
        max_y = points[:, 1].max()
        bottom_points = [p for p in points if p[1] >= max_y - 5]

        if not bottom_points:
            return None

        bottom_points = np.array(bottom_points)
        contact_x = int(np.mean(bottom_points[:, 0]))
        contact_y = int(np.max(bottom_points[:, 1]))

        return (contact_x, contact_y)

    def get_wheel_centroid(self, mask):
        """Get the centroid of a wheel mask."""
        if mask is None or mask.sum() == 0:
            return None

        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
        moments = cv2.moments(mask_uint8)

        if moments['m00'] == 0:
            return None

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        return (cx, cy)


class VehicleWheelAssociator:
    """Associates detected wheels with their parent vehicles."""

    def associate_wheels(self, vehicle_boxes, wheel_data, frame_shape):
        """Associate wheel detections with vehicle bounding boxes."""
        associations = {i: [] for i in range(len(vehicle_boxes))}

        for wheel in wheel_data:
            centroid = wheel.get('centroid')
            if centroid is None:
                continue

            best_vehicle_idx = None
            best_score = float('inf')

            for v_idx, box in enumerate(vehicle_boxes):
                x, y, w, h = box
                margin = 20
                x1, y1 = x - w/2 - margin, y - h/2 - margin
                x2, y2 = x + w/2 + margin, y + h/2 + margin

                cx, cy = centroid
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < best_score:
                        best_score = dist
                        best_vehicle_idx = v_idx

            if best_vehicle_idx is not None:
                associations[best_vehicle_idx].append(wheel)

        return associations


def draw_annotations(image, vehicle_boxes, track_ids, wheel_associations, speeds, methods_used):
    """Draw annotations with color coding by method."""
    annotated = image.copy()

    for v_idx, (box, track_id, speed) in enumerate(zip(vehicle_boxes, track_ids, speeds)):
        x, y, w, h = box
        method = methods_used[v_idx] if v_idx < len(methods_used) else 'unknown'

        # Color by method
        if method == 'wheel_seg':
            box_color = (0, 255, 0)  # Green
        elif method == 'keypoint':
            box_color = (255, 165, 0)  # Orange
        else:
            box_color = (128, 128, 128)  # Gray

        # Draw bounding box
        cv2.rectangle(annotated,
                     (int(x - w/2), int(y - h/2)),
                     (int(x + w/2), int(y + h/2)),
                     box_color, 2)

        # Draw label
        label = f"ID:{track_id} {speed:.1f}km/h [{method[:3]}]"
        cv2.putText(annotated, label,
                   (int(x - w/2), int(y - h/2 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Draw contact points
        wheels = wheel_associations.get(v_idx, [])
        for wheel in wheels:
            cp = wheel.get('contact_point')
            class_name = wheel.get('class_name', '')

            if cp is not None:
                if class_name == 'keypoint':
                    cv2.circle(annotated, cp, 8, (0, 165, 255), -1)
                    cv2.circle(annotated, cp, 4, (255, 255, 255), -1)
                else:
                    cv2.circle(annotated, cp, 8, (0, 0, 255), -1)
                    cv2.circle(annotated, cp, 4, (0, 255, 255), -1)

    return annotated


def process_wheel_segmentation(wheel_results, display_size):
    """Process wheel segmentation results."""
    extractor = WheelContactPointExtractor()
    wheel_data = []

    if wheel_results[0].masks is None:
        return wheel_data

    masks = wheel_results[0].masks.data.cpu().numpy()
    class_ids = wheel_results[0].boxes.cls.cpu().numpy().astype(int)

    for mask, class_id in zip(masks, class_ids):
        mask_resized = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST)
        contact_point = extractor.get_contact_point(mask_resized)
        centroid = extractor.get_wheel_centroid(mask_resized)

        wheel_data.append({
            'contact_point': contact_point,
            'centroid': centroid,
            'class_id': class_id,
            'class_name': ['backwheel', 'frontwheel', 'middlewheel'][class_id]
        })

    return wheel_data


def main():
    print("=" * 60)
    print("SEGMENTATION & KEYPOINT SPEED ESTIMATION PIPELINE")
    print("=" * 60)

    # Load models
    print(f"\nLoading vehicle model: {VEHICLE_MODEL_PATH}")
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)

    print(f"Loading wheel segmentation model: {WHEEL_SEG_MODEL_PATH}")
    wheel_model = YOLO(WHEEL_SEG_MODEL_PATH)

    # Set device
    try:
        vehicle_model.to("cuda")
        wheel_model.to("cuda")
        print("Using device: CUDA")
    except:
        print("Using device: CPU")

    # Load calibration
    print(f"\nLoading calibration: {CALIBRATION_FILE}")
    K, D, DIM = load_calibration_data(CALIBRATION_FILE)
    if K is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Initialize components
    print(f"Loading coordinate mapping: {MAPPING_FILE}")
    transformer = CoordinateTransformer(MAPPING_FILE)
    speed_tracker = SpeedTracker()
    wheel_associator = VehicleWheelAssociator()
    keypoint_extractor = KeypointContactPointExtractor(KEYPOINT_CONFIDENCE_THRESHOLD)

    # Open video
    print(f"\nOpening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"Error: Could not open video: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0

    print(f"Video: {total_frames} frames @ {fps} FPS")

    # CSV export
    header = ['frame', 'id', 'world_x', 'world_y', 'speed_kmh', 'num_wheels', 'method', 'contact_points']
    exporter = CSVExporter(OUTPUT_CSV, header)

    frame_count = 0
    print("\nProcessing... Press 'q' to quit")
    print("Color legend: Green=Segmentation | Orange=Keypoint | Gray=BBox\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")

        # Preprocess
        recognition_frame, display_frame = preprocess_frame(
            frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        # Vehicle detection
        vehicle_results = vehicle_model.track(recognition_frame, persist=True, verbose=False)

        # Wheel segmentation
        wheel_results = wheel_model(recognition_frame, verbose=False)
        wheel_data = process_wheel_segmentation(wheel_results, DISPLAY_SIZE)

        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

            keypoints = None
            if vehicle_results[0].keypoints is not None:
                keypoints = vehicle_results[0].keypoints.data.cpu().numpy()

            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            wheel_associations = wheel_associator.associate_wheels(
                scaled_boxes, wheel_data, display_frame.shape
            )

            # Calculate positions (Priority: Seg -> Keypoint -> BBox)
            real_world_coords = []
            methods_used = []
            contact_points_used = []

            for v_idx, box in enumerate(scaled_boxes):
                associated_wheels = wheel_associations.get(v_idx, [])
                method = 'bbox_fallback'
                contact_pt = None
                world_coord = None

                # Method 1: Wheel segmentation
                if associated_wheels:
                    contact_points = [w['contact_point'] for w in associated_wheels if w.get('contact_point')]
                    if contact_points:
                        avg_x = np.mean([p[0] for p in contact_points])
                        avg_y = np.mean([p[1] for p in contact_points])
                        world_coord = transformer.pixel_to_world(avg_x, avg_y)
                        method = 'wheel_seg'
                        contact_pt = ';'.join([f"{p[0]},{p[1]}" for p in contact_points])

                # Method 2: Keypoints
                if world_coord is None and keypoints is not None and v_idx < len(keypoints):
                    kp_contact = keypoint_extractor.get_ground_contact_point(
                        keypoints[v_idx], RECOGNITION_SIZE, DISPLAY_SIZE
                    )
                    if kp_contact:
                        world_coord = transformer.pixel_to_world(kp_contact[0], kp_contact[1])
                        method = 'keypoint'
                        contact_pt = f"{kp_contact[0]},{kp_contact[1]}"

                        wheel_associations[v_idx].append({
                            'contact_point': kp_contact,
                            'centroid': None,
                            'class_name': 'keypoint'
                        })

                # Method 3: BBox fallback
                if world_coord is None:
                    x, y, w, h = box
                    world_coord = transformer.pixel_to_world(x, y + h/2)
                    method = 'bbox_fallback'
                    contact_pt = f"{int(x)},{int(y + h/2)}"

                real_world_coords.append(world_coord if world_coord else (0, 0))
                methods_used.append(method)
                contact_points_used.append(contact_pt or '')

            # Calculate speeds
            speeds = speed_tracker.get_speeds(track_ids, real_world_coords, frame_count, fps)

            # Export data
            for v_idx, (track_id, world_coord, speed, method, contact_pt) in enumerate(
                zip(track_ids, real_world_coords, speeds, methods_used, contact_points_used)
            ):
                wheels = wheel_associations.get(v_idx, [])
                num_wheels = len([w for w in wheels if w.get('class_name') != 'keypoint'])

                exporter.write_row([
                    frame_count, track_id,
                    world_coord[0], world_coord[1],
                    speed, num_wheels, method, contact_pt
                ])

            # Draw annotations
            annotated_frame = draw_annotations(
                display_frame, scaled_boxes, track_ids,
                wheel_associations, speeds, methods_used
            )
        else:
            annotated_frame = display_frame
            methods_used = []

        # Frame info
        if methods_used:
            method_counts = {m: methods_used.count(m) for m in set(methods_used)}
            method_str = ' | '.join([f"{k}:{v}" for k, v in method_counts.items()])
        else:
            method_str = "No detections"

        cv2.putText(annotated_frame, f"Frame: {frame_count} | {method_str}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Green=Seg | Orange=Keypoint | Gray=BBox",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Display
        cv2.imshow("Segmentation Speed Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    exporter.close()

    print(f"\nProcessing complete!")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
