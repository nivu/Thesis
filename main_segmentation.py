"""
Segmentation & Keypoint-Based Vehicle Speed Estimation Pipeline

This pipeline uses multiple methods to detect tire-ground contact points
for more accurate vehicle localization and speed calculation.

Methods (in priority order):
1. Wheel Segmentation: Extract contact points from segmentation masks
2. Keypoint Detection: Use wheel keypoints from pose model (best.pt has 10 keypoints)
3. Bounding Box Fallback: Use bottom-center of vehicle bounding box

The pose model (best.pt) detects:
- Vehicle classes: vehicle_2_wheels, vehicle_4_wheels, vehicle_6_wheels, etc.
- 10 keypoints per vehicle representing wheel positions

This approach is compared against the bounding-box-only method in main.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates
from config import VIDEO_PATH, RECOGNITION_SIZE, DISPLAY_SIZE, MAPPING_FILE
from data_export import CSVExporter
from coordinate_transformer import (
    CoordinateTransformer,
    calculate_real_world_coordinates,
    calculate_point_real_world
)
from speed_utils import SpeedTracker


class KeypointContactPointExtractor:
    """Extracts tire-ground contact points from vehicle keypoints."""

    def __init__(self, confidence_threshold=0.3):
        """
        Args:
            confidence_threshold: Minimum confidence for a keypoint to be considered valid
        """
        self.confidence_threshold = confidence_threshold

    def get_contact_points_from_keypoints(self, keypoints, box, recognition_size, display_size):
        """
        Extract tire-ground contact points from vehicle keypoints.

        The pose model provides 10 keypoints per vehicle representing wheel positions.
        We filter by confidence and find the bottom-most visible points.

        Args:
            keypoints: Array of shape (10, 3) with [x, y, confidence] for each keypoint
            box: Bounding box [x_center, y_center, w, h] in recognition size
            recognition_size: Size of recognition frame
            display_size: Size of display frame

        Returns:
            List of (x, y) contact points in display coordinates
        """
        if keypoints is None or len(keypoints) == 0:
            return []

        contact_points = []
        scale_x = display_size[0] / recognition_size[0]
        scale_y = display_size[1] / recognition_size[1]

        for kp in keypoints:
            x, y, conf = kp[0], kp[1], kp[2]

            if conf >= self.confidence_threshold:
                # Scale to display size
                display_x = int(x * scale_x)
                display_y = int(y * scale_y)
                contact_points.append((display_x, display_y, conf))

        return contact_points

    def get_ground_contact_point(self, keypoints, box, recognition_size, display_size):
        """
        Get the best ground contact point from keypoints.

        Uses the bottom-most high-confidence keypoint as the contact point.

        Returns:
            Tuple of (x, y) or None if no valid keypoints
        """
        contact_points = self.get_contact_points_from_keypoints(
            keypoints, box, recognition_size, display_size
        )

        if not contact_points:
            return None

        # Find bottom-most point (highest y value)
        bottom_point = max(contact_points, key=lambda p: p[1])
        return (bottom_point[0], bottom_point[1])


class WheelContactPointExtractor:
    """Extracts tire-ground contact points from wheel segmentation masks."""

    def __init__(self):
        self.wheel_classes = ['backwheel', 'frontwheel', 'middlewheel']

    def get_contact_point(self, mask, class_id):
        """
        Extract the tire-ground contact point from a segmentation mask.
        The contact point is the bottom-center of the wheel mask.

        Args:
            mask: Binary segmentation mask (numpy array)
            class_id: Class ID of the wheel (0=backwheel, 1=frontwheel, 2=middlewheel)

        Returns:
            Tuple of (x, y) contact point coordinates, or None if invalid mask
        """
        if mask is None or mask.sum() == 0:
            return None

        # Find the contours of the mask
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Ensure mask is uint8
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 10:  # Minimum area threshold
            return None

        # Find the bottom-most points of the contour
        bottom_points = []
        points = largest_contour.reshape(-1, 2)

        # Get the maximum y-coordinate (bottom of wheel)
        max_y = points[:, 1].max()

        # Find all points near the bottom (within 5 pixels)
        for point in points:
            if point[1] >= max_y - 5:
                bottom_points.append(point)

        if not bottom_points:
            return None

        # Calculate the center of the bottom points (contact region)
        bottom_points = np.array(bottom_points)
        contact_x = int(np.mean(bottom_points[:, 0]))
        contact_y = int(np.max(bottom_points[:, 1]))

        return (contact_x, contact_y)

    def get_wheel_centroid(self, mask):
        """Get the centroid of a wheel mask for association purposes."""
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

    def __init__(self, iou_threshold=0.3, distance_threshold=200):
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold

    def associate_wheels_to_vehicles(self, vehicle_boxes, wheel_data, frame_shape):
        """
        Associate wheel detections with vehicle bounding boxes.

        Args:
            vehicle_boxes: List of vehicle bounding boxes [x_center, y_center, w, h]
            wheel_data: List of dicts with 'contact_point', 'centroid', 'class_id', 'mask'
            frame_shape: Shape of the frame (height, width)

        Returns:
            Dict mapping vehicle index to list of associated wheel data
        """
        associations = {i: [] for i in range(len(vehicle_boxes))}

        for wheel in wheel_data:
            centroid = wheel.get('centroid')
            if centroid is None:
                continue

            best_vehicle_idx = None
            best_score = float('inf')

            for v_idx, box in enumerate(vehicle_boxes):
                x, y, w, h = box

                # Check if wheel centroid is inside or near vehicle bounding box
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2

                # Expand bounding box slightly for better association
                margin = 20
                x1, y1 = x1 - margin, y1 - margin
                x2, y2 = x2 + margin, y2 + margin

                cx, cy = centroid

                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    # Calculate distance from wheel to vehicle center
                    dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < best_score:
                        best_score = dist
                        best_vehicle_idx = v_idx

            if best_vehicle_idx is not None:
                associations[best_vehicle_idx].append(wheel)

        return associations


def calculate_vehicle_ground_position(wheel_associations, transformer):
    """
    Calculate the ground position of a vehicle based on its associated wheels.

    Uses the average of all visible wheel contact points to determine
    the vehicle's position on the ground plane.

    Args:
        wheel_associations: List of wheel data dicts with 'contact_point'
        transformer: CoordinateTransformer instance

    Returns:
        Tuple of (world_x, world_y) or None if no valid contact points
    """
    contact_points = []

    for wheel in wheel_associations:
        cp = wheel.get('contact_point')
        if cp is not None:
            contact_points.append(cp)

    if not contact_points:
        return None

    # Calculate average contact point
    avg_x = np.mean([p[0] for p in contact_points])
    avg_y = np.mean([p[1] for p in contact_points])

    # Transform to real-world coordinates
    return transformer.pixel_to_world(avg_x, avg_y)


def draw_segmentation_annotations(image, vehicle_boxes, track_ids, wheel_associations, speeds, methods_used=None):
    """Draw annotations including wheel segmentation and keypoint results."""
    annotated = image.copy()

    if methods_used is None:
        methods_used = ['unknown'] * len(vehicle_boxes)

    for v_idx, (box, track_id, speed) in enumerate(zip(vehicle_boxes, track_ids, speeds)):
        x, y, w, h = box
        method = methods_used[v_idx] if v_idx < len(methods_used) else 'unknown'

        # Color coding by method
        if method == 'wheel_seg':
            box_color = (0, 255, 0)  # Green for segmentation
        elif method == 'keypoint':
            box_color = (255, 165, 0)  # Orange for keypoints
        else:
            box_color = (128, 128, 128)  # Gray for bbox fallback

        # Draw vehicle bounding box
        cv2.rectangle(annotated,
                     (int(x - w/2), int(y - h/2)),
                     (int(x + w/2), int(y + h/2)),
                     box_color, 2)

        # Draw track ID, speed and method
        label = f"ID:{track_id} {speed:.1f}km/h [{method[:3]}]"
        cv2.putText(annotated, label,
                   (int(x - w/2), int(y - h/2 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Draw wheel contact points
        wheels = wheel_associations.get(v_idx, [])
        for wheel in wheels:
            cp = wheel.get('contact_point')
            class_name = wheel.get('class_name', '')

            if cp is not None:
                if class_name == 'keypoint':
                    # Draw keypoint contact (orange)
                    cv2.circle(annotated, cp, 8, (0, 165, 255), -1)
                    cv2.circle(annotated, cp, 4, (255, 255, 255), -1)
                else:
                    # Draw segmentation contact (red)
                    cv2.circle(annotated, cp, 8, (0, 0, 255), -1)
                    cv2.circle(annotated, cp, 4, (0, 255, 255), -1)

            centroid = wheel.get('centroid')
            if centroid is not None:
                # Draw wheel centroid (blue)
                cv2.circle(annotated, centroid, 5, (255, 0, 0), -1)

    return annotated


def process_wheel_segmentation(wheel_results, frame_shape, recognition_size, display_size):
    """
    Process wheel segmentation results to extract contact points.

    Args:
        wheel_results: YOLO segmentation results
        frame_shape: Original frame shape
        recognition_size: Size used for recognition
        display_size: Size used for display

    Returns:
        List of wheel data dicts
    """
    extractor = WheelContactPointExtractor()
    wheel_data = []

    if wheel_results[0].masks is None:
        return wheel_data

    masks = wheel_results[0].masks.data.cpu().numpy()
    boxes = wheel_results[0].boxes
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
        # Resize mask to display size
        mask_resized = cv2.resize(mask, display_size, interpolation=cv2.INTER_NEAREST)

        contact_point = extractor.get_contact_point(mask_resized, class_id)
        centroid = extractor.get_wheel_centroid(mask_resized)

        wheel_data.append({
            'contact_point': contact_point,
            'centroid': centroid,
            'class_id': class_id,
            'class_name': ['backwheel', 'frontwheel', 'middlewheel'][class_id]
        })

    return wheel_data


def main():
    # Load models
    print("Loading models...")
    vehicle_model = YOLO("best.pt")  # Vehicle detection with keypoints
    wheel_model = YOLO("runs/segment/wheel_seg/weights/best.pt")  # Wheel segmentation

    # Set device
    device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
    try:
        vehicle_model.to(device)
        wheel_model.to(device)
    except:
        device = "cpu"
        print("CUDA not available, using CPU")

    print(f"Using device: {device}")

    # Load calibration data
    K, D, DIM = load_calibration_data()
    if K is None or D is None or DIM is None:
        print("Failed to load calibration data. Exiting.")
        return

    # Initialize components
    transformer = CoordinateTransformer(MAPPING_FILE)
    speed_tracker = SpeedTracker()
    wheel_associator = VehicleWheelAssociator()
    keypoint_extractor = KeypointContactPointExtractor(confidence_threshold=0.3)

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"Warning: Invalid FPS ({fps}), defaulting to 30")
        fps = 30.0

    # CSV export setup
    seg_header = ['frame', 'id', 'world_x', 'world_y', 'speed_kmh',
                  'num_wheels', 'method', 'contact_points']
    seg_exporter = CSVExporter('segmentation_results.csv', seg_header)

    frame_count = 0
    print(f"\nProcessing video: {VIDEO_PATH}")
    print("Press 'q' to quit\n")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Preprocess frame
        recognition_frame, display_frame = preprocess_frame(
            frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        # Run vehicle detection with tracking
        vehicle_results = vehicle_model.track(recognition_frame, persist=True)

        # Run wheel segmentation
        wheel_results = wheel_model(recognition_frame, verbose=False)

        # Process wheel segmentation results
        wheel_data = process_wheel_segmentation(
            wheel_results, frame.shape, RECOGNITION_SIZE, DISPLAY_SIZE
        )

        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

            # Get keypoints from pose model (if available)
            keypoints = None
            if vehicle_results[0].keypoints is not None:
                keypoints = vehicle_results[0].keypoints.data.cpu().numpy()

            # Rescale boxes to display size
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            # Associate wheels with vehicles (from segmentation)
            wheel_associations = wheel_associator.associate_wheels_to_vehicles(
                scaled_boxes, wheel_data, display_frame.shape
            )

            # Calculate positions and speeds for each vehicle
            # Priority: 1. Segmentation, 2. Keypoints, 3. BBox
            real_world_coords = []
            methods_used = []
            contact_points_used = []

            for v_idx, box in enumerate(scaled_boxes):
                associated_wheels = wheel_associations.get(v_idx, [])
                method = 'bbox_fallback'
                contact_pt = None

                if associated_wheels:
                    # Method 1: Use wheel segmentation contact points
                    world_coord = calculate_vehicle_ground_position(
                        associated_wheels, transformer
                    )
                    if world_coord:
                        method = 'wheel_seg'
                        contact_pt = ';'.join([
                            f"{w['contact_point'][0]},{w['contact_point'][1]}"
                            for w in associated_wheels if w.get('contact_point')
                        ])
                else:
                    world_coord = None

                if world_coord is None and keypoints is not None and v_idx < len(keypoints):
                    # Method 2: Use keypoints from pose model
                    kp_contact = keypoint_extractor.get_ground_contact_point(
                        keypoints[v_idx], boxes[v_idx], RECOGNITION_SIZE, DISPLAY_SIZE
                    )
                    if kp_contact:
                        world_coord = transformer.pixel_to_world(kp_contact[0], kp_contact[1])
                        method = 'keypoint'
                        contact_pt = f"{kp_contact[0]},{kp_contact[1]}"

                        # Add keypoint data to wheel_associations for visualization
                        wheel_associations[v_idx].append({
                            'contact_point': kp_contact,
                            'centroid': None,
                            'class_name': 'keypoint'
                        })

                if world_coord is None:
                    # Method 3: Fallback to bounding box bottom-center
                    x, y, w, h = box
                    bottom_center_x = x
                    bottom_center_y = y + h / 2
                    world_coord = transformer.pixel_to_world(bottom_center_x, bottom_center_y)
                    method = 'bbox_fallback'
                    contact_pt = f"{int(bottom_center_x)},{int(bottom_center_y)}"

                real_world_coords.append(world_coord if world_coord else (0, 0))
                methods_used.append(method)
                contact_points_used.append(contact_pt or '')

            # Calculate speeds
            speeds = speed_tracker.get_speeds(track_ids, real_world_coords, frame_count, fps)

            # Export data
            for v_idx, (box, track_id, world_coord, speed, method, contact_pt) in enumerate(
                zip(scaled_boxes, track_ids, real_world_coords, speeds, methods_used, contact_points_used)
            ):
                wheels = wheel_associations.get(v_idx, [])
                num_wheels = len([w for w in wheels if w.get('class_name') != 'keypoint'])

                seg_exporter.write_row([
                    frame_count, track_id,
                    world_coord[0], world_coord[1],
                    speed, num_wheels, method, contact_pt
                ])

            # Draw annotations
            annotated_frame = draw_segmentation_annotations(
                display_frame, scaled_boxes, track_ids,
                wheel_associations, speeds, methods_used
            )
        else:
            annotated_frame = display_frame
            speeds = []
            methods_used = []

        # Add frame info overlay
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
        cv2.imshow("Wheel Segmentation Speed Estimation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    seg_exporter.close()

    print(f"\nProcessing complete!")
    print(f"Results saved to: segmentation_results.csv")


if __name__ == "__main__":
    main()
