"""
Comparison Script: Object Detection vs Wheel Segmentation Approaches

This script runs both pipelines on the same video and compares:
1. Speed estimation accuracy
2. Position accuracy
3. Computational performance
4. Reliability (detection rate, fallback rate)

Usage:
    python compare_approaches.py [--video VIDEO_PATH] [--output OUTPUT_DIR]
"""

import cv2
import numpy as np
import time
import argparse
import json
import sys
import os
from collections import defaultdict

# Add parent directory to path for imports
COMPARISON_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(COMPARISON_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from utils import (
    preprocess_frame, load_calibration_data, rescale_coordinates,
    CoordinateTransformer, calculate_real_world_coordinates,
    SpeedTracker, CSVExporter
)

# Import from local comparison config, not project root config
import importlib.util
spec = importlib.util.spec_from_file_location("comparison_config", os.path.join(COMPARISON_DIR, "config.py"))
comparison_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison_config)

VIDEO_PATH = comparison_config.VIDEO_PATH
RECOGNITION_SIZE = comparison_config.RECOGNITION_SIZE
DISPLAY_SIZE = comparison_config.DISPLAY_SIZE
MAPPING_FILE = comparison_config.MAPPING_FILE
VEHICLE_MODEL_PATH = comparison_config.VEHICLE_MODEL_PATH
WHEEL_SEG_MODEL_PATH = comparison_config.WHEEL_SEG_MODEL_PATH
CALIBRATION_FILE = comparison_config.CALIBRATION_FILE

# Import 3DBB corner-based speed estimator
from comparison.speed_from_corners import CornerBasedSpeedEstimator


class WheelContactPointExtractor:
    """Extracts tire-ground contact points from wheel segmentation masks."""

    def get_contact_point(self, mask, class_id):
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

    def associate_wheels_to_vehicles(self, vehicle_boxes, wheel_data, frame_shape):
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
                x1, y1 = x - w/2 - 20, y - h/2 - 20
                x2, y2 = x + w/2 + 20, y + h/2 + 20

                cx, cy = centroid

                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                    if dist < best_score:
                        best_score = dist
                        best_vehicle_idx = v_idx

            if best_vehicle_idx is not None:
                associations[best_vehicle_idx].append(wheel)

        return associations


class KeypointExtractor:
    """Extracts contact points from vehicle keypoints."""

    def __init__(self, confidence_threshold=0.3):
        self.confidence_threshold = confidence_threshold

    def get_ground_contact(self, keypoints, recognition_size, display_size):
        """Get bottom-most high-confidence keypoint."""
        if keypoints is None or len(keypoints) == 0:
            return None

        scale_x = display_size[0] / recognition_size[0]
        scale_y = display_size[1] / recognition_size[1]

        valid_points = []
        for kp in keypoints:
            if kp[2] >= self.confidence_threshold:
                valid_points.append((int(kp[0] * scale_x), int(kp[1] * scale_y)))

        if not valid_points:
            return None

        # Return bottom-most point
        return max(valid_points, key=lambda p: p[1])


class ApproachComparator:
    """Compares object detection, keypoint, and segmentation approaches."""

    def __init__(self, video_path, mapping_file):
        self.video_path = video_path
        self.mapping_file = mapping_file

        # Load models
        print("Loading models...")
        self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        self.wheel_model = YOLO(WHEEL_SEG_MODEL_PATH)

        # Set device
        try:
            self.vehicle_model.to("cuda")
            self.wheel_model.to("cuda")
            self.device = "cuda"
        except:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Load calibration
        self.K, self.D, self.DIM = load_calibration_data(CALIBRATION_FILE)
        self.transformer = CoordinateTransformer(mapping_file)

        # Initialize trackers (separate for each approach)
        self.bbox_speed_tracker = SpeedTracker()
        self.keypoint_speed_tracker = SpeedTracker()
        self.seg_speed_tracker = SpeedTracker()

        # Keypoint extractor
        self.kp_extractor = KeypointExtractor(confidence_threshold=0.3)

        # Results storage
        self.bbox_results = defaultdict(list)
        self.keypoint_results = defaultdict(list)
        self.seg_results = defaultdict(list)
        self.corner3d_results = defaultdict(list)

        # 3DBB corner-based speed estimator
        self.corner_estimator = None
        self.corner3d_data_file = os.path.join(os.path.dirname(__file__), '3dbb_results.json')
        if os.path.exists(self.corner3d_data_file):
            with open(self.corner3d_data_file, 'r') as f:
                self.corner3d_data = json.load(f)
            self.corner_estimator = CornerBasedSpeedEstimator()
            print(f"Loaded 3DBB corner data: {len(self.corner3d_data)} frames")
        else:
            self.corner3d_data = []
            print("Warning: 3DBB results not found, corner-based approach disabled")

        # Metrics
        self.metrics = {
            'bbox': {'frames': 0, 'detections': 0, 'total_time': 0},
            'keypoint': {'frames': 0, 'detections': 0, 'valid_keypoints': 0, 'total_time': 0},
            'seg': {'frames': 0, 'detections': 0, 'wheels_detected': 0,
                   'seg_used': 0, 'fallback_used': 0, 'total_time': 0},
            'corner3d': {'frames': 0, 'detections': 0, 'vehicles_tracked': 0, 'total_time': 0}
        }

        # Calibration bounds (filter unrealistic coordinates)
        self.world_x_bounds = (-25, 25)  # meters
        self.world_y_bounds = (-10, 10)  # meters
        self.max_speed = 150  # km/h - filter outliers

    def is_valid_coordinate(self, world_coord):
        """Check if world coordinate is within calibrated area."""
        if world_coord is None:
            return False
        x, y = world_coord
        return (self.world_x_bounds[0] <= x <= self.world_x_bounds[1] and
                self.world_y_bounds[0] <= y <= self.world_y_bounds[1])

    def process_wheel_segmentation(self, wheel_results, frame_shape):
        """Process wheel segmentation results."""
        extractor = WheelContactPointExtractor()
        wheel_data = []

        if wheel_results[0].masks is None:
            return wheel_data

        masks = wheel_results[0].masks.data.cpu().numpy()
        boxes = wheel_results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)

        for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
            mask_resized = cv2.resize(mask, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
            contact_point = extractor.get_contact_point(mask_resized, class_id)
            centroid = extractor.get_wheel_centroid(mask_resized)

            wheel_data.append({
                'contact_point': contact_point,
                'centroid': centroid,
                'class_id': class_id
            })

        return wheel_data

    def calculate_vehicle_ground_position(self, wheel_associations):
        """Calculate ground position from wheel contact points."""
        contact_points = []
        for wheel in wheel_associations:
            cp = wheel.get('contact_point')
            if cp is not None:
                contact_points.append(cp)

        if not contact_points:
            return None

        avg_x = np.mean([p[0] for p in contact_points])
        avg_y = np.mean([p[1] for p in contact_points])

        return self.transformer.pixel_to_world(avg_x, avg_y)

    def run_comparison(self, max_frames=None, show_video=True):
        """Run both approaches and collect comparison data."""
        cap = cv2.VideoCapture(self.video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        print(f"\nProcessing {total_frames} frames at {fps} FPS")
        print("=" * 60)

        wheel_associator = VehicleWheelAssociator()
        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            # Progress
            if frame_count % 50 == 0:
                print(f"Processing frame {frame_count}/{total_frames}")

            # Preprocess
            recognition_frame, display_frame = preprocess_frame(
                frame, self.K, self.D, self.DIM, RECOGNITION_SIZE, DISPLAY_SIZE
            )

            # ========== BOUNDING BOX APPROACH ==========
            bbox_start = time.time()
            vehicle_results = self.vehicle_model.track(recognition_frame, persist=True)
            bbox_time = time.time() - bbox_start

            self.metrics['bbox']['frames'] += 1
            self.metrics['bbox']['total_time'] += bbox_time

            if vehicle_results[0].boxes.id is not None:
                boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
                track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

                scaled_boxes = [
                    rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                    for box in boxes
                ]

                # Calculate positions using bbox bottom-center
                bbox_world_coords = calculate_real_world_coordinates(
                    scaled_boxes, self.transformer
                )

                bbox_speeds = self.bbox_speed_tracker.get_speeds(
                    track_ids, bbox_world_coords, frame_count, fps
                )

                self.metrics['bbox']['detections'] += len(track_ids)

                for track_id, world_coord, speed in zip(track_ids, bbox_world_coords, bbox_speeds):
                    # Filter unrealistic values
                    if self.is_valid_coordinate(world_coord) and speed <= self.max_speed:
                        self.bbox_results[track_id].append({
                            'frame': frame_count,
                            'world_x': world_coord[0],
                            'world_y': world_coord[1],
                            'speed': speed
                        })

            # ========== KEYPOINT APPROACH ==========
            self.metrics['keypoint']['frames'] += 1

            if vehicle_results[0].boxes.id is not None and vehicle_results[0].keypoints is not None:
                keypoints_data = vehicle_results[0].keypoints.data.cpu().numpy()

                kp_world_coords = []
                for v_idx, (box, kps) in enumerate(zip(boxes, keypoints_data)):
                    contact_pt = self.kp_extractor.get_ground_contact(
                        kps, RECOGNITION_SIZE, DISPLAY_SIZE
                    )

                    if contact_pt:
                        world_coord = self.transformer.pixel_to_world(contact_pt[0], contact_pt[1])
                        self.metrics['keypoint']['valid_keypoints'] += 1
                    else:
                        # Fallback to bbox bottom center
                        scaled_box = scaled_boxes[v_idx]
                        x, y, w, h = scaled_box
                        world_coord = self.transformer.pixel_to_world(x, y + h/2)

                    kp_world_coords.append(world_coord)

                kp_speeds = self.keypoint_speed_tracker.get_speeds(
                    track_ids, kp_world_coords, frame_count, fps
                )

                self.metrics['keypoint']['detections'] += len(track_ids)

                for track_id, world_coord, speed in zip(track_ids, kp_world_coords, kp_speeds):
                    if self.is_valid_coordinate(world_coord) and speed <= self.max_speed:
                        self.keypoint_results[track_id].append({
                            'frame': frame_count,
                            'world_x': world_coord[0],
                            'world_y': world_coord[1],
                            'speed': speed
                        })

            # ========== SEGMENTATION APPROACH ==========
            seg_start = time.time()
            wheel_results = self.wheel_model(recognition_frame, verbose=False)
            wheel_data = self.process_wheel_segmentation(wheel_results, frame.shape)
            seg_time = time.time() - seg_start

            self.metrics['seg']['frames'] += 1
            self.metrics['seg']['total_time'] += seg_time
            self.metrics['seg']['wheels_detected'] += len(wheel_data)

            if vehicle_results[0].boxes.id is not None:
                wheel_associations = wheel_associator.associate_wheels_to_vehicles(
                    scaled_boxes, wheel_data, display_frame.shape
                )

                seg_world_coords = []
                for v_idx, box in enumerate(scaled_boxes):
                    associated_wheels = wheel_associations.get(v_idx, [])

                    if associated_wheels:
                        world_coord = self.calculate_vehicle_ground_position(associated_wheels)
                        self.metrics['seg']['seg_used'] += 1
                    else:
                        world_coord = None
                        self.metrics['seg']['fallback_used'] += 1

                    if world_coord is None:
                        x, y, w, h = box
                        world_coord = self.transformer.pixel_to_world(x, y + h/2)

                    seg_world_coords.append(world_coord)

                # Use separate tracker for segmentation
                seg_speeds = self.seg_speed_tracker.get_speeds(
                    track_ids, seg_world_coords, frame_count, fps
                )

                self.metrics['seg']['detections'] += len(track_ids)

                for track_id, world_coord, speed in zip(track_ids, seg_world_coords, seg_speeds):
                    if self.is_valid_coordinate(world_coord) and speed <= self.max_speed:
                        self.seg_results[track_id].append({
                            'frame': frame_count,
                            'world_x': world_coord[0],
                            'world_y': world_coord[1],
                            'speed': speed
                        })

            # ========== 3D BOUNDING BOX CORNER APPROACH ==========
            if self.corner_estimator and frame_count <= len(self.corner3d_data):
                corner3d_start = time.time()

                # Get frame data from pre-computed 3DBB results
                frame_idx = frame_count - 1  # 0-indexed
                if frame_idx < len(self.corner3d_data):
                    frame_data = self.corner3d_data[frame_idx]
                    detections = frame_data.get('detections', [])
                    frame_fps = frame_data.get('fps', fps)

                    # Process with corner-based estimator
                    corner_speeds = self.corner_estimator.process_frame(
                        frame_count, detections, frame_fps
                    )

                    self.metrics['corner3d']['frames'] += 1
                    self.metrics['corner3d']['detections'] += len(detections)

                    for speed_data in corner_speeds:
                        track_id = speed_data['track_id']
                        speed = speed_data['speed_kmh']
                        centroid = speed_data.get('centroid', {})

                        # Store results (using camera frame coordinates)
                        if speed <= self.max_speed:
                            self.corner3d_results[track_id].append({
                                'frame': frame_count,
                                'world_x': centroid.get('x', 0),
                                'world_y': centroid.get('y', 0),
                                'speed': speed,
                                'depth': speed_data.get('depth', 0),
                                'corners': speed_data.get('corners', [])
                            })

                corner3d_time = time.time() - corner3d_start
                self.metrics['corner3d']['total_time'] += corner3d_time

            # Visualization (side by side)
            if show_video:
                vis_frame = self.create_comparison_visualization(
                    display_frame, vehicle_results, wheel_data,
                    wheel_associations if vehicle_results[0].boxes.id is not None else {},
                    frame_count
                )
                cv2.imshow("Comparison: BBox (left) vs Segmentation (right)", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        return self.generate_comparison_report()

    def create_comparison_visualization(self, frame, vehicle_results, wheel_data,
                                       wheel_associations, frame_count):
        """Create side-by-side visualization."""
        h, w = frame.shape[:2]
        vis = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Left: BBox approach
        left_frame = frame.copy()
        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()
            scaled_boxes = [
                rescale_coordinates(box.tolist(), RECOGNITION_SIZE, DISPLAY_SIZE)
                for box in boxes
            ]

            for box, track_id in zip(scaled_boxes, track_ids):
                x, y, bw, bh = box
                cv2.rectangle(left_frame,
                            (int(x - bw/2), int(y - bh/2)),
                            (int(x + bw/2), int(y + bh/2)),
                            (0, 255, 0), 2)
                # Draw bottom center point
                cv2.circle(left_frame, (int(x), int(y + bh/2)), 8, (0, 255, 255), -1)
                cv2.putText(left_frame, f"ID:{track_id}",
                           (int(x - bw/2), int(y - bh/2 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(left_frame, "BBox Approach", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Right: Segmentation approach
        right_frame = frame.copy()
        if vehicle_results[0].boxes.id is not None:
            for v_idx, (box, track_id) in enumerate(zip(scaled_boxes, track_ids)):
                x, y, bw, bh = box
                cv2.rectangle(right_frame,
                            (int(x - bw/2), int(y - bh/2)),
                            (int(x + bw/2), int(y + bh/2)),
                            (0, 255, 0), 2)

                # Draw wheel contact points
                wheels = wheel_associations.get(v_idx, [])
                for wheel in wheels:
                    cp = wheel.get('contact_point')
                    if cp:
                        cv2.circle(right_frame, cp, 8, (0, 0, 255), -1)
                        cv2.circle(right_frame, cp, 4, (0, 255, 255), -1)

                cv2.putText(right_frame, f"ID:{track_id} W:{len(wheels)}",
                           (int(x - bw/2), int(y - bh/2 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(right_frame, "Segmentation Approach", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        vis[:, :w] = left_frame
        vis[:, w:] = right_frame

        # Add frame counter
        cv2.putText(vis, f"Frame: {frame_count}", (w - 100, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def generate_comparison_report(self):
        """Generate detailed comparison metrics for all four approaches."""
        print("\n" + "=" * 70)
        print("COMPARISON REPORT: BBox vs Keypoint vs Segmentation vs 3DBB Corner")
        print("=" * 70)

        # Performance metrics
        print("\n1. PERFORMANCE METRICS")
        print("-" * 50)
        bbox_fps = self.metrics['bbox']['frames'] / self.metrics['bbox']['total_time'] if self.metrics['bbox']['total_time'] > 0 else 0
        seg_fps = self.metrics['seg']['frames'] / self.metrics['seg']['total_time'] if self.metrics['seg']['total_time'] > 0 else 0
        corner3d_fps = self.metrics['corner3d']['frames'] / self.metrics['corner3d']['total_time'] if self.metrics['corner3d']['total_time'] > 0 else 0
        print(f"BBox Approach:         {bbox_fps:.2f} FPS")
        print(f"Segmentation Approach: {seg_fps:.2f} FPS")
        print(f"3DBB Corner Approach:  {corner3d_fps:.2f} FPS (post-processing only)")

        # Detection metrics
        print("\n2. DETECTION METRICS")
        print("-" * 50)
        print(f"BBox Total Detections:     {self.metrics['bbox']['detections']}")
        print(f"Keypoint Detections:       {self.metrics['keypoint']['detections']}")
        print(f"  Valid Keypoints Found:   {self.metrics['keypoint']['valid_keypoints']}")
        print(f"Seg Total Detections:      {self.metrics['seg']['detections']}")
        print(f"  Wheels Detected:         {self.metrics['seg']['wheels_detected']}")
        print(f"  Segmentation Used:       {self.metrics['seg']['seg_used']} times")
        print(f"  Fallback to BBox:        {self.metrics['seg']['fallback_used']} times")

        print(f"3DBB Corner Detections:    {self.metrics['corner3d']['detections']}")
        print(f"  Frames Processed:        {self.metrics['corner3d']['frames']}")

        seg_rate = self.metrics['seg']['seg_used'] / (self.metrics['seg']['seg_used'] + self.metrics['seg']['fallback_used']) * 100 if (self.metrics['seg']['seg_used'] + self.metrics['seg']['fallback_used']) > 0 else 0
        kp_rate = self.metrics['keypoint']['valid_keypoints'] / self.metrics['keypoint']['detections'] * 100 if self.metrics['keypoint']['detections'] > 0 else 0
        print(f"Wheel Seg Detection Rate:  {seg_rate:.1f}%")
        print(f"Keypoint Detection Rate:   {kp_rate:.1f}%")

        # Compare BBox vs Keypoint
        print("\n3. BBOX vs KEYPOINT COMPARISON")
        print("-" * 50)

        bbox_kp_speed_diffs = []
        bbox_kp_pos_diffs = []

        for track_id in self.bbox_results:
            if track_id in self.keypoint_results:
                bbox_data = {d['frame']: d for d in self.bbox_results[track_id]}
                kp_data = {d['frame']: d for d in self.keypoint_results[track_id]}
                common_frames = set(bbox_data.keys()) & set(kp_data.keys())

                for frame in common_frames:
                    bbox_speed = bbox_data[frame]['speed']
                    kp_speed = kp_data[frame]['speed']
                    if bbox_speed > 0 and kp_speed > 0:
                        bbox_kp_speed_diffs.append(abs(bbox_speed - kp_speed))

                    bbox_pos = (bbox_data[frame]['world_x'], bbox_data[frame]['world_y'])
                    kp_pos = (kp_data[frame]['world_x'], kp_data[frame]['world_y'])
                    pos_diff = np.sqrt((bbox_pos[0] - kp_pos[0])**2 + (bbox_pos[1] - kp_pos[1])**2)
                    bbox_kp_pos_diffs.append(pos_diff)

        if bbox_kp_speed_diffs:
            print(f"Speed Difference (BBox vs Keypoint):")
            print(f"  Mean: {np.mean(bbox_kp_speed_diffs):.2f} km/h")
            print(f"  Std:  {np.std(bbox_kp_speed_diffs):.2f} km/h")
            print(f"  Max:  {np.max(bbox_kp_speed_diffs):.2f} km/h")
        if bbox_kp_pos_diffs:
            print(f"Position Difference (BBox vs Keypoint):")
            print(f"  Mean: {np.mean(bbox_kp_pos_diffs):.4f} m")
            print(f"  Std:  {np.std(bbox_kp_pos_diffs):.4f} m")

        # Compare BBox vs Segmentation
        print("\n4. BBOX vs SEGMENTATION COMPARISON")
        print("-" * 50)

        bbox_seg_speed_diffs = []
        bbox_seg_pos_diffs = []

        for track_id in self.bbox_results:
            if track_id in self.seg_results:
                bbox_data = {d['frame']: d for d in self.bbox_results[track_id]}
                seg_data = {d['frame']: d for d in self.seg_results[track_id]}
                common_frames = set(bbox_data.keys()) & set(seg_data.keys())

                for frame in common_frames:
                    bbox_speed = bbox_data[frame]['speed']
                    seg_speed = seg_data[frame]['speed']
                    if bbox_speed > 0 and seg_speed > 0:
                        bbox_seg_speed_diffs.append(abs(bbox_speed - seg_speed))

                    bbox_pos = (bbox_data[frame]['world_x'], bbox_data[frame]['world_y'])
                    seg_pos = (seg_data[frame]['world_x'], seg_data[frame]['world_y'])
                    pos_diff = np.sqrt((bbox_pos[0] - seg_pos[0])**2 + (bbox_pos[1] - seg_pos[1])**2)
                    bbox_seg_pos_diffs.append(pos_diff)

        if bbox_seg_speed_diffs:
            print(f"Speed Difference (BBox vs Seg):")
            print(f"  Mean: {np.mean(bbox_seg_speed_diffs):.2f} km/h")
            print(f"  Std:  {np.std(bbox_seg_speed_diffs):.2f} km/h")
        if bbox_seg_pos_diffs:
            print(f"Position Difference (BBox vs Seg):")
            print(f"  Mean: {np.mean(bbox_seg_pos_diffs):.4f} m")

        # 3DBB Corner Analysis
        print("\n5. 3D BOUNDING BOX CORNER APPROACH ANALYSIS")
        print("-" * 50)

        if self.corner3d_results:
            corner3d_all_speeds = []
            for track_id, results in self.corner3d_results.items():
                speeds = [r['speed'] for r in results if r['speed'] > 0]
                corner3d_all_speeds.extend(speeds)

            if corner3d_all_speeds:
                print(f"3DBB Corner Speed Statistics:")
                print(f"  Total Measurements:  {len(corner3d_all_speeds)}")
                print(f"  Vehicles Tracked:    {len(self.corner3d_results)}")
                print(f"  Mean Speed:          {np.mean(corner3d_all_speeds):.2f} km/h")
                print(f"  Median Speed:        {np.median(corner3d_all_speeds):.2f} km/h")
                print(f"  Std Deviation:       {np.std(corner3d_all_speeds):.2f} km/h")
                print(f"  Range:               {np.min(corner3d_all_speeds):.2f} - {np.max(corner3d_all_speeds):.2f} km/h")

                # Top vehicles by sample count
                print("\n  Per-vehicle breakdown (top 5 by samples):")
                vehicle_stats = []
                for track_id, results in self.corner3d_results.items():
                    speeds = [r['speed'] for r in results if r['speed'] > 0]
                    if speeds:
                        vehicle_stats.append((track_id, len(speeds), np.mean(speeds), np.median(speeds)))

                vehicle_stats.sort(key=lambda x: x[1], reverse=True)
                for track_id, count, mean, median in vehicle_stats[:5]:
                    print(f"    {track_id}: {count} samples, mean={mean:.1f}, median={median:.1f} km/h")
        else:
            print("No 3DBB corner data available")

        # Per-vehicle analysis
        print("\n6. PER-VEHICLE SPEED ANALYSIS (Top 5 vehicles with valid data)")
        print("-" * 50)

        valid_vehicles = []
        for track_id in self.bbox_results:
            bbox_speeds = [d['speed'] for d in self.bbox_results[track_id] if d['speed'] > 0]
            if len(bbox_speeds) >= 5:  # At least 5 speed readings
                valid_vehicles.append((track_id, np.mean(bbox_speeds), len(bbox_speeds)))

        valid_vehicles.sort(key=lambda x: x[2], reverse=True)  # Sort by number of readings

        for track_id, bbox_avg, count in valid_vehicles[:5]:
            print(f"\nVehicle ID {track_id} ({count} readings):")
            bbox_speeds = [d['speed'] for d in self.bbox_results[track_id] if d['speed'] > 0]
            print(f"  BBox Avg Speed:     {np.mean(bbox_speeds):.2f} km/h")

            if track_id in self.keypoint_results:
                kp_speeds = [d['speed'] for d in self.keypoint_results[track_id] if d['speed'] > 0]
                if kp_speeds:
                    print(f"  Keypoint Avg Speed: {np.mean(kp_speeds):.2f} km/h")
                    print(f"  BBox vs KP Diff:    {abs(np.mean(bbox_speeds) - np.mean(kp_speeds)):.2f} km/h")

            if track_id in self.seg_results:
                seg_speeds = [d['speed'] for d in self.seg_results[track_id] if d['speed'] > 0]
                if seg_speeds:
                    print(f"  Seg Avg Speed:      {np.mean(seg_speeds):.2f} km/h")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 70)

        print("\nApproach Comparison:")
        print(f"  1. BBox (Baseline):    Always available, uses bottom-center of box")
        print(f"  2. Keypoint:           Uses wheel keypoints from pose model ({kp_rate:.1f}% valid)")
        print(f"  3. Segmentation:       Uses wheel masks ({seg_rate:.1f}% detected)")
        print(f"  4. 3DBB Corners:       Uses 4 bottom corners of 3D bounding box")

        if bbox_kp_pos_diffs:
            avg_kp_diff = np.mean(bbox_kp_pos_diffs)
            print(f"\nKeypoint vs BBox position difference: {avg_kp_diff:.4f}m average")
            if avg_kp_diff < 0.5:
                print("  -> Keypoints provide similar localization to BBox")
            else:
                print("  -> Keypoints provide different (potentially more accurate) localization")

        # Calculate 3DBB corner stats for report
        corner3d_stats = {}
        if self.corner3d_results:
            corner3d_all_speeds = []
            for track_id, results in self.corner3d_results.items():
                speeds = [r['speed'] for r in results if r['speed'] > 0]
                corner3d_all_speeds.extend(speeds)

            if corner3d_all_speeds:
                corner3d_stats = {
                    'total_measurements': len(corner3d_all_speeds),
                    'vehicles_tracked': len(self.corner3d_results),
                    'mean_speed': float(np.mean(corner3d_all_speeds)),
                    'median_speed': float(np.median(corner3d_all_speeds)),
                    'std_speed': float(np.std(corner3d_all_speeds)),
                    'min_speed': float(np.min(corner3d_all_speeds)),
                    'max_speed': float(np.max(corner3d_all_speeds))
                }

        report = {
            'performance': {
                'bbox_fps': bbox_fps,
                'seg_fps': seg_fps,
                'corner3d_fps': corner3d_fps
            },
            'detection': {
                'bbox_detections': self.metrics['bbox']['detections'],
                'keypoint_detections': self.metrics['keypoint']['detections'],
                'keypoint_valid_rate': kp_rate,
                'seg_detections': self.metrics['seg']['detections'],
                'wheels_detected': self.metrics['seg']['wheels_detected'],
                'seg_usage_rate': seg_rate,
                'corner3d_detections': self.metrics['corner3d']['detections'],
                'corner3d_frames': self.metrics['corner3d']['frames']
            },
            'accuracy': {
                'bbox_vs_keypoint': {
                    'mean_speed_diff': float(np.mean(bbox_kp_speed_diffs)) if bbox_kp_speed_diffs else None,
                    'mean_position_diff': float(np.mean(bbox_kp_pos_diffs)) if bbox_kp_pos_diffs else None
                },
                'bbox_vs_seg': {
                    'mean_speed_diff': float(np.mean(bbox_seg_speed_diffs)) if bbox_seg_speed_diffs else None,
                    'mean_position_diff': float(np.mean(bbox_seg_pos_diffs)) if bbox_seg_pos_diffs else None
                }
            },
            'corner3d_analysis': corner3d_stats
        }

        # Save report
        with open('comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("\nReport saved to: comparison_report.json")

        return report


def main():
    parser = argparse.ArgumentParser(description='Compare detection approaches')
    parser.add_argument('--video', type=str, default=VIDEO_PATH,
                       help='Path to video file')
    parser.add_argument('--mapping', type=str, default=MAPPING_FILE,
                       help='Path to coordinate mapping file')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    args = parser.parse_args()

    comparator = ApproachComparator(args.video, args.mapping)
    report = comparator.run_comparison(
        max_frames=args.max_frames,
        show_video=not args.no_display
    )


if __name__ == "__main__":
    main()
