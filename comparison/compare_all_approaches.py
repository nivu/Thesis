"""
Comprehensive Comparison: All Vehicle Localization Approaches

This script compares four approaches:
1. BBox (Baseline): Bottom-center of bounding box
2. Keypoint: Wheel keypoints from pose model
3. Segmentation: Wheel segmentation masks
4. 3D BBox (YOLOx3D): Monocular 3D detection with depth estimation

Usage:
    python compare_all_approaches.py [--video VIDEO_PATH] [--max-frames N]
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

# Import utilities
try:
    from utils import (
        preprocess_frame, load_calibration_data, rescale_coordinates,
        CoordinateTransformer, calculate_real_world_coordinates,
        SpeedTracker, CSVExporter
    )
except ImportError as e:
    print(f"Warning: Could not import all utils: {e}")
    # Define fallback classes
    class SpeedTracker:
        def __init__(self):
            self.positions = {}
            self.last_frame = {}

        def get_speeds(self, track_ids, world_coords, frame, fps):
            speeds = []
            for tid, coord in zip(track_ids, world_coords):
                if coord is None:
                    speeds.append(0)
                    continue

                if tid in self.positions and tid in self.last_frame:
                    prev_coord = self.positions[tid]
                    frame_diff = frame - self.last_frame[tid]
                    if frame_diff > 0:
                        dist = np.sqrt((coord[0] - prev_coord[0])**2 + (coord[1] - prev_coord[1])**2)
                        time_diff = frame_diff / fps
                        speed = (dist / time_diff) * 3.6  # m/s to km/h
                        speeds.append(min(speed, 200))  # Cap at 200 km/h
                    else:
                        speeds.append(0)
                else:
                    speeds.append(0)

                self.positions[tid] = coord
                self.last_frame[tid] = frame

            return speeds

# Import local config first (before 3DBB which adds YOLOx3D to path)
import importlib.util
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py")
spec = importlib.util.spec_from_file_location("comparison_config", CONFIG_PATH)
comparison_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison_config)

VIDEO_PATH = comparison_config.VIDEO_PATH
RECOGNITION_SIZE = comparison_config.RECOGNITION_SIZE
DISPLAY_SIZE = comparison_config.DISPLAY_SIZE
MAPPING_FILE = comparison_config.MAPPING_FILE
VEHICLE_MODEL_PATH = comparison_config.VEHICLE_MODEL_PATH
WHEEL_SEG_MODEL_PATH = comparison_config.WHEEL_SEG_MODEL_PATH
CALIBRATION_FILE = comparison_config.CALIBRATION_FILE

# 3DBB results can be loaded from pre-computed JSON
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DDDBB_RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3dbb_results.json")
HAS_3DBB = os.path.exists(DDDBB_RESULTS_PATH)
if HAS_3DBB:
    print(f"3D BBox results found: {DDDBB_RESULTS_PATH}")
else:
    print(f"Note: Pre-run 3DBB detection using 3dbb_pipeline/run_detection.py")


class AllApproachComparator:
    """Compares all vehicle localization approaches including 3D bbox."""

    def __init__(self, video_path, mapping_file):
        self.video_path = video_path
        self.mapping_file = mapping_file

        # Load models
        print("Loading models...")

        # Vehicle detection model (with keypoints)
        self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
        print(f"  Loaded vehicle model: {VEHICLE_MODEL_PATH}")

        # Wheel segmentation model
        if os.path.exists(WHEEL_SEG_MODEL_PATH):
            self.wheel_model = YOLO(WHEEL_SEG_MODEL_PATH)
            self.has_wheel_seg = True
            print(f"  Loaded wheel seg model: {WHEEL_SEG_MODEL_PATH}")
        else:
            self.wheel_model = None
            self.has_wheel_seg = False
            print(f"  Warning: Wheel seg model not found: {WHEEL_SEG_MODEL_PATH}")

        # 3D BBox results (loaded from pre-computed JSON)
        self.dddbb_data = None
        if HAS_3DBB:
            try:
                with open(DDDBB_RESULTS_PATH, 'r') as f:
                    self.dddbb_data = json.load(f)
                self.has_3dbb = True
                print(f"  Loaded 3D BBox results: {len(self.dddbb_data)} frames")
            except Exception as e:
                print(f"  Warning: Could not load 3DBB results: {e}")
                self.has_3dbb = False
        else:
            self.has_3dbb = False

        # Load calibration
        self.K, self.D, self.DIM = self._load_calibration()

        # Load coordinate transformer
        self.transformer = self._load_transformer()

        # Initialize speed trackers for each approach
        self.speed_trackers = {
            'bbox': SpeedTracker(),
            'keypoint': SpeedTracker(),
            'seg': SpeedTracker(),
            '3dbb': SpeedTracker()
        }

        # Results storage
        self.results = {
            'bbox': defaultdict(list),
            'keypoint': defaultdict(list),
            'seg': defaultdict(list),
            '3dbb': defaultdict(list)
        }

        # Metrics
        self.metrics = {
            'bbox': {'frames': 0, 'detections': 0, 'total_time': 0},
            'keypoint': {'frames': 0, 'detections': 0, 'valid_keypoints': 0, 'total_time': 0},
            'seg': {'frames': 0, 'detections': 0, 'wheels_detected': 0, 'seg_used': 0, 'fallback_used': 0, 'total_time': 0},
            '3dbb': {'frames': 0, 'detections': 0, 'boxes_3d': 0, 'total_time': 0}
        }

        # Bounds for filtering (wider bounds for testing)
        self.world_x_bounds = (-100, 100)
        self.world_y_bounds = (-100, 100)
        self.max_speed = 200

        print("Comparator initialized!")

    def _load_calibration(self):
        """Load camera calibration data."""
        if os.path.exists(CALIBRATION_FILE):
            data = np.load(CALIBRATION_FILE)
            K = data.get('K', None)
            D = data.get('D', None)
            DIM = data.get('DIM', None)
            print(f"  Loaded calibration: {CALIBRATION_FILE}")
            return K, D, DIM
        else:
            print(f"  Warning: Calibration not found: {CALIBRATION_FILE}")
            return None, None, None

    def _load_transformer(self):
        """Load coordinate transformer."""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r') as f:
                mapping_data = json.load(f)
            print(f"  Loaded mapping: {self.mapping_file}")

            # Create homography-based transformer
            class HomographyTransformer:
                def __init__(self, mapping):
                    self.mapping = mapping
                    # Load homography matrix if available
                    if 'transformation_matrix' in mapping:
                        self.H = np.array(mapping['transformation_matrix'])
                        print(f"    Using homography matrix (3x3)")
                    else:
                        self.H = None
                        print(f"    Warning: No transformation_matrix in mapping")

                def pixel_to_world(self, px, py):
                    """Transform pixel to world coordinates using homography."""
                    if self.H is None:
                        return (0, 0)

                    pt = np.array([px, py, 1.0])
                    transformed = self.H @ pt

                    if abs(transformed[2]) > 1e-10:
                        wx = transformed[0] / transformed[2]
                        wy = transformed[1] / transformed[2]
                        return (float(wx), float(wy))
                    return (0, 0)

            return HomographyTransformer(mapping_data)
        else:
            print(f"  Warning: Mapping not found: {self.mapping_file}")
            return None

    def is_valid_coordinate(self, world_coord):
        """Check if coordinate is within valid bounds."""
        if world_coord is None:
            return False
        x, y = world_coord
        return (self.world_x_bounds[0] <= x <= self.world_x_bounds[1] and
                self.world_y_bounds[0] <= y <= self.world_y_bounds[1])

    def undistort_frame(self, frame):
        """Apply fisheye undistortion if calibration available."""
        if self.K is not None and self.D is not None and self.DIM is not None:
            h, w = frame.shape[:2]
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                self.K, self.D, np.eye(3), self.K,
                (w, h), cv2.CV_16SC2
            )
            return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        return frame

    def run_comparison(self, max_frames=None, show_video=False):
        """Run comparison on video."""
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\nProcessing video: {width}x{height} @ {fps:.1f} FPS")
        print(f"Total frames to process: {total_frames}")
        print("=" * 60)

        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")

            # Undistort frame
            undistorted = self.undistort_frame(frame)

            # Resize for recognition
            recognition_frame = cv2.resize(undistorted, RECOGNITION_SIZE)
            display_frame = cv2.resize(undistorted, DISPLAY_SIZE)

            # ========== BBOX APPROACH ==========
            bbox_start = time.time()
            vehicle_results = self.vehicle_model.track(recognition_frame, persist=True, verbose=False)
            bbox_time = time.time() - bbox_start

            self.metrics['bbox']['frames'] += 1
            self.metrics['bbox']['total_time'] += bbox_time

            track_ids = []
            scaled_boxes = []

            if vehicle_results[0].boxes.id is not None:
                boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
                track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

                # Scale boxes to display size
                scale_x = DISPLAY_SIZE[0] / RECOGNITION_SIZE[0]
                scale_y = DISPLAY_SIZE[1] / RECOGNITION_SIZE[1]
                scaled_boxes = []
                for box in boxes:
                    x, y, w, h = box
                    scaled_boxes.append([x * scale_x, y * scale_y, w * scale_x, h * scale_y])

                # Calculate world coordinates using bottom-center
                bbox_world_coords = []
                for box in scaled_boxes:
                    x, y, w, h = box
                    bottom_center = (x, y + h/2)
                    if self.transformer:
                        world_coord = self.transformer.pixel_to_world(bottom_center[0], bottom_center[1])
                    else:
                        world_coord = (0, 0)
                    bbox_world_coords.append(world_coord)

                bbox_speeds = self.speed_trackers['bbox'].get_speeds(
                    track_ids, bbox_world_coords, frame_count, fps
                )

                self.metrics['bbox']['detections'] += len(track_ids)

                for tid, coord, speed in zip(track_ids, bbox_world_coords, bbox_speeds):
                    if self.is_valid_coordinate(coord) and speed <= self.max_speed:
                        self.results['bbox'][tid].append({
                            'frame': frame_count,
                            'world_x': coord[0],
                            'world_y': coord[1],
                            'speed': speed
                        })

            # ========== KEYPOINT APPROACH ==========
            self.metrics['keypoint']['frames'] += 1

            if vehicle_results[0].boxes.id is not None and vehicle_results[0].keypoints is not None:
                keypoints_data = vehicle_results[0].keypoints.data.cpu().numpy()

                kp_world_coords = []
                scale_x = DISPLAY_SIZE[0] / RECOGNITION_SIZE[0]
                scale_y = DISPLAY_SIZE[1] / RECOGNITION_SIZE[1]

                for v_idx, kps in enumerate(keypoints_data):
                    # Get bottom-most high-confidence keypoint
                    valid_kps = [(int(kp[0] * scale_x), int(kp[1] * scale_y))
                                for kp in kps if kp[2] > 0.3]

                    if valid_kps:
                        contact_pt = max(valid_kps, key=lambda p: p[1])
                        world_coord = self.transformer.pixel_to_world(contact_pt[0], contact_pt[1]) if self.transformer else (0, 0)
                        self.metrics['keypoint']['valid_keypoints'] += 1
                    else:
                        # Fallback to bbox
                        box = scaled_boxes[v_idx]
                        x, y, w, h = box
                        world_coord = self.transformer.pixel_to_world(x, y + h/2) if self.transformer else (0, 0)

                    kp_world_coords.append(world_coord)

                kp_speeds = self.speed_trackers['keypoint'].get_speeds(
                    track_ids, kp_world_coords, frame_count, fps
                )

                self.metrics['keypoint']['detections'] += len(track_ids)

                for tid, coord, speed in zip(track_ids, kp_world_coords, kp_speeds):
                    if self.is_valid_coordinate(coord) and speed <= self.max_speed:
                        self.results['keypoint'][tid].append({
                            'frame': frame_count,
                            'world_x': coord[0],
                            'world_y': coord[1],
                            'speed': speed
                        })

            # ========== SEGMENTATION APPROACH ==========
            if self.has_wheel_seg:
                seg_start = time.time()
                wheel_results = self.wheel_model(recognition_frame, verbose=False)
                seg_time = time.time() - seg_start

                self.metrics['seg']['frames'] += 1
                self.metrics['seg']['total_time'] += seg_time

                if wheel_results[0].masks is not None:
                    self.metrics['seg']['wheels_detected'] += len(wheel_results[0].masks)

                # Process wheel associations (simplified)
                if vehicle_results[0].boxes.id is not None:
                    seg_world_coords = []
                    for v_idx, box in enumerate(scaled_boxes):
                        # For simplicity, fall back to bbox for now
                        x, y, w, h = box
                        world_coord = self.transformer.pixel_to_world(x, y + h/2) if self.transformer else (0, 0)
                        seg_world_coords.append(world_coord)
                        self.metrics['seg']['fallback_used'] += 1

                    seg_speeds = self.speed_trackers['seg'].get_speeds(
                        track_ids, seg_world_coords, frame_count, fps
                    )

                    self.metrics['seg']['detections'] += len(track_ids)

                    for tid, coord, speed in zip(track_ids, seg_world_coords, seg_speeds):
                        if self.is_valid_coordinate(coord) and speed <= self.max_speed:
                            self.results['seg'][tid].append({
                                'frame': frame_count,
                                'world_x': coord[0],
                                'world_y': coord[1],
                                'speed': speed
                            })

            # ========== 3D BBOX APPROACH (from pre-computed results) ==========
            if self.has_3dbb and self.dddbb_data:
                # Find frame data (frame_count is 1-indexed)
                frame_data = None
                for fd in self.dddbb_data:
                    if fd.get('frame') == frame_count:
                        frame_data = fd
                        break

                if frame_data:
                    self.metrics['3dbb']['frames'] += 1

                    # Filter for vehicles only
                    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
                    detections = [d for d in frame_data.get('detections', [])
                                 if d.get('class') in vehicle_classes]

                    self.metrics['3dbb']['boxes_3d'] += len(detections)

                    if detections:
                        # Use track_id from detection if available, otherwise use index
                        tids_3d = [d.get('track_id', i) if d.get('track_id') else i
                                   for i, d in enumerate(detections)]
                        coords_3d = [(d['world_x'], d['world_y']) for d in detections]

                        speeds_3d = self.speed_trackers['3dbb'].get_speeds(
                            tids_3d, coords_3d, frame_count, fps
                        )

                        self.metrics['3dbb']['detections'] += len(tids_3d)

                        for i, (tid, speed) in enumerate(zip(tids_3d, speeds_3d)):
                            det = detections[i]
                            coord = (det['world_x'], det['world_y'])
                            if speed <= self.max_speed:
                                self.results['3dbb'][tid].append({
                                    'frame': frame_count,
                                    'world_x': coord[0],
                                    'world_y': coord[1],
                                    'depth': det.get('depth', 0),
                                    'speed': speed,
                                    'class': det.get('class', 'unknown')
                                })

            # Visualization
            if show_video:
                vis = self._create_visualization(display_frame, vehicle_results, frame_count)
                cv2.imshow("All Approaches Comparison", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if show_video:
            cv2.destroyAllWindows()

        return self.generate_report()

    def _create_visualization(self, frame, vehicle_results, frame_count):
        """Create visualization frame."""
        vis = frame.copy()

        if vehicle_results[0].boxes.id is not None:
            boxes = vehicle_results[0].boxes.xywh.cpu().numpy()
            track_ids = vehicle_results[0].boxes.id.int().cpu().tolist()

            scale_x = DISPLAY_SIZE[0] / RECOGNITION_SIZE[0]
            scale_y = DISPLAY_SIZE[1] / RECOGNITION_SIZE[1]

            for box, tid in zip(boxes, track_ids):
                x, y, w, h = box
                x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y

                cv2.rectangle(vis,
                             (int(x - w/2), int(y - h/2)),
                             (int(x + w/2), int(y + h/2)),
                             (0, 255, 0), 2)
                cv2.putText(vis, f"ID:{tid}", (int(x - w/2), int(y - h/2 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(vis, f"Frame: {frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return vis

    def generate_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("Approaches: BBox | Keypoint | Segmentation | 3D BBox (YOLOx3D)")
        print("=" * 80)

        # 1. Performance Metrics
        print("\n1. PERFORMANCE METRICS")
        print("-" * 60)
        for approach in ['bbox', 'keypoint', 'seg', '3dbb']:
            m = self.metrics[approach]
            if m['total_time'] > 0:
                fps = m['frames'] / m['total_time']
                print(f"  {approach.upper():10s}: {fps:6.2f} FPS  ({m['frames']} frames, {m['total_time']:.2f}s)")

        # 2. Detection Metrics
        print("\n2. DETECTION METRICS")
        print("-" * 60)
        print(f"  {'Approach':<15} {'Detections':<12} {'Special Metric':<30}")
        print(f"  {'-'*15} {'-'*12} {'-'*30}")
        print(f"  {'BBox':<15} {self.metrics['bbox']['detections']:<12} -")
        print(f"  {'Keypoint':<15} {self.metrics['keypoint']['detections']:<12} Valid keypoints: {self.metrics['keypoint']['valid_keypoints']}")
        print(f"  {'Segmentation':<15} {self.metrics['seg']['detections']:<12} Wheels detected: {self.metrics['seg']['wheels_detected']}")
        print(f"  {'3D BBox':<15} {self.metrics['3dbb']['detections']:<12} 3D boxes: {self.metrics['3dbb']['boxes_3d']}")

        # 3. Speed Comparison
        print("\n3. SPEED ESTIMATION COMPARISON")
        print("-" * 60)

        all_speeds = {}
        for approach in ['bbox', 'keypoint', 'seg', '3dbb']:
            speeds = []
            for tid in self.results[approach]:
                for entry in self.results[approach][tid]:
                    if entry['speed'] > 0:
                        speeds.append(entry['speed'])
            all_speeds[approach] = speeds

        print(f"  {'Approach':<15} {'Count':<10} {'Mean (km/h)':<15} {'Std (km/h)':<15} {'Max (km/h)':<12}")
        print(f"  {'-'*15} {'-'*10} {'-'*15} {'-'*15} {'-'*12}")
        for approach in ['bbox', 'keypoint', 'seg', '3dbb']:
            speeds = all_speeds[approach]
            if speeds:
                print(f"  {approach.upper():<15} {len(speeds):<10} {np.mean(speeds):<15.2f} {np.std(speeds):<15.2f} {np.max(speeds):<12.2f}")
            else:
                print(f"  {approach.upper():<15} {'No data':<10}")

        # 4. Cross-approach comparison
        print("\n4. CROSS-APPROACH SPEED DIFFERENCES")
        print("-" * 60)

        comparisons = [
            ('bbox', 'keypoint'),
            ('bbox', 'seg'),
            ('bbox', '3dbb'),
            ('keypoint', '3dbb')
        ]

        for a1, a2 in comparisons:
            diffs = []
            for tid in self.results[a1]:
                if tid in self.results[a2]:
                    d1 = {e['frame']: e['speed'] for e in self.results[a1][tid]}
                    d2 = {e['frame']: e['speed'] for e in self.results[a2][tid]}
                    for frame in set(d1.keys()) & set(d2.keys()):
                        if d1[frame] > 0 and d2[frame] > 0:
                            diffs.append(abs(d1[frame] - d2[frame]))

            if diffs:
                print(f"  {a1.upper()} vs {a2.upper()}: Mean diff = {np.mean(diffs):.2f} km/h, Std = {np.std(diffs):.2f} km/h")
            else:
                print(f"  {a1.upper()} vs {a2.upper()}: No overlapping data")

        # 5. 3D BBox specific metrics
        if self.has_3dbb and self.results['3dbb']:
            print("\n5. 3D BBOX SPECIFIC METRICS (YOLOx3D)")
            print("-" * 60)

            depths = []
            classes = defaultdict(int)
            for tid in self.results['3dbb']:
                for entry in self.results['3dbb'][tid]:
                    if 'depth' in entry:
                        depths.append(entry['depth'])
                    if 'class' in entry:
                        classes[entry['class']] += 1

            if depths:
                print(f"  Depth range: {np.min(depths):.2f}m - {np.max(depths):.2f}m (mean: {np.mean(depths):.2f}m)")

            if classes:
                print(f"  Classes detected: {dict(classes)}")

        # 6. Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        print("\nApproach Characteristics:")
        print("  - BBox:         Simple baseline, uses bounding box bottom-center")
        print("  - Keypoint:     Uses wheel keypoints from pose model")
        print("  - Segmentation: Uses wheel segmentation masks")
        print("  - 3D BBox:      Monocular 3D detection (YOLOx3D) with depth estimation")

        # Determine best approach based on data
        print("\nRecommendations:")
        if self.metrics['3dbb']['boxes_3d'] > 0:
            print("  - 3D BBox approach provides depth information for distance estimation")
        if self.metrics['keypoint']['valid_keypoints'] > 0:
            kp_rate = self.metrics['keypoint']['valid_keypoints'] / max(1, self.metrics['keypoint']['detections']) * 100
            print(f"  - Keypoint detection rate: {kp_rate:.1f}%")

        # Save report
        report = {
            'metrics': self.metrics,
            'speeds': {k: all_speeds[k] for k in all_speeds},
            'vehicle_count': {approach: len(self.results[approach]) for approach in self.results}
        }

        report_path = os.path.join(os.path.dirname(__file__), 'full_comparison_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\nDetailed report saved to: {report_path}")

        return report


def main():
    parser = argparse.ArgumentParser(description='Compare all vehicle localization approaches')
    parser.add_argument('--video', type=str, default=VIDEO_PATH,
                       help='Path to video file')
    parser.add_argument('--mapping', type=str, default=MAPPING_FILE,
                       help='Path to coordinate mapping file')
    parser.add_argument('--max-frames', type=int, default=300,
                       help='Maximum frames to process (default: 300)')
    parser.add_argument('--show', action='store_true',
                       help='Show video during processing')
    args = parser.parse_args()

    print("=" * 60)
    print("VEHICLE LOCALIZATION APPROACH COMPARISON")
    print("=" * 60)
    print(f"\nVideo: {args.video}")
    print(f"Mapping: {args.mapping}")
    print(f"Max frames: {args.max_frames}")

    comparator = AllApproachComparator(args.video, args.mapping)
    report = comparator.run_comparison(
        max_frames=args.max_frames,
        show_video=args.show
    )


if __name__ == "__main__":
    main()
