"""
3D Bounding Box Pipeline - Main Entry Point

Uses YOLO tracking + homography-based 3D estimation for accurate vehicle localization.
"""

import os
import sys
import cv2
import numpy as np
import json
from collections import deque
from scipy.ndimage import uniform_filter1d

# Add YOLOx3D to path
YOLOX3D_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "YOLOx3D")
sys.path.insert(0, os.path.join(YOLOX3D_PATH, "src"))

from models.yolo_v11 import YOLOv11

# Vehicle classes to detect (COCO class names)
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}

# Standard vehicle dimensions (height, width, length) in meters
VEHICLE_DIMENSIONS = {
    'car': (1.5, 1.8, 4.5),
    'truck': (2.5, 2.5, 8.0),
    'bus': (3.0, 2.5, 12.0),
    'motorcycle': (1.2, 0.8, 2.0),
    'bicycle': (1.0, 0.5, 1.8),
}

# Import local config
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_config", os.path.join(PIPELINE_DIR, "config.py"))
pipeline_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_config)

# Import coordinate transformer from utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
try:
    from utils.coordinate_transformer import CoordinateTransformer
except ImportError:
    # Create a placeholder if not available
    class CoordinateTransformer:
        def __init__(self, mapping_file):
            self.transformation_matrix = None
        def pixel_to_world(self, x, y):
            return (0, 0)
        def world_to_pixel(self, x, y):
            return (0, 0)


class Box3DSmoother:
    """Temporal smoothing for 3D bounding box parameters."""

    def __init__(self, window_size=5):
        self.window_size = window_size
        # Store history per track_id: {track_id: {'positions': deque, 'dimensions': deque, 'yaw': deque}}
        self.history = {}

    def update(self, track_id, world_x, world_y, dimensions, yaw):
        """Update history and return smoothed values."""
        if track_id not in self.history:
            self.history[track_id] = {
                'world_x': deque(maxlen=self.window_size),
                'world_y': deque(maxlen=self.window_size),
                'yaw': deque(maxlen=self.window_size),
            }

        h = self.history[track_id]
        h['world_x'].append(world_x)
        h['world_y'].append(world_y)
        h['yaw'].append(yaw)

        # Return smoothed values (simple moving average)
        smooth_x = np.mean(h['world_x'])
        smooth_y = np.mean(h['world_y'])
        smooth_yaw = np.mean(h['yaw'])

        return smooth_x, smooth_y, dimensions, smooth_yaw

    def cleanup_stale(self, active_ids):
        """Remove tracks that are no longer active."""
        stale = [tid for tid in self.history if tid not in active_ids]
        for tid in stale:
            del self.history[tid]


class SpeedTracker:
    """
    Tracks vehicle positions across frames and calculates speed.

    Uses a sliding window approach to smooth speed estimates.
    """

    def __init__(self, fps, history_length=10, smoothing_window=5):
        """
        Initialize speed tracker.

        Args:
            fps: Video frames per second (for time delta calculation)
            history_length: Number of frames to keep in history
            smoothing_window: Number of speed samples to average for smoothing
        """
        self.fps = fps
        self.time_per_frame = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.history_length = history_length
        self.smoothing_window = smoothing_window

        # Track history: {track_id: deque of (frame_num, world_x, world_y)}
        self.position_history = {}
        # Speed history for smoothing: {track_id: deque of speed values}
        self.speed_history = {}

    def update(self, frame_num, detections):
        """
        Update tracker with new detections and calculate speeds.

        Args:
            frame_num: Current frame number
            detections: List of detection dicts with track_id, world_x, world_y

        Returns:
            Dict mapping track_id to speed in km/h
        """
        speeds = {}

        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue

            world_x = det['world_x']
            world_y = det['world_y']

            # Initialize history for new tracks
            if track_id not in self.position_history:
                self.position_history[track_id] = deque(maxlen=self.history_length)
                self.speed_history[track_id] = deque(maxlen=self.smoothing_window)

            # Add current position
            self.position_history[track_id].append((frame_num, world_x, world_y))

            # Calculate speed if we have enough history
            if len(self.position_history[track_id]) >= 2:
                speed = self._calculate_speed(track_id)
                if speed is not None:
                    self.speed_history[track_id].append(speed)
                    # Return smoothed speed
                    speeds[track_id] = self._get_smoothed_speed(track_id)
            else:
                speeds[track_id] = 0.0

        return speeds

    def _calculate_speed(self, track_id):
        """Calculate instantaneous speed from position history."""
        history = self.position_history[track_id]
        if len(history) < 2:
            return None

        # Use first and last position in history for more stable estimate
        frame1, x1, y1 = history[0]
        frame2, x2, y2 = history[-1]

        # Calculate displacement in meters
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        # Calculate time delta
        frame_diff = frame2 - frame1
        if frame_diff == 0:
            return None
        time_delta = frame_diff * self.time_per_frame

        # Speed in m/s
        speed_ms = distance / time_delta

        # Convert to km/h
        speed_kmh = speed_ms * 3.6

        return speed_kmh

    def _get_smoothed_speed(self, track_id):
        """Get smoothed speed by averaging recent values."""
        if track_id not in self.speed_history or len(self.speed_history[track_id]) == 0:
            return 0.0
        return np.mean(self.speed_history[track_id])

    def cleanup_stale_tracks(self, current_frame, max_age=30):
        """Remove tracks that haven't been updated recently."""
        stale_ids = []
        for track_id, history in self.position_history.items():
            if len(history) > 0:
                last_frame = history[-1][0]
                if current_frame - last_frame > max_age:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.position_history[track_id]
            if track_id in self.speed_history:
                del self.speed_history[track_id]


class Pipeline3DBB:
    """
    3D Bounding Box Pipeline for vehicle localization.

    Uses YOLO tracking + homography-based 3D estimation for accurate results.
    """

    def __init__(self, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the pipeline with tracking and homography support.

        Args:
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Initialize YOLO detector with tracking enabled
        print("Initializing YOLO detector with tracking...")
        self.detector = YOLOv11(
            model_size='nano',
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            device='cpu'
        )

        # Initialize coordinate transformer for homography-based position estimation
        print("Initializing coordinate transformer...")
        self.transformer = None
        if os.path.exists(pipeline_config.MAPPING_FILE):
            self.transformer = CoordinateTransformer(pipeline_config.MAPPING_FILE)
            print(f"Loaded coordinate mapping from {pipeline_config.MAPPING_FILE}")
        else:
            print(f"Warning: Coordinate mapping file not found: {pipeline_config.MAPPING_FILE}")

        # Initialize 3D box smoother for temporal consistency
        self.smoother = Box3DSmoother(window_size=5)

        # Load camera calibration for projection
        self.K = None
        if os.path.exists(pipeline_config.CALIBRATION_FILE):
            calib_data = np.load(pipeline_config.CALIBRATION_FILE)
            self.K = calib_data.get('K', calib_data.get('camera_matrix', None))
            if self.K is not None:
                print(f"Loaded camera calibration from {pipeline_config.CALIBRATION_FILE}")
        if self.K is None:
            # Default camera matrix for 1920x1080
            self.K = np.array([
                [1000, 0, 960],
                [0, 1000, 540],
                [0, 0, 1]
            ], dtype=np.float64)
            print("Using default camera matrix")

        print("Pipeline initialized successfully!")

    def _rotation_matrix_y(self, theta):
        """Create rotation matrix around Y-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def get_3d_box_corners(self, center, dimensions, yaw):
        """
        Get all 8 corners of a 3D bounding box.

        Args:
            center: [x, y, z] center of the box
            dimensions: [h, w, l] height, width, length
            yaw: rotation around Y axis in radians

        Returns:
            8x3 array of corner coordinates
            Corner order: 0-3 are bottom face, 4-7 are top face
        """
        h, w, l = dimensions
        x, y, z = center

        # Corners in local frame (y=0 is bottom, y=-h is top in camera coords)
        # Bottom face corners (ground contact points)
        x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]
        z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

        corners = np.vstack([x_corners, y_corners, z_corners])

        # Apply rotation and translation
        R = self._rotation_matrix_y(yaw)
        rotated = R @ corners
        rotated[0, :] += x
        rotated[1, :] += y
        rotated[2, :] += z

        return rotated.T  # 8x3 array

    def get_bottom_corners(self, center, dimensions, yaw):
        """
        Get the 4 bottom corners of a 3D bounding box (ground contact points).

        Args:
            center: [x, y, z] center of the box
            dimensions: [h, w, l] height, width, length
            yaw: rotation around Y axis in radians

        Returns:
            4x3 array of bottom corner coordinates in camera frame
            Order: front-left, front-right, rear-right, rear-left
        """
        all_corners = self.get_3d_box_corners(center, dimensions, yaw)
        # Bottom corners are indices 0-3
        return all_corners[:4]

    def estimate_yaw_from_motion(self, track_id, world_x, world_y):
        """
        Estimate vehicle yaw (heading) from motion direction.

        Uses the smoother's position history to calculate direction of travel.
        """
        if track_id not in self.smoother.history:
            return 0.0

        h = self.smoother.history[track_id]
        if len(h['world_x']) < 2:
            return 0.0

        # Calculate motion direction from recent positions
        dx = world_x - h['world_x'][0]
        dy = world_y - h['world_y'][0]

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            # Not enough motion, use last known yaw
            return h['yaw'][-1] if len(h['yaw']) > 0 else 0.0

        # Calculate yaw from motion direction
        yaw = np.arctan2(dx, dy)  # atan2(x, y) gives heading from north
        return yaw

    def process_frame(self, frame, use_tracking=True):
        """
        Process a single frame using YOLO tracking and homography-based 3D estimation.

        Args:
            frame: Input image (BGR format)
            use_tracking: Whether to use YOLO tracking (default True)

        Returns:
            dict with:
                - boxes_3d: List of 3D bounding box detections (vehicles only)
                - detection_frame: Frame with 3D bounding boxes drawn
                - detections: Raw YOLO detections
        """
        result = {
            'boxes_3d': [],
            'detection_frame': frame.copy(),
            'detections': []
        }

        # Run YOLO detection with tracking
        _, detections = self.detector.detect(frame, track=use_tracking)

        # Filter to vehicles only
        vehicle_detections = [
            det for det in detections
            if det[3].lower() in VEHICLE_CLASSES
        ]
        result['detections'] = vehicle_detections

        # Convert 2D detections to 3D boxes using homography
        boxes_3d = []
        active_track_ids = set()

        for det in vehicle_detections:
            bbox, score, class_id, class_name, track_id = det
            x1, y1, x2, y2 = bbox

            # Get bottom center of bounding box (ground contact point)
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2  # Bottom of bbox

            # Convert to world coordinates using homography
            if self.transformer:
                world_x, world_y = self.transformer.pixel_to_world(bottom_center_x, bottom_center_y)
            else:
                # Fallback: estimate depth from bbox size
                bbox_height = y2 - y1
                # Rough depth estimation: smaller bbox = further away
                depth = max(5.0, 3000.0 / max(bbox_height, 1))
                world_x = (bottom_center_x - frame.shape[1]/2) * depth / 1000
                world_y = depth

            # Get standard dimensions for vehicle class
            h, w, l = VEHICLE_DIMENSIONS.get(class_name.lower(), (1.5, 1.8, 4.5))
            dimensions = np.array([h, w, l])

            # Estimate yaw from motion (if tracking)
            yaw = 0.0
            if track_id is not None:
                yaw = self.estimate_yaw_from_motion(track_id, world_x, world_y)
                active_track_ids.add(track_id)

                # Apply temporal smoothing
                world_x, world_y, dimensions, yaw = self.smoother.update(
                    track_id, world_x, world_y, dimensions, yaw
                )

            # Create 3D box representation
            # center_3d: [x, y, z] where y is height (negative = up), z is forward
            center_3d = np.array([world_x, -h/2, world_y])

            box_3d = {
                'bbox_2d': bbox,
                'center_3d': center_3d,
                'dimensions': dimensions,
                'yaw': yaw,
                'class_name': class_name,
                'object_id': track_id,
                'score': score,
                'world_x': world_x,
                'world_y': world_y,
            }
            boxes_3d.append(box_3d)

        # Cleanup stale tracks from smoother
        self.smoother.cleanup_stale(active_track_ids)

        result['boxes_3d'] = boxes_3d
        # Draw 3D bounding boxes
        result['detection_frame'] = self.draw_3d_boxes(frame.copy(), boxes_3d)

        return result

    def project_3d_to_2d_homography(self, world_x, world_y, height=0):
        """
        Project world coordinates back to pixel coordinates using inverse homography.

        Args:
            world_x, world_y: World coordinates in meters
            height: Height offset (for top of 3D box)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        if self.transformer:
            # For ground plane (height=0), use inverse homography
            px, py = self.transformer.world_to_pixel(world_x, world_y)
            # Adjust for height (move up in image for elevated points)
            # This is an approximation - proper handling needs camera extrinsics
            if height > 0:
                # Rough approximation: higher points appear higher in image
                # Scale by distance (further = less vertical shift)
                distance = np.sqrt(world_x**2 + world_y**2)
                height_shift = height * 200 / max(distance, 1)  # pixels per meter height
                py -= height_shift
            return px, py
        return 0, 0

    def draw_3d_boxes(self, frame, boxes_3d, thickness=2):
        """
        Draw 3D bounding boxes on frame using homography-based projection.

        Args:
            frame: Image to draw on
            boxes_3d: List of 3D bounding box detections
            thickness: Line thickness

        Returns:
            Annotated frame with 3D boxes
        """
        result = frame.copy()

        for box in boxes_3d:
            try:
                bbox_2d = box.get('bbox_2d')
                if bbox_2d is None:
                    continue

                x1, y1, x2, y2 = [int(v) for v in bbox_2d]
                world_x = box.get('world_x', 0)
                world_y = box.get('world_y', 0)
                h, w, l = box.get('dimensions', [1.5, 1.8, 4.5])
                yaw = box.get('yaw', 0)
                track_id = box.get('object_id')
                class_name = box.get('class_name', 'vehicle')

                # Color based on track ID for visual consistency
                if track_id is not None:
                    # Generate consistent color from track ID
                    np.random.seed(int(track_id) % 1000)
                    color = tuple(int(c) for c in np.random.randint(100, 255, 3))
                else:
                    color = (0, 255, 0)

                # Calculate 3D box corners in world coordinates
                # Bottom 4 corners (ground plane)
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                half_l, half_w = l/2, w/2

                # Local corners relative to center
                local_corners = np.array([
                    [ half_l,  half_w],  # front-right
                    [ half_l, -half_w],  # front-left
                    [-half_l, -half_w],  # rear-left
                    [-half_l,  half_w],  # rear-right
                ])

                # Rotate by yaw
                rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
                rotated_corners = local_corners @ rot.T

                # Translate to world position
                bottom_corners_world = rotated_corners + np.array([world_x, world_y])

                # Project bottom corners to image
                bottom_corners_2d = []
                for corner in bottom_corners_world:
                    px, py = self.project_3d_to_2d_homography(corner[0], corner[1], 0)
                    bottom_corners_2d.append([px, py])
                bottom_corners_2d = np.array(bottom_corners_2d, dtype=np.int32)

                # Project top corners (with height offset)
                top_corners_2d = []
                for corner in bottom_corners_world:
                    px, py = self.project_3d_to_2d_homography(corner[0], corner[1], h)
                    top_corners_2d.append([px, py])
                top_corners_2d = np.array(top_corners_2d, dtype=np.int32)

                # Check if projection is valid (corners should be within reasonable bounds)
                all_corners = np.vstack([bottom_corners_2d, top_corners_2d])
                if (np.any(all_corners < -1000) or np.any(all_corners > frame.shape[1] + 1000)):
                    # Projection failed, fall back to 2D bbox with pseudo-3D effect
                    self._draw_pseudo_3d_box(result, bbox_2d, h, color, thickness, class_name, world_y, track_id)
                    continue

                # Draw bottom face
                for i in range(4):
                    pt1 = tuple(bottom_corners_2d[i])
                    pt2 = tuple(bottom_corners_2d[(i + 1) % 4])
                    cv2.line(result, pt1, pt2, color, thickness)

                # Draw top face
                for i in range(4):
                    pt1 = tuple(top_corners_2d[i])
                    pt2 = tuple(top_corners_2d[(i + 1) % 4])
                    cv2.line(result, pt1, pt2, color, thickness)

                # Draw vertical edges
                for i in range(4):
                    pt1 = tuple(bottom_corners_2d[i])
                    pt2 = tuple(top_corners_2d[i])
                    cv2.line(result, pt1, pt2, color, thickness)

                # Draw front face with semi-transparent fill
                pts_front = np.array([bottom_corners_2d[0], bottom_corners_2d[1],
                                     top_corners_2d[1], top_corners_2d[0]], np.int32)
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts_front], color)
                cv2.addWeighted(overlay, 0.25, result, 0.75, 0, result)

                # Draw label
                label = f"{class_name}"
                if track_id is not None:
                    label = f"ID:{track_id} {class_name}"
                label += f" {world_y:.1f}m"

                label_pos = (x1, max(y1 - 10, 20))
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(result,
                            (label_pos[0] - 2, label_pos[1] - text_h - 2),
                            (label_pos[0] + text_w + 2, label_pos[1] + 2),
                            (0, 0, 0), -1)
                cv2.putText(result, label, label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            except Exception as e:
                # Skip boxes that can't be drawn
                continue

        return result

    def _draw_pseudo_3d_box(self, frame, bbox_2d, height, color, thickness, class_name, depth, track_id):
        """
        Draw a pseudo-3D box when homography projection fails.
        Uses 2D bbox with vertical offset for 3D effect.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox_2d]
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        # Estimate vertical offset based on vehicle height and distance
        v_offset = int(min(bbox_h * 0.4, 100))  # Cap at 100 pixels

        # Draw bottom rectangle
        cv2.rectangle(frame, (x1, y1 + v_offset), (x2, y2), color, thickness)

        # Draw top rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2 - v_offset), color, thickness)

        # Draw vertical edges
        cv2.line(frame, (x1, y1), (x1, y1 + v_offset), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + v_offset), color, thickness)
        cv2.line(frame, (x1, y2 - v_offset), (x1, y2), color, thickness)
        cv2.line(frame, (x2, y2 - v_offset), (x2, y2), color, thickness)

        # Draw label
        label = f"{class_name}"
        if track_id is not None:
            label = f"ID:{track_id} {class_name}"
        label += f" {depth:.1f}m"

        label_pos = (x1, max(y1 - 10, 20))
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame,
                    (label_pos[0] - 2, label_pos[1] - text_h - 2),
                    (label_pos[0] + text_w + 2, label_pos[1] + 2),
                    (0, 0, 0), -1)
        cv2.putText(frame, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_speed_annotations(self, frame, boxes_3d, speeds):
        """
        Draw speed annotations on frame for each detected vehicle.

        Args:
            frame: Image to annotate (will be modified in place)
            boxes_3d: List of 3D bounding box detections
            speeds: Dict mapping track_id to speed in km/h

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for box in boxes_3d:
            track_id = box.get('object_id')
            if track_id is None or track_id not in speeds:
                continue

            speed = speeds[track_id]
            bbox_2d = box.get('bbox_2d')

            if bbox_2d is None:
                continue

            # Get position for text (top of bounding box)
            x1, y1, x2, y2 = [int(v) for v in bbox_2d]

            # Format speed text
            speed_text = f"{speed:.1f} km/h"

            # Text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Get text size for background
            (text_w, text_h), baseline = cv2.getTextSize(speed_text, font, font_scale, thickness)

            # Position text above bounding box
            text_x = x1
            text_y = y1 - 10

            # Ensure text stays within frame
            if text_y - text_h < 0:
                text_y = y2 + text_h + 10

            # Draw background rectangle for better visibility
            padding = 4
            cv2.rectangle(
                annotated,
                (text_x - padding, text_y - text_h - padding),
                (text_x + text_w + padding, text_y + padding),
                (0, 0, 0),  # Black background
                -1  # Filled
            )

            # Color based on speed (green->yellow->red)
            if speed < 30:
                color = (0, 255, 0)  # Green
            elif speed < 50:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw speed text
            cv2.putText(
                annotated,
                speed_text,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

        return annotated

    def get_world_coordinates(self, boxes_3d):
        """
        Extract world coordinates from 3D boxes (already computed via homography).

        Args:
            boxes_3d: List of 3D bounding box detections

        Returns:
            List of dicts with world coordinates for each detection
        """
        world_coords = []

        for box in boxes_3d:
            dimensions = box.get('dimensions', np.array([1.5, 1.8, 4.5]))
            yaw = box.get('yaw', 0)
            world_x = box.get('world_x', 0)
            world_y = box.get('world_y', 0)

            world_coords.append({
                'track_id': box.get('object_id'),
                'class_name': box.get('class_name', 'unknown'),
                'confidence': box.get('score', 0),
                'world_x': world_x,
                'world_y': world_y,
                'depth': world_y,  # In our coordinate system, world_y is forward distance
                'dimensions': {
                    'height': float(dimensions[0]) if len(dimensions) > 0 else 1.5,
                    'width': float(dimensions[1]) if len(dimensions) > 1 else 1.8,
                    'length': float(dimensions[2]) if len(dimensions) > 2 else 4.5,
                },
                'yaw': float(yaw),
            })

        return world_coords

    def process_video(self, video_path, output_path=None, max_frames=None, enable_speed=True):
        """
        Process a video file and save results with speed annotations.

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            max_frames: Maximum frames to process (optional)
            enable_speed: Enable speed calculation and annotation (default: True)

        Returns:
            List of results per frame (includes speed for each detection)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Initialize speed tracker
        speed_tracker = SpeedTracker(fps) if enable_speed else None

        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Process frame
            result = self.process_frame(frame)
            world_coords = self.get_world_coordinates(result['boxes_3d'])

            # Calculate speeds
            speeds = {}
            if speed_tracker:
                speeds = speed_tracker.update(frame_count, world_coords)
                # Clean up old tracks periodically
                if frame_count % 30 == 0:
                    speed_tracker.cleanup_stale_tracks(frame_count)

            # Add speed to world_coords
            for det in world_coords:
                track_id = det.get('track_id')
                det['speed_kmh'] = speeds.get(track_id, 0.0) if track_id else 0.0

            all_results.append({
                'frame': frame_count,
                'detections': world_coords
            })

            # Write annotated frame with speed
            if writer:
                output_frame = result['detection_frame']
                if enable_speed and speeds:
                    output_frame = self.draw_speed_annotations(
                        output_frame, result['boxes_3d'], speeds
                    )
                writer.write(output_frame)

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        cap.release()
        if writer:
            writer.release()

        print(f"Finished processing {frame_count} frames")
        return all_results


def main():
    """Main entry point for testing the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="3D Bounding Box Pipeline with Tracking")
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max-frames', type=int, help='Max frames to process')
    args = parser.parse_args()

    # Initialize pipeline with tracking + homography
    pipeline = Pipeline3DBB(
        conf_threshold=args.conf,
        iou_threshold=0.45
    )

    if args.video:
        results = pipeline.process_video(
            args.video,
            output_path=args.output,
            max_frames=args.max_frames
        )

        # Save results to CSV (includes speed and track IDs)
        output_csv = args.output.replace('.mp4', '_results.csv') if args.output else 'results.csv'
        with open(output_csv, 'w') as f:
            f.write('frame,track_id,class,confidence,world_x,world_y,depth,length,width,height,yaw,speed_kmh\n')
            for frame_result in results:
                frame_num = frame_result['frame']
                for det in frame_result['detections']:
                    track_id = det.get('track_id', '')
                    f.write(f"{frame_num},{track_id if track_id else ''},{det['class_name']},"
                            f"{det['confidence']:.3f},{det['world_x']:.3f},{det['world_y']:.3f},"
                            f"{det['depth']:.3f},{det['dimensions']['length']:.3f},"
                            f"{det['dimensions']['width']:.3f},{det['dimensions']['height']:.3f},"
                            f"{det['yaw']:.3f},{det.get('speed_kmh', 0.0):.1f}\n")
        print(f"Results saved to {output_csv}")

    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image {args.image}")
            return

        result = pipeline.process_frame(frame, use_tracking=False)
        world_coords = pipeline.get_world_coordinates(result['boxes_3d'])

        print(f"\nDetected {len(world_coords)} vehicles:")
        for i, det in enumerate(world_coords):
            print(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f})")
            print(f"      Position: ({det['world_x']:.2f}, {det['world_y']:.2f}) m")
            print(f"      Dimensions: {det['dimensions']['length']:.2f} x {det['dimensions']['width']:.2f} x {det['dimensions']['height']:.2f} m")

        if args.output:
            cv2.imwrite(args.output, result['detection_frame'])
            print(f"\nAnnotated image saved to {args.output}")
    else:
        print("Please specify --video or --image")
        parser.print_help()


if __name__ == "__main__":
    main()
