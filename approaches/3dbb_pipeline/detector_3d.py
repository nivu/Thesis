"""
3D Bounding Box Detector Module

Uses YOLOx3D approach: YOLO detection + Depth estimation + 3D regression
to estimate 3D bounding boxes from monocular images.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from PIL import Image


@dataclass
class Box3D:
    """3D Bounding Box representation."""

    # Center position in camera frame (meters)
    center: Tuple[float, float, float]  # (x, y, z)

    # Dimensions in meters
    dimensions: Tuple[float, float, float]  # (length, width, height)

    # Orientation (yaw angle in radians)
    yaw: float

    # Detection confidence
    confidence: float

    # Track ID (if available)
    track_id: Optional[int] = None

    # 2D bounding box (for association with existing pipeline)
    bbox_2d: Optional[Tuple[float, float, float, float]] = None  # (x, y, w, h)

    def get_corners_3d(self) -> np.ndarray:
        """
        Get 8 corners of the 3D bounding box in camera frame.

        Returns:
            np.ndarray: (8, 3) array of corner coordinates
        """
        l, w, h = self.dimensions
        x, y, z = self.center

        # Half dimensions
        l2, w2, h2 = l / 2, w / 2, h / 2

        # 8 corners before rotation (centered at origin)
        corners = np.array([
            [-l2, -w2, -h2],  # 0: back-left-bottom
            [ l2, -w2, -h2],  # 1: front-left-bottom
            [ l2,  w2, -h2],  # 2: front-right-bottom
            [-l2,  w2, -h2],  # 3: back-right-bottom
            [-l2, -w2,  h2],  # 4: back-left-top
            [ l2, -w2,  h2],  # 5: front-left-top
            [ l2,  w2,  h2],  # 6: front-right-top
            [-l2,  w2,  h2],  # 7: back-right-top
        ])

        # Rotation matrix (yaw only, around vertical axis)
        cos_yaw = np.cos(self.yaw)
        sin_yaw = np.sin(self.yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0,        0,       1]
        ])

        # Rotate and translate
        corners_rotated = (rotation @ corners.T).T
        corners_world = corners_rotated + np.array([x, y, z])

        return corners_world

    def get_ground_corners(self) -> np.ndarray:
        """
        Get 4 bottom corners of the box (ground plane footprint).

        Returns:
            np.ndarray: (4, 3) array of ground corner coordinates
        """
        corners = self.get_corners_3d()
        return corners[:4]  # Bottom 4 corners

    def get_ground_center(self) -> Tuple[float, float]:
        """
        Get center of vehicle footprint on ground plane.

        Returns:
            Tuple[float, float]: (x, y) ground position
        """
        ground_corners = self.get_ground_corners()
        center_x = np.mean(ground_corners[:, 0])
        center_y = np.mean(ground_corners[:, 1])
        return (center_x, center_y)


class Detector3D:
    """
    3D Bounding Box Detector using monocular depth estimation.

    Pipeline:
    1. YOLOv11 → 2D detection
    2. Depth Anything v2 → Depth map
    3. Multi-bin CNN → Orientation + dimensions
    4. Geometric fusion → 3D bounding box
    """

    def __init__(self, camera_matrix: np.ndarray, depth_scale: float = 1.0):
        """
        Initialize 3D detector.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix (K)
            depth_scale: Scale factor to convert relative depth to absolute (meters)
        """
        self.camera_matrix = camera_matrix
        self.depth_scale = depth_scale
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

        self.yolo_model = None
        self.depth_model = None
        self.dimension_model = None

        self._models_loaded = False

    def load_models(self, yolo_path: str = None, depth_model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
        """
        Load all required models.

        Args:
            yolo_path: Path to YOLO model weights (optional, uses existing detection)
            depth_model_name: HuggingFace model name for depth estimation
        """
        from transformers import pipeline

        print("Loading Depth Anything v2 model...")
        self.depth_model = pipeline(
            task="depth-estimation",
            model=depth_model_name
        )
        print(f"  Depth model loaded: {depth_model_name}")

        # YOLO model is optional - we'll use external detection results
        if yolo_path:
            from ultralytics import YOLO
            self.yolo_model = YOLO(yolo_path)
            print(f"  YOLO model loaded: {yolo_path}")

        self._models_loaded = True
        print("3D detection models loaded successfully")

    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Get depth map from the frame using Depth Anything v2.

        Args:
            frame: Input image (BGR numpy array)

        Returns:
            depth_map: Depth map scaled by depth_scale factor
        """
        if self.depth_model is None:
            raise RuntimeError("Depth model not loaded. Call load_models() first.")

        # Convert BGR to RGB and to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run depth estimation
        depth_result = self.depth_model(pil_image)
        depth_map = np.array(depth_result['depth'])

        # Normalize and scale
        if depth_map.max() > 0:
            # Depth Anything outputs inverse depth - higher values = closer
            # We need to invert and scale to get actual depth in meters
            depth_map = depth_map / depth_map.max()  # Normalize to 0-1

        return depth_map

    def estimate_yaw_from_bbox(self, width: float, height: float) -> float:
        """
        Estimate vehicle orientation (yaw) from bounding box aspect ratio.

        This is a heuristic approach:
        - Wide bbox (ratio > 1.5) = vehicle perpendicular to camera (side view)
        - Narrow bbox (ratio < 0.8) = vehicle parallel (front/back view)
        - In between = diagonal view

        Args:
            width: Bounding box width
            height: Bounding box height

        Returns:
            yaw: Estimated yaw angle in radians
        """
        ratio = width / max(height, 1)

        if ratio > 1.5:
            # Side view - vehicle perpendicular to camera viewing direction
            return 0.0
        elif ratio < 0.8:
            # Front/back view - vehicle parallel to camera viewing direction
            return np.pi / 2
        else:
            # Diagonal view - interpolate
            t = (ratio - 0.8) / 0.7  # Normalize to 0-1
            return np.pi / 2 * (1 - t)

    def get_default_dimensions(self, class_name: str = "car") -> Tuple[float, float, float]:
        """
        Get default vehicle dimensions based on class.

        Args:
            class_name: Vehicle class name

        Returns:
            (length, width, height) in meters
        """
        # Average dimensions for different vehicle types
        dimensions = {
            "car": (4.5, 1.8, 1.5),
            "sedan": (4.7, 1.8, 1.45),
            "suv": (4.8, 1.9, 1.7),
            "truck": (5.5, 2.0, 2.0),
            "van": (5.0, 2.0, 2.0),
            "motorcycle": (2.2, 0.8, 1.1),
            "bicycle": (1.8, 0.5, 1.0),
            "bus": (12.0, 2.5, 3.2),
        }
        return dimensions.get(class_name.lower(), (4.5, 1.8, 1.5))

    def detect(self, frame: np.ndarray) -> List[Box3D]:
        """
        Detect vehicles and estimate 3D bounding boxes.

        This method requires YOLO model to be loaded. For integration with
        external detection, use detect_from_boxes() instead.

        Args:
            frame: Input image (BGR format, undistorted)

        Returns:
            List of Box3D detections
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        if self.yolo_model is None:
            raise RuntimeError("YOLO model not loaded. Use detect_from_boxes() for external detection.")

        # Step 1: Run 2D detection
        results = self.yolo_model(frame, verbose=False)

        # Step 2: Get depth map
        depth_map = self.get_depth_map(frame)

        # Step 3: Convert 2D detections to 3D
        boxes_2d = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x, y, w, h = box.xywh[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                boxes_2d.append((x, y, w, h, conf))

        return self.detect_from_boxes(frame, boxes_2d, depth_map)

    def detect_from_boxes(self, frame: np.ndarray, boxes_2d: List[Tuple],
                          depth_map: np.ndarray = None,
                          track_ids: List[int] = None) -> List[Box3D]:
        """
        Create 3D bounding boxes from 2D detection results.

        This method is designed to work with external detection results
        (e.g., from the existing YOLOv8 pose model).

        Args:
            frame: Input image (BGR format, undistorted)
            boxes_2d: List of (x, y, w, h, confidence) tuples (center format)
            depth_map: Pre-computed depth map (optional, will compute if None)
            track_ids: Optional list of track IDs for each box

        Returns:
            List of Box3D detections
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Get depth map if not provided
        if depth_map is None:
            depth_map = self.get_depth_map(frame)

        detections = []
        frame_h, frame_w = frame.shape[:2]
        depth_h, depth_w = depth_map.shape[:2]

        for idx, box in enumerate(boxes_2d):
            if len(box) >= 5:
                cx, cy, w, h, conf = box[:5]
            else:
                cx, cy, w, h = box[:4]
                conf = 0.5

            # Scale coordinates to depth map size
            scale_x = depth_w / frame_w
            scale_y = depth_h / frame_h

            depth_cx = int(cx * scale_x)
            depth_cy = int(cy * scale_y)

            # Ensure within bounds
            depth_cx = max(0, min(depth_cx, depth_w - 1))
            depth_cy = max(0, min(depth_cy, depth_h - 1))

            # Sample depth at bbox center (use small neighborhood for robustness)
            neighborhood = 3
            x1 = max(0, depth_cx - neighborhood)
            x2 = min(depth_w, depth_cx + neighborhood + 1)
            y1 = max(0, depth_cy - neighborhood)
            y2 = min(depth_h, depth_cy + neighborhood + 1)

            relative_depth = np.median(depth_map[y1:y2, x1:x2])

            # Convert relative depth to absolute depth
            # Depth Anything outputs inverse depth (higher = closer)
            # Use scale factor to convert to meters
            if relative_depth > 0.01:
                depth = self.depth_scale * relative_depth
            else:
                depth = self.depth_scale  # Default depth

            # Clamp depth to reasonable range
            depth = max(5.0, min(depth, 50.0))  # 5-50 meters

            # Backproject to 3D
            center_3d = self.backproject_to_3d(cx, cy, depth)

            # Estimate dimensions and orientation
            dimensions = self.get_default_dimensions("car")
            yaw = self.estimate_yaw_from_bbox(w, h)

            # Create Box3D
            box_3d = Box3D(
                center=tuple(center_3d),
                dimensions=dimensions,
                yaw=yaw,
                confidence=conf,
                track_id=track_ids[idx] if track_ids and idx < len(track_ids) else None,
                bbox_2d=(cx, cy, w, h)
            )
            detections.append(box_3d)

        return detections

    def project_to_image(self, point_3d: np.ndarray) -> Tuple[float, float]:
        """
        Project 3D point to image coordinates.

        Args:
            point_3d: (3,) array of (x, y, z) in camera frame

        Returns:
            (u, v) pixel coordinates
        """
        x, y, z = point_3d
        if z <= 0:
            return (0, 0)

        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy

        return (u, v)

    def backproject_to_3d(self, u: float, v: float, depth: float) -> np.ndarray:
        """
        Backproject image point to 3D using depth.

        Args:
            u, v: Pixel coordinates
            depth: Depth in meters

        Returns:
            (3,) array of (x, y, z) in camera frame
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return np.array([x, y, z])


def load_or_calibrate_depth_scale(calibration_file: str = None,
                                   default_scale: float = 15.0) -> float:
    """
    Load depth scale from calibration file or return default.

    Args:
        calibration_file: Path to depth_calibration.json
        default_scale: Default scale factor if calibration not found

    Returns:
        Depth scale factor
    """
    import json
    import os

    if calibration_file is None:
        # Default location
        pipeline_dir = os.path.dirname(os.path.abspath(__file__))
        calibration_file = os.path.join(pipeline_dir, "depth_calibration.json")

    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                data = json.load(f)
            scale = data.get('scale_factor', default_scale)
            print(f"Loaded depth scale from calibration: {scale:.4f}")
            return scale
        except Exception as e:
            print(f"Warning: Could not load calibration file: {e}")

    print(f"Using default depth scale: {default_scale}")
    return default_scale
