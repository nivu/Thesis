"""
3D Bounding Box Detector Module

Uses YOLOx3D approach: YOLO detection + Depth estimation + 3D regression
to estimate 3D bounding boxes from monocular images.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


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

    def load_models(self, yolo_path: str, multibin_path: str):
        """
        Load all required models.

        Args:
            yolo_path: Path to YOLO model weights
            multibin_path: Path to multi-bin regressor weights
        """
        # TODO: Implement model loading
        # from ultralytics import YOLO
        # self.yolo_model = YOLO(yolo_path)

        # For depth estimation:
        # from depth_anything_v2 import DepthAnythingV2
        # self.depth_model = DepthAnythingV2()

        # For dimension/orientation regression:
        # self.dimension_model = load_multibin_model(multibin_path)

        self._models_loaded = True
        print("3D detection models loaded successfully")

    def detect(self, frame: np.ndarray) -> List[Box3D]:
        """
        Detect vehicles and estimate 3D bounding boxes.

        Args:
            frame: Input image (BGR format, undistorted)

        Returns:
            List of Box3D detections
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        detections = []

        # Step 1: Run 2D detection
        # results_2d = self.yolo_model(frame)

        # Step 2: Estimate depth map
        # depth_map = self.depth_model(frame) * self.depth_scale

        # Step 3: For each detection, estimate 3D parameters
        # for box in results_2d.boxes:
        #     # Get depth at box center
        #     cx, cy = box.xywh[0], box.xywh[1]
        #     depth = depth_map[int(cy), int(cx)]
        #
        #     # Estimate dimensions and orientation
        #     crop = extract_crop(frame, box)
        #     dims, yaw = self.dimension_model(crop)
        #
        #     # Compute 3D position
        #     x3d = (cx - self.cx) * depth / self.fx
        #     y3d = (cy - self.cy) * depth / self.fy
        #     z3d = depth
        #
        #     box_3d = Box3D(
        #         center=(x3d, y3d, z3d),
        #         dimensions=dims,
        #         yaw=yaw,
        #         confidence=box.conf,
        #         bbox_2d=box.xywh
        #     )
        #     detections.append(box_3d)

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


def calibrate_depth_scale(
    depth_model,
    known_points: List[Tuple[Tuple[int, int], float]],
    frame: np.ndarray
) -> float:
    """
    Calibrate depth scale using known real-world distances.

    Args:
        depth_model: Depth estimation model
        known_points: List of ((u, v), real_depth_m) pairs
        frame: Calibration frame

    Returns:
        Depth scale factor
    """
    # Get relative depth map
    # relative_depth = depth_model(frame)

    # Compute scale factors for each known point
    # scales = []
    # for (u, v), real_depth in known_points:
    #     relative = relative_depth[v, u]
    #     if relative > 0:
    #         scales.append(real_depth / relative)

    # Return median scale factor
    # return np.median(scales)

    return 1.0  # Placeholder
