"""
Coordinate Fusion Module

Fuses 3D detection results with 2D homography-based estimates
to produce more accurate real-world coordinates.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.coordinate_transformer import CoordinateTransformer
from detector_3d import Box3D


@dataclass
class FusedResult:
    """Result of fusing 3D and 2D coordinate estimates."""

    # Final fused position (meters)
    position: Tuple[float, float]

    # Vehicle ground footprint corners (if available from 3D)
    ground_corners: Optional[np.ndarray]

    # Orientation (yaw in radians, if available from 3D)
    orientation: Optional[float]

    # Confidence score (0-1)
    confidence: float

    # Individual estimates for debugging
    position_3d: Optional[Tuple[float, float]]
    position_2d: Tuple[float, float]

    # Discrepancy between methods (meters)
    discrepancy: float

    # Method used: 'fused', '3d_only', '2d_only'
    method: str


class CoordinateFusion:
    """
    Fuses 3D detection with homography-based 2D estimation.

    Strategy:
    - Use both methods when available
    - Weight based on confidence and proximity to calibration points
    - Flag large discrepancies for review
    """

    def __init__(
        self,
        transformer: CoordinateTransformer,
        weight_3d: float = 0.6,
        weight_2d: float = 0.4,
        max_discrepancy: float = 1.0
    ):
        """
        Initialize fusion module.

        Args:
            transformer: Existing homography transformer
            weight_3d: Default weight for 3D estimates (0-1)
            weight_2d: Default weight for 2D estimates (0-1)
            max_discrepancy: Maximum allowed discrepancy before flagging (meters)
        """
        self.transformer = transformer
        self.weight_3d = weight_3d
        self.weight_2d = weight_2d
        self.max_discrepancy = max_discrepancy

        # Store calibration points for confidence calculation
        self.calibration_points = transformer.real_world_points

    def fuse(
        self,
        detection_3d: Optional[Box3D],
        contact_point_2d: Tuple[float, float],
        confidence_2d: float = 0.8,
        camera_matrix: np.ndarray = None
    ) -> FusedResult:
        """
        Fuse 3D and 2D coordinate estimates.

        Args:
            detection_3d: 3D bounding box detection (or None if not available)
            contact_point_2d: (u, v) pixel coordinates of ground contact point
            confidence_2d: Confidence of 2D detection (0-1)
            camera_matrix: Camera intrinsic matrix for 3D projection (optional)

        Returns:
            FusedResult with combined estimate
        """
        # Get 2D homography estimate
        pos_2d = self.transformer.pixel_to_world(
            contact_point_2d[0],
            contact_point_2d[1]
        )

        # If no 3D detection, return 2D only
        if detection_3d is None:
            return FusedResult(
                position=pos_2d,
                ground_corners=None,
                orientation=None,
                confidence=confidence_2d,
                position_3d=None,
                position_2d=pos_2d,
                discrepancy=0.0,
                method='2d_only'
            )

        # Get 3D estimate - transform to world coordinates via homography
        # The 3D box center is in camera frame, project to image then use homography
        if camera_matrix is not None and detection_3d.bbox_2d is not None:
            # Use the bbox center projected position with depth-adjusted offset
            cx, cy, w, h = detection_3d.bbox_2d
            # Project the 3D ground center to image coordinates
            center_3d = detection_3d.get_ground_center()  # (x, y) in camera frame at ground level

            # For surveillance camera, the depth gives us distance info
            # Transform the 3D position to world coords using homography of the projected point
            # Simple approach: use bbox bottom center (which we detected as contact point)
            # and adjust based on 3D depth information
            pos_3d = self.transformer.pixel_to_world(cx, cy + h/2)
        else:
            # Fallback: just use bbox center through homography
            if detection_3d.bbox_2d is not None:
                cx, cy, w, h = detection_3d.bbox_2d
                pos_3d = self.transformer.pixel_to_world(cx, cy + h/2)
            else:
                pos_3d = pos_2d

        conf_3d = detection_3d.confidence

        # Calculate discrepancy
        discrepancy = np.sqrt(
            (pos_3d[0] - pos_2d[0])**2 +
            (pos_3d[1] - pos_2d[1])**2
        )

        # Adjust weights based on confidence and discrepancy
        w3d, w2d = self._compute_weights(
            pos_3d, pos_2d, conf_3d, confidence_2d, discrepancy
        )

        # Weighted fusion
        total_weight = w3d + w2d
        fused_x = (pos_3d[0] * w3d + pos_2d[0] * w2d) / total_weight
        fused_y = (pos_3d[1] * w3d + pos_2d[1] * w2d) / total_weight
        fused_pos = (fused_x, fused_y)

        # Combined confidence
        fused_conf = (conf_3d * w3d + confidence_2d * w2d) / total_weight

        # Reduce confidence if large discrepancy
        if discrepancy > self.max_discrepancy:
            fused_conf *= 0.5

        # Get ground corners in world coordinates if camera matrix available
        if camera_matrix is not None:
            ground_corners = self.transform_3d_corners_to_world(detection_3d, camera_matrix)
        else:
            ground_corners = None

        return FusedResult(
            position=fused_pos,
            ground_corners=ground_corners,
            orientation=detection_3d.yaw,
            confidence=fused_conf,
            position_3d=pos_3d,
            position_2d=pos_2d,
            discrepancy=discrepancy,
            method='fused'
        )

    def _compute_weights(
        self,
        pos_3d: Tuple[float, float],
        pos_2d: Tuple[float, float],
        conf_3d: float,
        conf_2d: float,
        discrepancy: float
    ) -> Tuple[float, float]:
        """
        Compute adaptive weights for fusion.

        Weights are adjusted based on:
        - Detection confidence
        - Proximity to calibration points (favors 2D near known points)
        - Discrepancy between methods
        """
        # Start with default weights
        w3d = self.weight_3d * conf_3d
        w2d = self.weight_2d * conf_2d

        # Adjust based on proximity to calibration points
        if self.calibration_points is not None:
            min_dist_to_calib = self._min_distance_to_calibration(pos_2d)

            # If close to calibration points, trust homography more
            if min_dist_to_calib < 2.0:  # Within 2m of calibration point
                w2d *= 1.5
            elif min_dist_to_calib > 10.0:  # Far from calibration
                w3d *= 1.3

        # If large discrepancy, reduce weight of less confident method
        if discrepancy > self.max_discrepancy:
            if conf_3d > conf_2d:
                w2d *= 0.5
            else:
                w3d *= 0.5

        return w3d, w2d

    def _min_distance_to_calibration(self, position: Tuple[float, float]) -> float:
        """Calculate minimum distance to any calibration point."""
        if self.calibration_points is None:
            return float('inf')

        distances = np.sqrt(
            (self.calibration_points[:, 0] - position[0])**2 +
            (self.calibration_points[:, 1] - position[1])**2
        )
        return np.min(distances)

    def transform_3d_corners_to_world(
        self,
        box_3d: Box3D,
        camera_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Transform 3D box ground corners to real-world coordinates.

        This method projects the bottom 4 corners of the 3D bounding box
        back to image coordinates, then uses the homography transformation
        to get real-world coordinates.

        Args:
            box_3d: 3D bounding box detection
            camera_matrix: 3x3 camera intrinsic matrix (K)

        Returns:
            np.ndarray: (4, 2) array of ground corner coordinates in meters
        """
        # Extract camera intrinsics
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Get 3D ground corners in camera frame
        corners_3d = box_3d.get_ground_corners()  # (4, 3) array

        corners_world = []
        for corner in corners_3d:
            x, y, z = corner

            # Project 3D point to 2D pixel coordinates
            if z > 0:
                u = fx * x / z + cx
                v = fy * y / z + cy
            else:
                # Use bbox center as fallback
                if box_3d.bbox_2d is not None:
                    u, v = box_3d.bbox_2d[0], box_3d.bbox_2d[1]
                else:
                    continue

            # Use homography to transform to real-world coordinates
            world_pt = self.transformer.pixel_to_world(u, v)
            corners_world.append(world_pt)

        if len(corners_world) == 4:
            return np.array(corners_world)
        else:
            # Fallback: return estimated rectangle from center and dimensions
            center = box_3d.get_ground_center()
            l, w, _ = box_3d.dimensions
            yaw = box_3d.yaw

            # Create corners around center
            corners_local = np.array([
                [-l/2, -w/2],
                [l/2, -w/2],
                [l/2, w/2],
                [-l/2, w/2]
            ])

            # Rotate by yaw
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners_rotated = (rotation @ corners_local.T).T

            # Translate to center
            corners_world = corners_rotated + np.array([center[0], center[1]])
            return corners_world

    def get_vehicle_footprint(
        self,
        box_3d: Box3D,
        camera_matrix: np.ndarray
    ) -> dict:
        """
        Get complete vehicle footprint information.

        Args:
            box_3d: 3D bounding box detection
            camera_matrix: Camera intrinsic matrix

        Returns:
            dict with:
                - center: (x, y) center in meters
                - corners: (4, 2) corner coordinates in meters
                - length: Vehicle length in meters
                - width: Vehicle width in meters
                - yaw: Orientation in radians
        """
        corners = self.transform_3d_corners_to_world(box_3d, camera_matrix)
        center = box_3d.get_ground_center()
        l, w, _ = box_3d.dimensions

        return {
            'center': center,
            'corners': corners,
            'length': l,
            'width': w,
            'yaw': box_3d.yaw
        }


def associate_detections(
    detections_2d: List[dict],
    detections_3d: List[Box3D],
    iou_threshold: float = 0.5
) -> List[Tuple[dict, Optional[Box3D]]]:
    """
    Associate 3D detections with 2D tracked detections.

    Uses IoU of 2D bounding boxes for matching.

    Args:
        detections_2d: List of 2D detections with 'bbox' and 'track_id'
        detections_3d: List of 3D detections
        iou_threshold: Minimum IoU for association

    Returns:
        List of (detection_2d, detection_3d or None) pairs
    """
    results = []

    # Mark which 3D detections have been matched
    matched_3d = set()

    for det_2d in detections_2d:
        bbox_2d = det_2d['bbox']  # (x, y, w, h)
        best_match = None
        best_iou = iou_threshold

        for i, det_3d in enumerate(detections_3d):
            if i in matched_3d:
                continue

            if det_3d.bbox_2d is not None:
                iou = compute_iou(bbox_2d, det_3d.bbox_2d)
                if iou > best_iou:
                    best_iou = iou
                    best_match = (i, det_3d)

        if best_match is not None:
            matched_3d.add(best_match[0])
            results.append((det_2d, best_match[1]))
        else:
            results.append((det_2d, None))

    return results


def compute_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """
    Compute IoU between two boxes in (x_center, y_center, w, h) format.
    """
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    # Convert to (x1, y1, x2, y2) format
    box1_xyxy = (x1 - w1/2, y1 - h1/2, x1 + w1/2, y1 + h1/2)
    box2_xyxy = (x2 - w2/2, y2 - h2/2, x2 + w2/2, y2 + h2/2)

    # Intersection
    xi1 = max(box1_xyxy[0], box2_xyxy[0])
    yi1 = max(box1_xyxy[1], box2_xyxy[1])
    xi2 = min(box1_xyxy[2], box2_xyxy[2])
    yi2 = min(box1_xyxy[3], box2_xyxy[3])

    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0

    intersection = (xi2 - xi1) * (yi2 - yi1)

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
