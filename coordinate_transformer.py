"""
Coordinate Transformer Module
Handles conversion from pixel coordinates to real-world coordinates using homography transformation.
"""

import json
import numpy as np


class CoordinateTransformer:
    """Transforms pixel coordinates to real-world coordinates using homography matrix."""

    def __init__(self, mapping_file):
        """
        Initialize the coordinate transformer with a mapping file.

        Args:
            mapping_file: Path to JSON file containing transformation_matrix,
                         image_points, and real_world_points
        """
        self.transformation_matrix = None
        self.image_points = None
        self.real_world_points = None
        self._load_mapping(mapping_file)

    def _load_mapping(self, mapping_file):
        """Load the homography transformation matrix from JSON file."""
        try:
            with open(mapping_file, 'r') as f:
                data = json.load(f)

            self.transformation_matrix = np.array(data['transformation_matrix'])
            self.image_points = np.array(data['image_points'])
            self.real_world_points = np.array(data['real_world_points'])
            print(f"Coordinate mapping loaded successfully from {mapping_file}")

        except FileNotFoundError:
            print(f"Error: Mapping file '{mapping_file}' not found.")
            raise
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            raise

    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates to real-world coordinates (meters).

        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels

        Returns:
            Tuple of (world_x, world_y) in meters
        """
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix not loaded")

        # Create homogeneous pixel coordinate
        pixel_coord = np.array([pixel_x, pixel_y, 1.0])

        # Apply homography transformation
        world_homogeneous = self.transformation_matrix @ pixel_coord

        # Normalize by the third coordinate
        if world_homogeneous[2] != 0:
            world_x = world_homogeneous[0] / world_homogeneous[2]
            world_y = world_homogeneous[1] / world_homogeneous[2]
        else:
            world_x, world_y = 0.0, 0.0

        return (world_x, world_y)

    def world_to_pixel(self, world_x, world_y):
        """
        Convert real-world coordinates to pixel coordinates (inverse transform).

        Args:
            world_x: X coordinate in meters
            world_y: Y coordinate in meters

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        if self.transformation_matrix is None:
            raise ValueError("Transformation matrix not loaded")

        # Compute inverse of transformation matrix
        inv_matrix = np.linalg.inv(self.transformation_matrix)

        # Create homogeneous world coordinate
        world_coord = np.array([world_x, world_y, 1.0])

        # Apply inverse transformation
        pixel_homogeneous = inv_matrix @ world_coord

        # Normalize
        if pixel_homogeneous[2] != 0:
            pixel_x = pixel_homogeneous[0] / pixel_homogeneous[2]
            pixel_y = pixel_homogeneous[1] / pixel_homogeneous[2]
        else:
            pixel_x, pixel_y = 0.0, 0.0

        return (pixel_x, pixel_y)


def calculate_real_world_coordinates(scaled_boxes, transformer):
    """
    Calculate real-world coordinates for the bottom-center of each bounding box.
    This represents the approximate ground contact point of the vehicle.

    Args:
        scaled_boxes: List of bounding boxes in [x_center, y_center, width, height] format
        transformer: CoordinateTransformer instance

    Returns:
        List of (world_x, world_y) tuples in meters
    """
    real_world_coords = []

    for box in scaled_boxes:
        x, y, w, h = box
        # Calculate bottom-center point (vehicle ground contact approximation)
        bottom_center_x = x
        bottom_center_y = y + h / 2

        # Transform to real-world coordinates
        world_coord = transformer.pixel_to_world(bottom_center_x, bottom_center_y)
        real_world_coords.append(world_coord)

    return real_world_coords


def calculate_real_box_width(box, transformer):
    """
    Calculate the real-world width of a bounding box at its bottom edge.

    Args:
        box: Bounding box in [x_center, y_center, width, height] format
        transformer: CoordinateTransformer instance

    Returns:
        Real-world width in meters
    """
    x, y, w, h = box

    # Calculate bottom-left and bottom-right corners
    bottom_y = y + h / 2
    left_x = x - w / 2
    right_x = x + w / 2

    # Transform both corners to real-world coordinates
    left_world = transformer.pixel_to_world(left_x, bottom_y)
    right_world = transformer.pixel_to_world(right_x, bottom_y)

    # Calculate Euclidean distance in real-world coordinates
    real_width = np.sqrt(
        (right_world[0] - left_world[0])**2 +
        (right_world[1] - left_world[1])**2
    )

    return real_width


def calculate_point_real_world(pixel_x, pixel_y, transformer):
    """
    Calculate real-world coordinates for a single point.

    Args:
        pixel_x: X coordinate in pixels
        pixel_y: Y coordinate in pixels
        transformer: CoordinateTransformer instance

    Returns:
        Tuple of (world_x, world_y) in meters
    """
    return transformer.pixel_to_world(pixel_x, pixel_y)
