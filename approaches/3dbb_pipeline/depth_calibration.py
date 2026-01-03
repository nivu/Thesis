"""
Depth Calibration Module for 3D Bounding Box Pipeline

This module calibrates the depth scale factor using known real-world
reference points from the street plane calibration data.

Monocular depth estimation (Depth Anything v2) outputs relative depth.
This module computes the scale factor to convert to absolute depth in meters.
"""

import json
import numpy as np
import cv2
import os
from PIL import Image

# Camera position relative to the origin (from calibration data)
# The camera is mounted at approximately (-0.21m, -8.37m) from origin
CAMERA_POSITION = np.array([-0.21, -8.37])

# Estimated camera height above ground (meters)
# This is approximate - can be refined through calibration
CAMERA_HEIGHT = 10.0  # meters


def load_calibration_points(mapping_file):
    """
    Load calibration points from the coordinate mapping file.

    Args:
        mapping_file: Path to coordinate_mapping_2030.json

    Returns:
        image_points: List of (px, py) pixel coordinates
        real_world_points: List of (x, y) real-world coordinates in meters
    """
    with open(mapping_file, 'r') as f:
        data = json.load(f)

    image_points = [tuple(p) for p in data['image_points']]
    real_world_points = [tuple(p) for p in data['real_world_points']]

    return image_points, real_world_points


def compute_distance_from_camera(real_world_point, camera_height=CAMERA_HEIGHT):
    """
    Compute 3D distance from camera to a point on the ground plane.

    Args:
        real_world_point: (x, y) coordinates on ground plane in meters
        camera_height: Camera height above ground in meters

    Returns:
        distance: 3D Euclidean distance from camera to point
    """
    # Ground plane distance
    dx = real_world_point[0] - CAMERA_POSITION[0]
    dy = real_world_point[1] - CAMERA_POSITION[1]
    ground_dist = np.sqrt(dx**2 + dy**2)

    # 3D distance including height
    distance = np.sqrt(ground_dist**2 + camera_height**2)

    return distance


def calibrate_depth_scale(depth_model, frame, mapping_file, verbose=True):
    """
    Calibrate the depth scale factor using known reference points.

    The depth model outputs relative depth values. This function computes
    the scale factor to convert to absolute depth in meters.

    Args:
        depth_model: HuggingFace depth estimation pipeline
        frame: Input frame (BGR numpy array, undistorted)
        mapping_file: Path to coordinate mapping JSON file
        verbose: Print calibration details

    Returns:
        scale_factor: Multiply depth model output by this to get meters
        calibration_data: Dict with calibration details
    """
    # Load calibration points
    image_points, real_world_points = load_calibration_points(mapping_file)

    if verbose:
        print(f"Loaded {len(image_points)} calibration points")
        print(f"Camera position: {CAMERA_POSITION}")
        print(f"Camera height: {CAMERA_HEIGHT}m")

    # Convert frame to PIL Image for depth model
    if isinstance(frame, np.ndarray):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
    else:
        pil_image = frame

    # Get depth map from model
    if verbose:
        print("Running depth estimation...")

    depth_result = depth_model(pil_image)
    depth_map = np.array(depth_result['depth'])

    # Normalize depth map if needed
    if depth_map.max() > 1.0:
        depth_map = depth_map / depth_map.max()

    if verbose:
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth map range: [{depth_map.min():.4f}, {depth_map.max():.4f}]")

    # Compute scale factors for each calibration point
    scale_factors = []
    point_details = []

    for (px, py), (wx, wy) in zip(image_points, real_world_points):
        # Get real 3D distance to this point
        real_distance = compute_distance_from_camera((wx, wy))

        # Get depth value at this pixel
        # Note: px, py are in display coordinates, may need to scale
        h, w = depth_map.shape[:2]

        # Scale pixel coordinates to depth map size
        scale_x = w / 1920  # Assuming display size is 1920
        scale_y = h / 1080  # Assuming display size is 1080

        dpx = int(px * scale_x)
        dpy = int(py * scale_y)

        # Ensure within bounds
        dpx = max(0, min(dpx, w - 1))
        dpy = max(0, min(dpy, h - 1))

        # Sample depth at this location (use small neighborhood for robustness)
        neighborhood = 5
        x1 = max(0, dpx - neighborhood)
        x2 = min(w, dpx + neighborhood + 1)
        y1 = max(0, dpy - neighborhood)
        y2 = min(h, dpy + neighborhood + 1)

        relative_depth = np.median(depth_map[y1:y2, x1:x2])

        if relative_depth > 0.01:  # Avoid division by zero
            # Depth Anything outputs inverse depth (closer = higher value)
            # Scale factor = real_distance * relative_depth (for inverse depth)
            # or real_distance / relative_depth (for direct depth)

            # Depth Anything v2 typically outputs inverse depth
            # Higher values = closer objects
            scale = real_distance * relative_depth
            scale_factors.append(scale)

            point_details.append({
                'pixel': (px, py),
                'world': (wx, wy),
                'real_distance': real_distance,
                'relative_depth': float(relative_depth),
                'scale': scale
            })

            if verbose:
                print(f"  Point ({wx:.2f}, {wy:.2f}): "
                      f"real_dist={real_distance:.2f}m, "
                      f"rel_depth={relative_depth:.4f}, "
                      f"scale={scale:.2f}")

    if not scale_factors:
        raise ValueError("No valid calibration points found")

    # Use median for robustness against outliers
    final_scale = np.median(scale_factors)

    if verbose:
        print(f"\nScale factors: {[f'{s:.2f}' for s in scale_factors]}")
        print(f"Final scale factor (median): {final_scale:.4f}")
        print(f"Scale std dev: {np.std(scale_factors):.4f}")

    calibration_data = {
        'scale_factor': float(final_scale),
        'camera_height': CAMERA_HEIGHT,
        'camera_position': CAMERA_POSITION.tolist(),
        'num_points': len(scale_factors),
        'scale_std': float(np.std(scale_factors)),
        'point_details': point_details
    }

    return final_scale, calibration_data


def save_calibration(calibration_data, output_file):
    """Save calibration data to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    print(f"Calibration saved to: {output_file}")


def load_calibration(calibration_file):
    """
    Load previously computed calibration.

    Args:
        calibration_file: Path to depth_calibration.json

    Returns:
        scale_factor: Depth scale factor
        calibration_data: Full calibration data dict
    """
    if not os.path.exists(calibration_file):
        return None, None

    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)

    return calibration_data['scale_factor'], calibration_data


def run_calibration(video_path, calibration_file, mapping_file, output_file,
                    frame_index=0):
    """
    Run depth calibration on a video frame.

    Args:
        video_path: Path to video file
        calibration_file: Path to camera calibration npz
        mapping_file: Path to coordinate mapping JSON
        output_file: Path to save calibration results
        frame_index: Which frame to use for calibration
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.preprocess import preprocess_frame, load_calibration_data
    from transformers import pipeline

    print("=" * 60)
    print("DEPTH CALIBRATION")
    print("=" * 60)

    # Load camera calibration
    print(f"\nLoading camera calibration: {calibration_file}")
    K, D, DIM = load_calibration_data(calibration_file)

    # Open video and get calibration frame
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Skip to desired frame
    for _ in range(frame_index):
        cap.read()

    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(f"Could not read frame {frame_index}")

    # Preprocess frame (undistort fisheye)
    recognition_frame, display_frame = preprocess_frame(
        frame, K, D, DIM, (640, 640), (1920, 1080)
    )

    # Load depth model
    print("\nLoading Depth Anything v2 model...")
    depth_model = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf"
    )

    # Run calibration
    print("\nRunning calibration...")
    scale_factor, calibration_data = calibrate_depth_scale(
        depth_model, display_frame, mapping_file, verbose=True
    )

    # Save results
    save_calibration(calibration_data, output_file)

    print("\n" + "=" * 60)
    print(f"CALIBRATION COMPLETE")
    print(f"Scale factor: {scale_factor:.4f}")
    print("=" * 60)

    return scale_factor, calibration_data


if __name__ == "__main__":
    # Run calibration from command line
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate depth scale factor')
    parser.add_argument('--video', type=str,
                       default='traffic_analyis_data/Uni_west_1/GOPR0574.MP4',
                       help='Path to video file')
    parser.add_argument('--calibration', type=str,
                       default='gopro_calibration_fisheye.npz',
                       help='Path to camera calibration file')
    parser.add_argument('--mapping', type=str,
                       default='coordinate_mapping_2030.json',
                       help='Path to coordinate mapping file')
    parser.add_argument('--output', type=str,
                       default='3dbb_pipeline/depth_calibration.json',
                       help='Output calibration file')
    parser.add_argument('--frame', type=int, default=100,
                       help='Frame index to use for calibration')

    args = parser.parse_args()

    # Make paths absolute
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(base_dir, args.video)
    calibration_file = os.path.join(base_dir, args.calibration)
    mapping_file = os.path.join(base_dir, args.mapping)
    output_file = os.path.join(base_dir, args.output)

    run_calibration(video_path, calibration_file, mapping_file, output_file,
                   frame_index=args.frame)
