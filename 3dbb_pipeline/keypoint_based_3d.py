#!/usr/bin/env python3
"""
2D Detection + Calibration-based 3D Bounding Box Estimation

Uses 2D object detection and ground plane calibration to estimate 3D positions.
This approach is more suitable for static traffic cameras with downward angles.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
import argparse
from pathlib import Path


def load_calibration_lookup(calib_path):
    """Load the calibration lookup table (pixel -> real-world coordinates)."""
    lookup_table = np.load(calib_path)
    print(f"Loaded calibration lookup table: {lookup_table.shape}")
    return lookup_table


def pixel_to_world(pixel_x, pixel_y, lookup_table):
    """
    Convert pixel coordinates to real-world coordinates using lookup table.

    Args:
        pixel_x, pixel_y: Pixel coordinates
        lookup_table: Calibration lookup table (H, W, 3) -> (x, y, z) in meters

    Returns:
        world_x, world_y, world_z: Real-world coordinates in meters
    """
    h, w = lookup_table.shape[:2]

    # Clamp to valid range
    py = int(np.clip(pixel_y, 0, h - 1))
    px = int(np.clip(pixel_x, 0, w - 1))

    world_coords = lookup_table[py, px]
    return world_coords[0], world_coords[1], world_coords[2]


def estimate_3d_box_from_2d(bbox_2d, lookup_table, avg_car_dims=(4.5, 1.6, 1.8)):
    """
    Estimate 3D bounding box from 2D bbox using ground plane calibration.

    Args:
        bbox_2d: [x1, y1, x2, y2, conf, class_id]
        lookup_table: Calibration lookup table
        avg_car_dims: Average car dimensions (length, height, width) in meters

    Returns:
        box_3d: dict with 3D box information
    """
    x1, y1, x2, y2 = bbox_2d[:4]
    conf = bbox_2d[4] if len(bbox_2d) > 4 else 1.0

    # Bottom center point (where car touches ground)
    bottom_center_x = (x1 + x2) / 2
    bottom_center_y = y2  # Bottom of 2D box

    # Get real-world coordinates of bottom center
    world_x, world_y, world_z = pixel_to_world(bottom_center_x, bottom_center_y, lookup_table)

    # Estimate dimensions based on 2D box size and depth
    bbox_width_px = x2 - x1
    bbox_height_px = y2 - y1

    # Use average car dimensions (could be refined based on 2D box aspect ratio)
    length, height, width = avg_car_dims

    # Estimate orientation (simplified - assume parallel to road)
    rotation_y = 0.0  # This could be estimated from 2D box shape if needed

    box_3d = {
        'center': (world_x, world_y, world_z),
        'dimensions': (length, height, width),
        'rotation_y': rotation_y,
        'confidence': conf,
        'bottom_center_pixel': (bottom_center_x, bottom_center_y),
        'bbox_2d': (x1, y1, x2, y2)
    }

    return box_3d


def draw_3d_box_simplified(image, box_3d, lookup_table, color=(0, 255, 0)):
    """
    Draw a simplified 3D box by projecting 3D corners back to 2D.

    For a traffic camera view, we draw:
    - Ground rectangle (4 bottom corners)
    - Vertical lines to top corners
    - Top rectangle (4 top corners)
    """
    center = box_3d['center']
    length, height, width = box_3d['dimensions']
    rotation_y = box_3d['rotation_y']

    # Create 8 corners of 3D box in world coordinates
    # Center is at bottom-center of car
    cos_r = np.cos(rotation_y)
    sin_r = np.sin(rotation_y)

    # Half dimensions
    l2, h2, w2 = length / 2, height / 2, width / 2

    # 8 corners relative to center (bottom-center)
    corners_local = np.array([
        [-l2, 0, -w2],    # 0: front-left-bottom
        [ l2, 0, -w2],    # 1: front-right-bottom
        [ l2, 0,  w2],    # 2: back-right-bottom
        [-l2, 0,  w2],    # 3: back-left-bottom
        [-l2, -height, -w2],    # 4: front-left-top
        [ l2, -height, -w2],    # 5: front-right-top
        [ l2, -height,  w2],    # 6: back-right-top
        [-l2, -height,  w2],    # 7: back-left-top
    ])

    # Rotate corners
    rotation_matrix = np.array([
        [cos_r, 0, sin_r],
        [0, 1, 0],
        [-sin_r, 0, cos_r]
    ])
    corners_rotated = corners_local @ rotation_matrix.T

    # Translate to world position
    corners_world = corners_rotated + np.array(center)

    # Project back to pixel coordinates
    # We need to find pixels that correspond to these world coordinates
    # This is inverse of the lookup table - we'll use a simple approximation

    # For visualization, just draw the 2D bbox with height indication
    x1, y1, x2, y2 = box_3d['bbox_2d']

    # Draw 2D bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Draw bottom center marker
    bcx, bcy = box_3d['bottom_center_pixel']
    cv2.circle(image, (int(bcx), int(bcy)), 5, (0, 0, 255), -1)

    # Add text with 3D info
    cx, cy, cz = center
    text = f"({cx:.1f}, {cy:.1f}, {cz:.1f})m"
    cv2.putText(image, text, (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


def process_image(image_path, output_path, calib_path, model_name='yolo11n.pt', conf_threshold=0.3):
    """Process image with 2D detection + calibration-based 3D estimation."""

    # Load calibration
    print(f"Loading calibration from {calib_path}...")
    lookup_table = load_calibration_lookup(calib_path)

    # Load YOLO model
    print(f"Loading YOLO model: {model_name}...")
    model = YOLO(model_name)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = image.shape[:2]
    print(f"Image: {w}x{h}")

    # Run detection
    print("Running detection...")
    results = model(image, conf=conf_threshold, verbose=False)[0]

    # Filter for vehicles (car, truck, bus, motorcycle)
    vehicle_classes = [2, 5, 7, 3]  # COCO: car, bus, truck, motorcycle
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in vehicle_classes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf, cls])

    print(f"Detected {len(detections)} vehicles")

    # Estimate 3D boxes
    boxes_3d = []
    result_image = image.copy()

    for det in detections:
        box_3d = estimate_3d_box_from_2d(det, lookup_table)
        boxes_3d.append(box_3d)

        # Draw
        result_image = draw_3d_box_simplified(result_image, box_3d, lookup_table)

        # Print info
        cx, cy, cz = box_3d['center']
        print(f"  Vehicle at world coords: ({cx:.2f}, {cy:.2f}, {cz:.2f})m, "
              f"conf={box_3d['confidence']:.3f}")

    # Save result
    cv2.imwrite(output_path, result_image)
    print(f"\nOutput saved to: {output_path}")

    # Save JSON
    json_path = output_path.replace('.png', '.json').replace('.jpg', '.json')
    with open(json_path, 'w') as f:
        json.dump(boxes_3d, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"3D data saved to: {json_path}")

    return result_image, boxes_3d


def main():
    parser = argparse.ArgumentParser(description='2D Detection + Calibration-based 3D Estimation')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output image path')
    parser.add_argument('--calib', '-c', type=str,
                        default='../custom_dataset/calibration/calibration-lookup-table.npy',
                        help='Calibration lookup table path')
    parser.add_argument('--model', '-m', type=str, default='yolo11n.pt',
                        help='YOLO model name')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='Confidence threshold')

    args = parser.parse_args()

    if args.output is None:
        base = Path(args.input).stem
        args.output = f"{base}_calib3d.png"

    process_image(args.input, args.output, args.calib, args.model, args.conf)


if __name__ == '__main__':
    main()
