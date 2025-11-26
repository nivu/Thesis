#!/usr/bin/env python3
"""
Visualize calibration points on the example street image.
Reads coordinates from Max_Pla.txt and plots them on example-street.png
"""

import cv2
import numpy as np
import os

def load_3d_points(txt_file):
    """
    Load 3D calibration points from text file.
    Format: ID,X,Y,Z
    """
    points = []
    labels = []

    with open(txt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    label = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])

                    points.append([x, y, z])
                    labels.append(label)
                except ValueError:
                    print(f"Skipping invalid line: {line}")

    return np.array(points), labels

def visualize_calibration_points(image_path, points_file, output_path):
    """
    Visualize calibration points on the street image.
    Note: We don't have the 2D pixel coordinates yet, so we'll just
    show the 3D coordinate information.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image loaded: {image.shape[1]}x{image.shape[0]}")

    # Load 3D points
    points_3d, labels = load_3d_points(points_file)
    print(f"Loaded {len(points_3d)} calibration points")

    # Create a copy for annotation
    annotated = image.copy()

    # Create an info panel
    info_height = 800
    info_width = 500
    info_panel = np.ones((info_height, info_width, 3), dtype=np.uint8) * 240

    # Add title
    cv2.putText(info_panel, "Calibration Points", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(info_panel, f"Total: {len(points_3d)} points", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Add coordinate info
    y_offset = 100
    for i, (point, label) in enumerate(zip(points_3d[:20], labels[:20])):  # First 20 points
        text = f"{label}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"
        cv2.putText(info_panel, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        y_offset += 25

        if y_offset > info_height - 50:
            break

    if len(points_3d) > 20:
        cv2.putText(info_panel, f"... and {len(points_3d) - 20} more", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Add reference frame info
    y_offset = info_height - 150
    cv2.putText(info_panel, "Coordinate System:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y_offset += 30
    cv2.putText(info_panel, "X: Lateral (left-right)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)
    y_offset += 25
    cv2.putText(info_panel, "Y: Longitudinal (forward)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)
    y_offset += 25
    cv2.putText(info_panel, "Z: Height (elevation)", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 0), 1)
    y_offset += 35

    # Add coordinate range info
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]
    cv2.putText(info_panel, f"X range: [{x_coords.min():.1f}, {x_coords.max():.1f}]m",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    y_offset += 20
    cv2.putText(info_panel, f"Y range: [{y_coords.min():.1f}, {y_coords.max():.1f}]m",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)

    # Add a note about visualization
    note_y = 780
    cv2.putText(info_panel, "Note: Use calibration tool to", (10, note_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 0, 0), 1)
    cv2.putText(info_panel, "map these to image pixels", (10, note_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 0, 0), 1)

    # Resize image if needed to fit next to info panel
    max_height = info_height
    if annotated.shape[0] > max_height:
        scale = max_height / annotated.shape[0]
        new_width = int(annotated.shape[1] * scale)
        annotated = cv2.resize(annotated, (new_width, max_height))

    # Combine image and info panel
    combined = np.hstack([annotated, info_panel])

    # Save result
    cv2.imwrite(output_path, combined)
    print(f"✓ Saved visualization to: {output_path}")

    # Print statistics
    print(f"\nCalibration Statistics:")
    print(f"  X range: [{x_coords.min():.2f}, {x_coords.max():.2f}] meters")
    print(f"  Y range: [{y_coords.min():.2f}, {y_coords.max():.2f}] meters")
    print(f"  Coverage area: {x_coords.max() - x_coords.min():.1f}m × {y_coords.max() - y_coords.min():.1f}m")

    return combined

if __name__ == "__main__":
    image_path = "camera_calibration/Beispieldaten/example-street.png"
    points_file = "camera_calibration/Beispieldaten/Max_Pla.txt"
    output_path = "calibration_visualization.png"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        exit(1)

    if not os.path.exists(points_file):
        print(f"Error: Points file not found at {points_file}")
        exit(1)

    visualize_calibration_points(image_path, points_file, output_path)
