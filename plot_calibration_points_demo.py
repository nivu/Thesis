#!/usr/bin/env python3
"""
Demo: Plot sample calibration points on the street image.

NOTE: This is a DEMONSTRATION with manually estimated pixel coordinates.
In reality, these points need to be clicked using the calibration tool.
"""

import cv2
import numpy as np
import os

def create_sample_pixel_mapping():
    """
    Create sample pixel coordinates for demonstration.
    In real calibration, these would be clicked manually.
    """
    # Sample mapping: [pixel_x, pixel_y, label, world_x, world_y, world_z]
    sample_points = [
        # Format: [pixel_x, pixel_y, "label", world_x, world_y, world_z]
        # These are ESTIMATED for demo purposes only
        [950, 650, "TS0038", 6.46, 11.23, -0.09],
        [1050, 550, "TS0037", 10.50, 15.53, -0.08],
        [1200, 450, "TS0036", 23.23, 43.61, 0.79],
        [850, 700, "TS0034", 12.51, 26.51, 0.17],
        [500, 680, "TS0033", -1.04, 23.28, -0.16],
        [600, 620, "TS0032", 1.41, 20.07, -0.05],
        [450, 580, "TS0031", -0.87, 16.50, -0.07],
        [300, 550, "TS0030", -9.85, 10.15, -0.17],
        [250, 600, "TS0029", -13.11, 19.06, -0.38],
        [200, 520, "TS0028", -17.63, 15.04, -0.36],
        [400, 850, "TS0027", -14.85, -5.04, 0.20],
        [350, 900, "TS0026", -21.41, -8.41, 0.42],
        [1400, 500, "TS0013", 43.02, -36.63, 1.33],
        [1300, 550, "TS0012", 40.39, -40.60, 1.25],
        [950, 750, "TS0011", 23.89, -29.29, 0.77],
    ]
    return sample_points

def plot_points_on_image(image_path, output_dir):
    """
    Plot calibration points on the street image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")

    # Get sample points
    points = create_sample_pixel_mapping()

    # Create annotated version
    annotated = image.copy()

    # Plot each point
    for px, py, label, wx, wy, wz in points:
        # Draw red dot
        cv2.circle(annotated, (int(px), int(py)), 8, (0, 0, 255), -1)

        # Draw white outline
        cv2.circle(annotated, (int(px), int(py)), 8, (255, 255, 255), 2)

        # Add label
        cv2.putText(annotated, label, (int(px) + 15, int(py) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(annotated, label, (int(px) + 15, int(py) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Add world coordinates
        coord_text = f"({wx:.1f}, {wy:.1f})"
        cv2.putText(annotated, coord_text, (int(px) + 15, int(py) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 2)
        cv2.putText(annotated, coord_text, (int(px) + 15, int(py) + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    # Add title and info
    title = "DEMO: Calibration Points (Sample Locations)"
    cv2.putText(annotated, title, (30, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(annotated, title, (30, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    # Add disclaimer
    disclaimer = "NOTE: Pixel coordinates are estimated for demonstration"
    cv2.putText(annotated, disclaimer, (30, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(annotated, disclaimer, (30, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

    # Add legend
    legend_y = image.shape[0] - 120
    cv2.rectangle(annotated, (20, legend_y - 10), (450, image.shape[0] - 20),
                 (255, 255, 255), -1)
    cv2.rectangle(annotated, (20, legend_y - 10), (450, image.shape[0] - 20),
                 (0, 0, 0), 2)

    cv2.circle(annotated, (40, legend_y + 15), 8, (0, 0, 255), -1)
    cv2.circle(annotated, (40, legend_y + 15), 8, (255, 255, 255), 2)
    cv2.putText(annotated, "= Calibration point", (60, legend_y + 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.putText(annotated, f"Total shown: {len(points)} points", (60, legend_y + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.putText(annotated, "(Real calibration: 35 points in Max_Pla.txt)", (60, legend_y + 75),
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

    # Save annotated version
    output_path = os.path.join(output_dir, "calibration_points_annotated.png")
    cv2.imwrite(output_path, annotated)
    print(f"✓ Saved: {output_path}")

    # Create version with just dots (no labels)
    dots_only = image.copy()
    for px, py, label, wx, wy, wz in points:
        cv2.circle(dots_only, (int(px), int(py)), 10, (0, 0, 255), -1)
        cv2.circle(dots_only, (int(px), int(py)), 10, (255, 255, 255), 2)

    cv2.putText(dots_only, "Calibration Points", (30, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
    cv2.putText(dots_only, "Calibration Points", (30, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    output_dots = os.path.join(output_dir, "calibration_points_dots_only.png")
    cv2.imwrite(output_dots, dots_only)
    print(f"✓ Saved: {output_dots}")

    # Create zoomed versions for a few points
    create_zoomed_views(image, points[:5], output_dir)

    print(f"\n{'='*60}")
    print("IMPORTANT NOTE:")
    print("="*60)
    print("The pixel coordinates shown here are ESTIMATED for demo purposes.")
    print("To get accurate calibration, you must:")
    print("  1. Run: python camera_calibration/main.py")
    print("  2. Manually click each point visible in the image")
    print("  3. Match with corresponding coordinates from Max_Pla.txt")
    print("="*60)

def create_zoomed_views(image, points, output_dir):
    """Create zoomed in views of specific calibration points"""
    for i, (px, py, label, wx, wy, wz) in enumerate(points[:3]):
        # Extract region around point
        margin = 100
        x1 = max(0, int(px) - margin)
        y1 = max(0, int(py) - margin)
        x2 = min(image.shape[1], int(px) + margin)
        y2 = min(image.shape[0], int(py) + margin)

        zoomed = image[y1:y2, x1:x2].copy()

        # Adjust point coordinates for zoomed view
        px_local = int(px) - x1
        py_local = int(py) - y1

        # Draw crosshair
        cv2.line(zoomed, (px_local - 30, py_local), (px_local + 30, py_local),
                (0, 255, 0), 2)
        cv2.line(zoomed, (px_local, py_local - 30), (px_local, py_local + 30),
                (0, 255, 0), 2)
        cv2.circle(zoomed, (px_local, py_local), 12, (0, 0, 255), -1)
        cv2.circle(zoomed, (px_local, py_local), 12, (255, 255, 255), 3)

        # Add info
        info_text = f"{label}: pixel({int(px)},{int(py)}) -> world({wx:.1f},{wy:.1f})"
        cv2.putText(zoomed, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3)
        cv2.putText(zoomed, info_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        output_path = os.path.join(output_dir, f"calibration_point_{i+1}_zoomed.png")
        cv2.imwrite(output_path, zoomed)
        print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    image_path = "camera_calibration/Beispieldaten/example-street.png"
    output_dir = "camera_calibration"

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        exit(1)

    plot_points_on_image(image_path, output_dir)
