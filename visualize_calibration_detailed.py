#!/usr/bin/env python3
"""
Create a detailed visualization showing the calibration concept
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

def load_3d_points(txt_file):
    """Load 3D calibration points"""
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
                    continue

    return np.array(points), labels

def create_detailed_visualization(image_path, points_file, output_path):
    """Create a detailed multi-panel visualization"""

    # Load data
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    points_3d, labels = load_3d_points(points_file)

    print(f"Image: {image.shape[1]}x{image.shape[0]}")
    print(f"Points: {len(points_3d)}")

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Original street image
    ax1 = fig.add_subplot(gs[0, :])
    ax1.imshow(image_rgb)
    ax1.set_title('Example Street Image from Camera\n(Calibration points need to be manually marked on this image)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Pixel X', fontsize=10)
    ax1.set_ylabel('Pixel Y', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Add annotation
    ax1.text(50, 100, 'Step 1: Click reference points\nvisible in this image',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
             fontsize=11, color='red', fontweight='bold')

    # Panel 2: Top-down view of calibration points (X-Y plane)
    ax2 = fig.add_subplot(gs[1, 0])
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]

    # Plot points
    scatter = ax2.scatter(x_coords, y_coords, c=range(len(points_3d)),
                          cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=1)

    # Annotate some key points
    for i in [0, 5, 10, 15, 20]:
        if i < len(labels):
            ax2.annotate(labels[i], (x_coords[i], y_coords[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('X (meters) - Lateral', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Y (meters) - Longitudinal', fontsize=10, fontweight='bold')
    ax2.set_title('Top-Down View: Real-World Coordinates (Bird\'s Eye)',
                  fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Point Index', fontsize=9)

    # Panel 3: Coordinate statistics and info
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    info_text = f"""
CALIBRATION POINT SUMMARY
{'='*40}

Total Points: {len(points_3d)}

Coordinate Ranges:
  X: [{x_coords.min():.2f}, {x_coords.max():.2f}] m
  Y: [{y_coords.min():.2f}, {y_coords.max():.2f}] m
  Z: [{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}] m

Coverage Area:
  Width:  {x_coords.max() - x_coords.min():.1f} m
  Depth:  {y_coords.max() - y_coords.min():.1f} m

Sample Points:
"""

    # Add some sample points
    for i in range(min(10, len(points_3d))):
        info_text += f"\n  {labels[i]}: "
        info_text += f"({points_3d[i][0]:.1f}, {points_3d[i][1]:.1f}, {points_3d[i][2]:.1f})"

    if len(points_3d) > 10:
        info_text += f"\n  ... and {len(points_3d) - 10} more points"

    info_text += f"""

{'='*40}
HOW CALIBRATION WORKS:
{'='*40}

1. Open calibration tool with street image
2. For each point in Max_Pla.txt:
   - Find the point in the image
   - Click it to record pixel (x, y)
   - Link to real-world (X, Y, Z)
3. Compute homography matrix H
4. Result: pixel → world transformation

Formula:
  [X]   [H11 H12 H13]   [x]
  [Y] = [H21 H22 H23] × [y]
  [1]   [H31 H32 H33]   [1]

This enables: any street pixel → meters
"""

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Overall title
    fig.suptitle('Camera Calibration Data: Max Planck Street Example',
                fontsize=16, fontweight='bold', y=0.98)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed visualization to: {output_path}")
    plt.close()

    # Also save a simple version with just the image
    fig2, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.imshow(image_rgb)
    ax.set_title('Example Street Image - Ready for Calibration\n' +
                f'{len(points_3d)} reference points available in Max_Pla.txt',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Pixel X', fontsize=11)
    ax.set_ylabel('Pixel Y', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

    # Add instruction box
    instruction = ('CALIBRATION WORKFLOW:\n'
                  '1. Run: python camera_calibration/main.py\n'
                  '2. Click visible reference points in this image\n'
                  '3. Match with coordinates from Max_Pla.txt\n'
                  '4. System computes transformation matrix\n'
                  '5. Result: pixel → real-world coordinate mapping')

    ax.text(0.02, 0.98, instruction,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=1', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=2),
           family='monospace')

    simple_output = output_path.replace('.png', '_simple.png')
    plt.savefig(simple_output, dpi=150, bbox_inches='tight')
    print(f"✓ Saved simple visualization to: {simple_output}")
    plt.close()

if __name__ == "__main__":
    create_detailed_visualization(
        "camera_calibration/Beispieldaten/example-street.png",
        "camera_calibration/Beispieldaten/Max_Pla.txt",
        "calibration_visualization_detailed.png"
    )
