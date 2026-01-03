import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def _calculate_axis_limits(all_positions):
    if len(all_positions) <= 1:
        raise ValueError("Not enough objects to plot BEV comparison.")
    x_positions = [pos[0] for pos in all_positions]
    z_positions = [pos[1] for pos in all_positions]
    x_min, x_max = min(x_positions), max(x_positions)
    z_min, z_max = min(z_positions), max(z_positions)
    x_range = x_max - x_min
    z_range = z_max - z_min
    x_margin = max(x_range * 0.2, 10)
    z_margin = max(z_range * 0.2, 10)
    x_lim = [x_min - x_margin, x_max + x_margin]
    z_lim = [max(0, z_min - z_margin), z_max + z_margin]
    return x_lim, z_lim

def _plot_fov_lines(fx, image_shape, z_max):
    if image_shape is None:
        return
    img_width = image_shape[1]
    fov_x = 2 * np.arctan(img_width / (2 * fx))
    for angle in [-fov_x / 2, fov_x / 2]:
        x_line = z_max * np.tan(angle)
        plt.plot([0, x_line], [0, z_max], 'k--', alpha=0.3, linewidth=1)

def _create_legend():
    legend_elements = [
        Line2D([0], [0], color='black', label='GT Boxes', linewidth=2),
        Line2D([0], [0], color='green', label='Predicted Boxes', linewidth=2)
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

def _get_bev_corners(center_3d, dimensions, yaw):
    """Retourne les 4 coins au sol (X,Z) de la boîte 3D"""
    h, w, l = dimensions  # h vertical, w côté gauche/droite, l avant/arrière
    x, _, z = center_3d

    # Coins locaux (X,Z) — Y est ignoré pour BEV
    x_corners = [ l/2,  l/2, -l/2, -l/2]
    z_corners = [ w/2, -w/2, -w/2,  w/2]

    corners = np.array([x_corners, z_corners])
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    rotated = R @ corners
    rotated[0, :] += x
    rotated[1, :] += z
    return rotated.T  # shape (4,2)

def plot_objects_as_boxes(objects, color='blue'):
    positions = []
    for obj in objects:
        if 'center_3d' not in obj or 'dimensions' not in obj:
            continue
        corners = _get_bev_corners(obj['center_3d'], obj['dimensions'], obj['yaw'])
        plt.plot(*np.append(corners, [corners[0]], axis=0).T, color=color, linewidth=2)

        # Centre
        x, z = obj['center_3d'][0], obj['center_3d'][2]
        positions.append((x, z))

        # Label
        plt.text(x, z, f"{obj.get('class_name','?')}",
                 ha='center', va='center', fontsize=7, color=color, weight='bold')
    return positions

def plot_kitti_comparison_bev(predicted_objects, ground_truth_objects, fx, image_shape=None,
                              title="3D Object Detection - BEV"):
    plt.figure(figsize=(14, 12))
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('X (right, meters)', fontsize=12)
    plt.ylabel('Z (forward, meters)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Camera position
    plt.plot(0, 0, '^', color='black', markersize=15, label='Camera')
    
    # Plot GT and Pred boxes
    gt_positions = plot_objects_as_boxes(ground_truth_objects, color='black')
    pred_positions = plot_objects_as_boxes(predicted_objects, color='green')
    
    # Limits
    all_positions = [(0,0)] + gt_positions + pred_positions
    x_lim, z_lim = _calculate_axis_limits(all_positions)
    
    # FOV lines
    _plot_fov_lines(fx, image_shape, z_lim[1])
    
    # Legend
    _create_legend()
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_lim)
    plt.ylim(z_lim)
    plt.tight_layout()
    return plt.gcf()