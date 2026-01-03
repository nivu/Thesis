from email import errors
from typing import List, Dict, Optional, Tuple
from scipy.spatial import ConvexHull
import numpy as np
from utils.kitti_utils import get_3d_box_corners
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def calculate_iou_2d(bbox1, bbox2):
    """Calculate IoU between two 2D bounding boxes."""
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_iou_3d(box1, box2):
    """
    box1, box2 : dict avec keys :
        - center_3d: [x, y, z]
        - dimensions: [h, w, l]
        - yaw: float (rad)
    Returns IoU 3D
    """
    c1 = get_3d_box_corners(box1['center_3d'], box1['dimensions'], box1['yaw'])
    c2 = get_3d_box_corners(box2['center_3d'], box2['dimensions'], box2['yaw'])
    
    poly1 = Polygon(c1[[0, 1, 2, 3]][:, [0, 2]])
    poly2 = Polygon(c2[[0, 1, 2, 3]][:, [0, 2]])

    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    inter_area = poly1.intersection(poly2).area
    if inter_area == 0:
        return 0.0
    
    y1_min, y1_max = np.min(c1[:,1]), np.max(c1[:,1])
    y2_min, y2_max = np.min(c2[:,1]), np.max(c2[:,1])
    
    inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    if inter_h == 0:
        return 0.0
    
    inter_vol = inter_area * inter_h
    
    vol1 = np.prod(box1['dimensions'])
    vol2 = np.prod(box2['dimensions'])
    
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

def match_predictions_to_gt(predictions: List[Dict], gt_objects: List[Dict], iou_threshold: float = 0.5) -> Tuple[List[Tuple[Optional[Dict], Dict]], int]:
    """
    Match predictions to ground truth objects based on 2D IoU.

    Returns:
        - list of (matched_prediction, gt_object)
        - count of false negatives (GT with no matching prediction)
    """
    matches = []
    used_predictions = set()
    
    false_negatives = 0
    
    for gt_obj in gt_objects:
        gt_bbox = gt_obj['bbox_2d']
        best_match, best_iou, best_idx = None, 0.0, -1
        
        for i, pred_obj in enumerate(predictions):
            if i in used_predictions:
                continue
            iou = calculate_iou_2d(gt_bbox, pred_obj['bbox_2d'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou, best_match, best_idx = iou, pred_obj, i
        
        if best_match is not None:
            used_predictions.add(best_idx)
            matches.append((best_match, gt_obj))
        else:
            matches.append((None, gt_obj))
            false_negatives += 1
    
    return matches, false_negatives


def calculate_rmse(errors: List[float]) -> float:
    """Calculate Root Mean Square Error."""
    return np.sqrt(np.mean(np.array(errors)**2)) if errors else 0.0


def calculate_orientation_error(rotation_y_pred: float, rotation_y_gt: float) -> float:
    """Calculate orientation error in radians."""
    error = abs(rotation_y_pred - rotation_y_gt)
    return min(error, 2 * np.pi - error)


def calculate_delta_accuracy(list_depth_pred, list_depth_gt, tau: float = 1.25) -> float:
    """Calculate δ-accuracy for depth estimation."""
    pred = np.array(list_depth_pred, dtype=np.float32)
    gt = np.array(list_depth_gt, dtype=np.float32)
    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    ratios = np.maximum(pred / gt, gt / pred)
    return np.mean(ratios < tau)


def calculate_errors(matches: List[Tuple[Optional[Dict], Dict]]) -> Dict[str, List[float]]:
    """Calculate various errors for matched objects."""
    errors = {k: [] for k in ['x','y','z','euclidean','w','h','l','classes',
                              'iou_2d','iou_3d','orientation','depths_gt','depths_pred']}
    
    for pred_obj, gt_obj in matches:
        if pred_obj is None:
            continue
        
        pred_center, gt_center = pred_obj['center_3d'], gt_obj['center_3d']
        errors['x'].append(abs(pred_center[0] - gt_center[0]))
        errors['y'].append(abs(pred_center[1] - gt_center[1]))
        errors['z'].append(abs(pred_center[2] - gt_center[2]))
        errors['euclidean'].append(np.linalg.norm(np.array(pred_center) - np.array(gt_center)))
        
        errors['h'].append(abs(pred_obj['dimensions'][0] - gt_obj['dimensions'][0]))
        errors['w'].append(abs(pred_obj['dimensions'][1] - gt_obj['dimensions'][1]))
        errors['l'].append(abs(pred_obj['dimensions'][2] - gt_obj['dimensions'][2]))
        
        errors['classes'].append(gt_obj['class_name'])
        errors['iou_2d'].append(calculate_iou_2d(pred_obj['bbox_2d'], gt_obj['bbox_2d']))
        errors['iou_3d'].append(calculate_iou_3d(pred_obj, gt_obj))
        errors['orientation'].append(calculate_orientation_error(pred_obj['yaw'], gt_obj['yaw']))
        
        errors['depths_gt'].append(gt_center[2])
        errors['depths_pred'].append(pred_center[2])
    
    return errors

def calculate_precision_recall_3d(iou_3d_list: List[float], false_negatives: int, threshold: float = 0.5):
    """
    Calculate precision and recall for 3D detection based on IoU threshold.
    
    Args:
        iou_3d_list: List of IoU values of matched predictions
        false_negatives: Number of GT objects without matched predictions
        threshold: IoU threshold for a true positive
        
    Returns:
        precision, recall
    """
    tp = sum(iou >= threshold for iou in iou_3d_list)
    fp = sum(iou < threshold for iou in iou_3d_list)
    fn = false_negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall



def get_metrics(predictions: List[Dict], gt_objects: List[Dict], iou_threshold: float = 0.5):
    """Get evaluation metrics for predictions vs ground truth."""
    matches, false_negatives = match_predictions_to_gt(predictions, gt_objects, iou_threshold)
    errors = calculate_errors(matches)

    # Depth metrics
    delta1 = calculate_delta_accuracy(errors['depths_pred'], errors['depths_gt'], tau=1.25)
    delta2 = calculate_delta_accuracy(errors['depths_pred'], errors['depths_gt'], tau=1.25**2)
    delta3 = calculate_delta_accuracy(errors['depths_pred'], errors['depths_gt'], tau=1.25**3)
    depth_rmse = calculate_rmse([p - g for p, g in zip(errors['depths_pred'], errors['depths_gt'])])

    # Detection quality
    precision_3d, recall_3d = calculate_precision_recall_3d(errors['iou_3d'], false_negatives, threshold=iou_threshold)

    metrics = {
        'average_iou_2d': np.mean(errors['iou_2d']) if errors['iou_2d'] else 0,
        'average_iou_3d': np.mean(errors['iou_3d']) if errors['iou_3d'] else 0,
        'average_orientation_error': np.mean(errors['orientation']) if errors['orientation'] else 0,
        
        # Depth
        'depth_rmse': depth_rmse,
        'delta1': delta1,
        'delta2': delta2,
        'delta3': delta3,
        
        # Detection
        'precision_3d': precision_3d,
        'recall_3d': recall_3d
    }
    return metrics

def plot_error_distributions(errors: dict, bins: int = 30):
    """
    Plot histograms for different 3D detection error metrics in a 2x2 layout.

    Args:
        errors (dict): Contains 'euclidean', 'depths', 'orientation', 'iou_3d' 
                       (+ optionally 'depths_gt')
        bins (int): Number of bins for the histograms.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Distribution of 3D errors and metrics", fontsize=16, fontweight="bold")

    # 1 - Euclidean Error
    axs[0, 0].hist(errors['euclidean'], bins=bins, color='skyblue', edgecolor='black')
    axs[0, 0].axvline(np.mean(errors['euclidean']), color='red', linestyle='--',
                      label=f"Mean = {np.mean(errors['euclidean']):.2f}m")
    axs[0, 0].set_title("Euclidean Error (m)")
    axs[0, 0].set_xlabel("Error (m)")
    axs[0, 0].set_ylabel("Number of objects")
    axs[0, 0].legend()

    # 2 - Depth Error
    axs[0, 1].hist(errors['z'], bins=bins, color='lightgreen', edgecolor='black')
    axs[0, 1].axvline(np.mean(errors['z']), color='red', linestyle='--',
                      label=f"Mean = {np.mean(errors['z']):.2f}m")
    axs[0, 1].set_title("Depth Error")
    axs[0, 1].set_xlabel("Error (m)")
    axs[0, 1].legend()

    # 3 - Orientation Error
    if 'orientation' in errors and errors['orientation']:
        axs[1, 0].hist(np.degrees(errors['orientation']), bins=bins, color='orange', edgecolor='black')
        axs[1, 0].axvline(np.degrees(np.mean(errors['orientation'])), color='red', linestyle='--',
                          label=f"Mean = {np.degrees(np.mean(errors['orientation'])):.2f}°")
        axs[1, 0].set_title("Orientation Error")
        axs[1, 0].set_xlabel("Error (°)")
        axs[1, 0].set_ylabel("Number of objects")
        axs[1, 0].legend()
    else:
        axs[1, 0].set_visible(False)

    # 4 - 3D IoU
    axs[1, 1].hist(errors['iou_3d'], bins=bins, color='purple', edgecolor='black')
    axs[1, 1].axvline(np.mean(errors['iou_3d']), color='red', linestyle='--',
                      label=f"Mean = {np.mean(errors['iou_3d']):.2f}")
    axs[1, 1].set_title("3D IoU")
    axs[1, 1].set_xlabel("IoU")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def analyze_metric_by_depth_range(
    errors: Dict[str, List[float]], 
    metric: str,
    depth_interval: float = 5.0,
    stat: str = "rmse"
) -> Dict[str, Dict[str, float]]:
    """
    Analyze a metric by depth ranges.

    Args:
        errors: Dictionary containing at least 'depths' (estimated depth)
                and the metric to analyze (e.g., 'euclidean', 'orientation', 'iou_3d', etc.)
        metric: Name of the metric to analyze (key in errors)
        depth_interval: Depth range size in meters
        stat: "rmse" or "mean"

    Returns:
        dict with statistics for each depth range
    """
    if 'depths_gt' not in errors or metric not in errors:
        print(f"Error: Key 'depths_gt' or '{metric}' missing in errors")
        return {}

    depths = np.array(errors['depths_gt'])
    values = np.array(errors[metric])

    # Depth range boundaries
    min_depth = max(0, np.floor(depths.min() / depth_interval) * depth_interval)
    max_depth = np.ceil(depths.max() / depth_interval) * depth_interval

    results = {}
    for start in np.arange(min_depth, max_depth, depth_interval):
        end = start + depth_interval
        mask = (depths >= start) & (depths < end)
        if mask.sum() == 0:
            continue

        vals = values[mask]
        if stat == "rmse":
            metric_val = np.sqrt(np.mean(vals**2))
        else:  # simple mean
            metric_val = np.mean(vals)

        results[f"{start:.0f}-{end:.0f}m"] = {
            "count": mask.sum(),
            f"{metric}_{stat}": metric_val,
            "depth_center": start + depth_interval / 2
        }

    return results


def plot_metric_by_depth_range(
    errors: Dict[str, List[float]],
    metric: str,
    stat: str = "rmse",
    depth_interval: float = 5.0,
    save_path: Optional[str] = None
):
    """
    Analyze a metric by depth ranges and plot the results.

    Args:
        errors: Dictionary containing error lists and 'depths'
        metric: Name of the metric to analyze (key in errors)
        stat: "rmse" or "mean"
        depth_interval: Depth range width in meters
        save_path: Path to save the plot (optional)
    """
    # Perform the analysis
    depth_analysis = analyze_metric_by_depth_range(errors, metric, depth_interval, stat)

    if not depth_analysis:
        print("No data available for plotting.")
        return

    # Sort results by depth
    sorted_items = sorted(depth_analysis.items(), key=lambda x: x[1]['depth_center'])
    ranges = [k for k, _ in sorted_items]
    errors_vals = [v[f"{metric}_{stat}"] for _, v in sorted_items]
    counts = [v["count"] for _, v in sorted_items]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Metric bars
    bars = ax1.bar(ranges, errors_vals, color=plt.cm.viridis(np.linspace(0.3, 0.8, len(errors_vals))), edgecolor="black")
    ax1.set_ylabel(f"{metric} ({stat})", fontsize=12)
    ax1.set_title(f"{metric} error ({stat}) by depth range", fontsize=14, fontweight="bold")
    ax1.grid(alpha=0.3, axis="y")

    for b, val in zip(bars, errors_vals):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + max(errors_vals)*0.01, f"{val:.2f}", 
                 ha="center", va="bottom", fontsize=9)

    # Count bars
    bars2 = ax2.bar(ranges, counts, color="lightgray", edgecolor="black")
    ax2.set_xlabel("Depth range (m)", fontsize=12)
    ax2.set_ylabel("Number of objects", fontsize=12)
    ax2.grid(alpha=0.3, axis="y")

    for b, val in zip(bars2, counts):
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + max(counts)*0.01, str(val),
                 ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()