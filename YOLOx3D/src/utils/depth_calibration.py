import numpy as np
import json
import os
from pathlib import Path
import glob
from tqdm import tqdm
from utils.kitti_utils import get_kitti_directories, load_label
from utils.depth import get_depth_in_region
from models.depth_anything_v2 import DepthAnythingV2
import cv2

# Global calibration state
_global_calibration = {
    'coefficients': None,
    'is_calibrated': False,
    'calibration_file': None
}

def load_calibration(calibration_file):
    """
    Load calibration globally
    
    Args:
        calibration_file (str): Path to calibration file
    """
    global _global_calibration
    
    if not os.path.exists(calibration_file):
        print(f"Warning: Calibration file not found: {calibration_file}")
        return False
    
    try:
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        
        _global_calibration['coefficients'] = np.array(calibration_data['coefficients'])
        _global_calibration['is_calibrated'] = calibration_data['is_calibrated']
        _global_calibration['calibration_file'] = calibration_file
        
        print(f"✓ Global depth calibration loaded from: {calibration_file}")
        print(f"  Coefficients: {_global_calibration['coefficients']}")
        return True
        
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False

def save_calibration(coefficients, calibration_file):
    """
    Save calibration globally
    
    Args:
        coefficients (numpy.ndarray): Calibration coefficients
        calibration_file (str): Path to save calibration file
    """
    global _global_calibration
    
    calibration_data = {
        'coefficients': coefficients.tolist(),
        'is_calibrated': True,
        'calibration_type': 'polynomial_degree_2'
    }
    
    os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
    with open(calibration_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    # Update global state
    _global_calibration['coefficients'] = coefficients
    _global_calibration['is_calibrated'] = True
    _global_calibration['calibration_file'] = calibration_file
    
    print(f"✓ Global calibration saved to: {calibration_file}")

def calibrate_depth(relative_depths, absolute_depths, save_file=None):
    """
    Perform global depth calibration
    
    Args:
        relative_depths (list/numpy.ndarray): Relative depth values
        absolute_depths (list/numpy.ndarray): Absolute depth values
        save_file (str): Optional path to save calibration
        
    Returns:
        tuple: (coefficients, relative_depths_array, absolute_depths_array)
    """
    global _global_calibration
    
    relative_depths = np.array(relative_depths)
    absolute_depths = np.array(absolute_depths)
    
    if len(relative_depths) < 3:
        raise ValueError("At least 3 calibration points are required")
    
    # Fit polynomial
    coefficients = np.polyfit(relative_depths, absolute_depths, 2)
    
    # Update global state
    _global_calibration['coefficients'] = coefficients
    _global_calibration['is_calibrated'] = True
    
    # Save if requested
    if save_file:
        save_calibration(coefficients, save_file)
    
    print(f"✓ Global depth calibration completed with {len(relative_depths)} points")
    print(f"  Coefficients: {coefficients}")
    
    return coefficients, relative_depths, absolute_depths

def convert_depth_to_absolute(relative_depth):
    """
    Convert relative depth to absolute depth using global calibration
    
    Args:
        relative_depth (float/numpy.ndarray): Relative depth value(s)
        
    Returns:
        float/numpy.ndarray: Absolute depth if calibrated, relative depth otherwise
    """
    global _global_calibration
    
    if not _global_calibration['is_calibrated']:
        raise ValueError("There is no valid calibration. Please load calibration first.")

    c0, c1, c2 = _global_calibration['coefficients']
    absolute_depth = c0 * relative_depth**2 + c1 * relative_depth + c2
    
    return absolute_depth

def is_calibrated():
    """Check if global calibration is loaded"""
    return _global_calibration['is_calibrated']

def get_calibration_info():
    """Get current calibration information"""
    return _global_calibration.copy()

def reset_calibration():
    """Reset global calibration state"""
    global _global_calibration
    _global_calibration = {
        'coefficients': None,
        'is_calibrated': False,
        'calibration_file': None
    }
    print("✓ Global calibration reset")

def calibrate_with_kitti_3D_objects_dataset(kitti_base_dir,
                                depth_model_size="small",
                                device='cuda',
                                max_depth=150.0,
                                max_images=None,
                                save_file=None,
                                show_progress=True,
                                return_data=False):
    """
    Calibrate using entire KITTI dataset and set global calibration
    
    Args:
        kitti_base_dir (str): Base directory of 3D objects KITTI dataset
        depth_model_size (str): Depth model size
        device (str): Device for computation
        max_images (int): Maximum images to process
        save_file (str): Path to save calibration
        show_progress (bool): Show progress bar
        return_data (bool): Whether to return calibration data for plotting
        
    Returns:
        tuple: (coefficients, num_points) or (coefficients, num_points, relative_depths, absolute_depths)
    """

    paths = get_kitti_directories(kitti_base_dir)
    kitti_images_dir = paths['images']
    kitti_labels_dir = paths['labels']
    kitti_calib_dir = paths['calib']
    
    print(f"Starting KITTI dataset calibration...")
    
    for dir_path, dir_name in [(kitti_images_dir, "images"), 
                               (kitti_labels_dir, "labels"), 
                               (kitti_calib_dir, "calibration")]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"KITTI {dir_name} directory not found: {dir_path}")
        
    depth_estimator = DepthAnythingV2(model_size=depth_model_size, device=device)
    
    image_files = sorted(glob.glob(os.path.join(kitti_images_dir, "*.png")))
    if max_images is not None:
        image_files = image_files[:max_images]
    
    relative_depths = []
    absolute_depths = []
    total_objects = 0
    filtered_objects = 0
    
    iterator = tqdm(image_files, desc="Processing KITTI images") if show_progress else image_files
    
    for image_path in iterator:
        image_name = Path(image_path).stem
        label_path = os.path.join(kitti_labels_dir, f"{image_name}.txt")
        calib_path = os.path.join(kitti_calib_dir, f"{image_name}.txt")
        
        if not os.path.exists(label_path) or not os.path.exists(calib_path):
            print(f"Warning: Missing label or calibration file for {image_name}. Skipping.")
            continue
        
        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]
            
        gt_objects = load_label(label_path)
        
        depth_map = depth_estimator.estimate_depth(image)
        
        for gt_obj in gt_objects:
            total_objects += 1
            z_abs = gt_obj['center_3d'][2]

            if z_abs >= max_depth:
                continue

            bbox_2d = gt_obj['bbox_2d']
            x1, y1, x2, y2 = bbox_2d
            
            # Filter out objects that touch image edges
            if x1 <= 0 or y1 <= 0 or x2 >= img_width or y2 >= img_height:
                filtered_objects += 1
                continue

            z_rel = get_depth_in_region(depth_map, bbox_2d, method='median')

            relative_depths.append(z_rel)
            absolute_depths.append(z_abs)
        
        if show_progress:
            iterator.set_description(f"Processing (Points: {len(relative_depths)}, Filtered: {filtered_objects})")
    
    print(f"Objects filtered (touching edges): {filtered_objects}/{total_objects} ({filtered_objects/total_objects*100:.1f}%)")
    
    coefficients, rel_depths_array, abs_depths_array = calibrate_depth(relative_depths, absolute_depths, save_file)
    
    print(f"✓ KITTI calibration completed with {len(relative_depths)} points")

    if return_data:
        return coefficients, len(relative_depths), rel_depths_array, abs_depths_array
    else:
        return coefficients, len(relative_depths)