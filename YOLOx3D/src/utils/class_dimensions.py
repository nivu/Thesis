import numpy as np
import json
import os
from pathlib import Path
import glob
from tqdm import tqdm
from utils.kitti_utils import get_kitti_directories, load_label

# Global dimensions state
_global_dimensions = {
    'class_dimensions': None,
    'is_loaded': False,
    'dimensions_file': None
}

def load_class_dimensions(dimensions_file):
    """
    Load class dimensions globally
    
    Args:
        dimensions_file (str): Path to dimensions file
        
    Returns:
        bool: True if loaded successfully, False otherwise
    """
    global _global_dimensions
    
    if not os.path.exists(dimensions_file):
        print(f"Warning: Dimensions file not found: {dimensions_file}")
        return False
    
    try:
        with open(dimensions_file, 'r') as f:
            dimensions_data = json.load(f)
        
        _global_dimensions['class_dimensions'] = dimensions_data['class_dimensions']
        _global_dimensions['is_loaded'] = dimensions_data['is_loaded']
        _global_dimensions['dimensions_file'] = dimensions_file
        
        return True
        
    except Exception as e:
        print(f"Error loading class dimensions: {e}")
        return False

def save_class_dimensions(class_dimensions, dimensions_file):
    """
    Save class dimensions globally
    
    Args:
        class_dimensions (dict): Dictionary mapping class names to dimensions
        dimensions_file (str): Path to save dimensions file
    """
    global _global_dimensions
    
    dimensions_data = {
        'class_dimensions': class_dimensions,
        'is_loaded': True,
        'computation_type': 'kitti_dataset_average'
    }
    
    os.makedirs(os.path.dirname(dimensions_file), exist_ok=True)
    with open(dimensions_file, 'w') as f:
        json.dump(dimensions_data, f, indent=2)
    
    # Update global state
    _global_dimensions['class_dimensions'] = class_dimensions
    _global_dimensions['is_loaded'] = True
    _global_dimensions['dimensions_file'] = dimensions_file

def compute_class_dimensions_from_kitti(kitti_base_dir, save_file=None, show_progress=True):
    """
    Compute average dimensions for each class from KITTI dataset
    
    Args:
        kitti_base_dir (str): Base directory of KITTI dataset
        save_file (str): Optional path to save dimensions
        show_progress (bool): Show progress bar
        
    Returns:
        dict: Dictionary mapping class names to average dimensions [h, w, l]
    """
    global _global_dimensions
    
    paths = get_kitti_directories(kitti_base_dir)
    kitti_labels_dir = paths['labels']
    
    if not os.path.exists(kitti_labels_dir):
        raise FileNotFoundError(f"KITTI labels directory not found: {kitti_labels_dir}")
    
    # Collect dimensions for each class
    class_dimensions_data = {}
    
    label_files = sorted(glob.glob(os.path.join(kitti_labels_dir, "*.txt")))
    
    iterator = tqdm(label_files, desc="Processing KITTI labels") if show_progress else label_files
    
    for label_path in iterator:
        objects = load_label(label_path)
        
        for obj in objects:
            class_name = obj['class_name'].lower()
            dimensions = obj['dimensions']  # (h, w, l) from kitti_utils

            if class_name not in class_dimensions_data:
                class_dimensions_data[class_name] = []
            
            class_dimensions_data[class_name].append(dimensions)
        
        if show_progress:
            total_objects = sum(len(dims) for dims in class_dimensions_data.values())
            iterator.set_description(f"Processing (Objects: {total_objects})")
    
    # Compute averages
    class_dimensions = {}
    for class_name, dimensions_list in class_dimensions_data.items():
        if len(dimensions_list) > 0:
            avg_dimensions = np.mean(dimensions_list, axis=0)
            class_dimensions[class_name] = avg_dimensions.tolist()
    
    # Update global state
    _global_dimensions['class_dimensions'] = class_dimensions
    _global_dimensions['is_loaded'] = True
    
    # Save if requested
    if save_file:
        save_class_dimensions(class_dimensions, save_file)
    
    return class_dimensions

def get_class_dimensions(class_name):
    """
    Get dimensions for a specific class
    
    Args:
        class_name (str): Name of the class
        
    Returns:
        list: Dimensions [h, w, l] if available, default dimensions otherwise
    """
    global _global_dimensions
    
    if not _global_dimensions['is_loaded']:
        raise ValueError("Class dimensions not loaded. Please load dimensions first.")
    
    class_name_lower = class_name.lower()
    
    if class_name_lower in _global_dimensions['class_dimensions']:
        return _global_dimensions['class_dimensions'][class_name_lower]
    else:
        # Default dimensions if class not found
        default_dims = [1.5, 1.5, 3.0]  # w, h, l
        print(f"Warning: No dimensions found for class '{class_name}', using default: {default_dims}")
        return default_dims

def is_dimensions_loaded():
    """Check if class dimensions are loaded"""
    return _global_dimensions['is_loaded']

def get_all_class_dimensions():
    """Get all loaded class dimensions"""
    if not _global_dimensions['is_loaded']:
        return {}
    return _global_dimensions['class_dimensions'].copy()

def get_dimensions_info():
    """Get current dimensions information"""
    return _global_dimensions.copy()

def reset_dimensions():
    """Reset global dimensions state"""
    global _global_dimensions
    _global_dimensions = {
        'class_dimensions': None,
        'is_loaded': False,
        'dimensions_file': None
    }
    print("✓ Global class dimensions reset")
