import numpy as np
import cv2
import os

def colorize_depth(depth_map, cmap=cv2.COLORMAP_INFERNO):
    """
    Colorize depth map for visualization
    
    Args:
        depth_map (numpy.ndarray): Depth map (normalized to 0-1)
        cmap (int): OpenCV colormap
        
    Returns:
        numpy.ndarray: Colorized depth map (BGR format)
    """
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)

    depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
    return colored_depth

def get_depth_at_point(depth_map, x, y):
    """
    Get depth value at a specific point
    
    Args:
        depth_map (numpy.ndarray): Depth map
        x (int): X coordinate
        y (int): Y coordinate
        
    Returns:
        float: Depth value at (x, y) - absolute if calibrator provided, relative otherwise
    """
    if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
        relative_depth = depth_map[y, x]
        return relative_depth
    
    print(f"Warning: Coordinates ({x}, {y}) are out of bounds for depth map of shape {depth_map.shape}")
    return 0.0

def get_depth_in_region(depth_map, bbox, method='median'):
    """
    Get depth value in a region defined by a bounding box
    
    Args:
        depth_map (numpy.ndarray): Depth map
        bbox (list): Bounding box [x1, y1, x2, y2]
        method (str): Method to compute depth ('median', 'mean', 'min')
        
    Returns:
        float: Depth value in the region - absolute if calibrator provided, relative otherwise
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(depth_map.shape[1] - 1, x2)
    y2 = min(depth_map.shape[0] - 1, y2)
    
    # Extract region
    region = depth_map[y1:y2, x1:x2]
    
    if region.size == 0:
        return 0.0
    
    # Compute depth based on method
    if method == 'median':
        relative_depth = float(np.median(region))
    elif method == 'mean':
        relative_depth = float(np.mean(region))
    elif method == 'min':
        relative_depth = float(np.min(region))
    else:
        relative_depth = float(np.median(region))
    
    return relative_depth

class DepthCalibrator:
    """
    Legacy DepthCalibrator class for backward compatibility.
    Now uses the global calibration system internally.
    """
    
    def __init__(self, calibration_file=None):
        """
        Initialize with optional calibration file
        
        Args:
            calibration_file (str): Path to calibration file
        """
        self._is_calibrated = False
        if calibration_file and os.path.exists(calibration_file):
            from .depth_calibration import load_calibration
            self._is_calibrated = load_calibration(calibration_file)
    
    @property
    def is_calibrated(self):
        """Check if calibration is available"""
        from .depth_calibration import is_calibrated
        return is_calibrated()
    
    def calibrate(self, relative_depths, absolute_depths, save_file=None):
        """
        Perform calibration (delegates to global system)
        
        Args:
            relative_depths (list): Relative depth values
            absolute_depths (list): Absolute depth values
            save_file (str): Optional file to save calibration
        """
        from .depth_calibration import calibrate_depth
        calibrate_depth(relative_depths, absolute_depths, save_file)
        self._is_calibrated = True
    
    def convert_to_absolute(self, relative_depth):
        """
        Convert relative depth to absolute (delegates to global system)
        
        Args:
            relative_depth (float): Relative depth value
            
        Returns:
            float: Absolute depth if calibrated, relative depth otherwise
        """
        from .depth_calibration import convert_depth_to_absolute, is_calibrated
        
        if is_calibrated():
            return convert_depth_to_absolute(relative_depth)
        else:
            return relative_depth