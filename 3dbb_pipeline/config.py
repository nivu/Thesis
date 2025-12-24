"""
Configuration for 3D Bounding Box Pipeline

This pipeline integrates YOLO-based 3D detection with the existing
2D keypoint + homography approach for improved vehicle localization.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))

# Video input
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "traffic.mp4")

# Model paths
MODEL_2D_PATH = os.path.join(BASE_DIR, "best.pt")  # Existing YOLOv8 pose model
WEIGHTS_DIR = os.path.join(PIPELINE_DIR, "weights")
MODEL_3D_YOLO = os.path.join(WEIGHTS_DIR, "yolov11.pt")
MODEL_3D_MULTIBIN = os.path.join(WEIGHTS_DIR, "multibin_model.pt")

# Calibration files
CALIBRATION_FILE = os.path.join(BASE_DIR, "gopro_calibration_fisheye.npz")
MAPPING_FILE = os.path.join(BASE_DIR, "coordinate_mapping_2030.json")
DEPTH_SCALE_FILE = os.path.join(PIPELINE_DIR, "depth_calibration.json")

# Image sizes
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)

# 3D Detection parameters
DETECTION_3D_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "depth_model": "depth_anything_v2",
    "use_geometric_constraints": True,
}

# Fusion parameters
FUSION_CONFIG = {
    "enable_fusion": True,
    "weight_3d": 0.6,  # Weight for 3D estimate (0-1)
    "weight_2d": 0.4,  # Weight for 2D homography estimate
    "max_discrepancy_m": 1.0,  # Flag if estimates differ more than this
}

# Output paths
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
OUTPUT_TRACKING_CSV = os.path.join(OUTPUT_DIR, "3dbb_tracking_data.csv")
OUTPUT_WORLD_COORDS_CSV = os.path.join(OUTPUT_DIR, "3dbb_world_coordinates.csv")
OUTPUT_COMPARISON_CSV = os.path.join(OUTPUT_DIR, "3dbb_vs_2d_comparison.csv")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
