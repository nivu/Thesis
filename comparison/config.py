# Configuration for Comparison Tools
import os

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video settings
VIDEO_PATH = os.path.join(PROJECT_ROOT, "traffic_analyis_data/Uni_west_1/GOPR0574.MP4")

# Model settings
VEHICLE_MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")
WHEEL_SEG_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs/segment/wheel_seg/weights/best.pt")

# Calibration settings
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "gopro_calibration_fisheye.npz")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "coordinate_mapping_2030.json")

# Frame sizes
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)
