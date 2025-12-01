# Configuration for Bounding Box Pipeline
import os

# Paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Video settings
VIDEO_PATH = os.path.join(PROJECT_ROOT, "traffic_analyis_data/Uni_west_1/GOPR0574.MP4")

# Model settings
MODEL_PATH = os.path.join(PROJECT_ROOT, "best.pt")

# Calibration settings
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "gopro_calibration_fisheye.npz")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "coordinate_mapping_2030.json")

# Frame sizes
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)

# Output files
OUTPUT_TRACKING_CSV = "bbox_tracking_data.csv"
OUTPUT_WORLD_COORDS_CSV = "bbox_world_coordinates.csv"
