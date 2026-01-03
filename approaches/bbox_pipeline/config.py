# Configuration for Bounding Box Pipeline
import os

# Paths
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
APPROACHES_DIR = os.path.dirname(PIPELINE_DIR)
PROJECT_ROOT = os.path.dirname(APPROACHES_DIR)

# Video settings
VIDEO_PATH = os.path.join(PROJECT_ROOT, "traffic_analyis_data/Uni_west_1/GOPR0574.MP4")

# Model settings - use local models folder
MODEL_PATH = os.path.join(PIPELINE_DIR, "models/best.pt")

# Calibration settings
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "gopro_calibration_fisheye.npz")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "coordinate_mapping_2030.json")

# Frame sizes
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)

# Output files
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_TRACKING_CSV = os.path.join(OUTPUT_DIR, "bbox_tracking_data.csv")
OUTPUT_WORLD_COORDS_CSV = os.path.join(OUTPUT_DIR, "bbox_world_coordinates.csv")
