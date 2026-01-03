# Configuration for Wheel Segmentation Pipeline
import os

# Paths
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
APPROACHES_DIR = os.path.dirname(PIPELINE_DIR)
PROJECT_ROOT = os.path.dirname(APPROACHES_DIR)

# Video settings
VIDEO_PATH = os.path.join(PROJECT_ROOT, "traffic_analyis_data/Uni_west_1/GOPR0574.MP4")

# Model settings - use local models folder
VEHICLE_MODEL_PATH = os.path.join(PIPELINE_DIR, "models/vehicle_best.pt")
WHEEL_SEG_MODEL_PATH = os.path.join(PIPELINE_DIR, "models/wheel_seg_best.pt")

# Calibration settings
CALIBRATION_FILE = os.path.join(PROJECT_ROOT, "gopro_calibration_fisheye.npz")
MAPPING_FILE = os.path.join(PROJECT_ROOT, "coordinate_mapping_2030.json")

# Frame sizes
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)

# Keypoint detection settings
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

# Output files
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "segmentation_results.csv")
OUTPUT_WORLD_COORDS_CSV = os.path.join(OUTPUT_DIR, "seg_world_coordinates.csv")
