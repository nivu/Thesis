# Configuration variables
# Use traffic analysis video - update this to your desired video file
VIDEO_PATH = "traffic_analyis_data/Uni_west_1/GOPR0574.MP4"
RECOGNITION_SIZE = (640, 640)
DISPLAY_SIZE = (1920, 1080)
MAPPING_FILE = "coordinate_mapping_2030.json"

# Stabilizer configuration (not functioning)
#STABILIZER_SMOOTHING_WINDOW = 30  # Adjust this value based on your needs
# Higher values (e.g., 45-60) = smoother but more delayed stabilization
# Lower values (e.g., 15-20) = more responsive but less smooth stabilization