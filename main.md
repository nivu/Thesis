
The main file for calculating vehicle speed is:

main.py - This is the entry point that orchestrates the entire pipeline.

However, the actual speed calculation logic is in speed_utils.py.

### Detection based approach
How it works:

main.py (entry point)

- Loads YOLO model for vehicle detection
- Processes video frames
- Converts pixel coordinates to real-world coordinates using CoordinateTransformer
- Calls SpeedTracker.get_speeds() at line 89 to calculate speeds
- Exports results to world_coordinates.csv

speed_utils.py (calculation logic)

Contains the SpeedTracker class that:
- Tracks vehicle positions over a buffer of frames (default 10)
- Calculates speed using Euclidean distance between real-world positions
- Converts from m/s to km/h (line 16: speed_ms * 3.6)
- Returns smoothed average speed over the buffer

Key formula (line 12-16):
distance = sqrt((x2-x1)² + (y2-y1)²)  # meters
time_diff = 1/fps                      # seconds
speed_kmh = (distance/time_diff) * 3.6 # km/h


### Wheel Segmentation based approach 

I've successfully built a segmentation-based wheel detection pipeline and comparison system. Here's what was created:

  Files Created/Modified:

  | File                      | Purpose                                                               |
  |---------------------------|-----------------------------------------------------------------------|
  | coordinate_transformer.py | NEW - Missing module for pixel-to-world coordinate conversion         |
  | main_segmentation.py      | NEW - Multi-method pipeline (Segmentation → Keypoint → BBox fallback) |
  | compare_approaches.py     | NEW - Comprehensive comparison of all three approaches                |
  | validate_pipelines.py     | NEW - Validation script to verify all components work                 |
  | config.py                 | Updated video path to available video                                 |

  Three Approaches Compared:

  1. BBox Approach (main.py)
    - Uses bottom-center of bounding box
    - Always available, baseline method
  2. Keypoint Approach (from pose model best.pt)
    - Uses detected wheel keypoints (10 per vehicle)
    - 100% detection rate on traffic videos
    - ~0.31m position difference from BBox
  3. Segmentation Approach (wheel_seg model)
    - Uses wheel masks to find contact points
    - 0% detection rate on traffic videos (domain gap)
    - Trained on different data perspective

  Key Findings from 100-Frame Comparison:

  | Metric                 | BBox | Keypoint  | Segmentation      |
  |------------------------|------|-----------|-------------------|
  | Detection Rate         | 100% | 100%      | 0%                |
  | Avg Speed Diff vs BBox | -    | 1.67 km/h | 0 km/h (fallback) |
  | Position Diff vs BBox  | -    | 0.31m     | 0m (fallback)     |

  Recommendations:

  1. Use Keypoints as primary method - they provide wheel-level precision with 100% availability
  2. Retrain wheel segmentation model on traffic video data if you want segmentation to work
  3. Filter by calibration bounds - vehicles outside the calibrated area produce unrealistic coordinates

  Running the Pipelines:

  # Validate setup
  python validate_pipelines.py

  # Run BBox approach (original)
  python main.py

  # Run Segmentation+Keypoint approach (new)
  python main_segmentation.py

  # Compare all approaches
  python compare_approaches.py --max-frames 200

──────────────────────────────────────────
