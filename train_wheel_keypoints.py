"""
YOLOv8 Pose Detection for Vehicle Wheels
Train a keypoint detection model to locate wheel centers and vehicle corners
This is likely what the original best.pt model was doing
"""

import os
from ultralytics import YOLO
import torch


def train_wheel_keypoint_model():
    """
    Train YOLOv8 pose model for vehicle wheel keypoint detection

    Keypoint schema (10 keypoints):
    0-3: Four wheel centers (FL, FR, RL, RR)
    4-9: Six vehicle bounding points (front-left, front-right, rear-left,
         rear-right, top-center, bottom-center)
    """
    print("=" * 70)
    print("YOLOv8 Pose Detection for Vehicle Wheels")
    print("=" * 70)

    # Check for dataset
    data_yaml = "VehicleKeypointDataset/data.yaml"

    if not os.path.exists(data_yaml):
        print("\n⚠️  Vehicle keypoint dataset not found!")
        print("\nThis is the RECOMMENDED approach - matches original best.pt!")
        print("\nDataset structure needed:")
        print("VehicleKeypointDataset/")
        print("├── data.yaml")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── labels/  (keypoint format)")
        print("├── valid/")
        print("└── test/")

        print("\ndata.yaml example:")
        print("""
train: ../train/images
val: ../valid/images
test: ../test/images

# Number of classes
nc: 3
names: ['car', 'truck', 'bus']

# Keypoint configuration
kpt_shape: [10, 3]  # 10 keypoints, 3 values each (x, y, visibility)

# Keypoint names (optional but helpful)
kpt_names:
  - front_left_wheel
  - front_right_wheel
  - rear_left_wheel
  - rear_right_wheel
  - front_left_corner
  - front_right_corner
  - rear_left_corner
  - rear_right_corner
  - top_center
  - bottom_center
""")

        print("\nLabel format (labels/*.txt):")
        print("class_id x_center y_center width height kp1_x kp1_y kp1_vis ... kp10_x kp10_y kp10_vis")
        print("\nWhere:")
        print("  - Coordinates are normalized [0-1]")
        print("  - Visibility: 0=not visible, 1=occluded, 2=visible")

        print("\n" + "=" * 70)
        print("HOW TO ANNOTATE:")
        print("=" * 70)
        print("""
1. Use Label-Studio or CVAT with keypoint annotation mode
2. For each vehicle, mark 10 keypoints:
   - 4 wheel centers (where wheel touches ground)
   - 6 vehicle corners/reference points

3. Tools:
   - Label-Studio: https://labelstud.io/
   - CVAT: https://cvat.org/
   - Roboflow (supports keypoint annotation)

4. Convert to YOLO format if needed
""")

        print("\n" + "=" * 70)
        print("QUICK START - Use the Current Dataset:")
        print("=" * 70)
        print("""
Since you already have vehicle bounding boxes, you can:

1. OPTION A - Estimate wheel positions from bounding boxes:
   - Use a pre-trained wheel detector
   - Or use heuristics (wheels are at bottom corners)

2. OPTION B - Annotate wheels on existing 900 images:
   - Much faster than collecting new data
   - Only need to mark 4-10 points per vehicle
   - Can be done in ~2-3 hours for 900 images

3. OPTION C - Use Roboflow Auto-Annotate:
   - Upload your dataset to Roboflow
   - Use their AI-assisted annotation for keypoints
   - Review and correct
""")

        return None

    # Initialize YOLOv8 pose model
    print("\nInitializing YOLOv8n-pose model...")
    model = YOLO("yolov8n-pose.pt")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Training parameters
    epochs = 100
    print(f"\n" + "-" * 70)
    print("Training Configuration:")
    print("-" * 70)
    print("Model: YOLOv8n-pose")
    print("Task: Vehicle + wheel keypoint detection")
    print("Keypoints: 10 per vehicle")
    print(f"Epochs: {epochs}")
    print("Image size: 640x640")
    print("-" * 70)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,
        device=device,
        patience=20,
        save=True,
        plots=True,
        val=True,
        project="runs/pose",
        name="vehicle_wheels",
        exist_ok=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nBest model: runs/pose/vehicle_wheels/weights/best.pt")

    return results


def demonstrate_usage():
    """
    Show how to use keypoint model for speed/dimension estimation
    """
    print("\n" + "=" * 70)
    print("USING KEYPOINTS FOR SPEED & DIMENSION CALCULATION")
    print("=" * 70)

    print("""
IMPLEMENTATION EXAMPLE:

```python
from ultralytics import YOLO
import numpy as np
from coordinate_transformer import CoordinateTransformer

# Load model
model = YOLO("runs/pose/vehicle_wheels/weights/best.pt")
transformer = CoordinateTransformer("coordinate_mapping.json")

# Process frame
results = model(frame)

for result in results:
    keypoints = result.keypoints.xy.cpu().numpy()[0]  # 10 x 2

    # Extract wheel centers
    fl_wheel = keypoints[0]  # Front-left
    fr_wheel = keypoints[1]  # Front-right
    rl_wheel = keypoints[2]  # Rear-left
    rr_wheel = keypoints[3]  # Rear-right

    # 1. CALCULATE WHEELBASE (front to rear wheel distance)
    left_wheelbase_px = np.linalg.norm(fl_wheel - rl_wheel)
    right_wheelbase_px = np.linalg.norm(fr_wheel - rr_wheel)
    avg_wheelbase_px = (left_wheelbase_px + right_wheelbase_px) / 2

    # Convert to real-world using homography
    fl_world = transformer.pixel_to_world(fl_wheel)
    rl_world = transformer.pixel_to_world(rl_wheel)
    wheelbase_real = np.linalg.norm(fl_world - rl_world)  # meters

    # 2. CALCULATE TRACK WIDTH (left to right wheel distance)
    front_track_px = np.linalg.norm(fl_wheel - fr_wheel)
    rear_track_px = np.linalg.norm(rl_wheel - rr_wheel)

    # 3. ESTIMATE WHEEL DIAMETER
    # Use known typical wheel diameters:
    # Car: 0.60-0.70m, Truck: 0.80-1.00m, Bus: 0.90-1.10m
    wheel_diameter = 0.65  # meters (car)

    # 4. CALCULATE SPEED
    # Track rear wheel center (on ground, most stable)
    rear_center = (rl_wheel + rr_wheel) / 2
    rear_center_world = transformer.pixel_to_world(rear_center)

    # Between frames:
    displacement = current_position - previous_position
    speed_ms = displacement / time_delta
    speed_kmh = speed_ms * 3.6

    # 5. VEHICLE CLASSIFICATION BY DIMENSIONS
    if wheelbase_real > 3.5:
        vehicle_type = "truck/bus"
    elif wheelbase_real > 2.5:
        vehicle_type = "sedan/suv"
    else:
        vehicle_type = "compact car"
```

ADVANTAGES:
✓ Most accurate method (ground-truth wheel positions)
✓ Handles occlusion better (can estimate from visible wheels)  ✓ Provides real vehicle dimensions (wheelbase, track width)
✓ Better speed estimation (wheels on ground plane)
✓ Can classify vehicle types by dimensions

ACCURACY IMPROVEMENTS:
- Wheelbase known for specific vehicle models
- Can use wheel diameter as scale reference
- Ground plane constraint reduces homography errors
- Tracking 4 points instead of 1 centroid
""")


if __name__ == "__main__":
    result = train_wheel_keypoint_model()

    if result is None:
        demonstrate_usage()
