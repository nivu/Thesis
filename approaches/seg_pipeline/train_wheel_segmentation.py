"""
YOLOv8 Wheel Instance Segmentation Training
Train a model to detect and segment individual vehicle wheels
"""

import os
from ultralytics import YOLO
import torch


def train_wheel_segmentation_model():
    """
    Train YOLOv8 segmentation model for wheel detection

    Note: This requires a dataset with wheel instance segmentation masks
    Format: YOLO segmentation format with polygon annotations
    """
    print("=" * 70)
    print("YOLOv8 Wheel Segmentation Training")
    print("=" * 70)

    # Check for dataset
    data_yaml = "Wheel_seg-6/data.yaml"

    if not os.path.exists(data_yaml):
        print("\n⚠️  Wheel segmentation dataset not found!")
        print("\nTo use this approach, you need a dataset with wheel annotations.")
        print("\nDataset structure needed:")
        print("WheelDataset/")
        print("├── data.yaml")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── labels/  (polygon format)")
        print("├── valid/")
        print("└── test/")
        print("\nAnnotation format (labels/*.txt):")
        print("class_id x1 y1 x2 y2 x3 y3 x4 y4  (polygon points, normalized)")
        print("\nExample data.yaml:")
        print("""
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 1  # number of classes
names: ['wheel']

# Or if detecting different wheel types:
# nc: 4
# names: ['front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel']
""")
        print("\n" + "=" * 70)
        print("ALTERNATIVE SOLUTIONS:")
        print("=" * 70)
        print("\n1. Use existing wheel detection datasets from:")
        print("   - Roboflow Universe (search 'wheel detection')")
        print("   - COCO dataset (has some wheel annotations)")
        print("   - Custom annotation using tools like LabelImg, CVAT")

        print("\n2. Use YOLOv8 Pose instead (10 keypoints for vehicle):")
        print("   - Front-left wheel center")
        print("   - Front-right wheel center")
        print("   - Rear-left wheel center")
        print("   - Rear-right wheel center")
        print("   - Vehicle corners (for bounding)")

        print("\n3. Use pretrained wheel detector + current vehicle model")

        return None

    # Initialize YOLOv8 segmentation model
    print("\nInitializing YOLOv8n-seg model...")
    model = YOLO("yolov8n-seg.pt")  # Segmentation model

    # Check device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Training parameters (GPU can handle more)
    if device == "mps":
        epochs = 50
        batch = 16  # M2 Pro can handle this
    elif device == "cuda":
        epochs = 50
        batch = 16
    else:
        epochs = 30
        batch = 8

    print(f"\n" + "-" * 70)
    print("Training Configuration:")
    print("-" * 70)
    print("Model: YOLOv8n-seg (instance segmentation)")
    print("Task: Wheel detection and segmentation")
    print("Classes: backwheel, frontwheel, middlewheel")
    print(f"Dataset: 994 train, 285 valid, 141 test")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("Image size: 640x640")
    print(f"Batch size: {batch}")
    print("-" * 70)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        patience=15,
        save=True,
        plots=True,
        val=True,
        project="runs/segment",
        name="wheel_seg",
        exist_ok=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"\nBest model: runs/segment/wheel_seg/weights/best.pt")

    return results


def use_wheel_segmentation_for_speed():
    """
    Example of how to use wheel segmentation for speed/dimension estimation
    """
    print("\n" + "=" * 70)
    print("HOW TO USE WHEEL SEGMENTATION FOR SPEED & DIMENSION")
    print("=" * 70)

    print("""
1. DETECT & SEGMENT WHEELS:
   - Run YOLOv8-seg to get wheel masks
   - Get centroid of each wheel mask
   - Identify front/rear and left/right wheels

2. CALCULATE VEHICLE DIMENSIONS:
   - Wheelbase = distance(front_wheel_center, rear_wheel_center)
   - Track width = distance(left_wheel_center, right_wheel_center)
   - Known wheel diameter (e.g., 60-70cm for cars)
   - Use homography to convert pixel distances to real-world

3. ESTIMATE SPEED:
   - Track wheel centers across frames
   - Calculate displacement in real-world coordinates
   - Speed = displacement / time_between_frames
   - More accurate than tracking vehicle centroid

4. ADVANTAGES:
   - Wheels are on the ground plane (simplifies homography)
   - Standard wheel dimensions provide known reference
   - Better tracking stability
   - Can estimate vehicle type from wheelbase/track width

5. IMPLEMENTATION:

   from ultralytics import YOLO
   import numpy as np

   model = YOLO("runs/segment/wheel_seg/weights/best.pt")

   results = model(frame)

   for result in results:
       masks = result.masks.xy  # Wheel contours
       boxes = result.boxes.xywh  # Wheel bounding boxes

       # Get wheel centers
       wheel_centers = []
       for mask in masks:
           centroid = np.mean(mask, axis=0)
           wheel_centers.append(centroid)

       # Match wheels to vehicle
       # Calculate wheelbase, track width
       # Apply homography transformation
       # Estimate speed from frame-to-frame displacement

""")


if __name__ == "__main__":
    result = train_wheel_segmentation_model()

    if result is None:
        # Dataset not found, show usage guide
        use_wheel_segmentation_for_speed()
