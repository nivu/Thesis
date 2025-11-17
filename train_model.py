"""
YOLOv8 Training Script
Trains a YOLOv8 detection model on the Roboflow dataset
"""

import os
from ultralytics import YOLO
from pathlib import Path


def train_model():
    """
    Train YOLOv8 detection model on the Roboflow dataset
    """
    print("=" * 70)
    print("YOLOv8 Model Training")
    print("=" * 70)

    # Check if dataset exists
    data_yaml = "Dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration '{data_yaml}' not found!")
        return

    print(f"\nDataset config: {data_yaml}")

    # Initialize YOLOv8n model (nano version for faster training)
    # You can use 'yolov8s.pt', 'yolov8m.pt', etc. for larger models
    print("\nInitializing YOLOv8n detection model...")
    model = YOLO("yolov8n.pt")  # Start with pretrained YOLOv8n

    # Check available device
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. GPU is recommended.")

    # Training configuration
    epochs = 20  # Reduced for CPU training
    print("\n" + "-" * 70)
    print("Training Configuration:")
    print("-" * 70)
    print("Model: YOLOv8n (detection)")
    print("Classes: bus, cars, truck")
    print("Image size: 640x640")
    print("Batch size: 16")
    print(f"Epochs: {epochs}")
    print("Device:", device)
    print("-" * 70)

    # Train the model
    print("\nStarting training...")
    print("This may take a while depending on your hardware.")
    print("Press Ctrl+C to stop training early.\n")

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,       # Number of training epochs
            imgsz=640,           # Image size
            batch=16,            # Batch size (reduce if out of memory)
            device=device,       # Use GPU if available
            patience=10,         # Early stopping patience
            save=True,           # Save checkpoints
            plots=True,          # Save training plots
            verbose=True,        # Verbose output
            val=True,            # Validate during training
            project="runs/detect",  # Save location
            name="roboflow_train",  # Experiment name
            exist_ok=True,       # Overwrite existing folder
            pretrained=True,     # Use pretrained weights
            optimizer="auto",    # Auto optimizer
            seed=0,              # Random seed for reproducibility
            workers=8,           # Number of worker threads
            cache=False,         # Don't cache images (set True if enough RAM)
        )

        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"\nBest model saved to: runs/detect/roboflow_train/weights/best.pt")
        print(f"Last model saved to: runs/detect/roboflow_train/weights/last.pt")
        print(f"Training results saved to: runs/detect/roboflow_train/")
        print("\nTraining metrics:")
        print(f"  Final mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  Final mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Partial results may be saved in runs/detect/roboflow_train/")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise

    return results


if __name__ == "__main__":
    train_model()
