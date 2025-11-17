"""
YOLO Model Benchmarking Script
Evaluates the trained YOLOv8 model on the Roboflow test dataset
"""

import os
from ultralytics import YOLO
import json
from pathlib import Path


def benchmark_model(model_path="runs/detect/roboflow_train/weights/best.pt"):
    """
    Run benchmarking on the YOLO model using the test dataset

    Args:
        model_path: Path to the trained model (default: newly trained model)
    """
    print("=" * 70)
    print("YOLO Model Benchmarking")
    print("=" * 70)

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Have you trained the model yet? Run train_model.py first.")
        return

    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)

    # Set device
    device = "cuda" if model.device.type == "cuda" else "cpu"
    print(f"Using device: {device}")

    # Path to data.yaml
    data_yaml = "Dataset/data.yaml"
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration '{data_yaml}' not found!")
        return

    print(f"Dataset config: {data_yaml}")
    print(f"\nClasses: {model.names}")
    print(f"Number of classes: {len(model.names)}\n")

    # Run validation on test dataset
    print("-" * 70)
    print("Running validation on test dataset...")
    print("-" * 70)

    # Validate the model - this will use the 'test' split from data.yaml
    results = model.val(
        data=data_yaml,
        split='test',  # Use test split
        imgsz=640,     # Image size for inference
        batch=16,      # Batch size
        conf=0.25,     # Confidence threshold
        iou=0.45,      # IoU threshold for NMS
        device=device,
        plots=True,    # Save plots
        save_json=True # Save results in COCO JSON format
    )

    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Overall metrics
    print("\n--- Overall Metrics ---")
    print(f"mAP@50:     {results.box.map50:.4f}")
    print(f"mAP@50-95:  {results.box.map:.4f}")
    print(f"Precision:  {results.box.mp:.4f}")
    print(f"Recall:     {results.box.mr:.4f}")

    # Per-class metrics
    print("\n--- Per-Class Metrics ---")
    class_names = ['bus', 'cars', 'truck']

    if hasattr(results.box, 'ap_class_index'):
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = class_names[int(class_idx)]
            print(f"\n{class_name.upper()}:")
            print(f"  AP@50:     {results.box.ap50[i]:.4f}")
            print(f"  AP@50-95:  {results.box.ap[i]:.4f}")

    # Speed metrics
    print("\n--- Speed Metrics ---")
    if hasattr(results, 'speed'):
        print(f"Preprocess:  {results.speed['preprocess']:.2f} ms/image")
        print(f"Inference:   {results.speed['inference']:.2f} ms/image")
        print(f"Postprocess: {results.speed['postprocess']:.2f} ms/image")

    # Additional statistics
    print("\n--- Dataset Statistics ---")
    print(f"Total test images: 90")
    print(f"Image size: 640x640")

    print("\n" + "=" * 70)
    print("Validation plots and results saved in 'runs/detect/val/' directory")
    print("=" * 70)

    # Save summary to file
    save_summary(results, class_names)

    return results


def save_summary(results, class_names):
    """
    Save benchmark summary to a JSON file
    """
    summary = {
        "overall_metrics": {
            "mAP@50": float(results.box.map50),
            "mAP@50-95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr)
        },
        "per_class_metrics": {},
        "speed_metrics": {}
    }

    # Per-class metrics
    if hasattr(results.box, 'ap_class_index'):
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = class_names[int(class_idx)]
            summary["per_class_metrics"][class_name] = {
                "AP@50": float(results.box.ap50[i]),
                "AP@50-95": float(results.box.ap[i])
            }

    # Speed metrics
    if hasattr(results, 'speed'):
        summary["speed_metrics"] = {
            "preprocess_ms": float(results.speed['preprocess']),
            "inference_ms": float(results.speed['inference']),
            "postprocess_ms": float(results.speed['postprocess'])
        }

    # Save to file
    output_file = "benchmark_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"\nBenchmark summary saved to: {output_file}")


if __name__ == "__main__":
    benchmark_model()
