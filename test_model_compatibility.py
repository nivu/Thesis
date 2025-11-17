"""
Test compatibility between best.pt model and Wheel_seg-6 dataset
"""

from ultralytics import YOLO
import torch


def test_compatibility():
    print("=" * 70)
    print("Testing Model-Dataset Compatibility")
    print("=" * 70)

    # Load original best.pt model
    print("\n1. Loading original best.pt model...")
    model = YOLO("best.pt")

    # Check model details
    print(f"\nModel Information:")
    print(f"  Task: {model.task}")
    print(f"  Model type: {type(model.model).__name__}")

    # Get model metadata
    print(f"\nModel Configuration:")
    print(f"  Classes: {model.names}")
    print(f"  Number of classes: {len(model.names)}")

    # Check if it's a pose model
    if model.task == 'pose':
        print(f"  Keypoint shape: {model.model.kpt_shape if hasattr(model.model, 'kpt_shape') else 'N/A'}")
        print(f"  → This is a POSE model (expects keypoint annotations)")

    # Check dataset
    print("\n2. Checking Wheel_seg-6 dataset...")
    import yaml
    with open("Wheel_seg-6/data.yaml", 'r') as f:
        data_config = yaml.safe_load(f)

    print(f"\nDataset Configuration:")
    print(f"  Classes: {data_config['names']}")
    print(f"  Number of classes: {data_config['nc']}")
    print(f"  Has 'kpt_shape': {'kpt_shape' in data_config}")
    print(f"  → This is a SEGMENTATION dataset (polygon annotations)")

    # Test compatibility
    print("\n3. Testing compatibility...")
    print("-" * 70)

    try:
        # Try to run validation
        print("Attempting to run model on dataset...")
        results = model.val(
            data="Wheel_seg-6/data.yaml",
            split='test',
            imgsz=640,
            batch=8,
            verbose=False
        )
        print("✓ Compatible! Model ran successfully.")
        print(f"\nResults: {results}")

    except Exception as e:
        print(f"✗ Incompatible!")
        print(f"\nError: {type(e).__name__}")
        print(f"Message: {str(e)}")

        print("\n" + "=" * 70)
        print("COMPATIBILITY ANALYSIS")
        print("=" * 70)
        print("\nThe models are INCOMPATIBLE because:")
        print("  • best.pt is a YOLOv8-Pose model")
        print("    - Expects: 5 vehicle classes with 10 keypoints each")
        print("    - Format: Bounding box + keypoint coordinates (x, y, visibility)")
        print("\n  • Wheel_seg-6 is an instance segmentation dataset")
        print("    - Has: 3 wheel classes (backwheel, frontwheel, middlewheel)")
        print("    - Format: Polygon masks (multiple x,y points forming contours)")

        print("\n" + "=" * 70)
        print("SOLUTION OPTIONS")
        print("=" * 70)
        print("\n1. Train NEW wheel segmentation model (RECOMMENDED - IN PROGRESS)")
        print("   ✓ Currently training: runs/segment/wheel_seg/")
        print("   - Uses YOLOv8n-seg (segmentation architecture)")
        print("   - Will output wheel masks for backwheel, frontwheel, middlewheel")
        print("   - Can extract wheel centers from mask centroids")

        print("\n2. Keep both models and use them differently:")
        print("   • best.pt → For vehicle detection with keypoints (if you get original dataset)")
        print("   • New wheel_seg model → For wheel-specific segmentation")

        print("\n3. Alternative: Create keypoint dataset from wheel segmentation")
        print("   - Convert wheel polygon masks to centroid keypoints")
        print("   - Train YOLOv8-pose with 4 keypoints (4 wheel centers)")
        print("   - Matches original best.pt approach but with new data")

        print("\n" + "=" * 70)

    print("\nTest complete.")


if __name__ == "__main__":
    test_compatibility()
