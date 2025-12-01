"""
Quick Validation Script for Both Pipelines

This script validates that both the object detection and segmentation
pipelines work correctly by processing a few frames from the video.

Usage:
    python validate_pipelines.py [--frames N]
"""

import cv2
import numpy as np
import argparse
import sys
from pathlib import Path


def validate_imports():
    """Validate all required imports work."""
    print("Validating imports...")
    errors = []

    try:
        from ultralytics import YOLO
        print("  [OK] ultralytics")
    except ImportError as e:
        errors.append(f"  [FAIL] ultralytics: {e}")

    try:
        from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates
        print("  [OK] preprocess")
    except ImportError as e:
        errors.append(f"  [FAIL] preprocess: {e}")

    try:
        from config import VIDEO_PATH, RECOGNITION_SIZE, DISPLAY_SIZE, MAPPING_FILE
        print(f"  [OK] config (VIDEO_PATH={VIDEO_PATH})")
    except ImportError as e:
        errors.append(f"  [FAIL] config: {e}")

    try:
        from coordinate_transformer import (
            CoordinateTransformer,
            calculate_real_world_coordinates,
            calculate_real_box_width
        )
        print("  [OK] coordinate_transformer")
    except ImportError as e:
        errors.append(f"  [FAIL] coordinate_transformer: {e}")

    try:
        from speed_utils import SpeedTracker
        print("  [OK] speed_utils")
    except ImportError as e:
        errors.append(f"  [FAIL] speed_utils: {e}")

    try:
        from data_export import CSVExporter
        print("  [OK] data_export")
    except ImportError as e:
        errors.append(f"  [FAIL] data_export: {e}")

    if errors:
        print("\nImport errors:")
        for e in errors:
            print(e)
        return False

    print("All imports successful!\n")
    return True


def validate_files():
    """Validate all required files exist."""
    print("Validating files...")

    from config import VIDEO_PATH, MAPPING_FILE

    files = {
        'Video': VIDEO_PATH,
        'Vehicle Model': 'best.pt',
        'Wheel Seg Model': 'runs/segment/wheel_seg/weights/best.pt',
        'Calibration': 'gopro_calibration_fisheye.npz',
        'Coordinate Mapping': MAPPING_FILE
    }

    all_exist = True
    for name, path in files.items():
        exists = Path(path).exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False

    print()
    return all_exist


def validate_models():
    """Validate models load correctly."""
    print("Validating models...")

    from ultralytics import YOLO

    try:
        vehicle_model = YOLO("best.pt")
        print("  [OK] Vehicle detection model loaded")
    except Exception as e:
        print(f"  [FAIL] Vehicle model: {e}")
        return False

    try:
        wheel_model = YOLO("runs/segment/wheel_seg/weights/best.pt")
        print("  [OK] Wheel segmentation model loaded")
    except Exception as e:
        print(f"  [FAIL] Wheel model: {e}")
        return False

    print()
    return True


def validate_calibration():
    """Validate calibration data loads correctly."""
    print("Validating calibration...")

    from preprocess import load_calibration_data
    from coordinate_transformer import CoordinateTransformer
    from config import MAPPING_FILE

    K, D, DIM = load_calibration_data()
    if K is None:
        print("  [FAIL] Could not load camera calibration")
        return False
    print(f"  [OK] Camera calibration: K={K.shape}, D={D.shape}, DIM={DIM}")

    try:
        transformer = CoordinateTransformer(MAPPING_FILE)
        print(f"  [OK] Coordinate transformer loaded")

        # Test transformation
        test_x, test_y = 960, 540  # Center of 1920x1080
        world_x, world_y = transformer.pixel_to_world(test_x, test_y)
        print(f"  [OK] Test transform: pixel({test_x}, {test_y}) -> world({world_x:.2f}, {world_y:.2f})")
    except Exception as e:
        print(f"  [FAIL] Coordinate transformer: {e}")
        return False

    print()
    return True


def validate_video_processing(num_frames=5):
    """Validate video processing with both approaches."""
    print(f"Validating video processing ({num_frames} frames)...")

    from ultralytics import YOLO
    from preprocess import preprocess_frame, load_calibration_data, rescale_coordinates
    from config import VIDEO_PATH, RECOGNITION_SIZE, DISPLAY_SIZE, MAPPING_FILE
    from coordinate_transformer import CoordinateTransformer, calculate_real_world_coordinates
    from speed_utils import SpeedTracker

    # Load models
    vehicle_model = YOLO("best.pt")
    wheel_model = YOLO("runs/segment/wheel_seg/weights/best.pt")

    # Load calibration
    K, D, DIM = load_calibration_data()
    transformer = CoordinateTransformer(MAPPING_FILE)
    speed_tracker = SpeedTracker()

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  [FAIL] Could not open video: {VIDEO_PATH}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  [OK] Video opened: {total_frames} frames @ {fps} FPS")

    bbox_detections = 0
    wheel_detections = 0

    for i in range(num_frames):
        success, frame = cap.read()
        if not success:
            print(f"  [FAIL] Could not read frame {i}")
            return False

        # Preprocess
        try:
            recognition_frame, display_frame = preprocess_frame(
                frame, K, D, DIM, RECOGNITION_SIZE, DISPLAY_SIZE
            )
        except Exception as e:
            print(f"  [FAIL] Preprocessing error: {e}")
            return False

        # Vehicle detection
        try:
            vehicle_results = vehicle_model.track(recognition_frame, persist=True)
            if vehicle_results[0].boxes.id is not None:
                bbox_detections += len(vehicle_results[0].boxes)
        except Exception as e:
            print(f"  [FAIL] Vehicle detection error: {e}")
            return False

        # Wheel segmentation
        try:
            wheel_results = wheel_model(recognition_frame, verbose=False)
            if wheel_results[0].masks is not None:
                wheel_detections += len(wheel_results[0].masks)
        except Exception as e:
            print(f"  [FAIL] Wheel segmentation error: {e}")
            return False

    cap.release()

    print(f"  [OK] Processed {num_frames} frames")
    print(f"  [OK] Vehicle detections: {bbox_detections}")
    print(f"  [OK] Wheel detections: {wheel_detections}")
    print()
    return True


def main():
    parser = argparse.ArgumentParser(description='Validate pipelines')
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to test')
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE VALIDATION")
    print("=" * 60 + "\n")

    results = []

    # Run validations
    results.append(("Imports", validate_imports()))
    results.append(("Files", validate_files()))
    results.append(("Models", validate_models()))
    results.append(("Calibration", validate_calibration()))
    results.append(("Video Processing", validate_video_processing(args.frames)))

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All validations PASSED!")
        print("\nYou can now run:")
        print("  python main.py                    # Object detection approach")
        print("  python main_segmentation.py       # Segmentation approach")
        print("  python compare_approaches.py      # Compare both approaches")
    else:
        print("Some validations FAILED. Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
