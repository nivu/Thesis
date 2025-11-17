"""
Quick Test: Wheel Detection on Sample Frames
=============================================
Test the wheel segmentation model on a few frames from the dataset
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import random


def test_wheel_detection():
    """Quick test of wheel detection on dataset images"""
    print("=" * 70)
    print("Quick Wheel Detection Test")
    print("=" * 70)

    # Check if model exists
    model_path = "runs/segment/wheel_seg/weights/best.pt"
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not yet trained: {model_path}")
        print("Waiting for training to complete...")
        return

    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    print("✓ Model loaded!")

    # Find test images
    test_dir = Path("Wheel_seg-6/test/images")
    if not test_dir.exists():
        test_dir = Path("Dataset/test/images")

    if not test_dir.exists():
        print(f"\n⚠️  No test images found")
        return

    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not images:
        print("No images found in test directory")
        return

    # Select random images
    num_samples = min(5, len(images))
    sample_images = random.sample(images, num_samples)

    print(f"\nTesting on {num_samples} random images from {test_dir}")
    print("-" * 70)

    # Process each image
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n{i}. Processing: {img_path.name}")

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ✗ Failed to load image")
            continue

        # Run detection
        results = model(img, conf=0.3, verbose=False)

        # Check results
        if results[0].masks is None:
            print(f"   ✗ No wheels detected")
            continue

        num_wheels = len(results[0].boxes)
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        print(f"   ✓ Detected {num_wheels} wheels")

        # Count by type
        class_counts = {}
        for cls_id, conf in zip(classes, confidences):
            class_name = model.names[int(cls_id)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            print(f"      - {class_name}: confidence {conf:.2f}")

        # Visualize
        annotated = results[0].plot()

        # Save result
        output_path = f"test_output_{img_path.stem}.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"   Saved: {output_path}")

    print("\n" + "=" * 70)
    print("✓ Quick test complete!")
    print(f"Check the generated test_output_*.jpg files")
    print("=" * 70)


def test_on_video_frame():
    """Test on a single frame from reconstructed video"""
    print("\n" + "=" * 70)
    print("Testing on Video Frame")
    print("=" * 70)

    model_path = "runs/segment/wheel_seg/weights/best.pt"
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not yet trained")
        return

    # Find a reconstructed video
    video_dir = Path("reconstructed_videos")
    if not video_dir.exists():
        print(f"\n⚠️  No reconstructed videos found")
        return

    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print("No videos found")
        return

    # Use the largest video (most content)
    video = max(videos, key=lambda v: v.stat().st_size)
    print(f"\nUsing video: {video.name}")

    # Load model
    model = YOLO(model_path)

    # Open video and grab a frame from the middle
    cap = cv2.VideoCapture(str(video))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to read frame")
        return

    print(f"Processing frame {mid_frame}/{total_frames}")

    # Run detection
    results = model(frame, conf=0.3)

    if results[0].masks is None:
        print("No wheels detected in this frame")
        print("Try running on multiple frames or different video")
    else:
        num_wheels = len(results[0].boxes)
        print(f"✓ Detected {num_wheels} wheels!")

        # Save annotated frame
        annotated = results[0].plot()
        output_path = "test_video_frame.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    # Run both tests
    test_wheel_detection()
    test_on_video_frame()

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("\n1. Review the test_output_*.jpg images to verify wheel detection")
    print("2. Once satisfied, run the full pipeline:")
    print("   python demo_wheel_pipeline.py")
    print("\n3. This will process all reconstructed videos and generate:")
    print("   - Annotated videos with wheel detection overlay")
    print("   - CSV files with vehicle dimensions and speeds")
    print("   - Visual demonstration of the complete system")
    print("\nNote: Speed estimates are for demonstration only.")
    print("Ground truth validation requires calibrated test data.")
    print("=" * 70)
