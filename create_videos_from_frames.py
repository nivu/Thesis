"""
Create videos from dataset frames
Reconstructs original videos from the split train/valid/test frames
"""

import cv2
import os
from pathlib import Path
import re
from collections import defaultdict
import numpy as np


def collect_frames():
    """
    Collect all frames from train, valid, and test folders
    Returns a dict mapping video_name -> [(frame_num, file_path), ...]
    """
    frames_by_video = defaultdict(list)

    # Search all dataset folders
    dataset_root = Path("Dataset")

    for folder in ["train", "valid", "test"]:
        images_dir = dataset_root / folder / "images"
        if not images_dir.exists():
            continue

        for img_file in images_dir.glob("*.jpg"):
            # Extract video name and frame number
            # Format: GOPR0593_MP4-0000_jpg.rf.{hash}.jpg
            match = re.match(r'(GOPR\d+)_MP4-(\d+)_', img_file.name)
            if match:
                video_name = match.group(1)
                frame_num = int(match.group(2))
                frames_by_video[video_name].append((frame_num, str(img_file)))

    # Sort frames by frame number
    for video_name in frames_by_video:
        frames_by_video[video_name].sort(key=lambda x: x[0])

    return frames_by_video


def create_video(video_name, frames, output_dir="reconstructed_videos", fps=30):
    """
    Create a video from a list of frames

    Args:
        video_name: Name of the video (e.g., 'GOPR0593')
        frames: List of (frame_num, file_path) tuples
        output_dir: Directory to save the video
        fps: Frames per second
    """
    if not frames:
        print(f"No frames found for {video_name}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}.mp4")

    # Read first frame to get dimensions
    first_frame = cv2.imread(frames[0][1])
    if first_frame is None:
        print(f"Error: Could not read first frame for {video_name}")
        return

    height, width = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nCreating {video_name}.mp4:")
    print(f"  Frames: {len(frames)}")
    print(f"  Frame range: {frames[0][0]} to {frames[-1][0]}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    # Write frames
    for i, (frame_num, file_path) in enumerate(frames):
        img = cv2.imread(file_path)
        if img is None:
            print(f"  Warning: Could not read frame {frame_num} at {file_path}")
            continue

        # Resize if needed
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        out.write(img)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(frames)} frames")

    out.release()
    print(f"  ✓ Saved to: {output_path}")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.2f} MB")


def create_combined_video(frames_by_video, output_dir="reconstructed_videos", fps=30):
    """
    Create a single combined video from all sources
    """
    all_frames = []

    # Collect all frames with video prefix
    for video_name in sorted(frames_by_video.keys()):
        for frame_num, file_path in frames_by_video[video_name]:
            all_frames.append((video_name, frame_num, file_path))

    if not all_frames:
        print("No frames found!")
        return

    # Sort by video name then frame number
    all_frames.sort(key=lambda x: (x[0], x[1]))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_videos_combined.mp4")

    # Read first frame to get dimensions
    first_frame = cv2.imread(all_frames[0][2])
    if first_frame is None:
        print("Error: Could not read first frame")
        return

    height, width = first_frame.shape[:2]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nCreating combined video:")
    print(f"  Total frames: {len(all_frames)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Sources: {', '.join(sorted(frames_by_video.keys()))}")

    # Write frames
    for i, (video_name, frame_num, file_path) in enumerate(all_frames):
        img = cv2.imread(file_path)
        if img is None:
            print(f"  Warning: Could not read frame {frame_num} from {video_name}")
            continue

        # Resize if needed
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        out.write(img)

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(all_frames)} frames")

    out.release()
    print(f"  ✓ Saved to: {output_path}")

    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.2f} MB")


def main():
    """
    Main function to create videos from dataset frames
    """
    print("=" * 70)
    print("VIDEO RECONSTRUCTION FROM DATASET FRAMES")
    print("=" * 70)

    # Collect all frames
    print("\nScanning dataset folders...")
    frames_by_video = collect_frames()

    if not frames_by_video:
        print("No frames found in Dataset folder!")
        return

    print(f"\nFound {len(frames_by_video)} video sources:")
    total_frames = 0
    for video_name in sorted(frames_by_video.keys()):
        frame_count = len(frames_by_video[video_name])
        total_frames += frame_count
        print(f"  {video_name}: {frame_count} frames")
    print(f"  Total: {total_frames} frames")

    # Ask user what to create
    print("\n" + "-" * 70)
    print("Options:")
    print("  1. Create separate video for each source")
    print("  2. Create one combined video from all sources")
    print("  3. Create both")
    print("-" * 70)

    choice = input("Enter choice (1/2/3) [default: 3]: ").strip() or "3"

    fps = input("Enter FPS [default: 30]: ").strip() or "30"
    fps = int(fps)

    print("\n" + "=" * 70)

    if choice in ["1", "3"]:
        print("\nCreating individual videos...")
        for video_name in sorted(frames_by_video.keys()):
            create_video(video_name, frames_by_video[video_name], fps=fps)

    if choice in ["2", "3"]:
        create_combined_video(frames_by_video, fps=fps)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\nVideos saved in: reconstructed_videos/")


if __name__ == "__main__":
    main()
