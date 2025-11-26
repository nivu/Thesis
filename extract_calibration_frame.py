#!/usr/bin/env python3
"""
Extract a single frame from a video for calibration purposes.
"""
import cv2
import sys
import os

def extract_frame(video_path, output_path, frame_number=None):
    """
    Extract a frame from a video.

    Args:
        video_path: Path to input video
        output_path: Path to save the frame
        frame_number: Specific frame to extract (default: middle of video)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Resolution: {width}x{height}")

    # Use middle frame if not specified
    if frame_number is None:
        frame_number = total_frames // 2

    print(f"  Extracting frame: {frame_number}")

    # Set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_number}")
        cap.release()
        return False

    # Save the frame
    cv2.imwrite(output_path, frame)
    print(f"✓ Frame saved to: {output_path}")

    cap.release()
    return True

if __name__ == "__main__":
    video_path = "traffic_analyis_data/Uni_west_1/GOPR0574.MP4"
    output_path = "camera_calibration/calibration_frame.png"

    extract_frame(video_path, output_path)
