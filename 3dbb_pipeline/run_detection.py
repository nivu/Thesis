#!/usr/bin/env python
"""
Standalone 3D BBox detection script.
Run this from the 3dbb_pipeline directory.
"""
import os
import sys
import json
import cv2
import numpy as np

# Ensure correct path for YOLOx3D
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
YOLOX3D_SRC = os.path.join(os.path.dirname(SCRIPT_DIR), "YOLOx3D", "src")
sys.path.insert(0, YOLOX3D_SRC)

from main import Pipeline3DBB


def process_video(video_path, output_path, max_frames=None):
    """Process video and save detection results."""
    print(f"Processing video: {video_path}")

    pipeline = Pipeline3DBB(use_geometry=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames:
        total_frames = min(total_frames, max_frames)

    print(f"FPS: {fps}, Total frames: {total_frames}")

    results = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")

        # Process frame
        result = pipeline.process_frame(frame, conf_threshold=0.25)
        world_coords = pipeline.get_world_coordinates(result['boxes_3d'])

        frame_results = {
            'frame': frame_count,
            'fps': fps,
            'detections': []
        }

        for wc in world_coords:
            detection = {
                'track_id': wc.get('track_id'),
                'class': wc.get('class_name', 'unknown'),
                'confidence': float(wc.get('confidence', 0)),
                'world_x': float(wc.get('world_x', 0)),
                'world_y': float(wc.get('world_y', 0)),
                'depth': float(wc.get('depth', 0)),
                'dimensions': wc.get('dimensions', {}),
                'yaw': float(wc.get('yaw', 0)),
                'center_3d': wc.get('center_3d', {}),
                # Bottom 4 corners of the 3D bbox (ground contact points)
                # Each corner has x (lateral), y (forward/depth), height
                'bottom_corners': wc.get('bottom_corners', [])
            }
            frame_results['detections'].append(detection)

        results.append(frame_results)

    cap.release()

    # Save results (handle numpy types)
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="3D BBox Detection")
    parser.add_argument('--video', type=str, required=True, help='Path to video')
    parser.add_argument('--output', type=str, default='3dbb_results.json', help='Output JSON path')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    args = parser.parse_args()

    process_video(args.video, args.output, args.max_frames)
