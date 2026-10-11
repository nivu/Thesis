"""
Keypoint-Based Vehicle Localization Pipeline

Estimates vehicle tire contact points from 2D bounding boxes,
then converts to world coordinates using homography.
"""

import os
import sys
import cv2
import numpy as np
import json
from collections import deque

# Add YOLOx3D to path
YOLOX3D_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "YOLOx3D")
sys.path.insert(0, os.path.join(YOLOX3D_PATH, "src"))

from models.yolo_v11 import YOLOv11

# Import coordinate transformer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
try:
    from utils.coordinate_transformer import CoordinateTransformer
except ImportError:
    class CoordinateTransformer:
        def __init__(self, mapping_file):
            self.transformation_matrix = None
        def pixel_to_world(self, x, y):
            return (0, 0)

# Import config
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_config", os.path.join(PIPELINE_DIR, "config.py"))
pipeline_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_config)

# Vehicle classes
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}


class KeypointSpeedTracker:
    """Track vehicle positions and calculate speed from keypoints."""

    def __init__(self, fps, history_length=10):
        self.fps = fps
        self.time_per_frame = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.history_length = history_length
        self.tracks = {}  # track_id -> deque of (frame, world_x, world_y)

    def update(self, track_id, frame_num, world_x, world_y):
        """Update position and return smoothed speed."""
        if track_id not in self.tracks:
            self.tracks[track_id] = deque(maxlen=self.history_length)

        self.tracks[track_id].append((frame_num, world_x, world_y))

        if len(self.tracks[track_id]) < 2:
            return 0.0, world_x, world_y

        # Calculate speed from displacement
        hist = self.tracks[track_id]
        f1, x1, y1 = hist[0]
        f2, x2, y2 = hist[-1]

        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)

        frame_diff = f2 - f1
        if frame_diff == 0:
            return 0.0, world_x, world_y

        time_delta = frame_diff * self.time_per_frame
        speed_ms = distance / time_delta
        speed_kmh = speed_ms * 3.6

        # Smoothed position
        smooth_x = np.mean([p[1] for p in hist])
        smooth_y = np.mean([p[2] for p in hist])

        return speed_kmh, smooth_x, smooth_y


class KeypointPipeline:
    """
    Keypoint-based vehicle localization.

    Estimates tire contact points from bounding box corners,
    converts to world coordinates via homography.
    """

    def __init__(self, conf_threshold=0.25):
        print("Initializing Keypoint Pipeline...")

        # YOLO detector with tracking
        self.detector = YOLOv11(
            model_size='nano',
            conf_thres=conf_threshold,
            iou_thres=0.45,
            device='cpu'
        )

        # Coordinate transformer
        self.transformer = None
        if os.path.exists(pipeline_config.MAPPING_FILE):
            self.transformer = CoordinateTransformer(pipeline_config.MAPPING_FILE)
            print(f"Loaded homography from {pipeline_config.MAPPING_FILE}")

        print("Keypoint Pipeline initialized!")

    def estimate_tire_contact_points(self, bbox):
        """
        Estimate tire contact points from 2D bounding box.

        For a side view, tires are at bottom-left and bottom-right of bbox.
        For an angled view, we estimate based on perspective.

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            List of (x, y) pixel coordinates for estimated tire contact points
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        # Bottom of bounding box is ground contact plane
        bottom_y = y2

        # Estimate tire positions:
        # - Front tire: ~15% from front edge
        # - Rear tire: ~15% from rear edge
        front_x = x1 + width * 0.15
        rear_x = x2 - width * 0.15

        # For surveillance camera (looking down at angle):
        # The visible wheels are typically on one side
        # We estimate 2 contact points
        keypoints = [
            (front_x, bottom_y),  # Front tire contact
            (rear_x, bottom_y),   # Rear tire contact
        ]

        return keypoints

    def process_frame(self, frame, frame_num=0, speed_tracker=None):
        """Process frame and extract vehicle keypoints."""
        result = {
            'vehicles': [],
            'annotated_frame': frame.copy()
        }

        # Detect vehicles with tracking
        _, detections = self.detector.detect(frame, track=True)

        # Filter to vehicles
        vehicles = [d for d in detections if d[3].lower() in VEHICLE_CLASSES]

        for det in vehicles:
            bbox, score, class_id, class_name, track_id = det
            x1, y1, x2, y2 = bbox

            # Estimate tire contact keypoints
            keypoints_px = self.estimate_tire_contact_points(bbox)

            # Convert to world coordinates
            keypoints_world = []
            for kp in keypoints_px:
                if self.transformer:
                    wx, wy = self.transformer.pixel_to_world(kp[0], kp[1])
                else:
                    wx, wy = kp[0] / 100, kp[1] / 100  # Fallback
                keypoints_world.append((wx, wy))

            # Calculate vehicle center from keypoints
            center_px = np.mean(keypoints_px, axis=0)
            if self.transformer:
                center_world = self.transformer.pixel_to_world(center_px[0], center_px[1])
            else:
                center_world = (center_px[0] / 100, center_px[1] / 100)

            # Calculate wheelbase from keypoints
            if len(keypoints_world) >= 2:
                kp1, kp2 = keypoints_world[0], keypoints_world[1]
                wheelbase = np.sqrt((kp2[0] - kp1[0])**2 + (kp2[1] - kp1[1])**2)
            else:
                wheelbase = 0

            # Calculate speed if tracker provided
            speed = 0.0
            if speed_tracker and track_id is not None:
                speed, smooth_x, smooth_y = speed_tracker.update(
                    track_id, frame_num, center_world[0], center_world[1]
                )
                center_world = (smooth_x, smooth_y)

            result['vehicles'].append({
                'track_id': track_id,
                'class_name': class_name,
                'confidence': score,
                'bbox': bbox,
                'keypoints_px': keypoints_px,
                'keypoints_world': keypoints_world,
                'center_world': center_world,
                'wheelbase': wheelbase,
                'speed_kmh': speed,
            })

        # Draw annotations
        result['annotated_frame'] = self.draw_annotations(
            frame.copy(), result['vehicles']
        )

        return result

    def draw_annotations(self, frame, vehicles):
        """Draw keypoint annotations on frame."""
        for v in vehicles:
            track_id = v['track_id']
            bbox = v['bbox']
            keypoints = v['keypoints_px']
            speed = v['speed_kmh']
            wheelbase = v['wheelbase']
            class_name = v['class_name']
            center = v['center_world']

            x1, y1, x2, y2 = [int(b) for b in bbox]

            # Color based on track ID
            if track_id is not None:
                np.random.seed(int(track_id) % 1000)
                color = tuple(int(c) for c in np.random.randint(100, 255, 3))
            else:
                color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw keypoints (tire contact points)
            for i, kp in enumerate(keypoints):
                px, py = int(kp[0]), int(kp[1])
                # Draw tire contact point
                cv2.circle(frame, (px, py), 8, color, -1)
                cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

                # Draw vertical line to show ground contact
                cv2.line(frame, (px, py - 15), (px, py + 5), color, 2)

                # Label keypoint
                label = "Front" if i == 0 else "Rear"
                cv2.putText(frame, label, (px - 20, py - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw line connecting keypoints (wheelbase visualization)
            if len(keypoints) >= 2:
                kp1, kp2 = keypoints[0], keypoints[1]
                cv2.line(frame, (int(kp1[0]), int(kp1[1])),
                        (int(kp2[0]), int(kp2[1])), (255, 255, 0), 2)

            # Draw info label
            label_lines = []
            if track_id is not None:
                label_lines.append(f"ID:{track_id}")
            label_lines.append(f"{class_name}")
            if speed > 0:
                label_lines.append(f"{speed:.1f} km/h")
            if wheelbase > 0:
                label_lines.append(f"WB:{wheelbase:.2f}m")
            label_lines.append(f"({center[0]:.1f}, {center[1]:.1f})m")

            # Draw label background
            label_y = max(y1 - 10, 60)
            for i, line in enumerate(label_lines):
                y_pos = label_y - (len(label_lines) - i - 1) * 18
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1 - 2, y_pos - th - 2),
                             (x1 + tw + 2, y_pos + 2), (0, 0, 0), -1)
                cv2.putText(frame, line, (x1, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


def process_video(video_path, output_path, max_frames=None):
    """Process video with keypoint pipeline."""
    pipeline = KeypointPipeline()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing: {width}x{height} @ {fps:.1f} FPS, {total} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    speed_tracker = KeypointSpeedTracker(fps)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_num >= max_frames:
            break

        result = pipeline.process_frame(frame, frame_num, speed_tracker)
        writer.write(result['annotated_frame'])

        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Processed {frame_num}/{total if not max_frames else max_frames} frames")

    cap.release()
    writer.release()
    print(f"Done! Output: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--output', required=True, help='Output video path')
    parser.add_argument('--max-frames', type=int, help='Max frames to process')
    args = parser.parse_args()

    process_video(args.video, args.output, args.max_frames)
