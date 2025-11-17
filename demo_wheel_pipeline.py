"""
End-to-End Wheel-Based Speed & Dimension Estimation Demo
=========================================================
This script demonstrates the complete pipeline:
1. Wheel detection & segmentation using trained YOLOv8-seg model
2. Wheel center extraction from segmentation masks
3. Wheel grouping into vehicles
4. Wheelbase & track width calculation
5. Speed estimation via wheel tracking
6. Visualization with annotations

Note: This is a DEMONSTRATION without ground truth validation.
      Actual accuracy requires calibrated test data with known speeds.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import csv


@dataclass
class WheelDetection:
    """Represents a single detected wheel"""
    center: Tuple[float, float]  # (x, y) centroid
    mask: np.ndarray  # Segmentation mask
    class_name: str  # 'frontwheel', 'backwheel', 'middlewheel'
    confidence: float
    bbox: Tuple[float, float, float, float]  # x, y, w, h


@dataclass
class Vehicle:
    """Represents a vehicle with its wheels"""
    id: int
    wheels: List[WheelDetection]
    wheelbase: float  # Distance between front and rear axles (meters)
    track_width: float  # Distance between left and right wheels (meters)
    center: Tuple[float, float]  # Vehicle center
    speed_kmh: float  # Estimated speed


class WheelSegmentationPipeline:
    def __init__(
        self,
        model_path: str = "runs/segment/wheel_seg/weights/best.pt",
        homography_file: Optional[str] = None,
        wheel_diameter_m: float = 0.65,  # Average car wheel diameter
        conf_threshold: float = 0.3
    ):
        """
        Initialize the wheel-based pipeline

        Args:
            model_path: Path to trained wheel segmentation model
            homography_file: Optional homography transformation file
            wheel_diameter_m: Known wheel diameter for scale reference
            conf_threshold: Detection confidence threshold
        """
        print("=" * 70)
        print("Initializing Wheel Segmentation Pipeline")
        print("=" * 70)

        # Load model
        print(f"\nLoading model from: {model_path}")
        self.model = YOLO(model_path)
        print(f"Model loaded successfully!")

        # Configuration
        self.wheel_diameter = wheel_diameter_m
        self.conf_threshold = conf_threshold

        # Load homography if provided
        self.homography = None
        if homography_file and Path(homography_file).exists():
            with open(homography_file, 'r') as f:
                data = json.load(f)
                self.homography = np.array(data['transformation_matrix'])
            print(f"Loaded homography from: {homography_file}")
        else:
            print("No homography file - using pixel-based measurements")

        # Tracking state
        self.vehicle_tracks = {}  # track_id -> deque of positions
        self.next_vehicle_id = 0
        self.speed_buffer_size = 10

    def extract_wheel_center(self, mask: np.ndarray) -> Tuple[float, float]:
        """Extract centroid from wheel segmentation mask"""
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Get largest contour (main wheel)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        return (cx, cy)

    def detect_wheels(self, frame: np.ndarray) -> List[WheelDetection]:
        """Detect and segment wheels in frame"""
        results = self.model(frame, conf=self.conf_threshold, verbose=False)

        wheels = []
        if results[0].masks is None:
            return wheels

        # Extract detections
        boxes = results[0].boxes.xywh.cpu().numpy()
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        # Process each detection
        for box, mask, cls_id, conf in zip(boxes, masks, classes, confidences):
            # Resize mask to frame size
            mask_resized = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # Extract center
            center = self.extract_wheel_center(mask_resized)
            if center is None:
                continue

            # Get class name
            class_name = self.model.names[int(cls_id)]

            wheels.append(WheelDetection(
                center=center,
                mask=mask_resized,
                class_name=class_name,
                confidence=float(conf),
                bbox=tuple(box)
            ))

        return wheels

    def group_wheels_into_vehicles(
        self,
        wheels: List[WheelDetection],
        max_distance: float = 300
    ) -> List[List[WheelDetection]]:
        """
        Group wheels into vehicles based on proximity

        Args:
            wheels: List of detected wheels
            max_distance: Maximum pixel distance between wheels of same vehicle
        """
        if len(wheels) < 2:
            return [[w] for w in wheels]

        # Simple clustering based on distance
        clusters = []
        used = set()

        for i, wheel in enumerate(wheels):
            if i in used:
                continue

            cluster = [wheel]
            used.add(i)

            # Find nearby wheels
            for j, other_wheel in enumerate(wheels):
                if j in used:
                    continue

                dist = np.sqrt(
                    (wheel.center[0] - other_wheel.center[0])**2 +
                    (wheel.center[1] - other_wheel.center[1])**2
                )

                if dist < max_distance:
                    cluster.append(other_wheel)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def calculate_vehicle_dimensions(
        self,
        wheels: List[WheelDetection]
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate wheelbase and track width from wheel positions

        Returns:
            (wheelbase, track_width, center)
        """
        if len(wheels) < 2:
            return 0, 0, wheels[0].center if wheels else (0, 0)

        # Get wheel centers
        centers = np.array([w.center for w in wheels])

        # Sort by Y coordinate (top to bottom)
        sorted_by_y = centers[centers[:, 1].argsort()]

        # Wheelbase: distance between top and bottom wheels (front to rear)
        wheelbase_px = np.linalg.norm(sorted_by_y[0] - sorted_by_y[-1])

        # Track width: max horizontal distance
        sorted_by_x = centers[centers[:, 0].argsort()]
        track_width_px = abs(sorted_by_x[-1][0] - sorted_by_x[0][0])

        # Vehicle center
        center = tuple(np.mean(centers, axis=0))

        # Convert to meters if homography available
        if self.homography is not None:
            # Transform dimensions (simplified)
            wheelbase_m = wheelbase_px / 100  # Placeholder conversion
            track_width_m = track_width_px / 100
        else:
            # Use pixel-based with wheel diameter as reference
            # Assume average wheel appears ~30 pixels in diameter
            px_per_meter = 30 / self.wheel_diameter
            wheelbase_m = wheelbase_px / px_per_meter
            track_width_m = track_width_px / px_per_meter

        return wheelbase_m, track_width_m, center

    def apply_homography(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Apply homography transformation to convert pixel to real-world coordinates"""
        if self.homography is None:
            return point

        x, y = point
        denom = self.homography[2, 0] * x + self.homography[2, 1] * y + self.homography[2, 2]

        if abs(denom) < 1e-6:
            return point

        X = (self.homography[0, 0] * x + self.homography[0, 1] * y + self.homography[0, 2]) / denom
        Y = (self.homography[1, 0] * x + self.homography[1, 1] * y + self.homography[1, 2]) / denom

        return (X, Y)

    def calculate_speed(
        self,
        vehicle_id: int,
        current_pos: Tuple[float, float],
        fps: float
    ) -> float:
        """
        Calculate vehicle speed based on position tracking

        Args:
            vehicle_id: Vehicle tracking ID
            current_pos: Current position (real-world coordinates)
            fps: Video frame rate

        Returns:
            Speed in km/h
        """
        # Initialize tracking buffer for new vehicles
        if vehicle_id not in self.vehicle_tracks:
            self.vehicle_tracks[vehicle_id] = deque(maxlen=self.speed_buffer_size)

        track = self.vehicle_tracks[vehicle_id]

        # Add current position
        track.append(current_pos)

        # Need at least 2 positions to calculate speed
        if len(track) < 2:
            return 0.0

        # Calculate average speed over buffer
        speeds = []
        for i in range(len(track) - 1):
            pos1 = track[i]
            pos2 = track[i + 1]

            distance_m = np.sqrt(
                (pos2[0] - pos1[0])**2 +
                (pos2[1] - pos1[1])**2
            )

            time_s = 1.0 / fps if fps > 0 else 1.0
            speed_ms = distance_m / time_s
            speed_kmh = speed_ms * 3.6

            speeds.append(speed_kmh)

        return np.mean(speeds) if speeds else 0.0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        fps: float = 30.0
    ) -> Tuple[np.ndarray, List[Vehicle]]:
        """
        Process a single frame through the complete pipeline

        Returns:
            (annotated_frame, vehicles)
        """
        # Detect wheels
        wheels = self.detect_wheels(frame)

        # Group wheels into vehicles
        vehicle_wheel_groups = self.group_wheels_into_vehicles(wheels)

        # Process each vehicle
        vehicles = []
        for wheel_group in vehicle_wheel_groups:
            if len(wheel_group) < 2:  # Need at least 2 wheels
                continue

            # Calculate dimensions
            wheelbase, track_width, center_px = self.calculate_vehicle_dimensions(wheel_group)

            # Transform to real-world coordinates
            center_world = self.apply_homography(center_px)

            # Calculate speed
            vehicle_id = self.next_vehicle_id
            self.next_vehicle_id += 1

            speed_kmh = self.calculate_speed(vehicle_id, center_world, fps)

            vehicles.append(Vehicle(
                id=vehicle_id,
                wheels=wheel_group,
                wheelbase=wheelbase,
                track_width=track_width,
                center=center_px,
                speed_kmh=speed_kmh
            ))

        # Annotate frame
        annotated_frame = self.draw_annotations(frame.copy(), vehicles)

        return annotated_frame, vehicles

    def draw_annotations(
        self,
        frame: np.ndarray,
        vehicles: List[Vehicle]
    ) -> np.ndarray:
        """Draw visualizations on frame"""
        overlay = frame.copy()

        for vehicle in vehicles:
            # Draw wheels
            for wheel in vehicle.wheels:
                # Draw wheel mask
                mask_color = {
                    'frontwheel': (0, 255, 0),    # Green
                    'backwheel': (255, 0, 0),     # Blue
                    'middlewheel': (0, 255, 255)  # Yellow
                }.get(wheel.class_name, (255, 255, 255))

                # Apply semi-transparent mask
                mask_overlay = np.zeros_like(frame)
                mask_overlay[wheel.mask > 0.5] = mask_color
                cv2.addWeighted(overlay, 0.7, mask_overlay, 0.3, 0, overlay)

                # Draw wheel center
                cx, cy = int(wheel.center[0]), int(wheel.center[1])
                cv2.circle(overlay, (cx, cy), 5, mask_color, -1)
                cv2.circle(overlay, (cx, cy), 8, (255, 255, 255), 2)

            # Draw vehicle center
            vx, vy = int(vehicle.center[0]), int(vehicle.center[1])
            cv2.circle(overlay, (vx, vy), 10, (0, 0, 255), -1)
            cv2.circle(overlay, (vx, vy), 12, (255, 255, 255), 2)

            # Draw bounding box around all wheels
            wheel_centers = np.array([w.center for w in vehicle.wheels])
            x_min = int(wheel_centers[:, 0].min() - 50)
            y_min = int(wheel_centers[:, 1].min() - 50)
            x_max = int(wheel_centers[:, 0].max() + 50)
            y_max = int(wheel_centers[:, 1].max() + 50)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Draw info text
            info_lines = [
                f"ID: {vehicle.id}",
                f"Speed: {vehicle.speed_kmh:.1f} km/h",
                f"Wheelbase: {vehicle.wheelbase:.2f}m",
                f"Track: {vehicle.track_width:.2f}m",
                f"Wheels: {len(vehicle.wheels)}"
            ]

            y_offset = y_min - 10
            for i, line in enumerate(info_lines):
                y_pos = y_offset - (len(info_lines) - i) * 20
                cv2.putText(
                    overlay, line, (x_min, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
                )

        return overlay


def process_video(
    video_path: str,
    output_path: str,
    model_path: str = "runs/segment/wheel_seg/weights/best.pt",
    max_frames: Optional[int] = None,
    display: bool = False
):
    """
    Process a video through the wheel-based pipeline

    Args:
        video_path: Path to input video
        output_path: Path for output video
        model_path: Path to trained model
        max_frames: Optional limit on frames to process
        display: Whether to display frames during processing
    """
    print("\n" + "=" * 70)
    print(f"Processing Video: {video_path}")
    print("=" * 70)

    # Initialize pipeline
    pipeline = WheelSegmentationPipeline(model_path=model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total Frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Setup CSV export
    csv_path = output_path.replace('.mp4', '_data.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame', 'vehicle_id', 'num_wheels',
        'wheelbase_m', 'track_width_m',
        'center_x', 'center_y', 'speed_kmh'
    ])

    # Process frames
    frame_count = 0
    processed_count = 0

    print("\nProcessing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Limit frames if specified
        if max_frames and frame_count > max_frames:
            break

        # Process frame
        annotated_frame, vehicles = pipeline.process_frame(frame, frame_count, fps)

        # Write output
        out.write(annotated_frame)

        # Export data
        for vehicle in vehicles:
            csv_writer.writerow([
                frame_count,
                vehicle.id,
                len(vehicle.wheels),
                f"{vehicle.wheelbase:.2f}",
                f"{vehicle.track_width:.2f}",
                f"{vehicle.center[0]:.1f}",
                f"{vehicle.center[1]:.1f}",
                f"{vehicle.speed_kmh:.1f}"
            ])

        processed_count += 1

        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - Found {len(vehicles)} vehicles")

        # Display if requested
        if display:
            cv2.imshow('Wheel Detection Pipeline', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    out.release()
    csv_file.close()
    if display:
        cv2.destroyAllWindows()

    print(f"\n✓ Processing complete!")
    print(f"  Processed: {processed_count} frames")
    print(f"  Output video: {output_path}")
    print(f"  Output data: {csv_path}")


def main():
    """Main demo function"""
    print("\n" + "=" * 70)
    print("WHEEL-BASED SPEED & DIMENSION ESTIMATION DEMO")
    print("=" * 70)
    print("\nThis demo processes reconstructed videos using the wheel segmentation model.")
    print("NOTE: Without ground truth data, speeds are ESTIMATES for demonstration only.")
    print("\n" + "=" * 70)

    # Check if model exists
    model_path = "runs/segment/wheel_seg/weights/best.pt"
    if not Path(model_path).exists():
        print(f"\n⚠️  Model not found at: {model_path}")
        print("\nPlease wait for wheel segmentation training to complete.")
        print("Expected location: runs/segment/wheel_seg/weights/best.pt")
        return

    # Check for reconstructed videos
    video_dir = Path("reconstructed_videos")
    if not video_dir.exists():
        print(f"\n⚠️  Video directory not found: {video_dir}")
        print("Please run create_videos_from_frames.py first.")
        return

    # Get available videos
    videos = list(video_dir.glob("*.mp4"))
    if not videos:
        print(f"\n⚠️  No videos found in {video_dir}")
        return

    print(f"\nFound {len(videos)} videos:")
    for i, video in enumerate(videos, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {i}. {video.name} ({size_mb:.1f} MB)")

    # Process videos
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'-' * 70}")
    choice = input("\nProcess [1] Single video, [2] All videos, or [Q] Quit: ").strip()

    if choice.lower() == 'q':
        return
    elif choice == '1':
        # Single video
        video_idx = int(input(f"Enter video number (1-{len(videos)}): ")) - 1
        if 0 <= video_idx < len(videos):
            video = videos[video_idx]
            output_path = str(output_dir / f"demo_{video.name}")
            process_video(str(video), output_path, model_path, display=True)
    elif choice == '2':
        # All videos
        for video in videos:
            output_path = str(output_dir / f"demo_{video.name}")
            process_video(str(video), output_path, model_path)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
