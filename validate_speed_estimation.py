"""
Speed Estimation Validation Script
===================================
Validates the speed estimation pipeline against ground truth test videos

Usage:
    python validate_speed_estimation.py --videos 20kmph.mp4 30kmph.mp4 40kmph.mp4 50kmph.mp4

Or run automatically on all test videos in current directory:
    python validate_speed_estimation.py --auto
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
from collections import defaultdict
import pandas as pd
from datetime import datetime


class SpeedValidator:
    """Validates speed estimation against ground truth"""

    def __init__(
        self,
        model_path="runs/segment/wheel_seg/weights/best.pt",
        calibration_file="gopro_calibration_fisheye.npz",
        homography_file="coordinate_mapping_2030.json"
    ):
        print("=" * 70)
        print("Speed Estimation Validator")
        print("=" * 70)

        # Load model
        print(f"\nLoading model: {model_path}")
        self.model = YOLO(model_path)

        # Load calibration
        if Path(calibration_file).exists():
            print(f"Loading calibration: {calibration_file}")
            calib = np.load(calibration_file)
            self.K = calib['K']
            self.D = calib['D']
            self.DIM = calib['DIM']
            print(f"✓ Calibration loaded (RMS: {calib.get('rms', 'N/A'):.4f})")
        else:
            print(f"⚠️ Calibration file not found: {calibration_file}")
            self.K = self.D = self.DIM = None

        # Load homography (if available)
        if Path(homography_file).exists():
            with open(homography_file, 'r') as f:
                data = json.load(f)
                self.homography = np.array(data['transformation_matrix'])
            print(f"✓ Homography loaded")
        else:
            print(f"⚠️ Homography file not found: {homography_file}")
            self.homography = None

    def undistort_frame(self, frame):
        """Undistort fisheye frame"""
        if self.K is None or self.D is None:
            return frame

        h, w = frame.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.K,
            (w, h), cv2.CV_16SC2
        )
        return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

    def detect_and_track_vehicles(self, video_path, ground_truth_speed):
        """
        Process video and estimate speeds

        Returns:
            dict with statistics
        """
        print(f"\n{'-' * 70}")
        print(f"Processing: {Path(video_path).name}")
        print(f"Ground Truth: {ground_truth_speed} km/h")
        print(f"{'-' * 70}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Could not open video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"FPS: {fps:.2f}, Total Frames: {total_frames}")

        # Storage for tracking
        vehicle_tracks = {}
        all_speed_estimates = []

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Undistort
            if self.K is not None:
                frame = self.undistort_frame(frame)

            # Detect vehicles (using detection model for now)
            # TODO: Use wheel segmentation model when available
            results = self.model.track(frame, persist=True, verbose=False)

            # Extract detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box

                    # Apply homography if available
                    if self.homography is not None:
                        # Transform bottom center point
                        px, py = x, y + h/2
                        denom = (self.homography[2, 0] * px +
                                self.homography[2, 1] * py +
                                self.homography[2, 2])

                        if abs(denom) > 1e-6:
                            real_x = (self.homography[0, 0] * px +
                                    self.homography[0, 1] * py +
                                    self.homography[0, 2]) / denom
                            real_y = (self.homography[1, 0] * px +
                                    self.homography[1, 1] * py +
                                    self.homography[1, 2]) / denom
                        else:
                            real_x, real_y = px, py
                    else:
                        real_x, real_y = x, y

                    # Track position
                    if track_id not in vehicle_tracks:
                        vehicle_tracks[track_id] = []

                    vehicle_tracks[track_id].append({
                        'frame': frame_count,
                        'position': (real_x, real_y)
                    })

            # Progress
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%) - "
                      f"Tracking {len(vehicle_tracks)} vehicles")

        cap.release()

        # Calculate speeds from tracks
        print(f"\nCalculating speeds from {len(vehicle_tracks)} vehicle tracks...")

        for track_id, positions in vehicle_tracks.items():
            if len(positions) < 10:  # Need enough frames
                continue

            # Calculate speed over entire track
            speeds = []
            for i in range(len(positions) - 1):
                p1 = positions[i]
                p2 = positions[i + 1]

                # Distance in meters (if homography available, else pixels)
                dist = np.sqrt(
                    (p2['position'][0] - p1['position'][0])**2 +
                    (p2['position'][1] - p1['position'][1])**2
                )

                # Time difference
                frame_diff = p2['frame'] - p1['frame']
                time_s = frame_diff / fps

                if time_s > 0:
                    speed_ms = dist / time_s
                    speed_kmh = speed_ms * 3.6
                    speeds.append(speed_kmh)

            if speeds:
                avg_speed = np.mean(speeds)
                all_speed_estimates.append(avg_speed)
                print(f"  Vehicle {track_id}: {avg_speed:.2f} km/h "
                      f"(min: {np.min(speeds):.2f}, max: {np.max(speeds):.2f})")

        # Calculate statistics
        if all_speed_estimates:
            estimated_speed = np.mean(all_speed_estimates)
            std_dev = np.std(all_speed_estimates)
            error = abs(estimated_speed - ground_truth_speed)
            percent_error = (error / ground_truth_speed) * 100

            result = {
                'video': Path(video_path).name,
                'ground_truth_kmh': ground_truth_speed,
                'estimated_speed_kmh': estimated_speed,
                'std_dev_kmh': std_dev,
                'absolute_error_kmh': error,
                'percent_error': percent_error,
                'num_vehicles': len(vehicle_tracks),
                'num_estimates': len(all_speed_estimates),
                'min_estimate': np.min(all_speed_estimates),
                'max_estimate': np.max(all_speed_estimates)
            }

            print(f"\n{'✓' if error < 5 else '⚠️'} Results:")
            print(f"  Estimated Speed: {estimated_speed:.2f} ± {std_dev:.2f} km/h")
            print(f"  Ground Truth:    {ground_truth_speed:.2f} km/h")
            print(f"  Absolute Error:  {error:.2f} km/h ({percent_error:.1f}%)")

            return result
        else:
            print(f"✗ No speed estimates generated (no vehicles tracked)")
            return None

    def run_validation(self, test_videos):
        """
        Run validation on all test videos

        Args:
            test_videos: dict of {video_path: ground_truth_speed}

        Returns:
            pandas DataFrame with results
        """
        print("\n" + "=" * 70)
        print("Running Speed Validation")
        print("=" * 70)
        print(f"\nTest videos: {len(test_videos)}")

        results = []

        for video_path, ground_truth in test_videos.items():
            if not Path(video_path).exists():
                print(f"\n✗ Video not found: {video_path}")
                continue

            result = self.detect_and_track_vehicles(video_path, ground_truth)
            if result:
                results.append(result)

        # Create DataFrame
        if results:
            df = pd.DataFrame(results)

            # Calculate overall metrics
            print("\n" + "=" * 70)
            print("OVERALL VALIDATION RESULTS")
            print("=" * 70)

            mae = np.mean(df['absolute_error_kmh'])
            rmse = np.sqrt(np.mean(df['absolute_error_kmh']**2))
            mean_percent_error = np.mean(df['percent_error'])

            print(f"\nMean Absolute Error (MAE):  {mae:.2f} km/h")
            print(f"Root Mean Square Error (RMSE): {rmse:.2f} km/h")
            print(f"Mean Percentage Error: {mean_percent_error:.1f}%")

            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"validation_results_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to: {output_file}")

            return df
        else:
            print("\n✗ No valid results generated")
            return None


def main():
    parser = argparse.ArgumentParser(description='Validate speed estimation')
    parser.add_argument('--videos', nargs='+', help='Test video files')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect test videos in current directory')
    parser.add_argument('--model', default='runs/segment/wheel_seg/weights/best.pt',
                       help='Path to trained model')

    args = parser.parse_args()

    # Map video files to ground truth speeds
    test_videos = {}

    if args.auto:
        # Auto-detect videos with speed in filename
        current_dir = Path('.')
        for speed in [20, 30, 40, 50]:
            for pattern in [f"{speed}kmph.mp4", f"{speed}kph.mp4", f"{speed}_kmph.mp4"]:
                matches = list(current_dir.glob(pattern))
                if matches:
                    test_videos[str(matches[0])] = speed
                    print(f"Found: {matches[0].name} ({speed} km/h)")

        if not test_videos:
            print("\n✗ No test videos found with standard naming (20kmph.mp4, etc.)")
            print("Please use --videos option to specify files manually")
            return

    elif args.videos:
        # Parse from command line arguments
        for video_path in args.videos:
            # Extract speed from filename
            filename = Path(video_path).stem.lower()
            for speed in [20, 30, 40, 50]:
                if f"{speed}km" in filename or f"{speed}_km" in filename:
                    test_videos[video_path] = speed
                    break
            else:
                print(f"⚠️ Could not determine speed for: {video_path}")
                # Try to prompt user
                try:
                    speed = float(input(f"Enter ground truth speed for {video_path} (km/h): "))
                    test_videos[video_path] = speed
                except:
                    print(f"Skipping {video_path}")

    else:
        print("Error: Please specify --videos or --auto")
        parser.print_help()
        return

    if not test_videos:
        print("\n✗ No test videos to process")
        return

    # Run validation
    validator = SpeedValidator(model_path=args.model)
    results_df = validator.run_validation(test_videos)

    if results_df is not None:
        print("\n" + "=" * 70)
        print("✓ Validation Complete!")
        print("=" * 70)
        print("\nNext step: Run generate_validation_report.py to create full report")


if __name__ == "__main__":
    main()
