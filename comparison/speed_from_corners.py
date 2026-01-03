"""
Speed Estimation using 3D Bounding Box Bottom Corners

This script calculates vehicle speed by tracking the movement of
the 4 bottom corners (ground contact points) of the 3D bounding box.
"""

import json
import numpy as np
import argparse
from collections import defaultdict


class CornerBasedSpeedEstimator:
    """
    Estimates vehicle speed using the bottom 4 corners of 3D bounding boxes.

    The bottom corners represent the vehicle's ground footprint:
    - Corner 0: Front-left
    - Corner 1: Front-right
    - Corner 2: Rear-right
    - Corner 3: Rear-left
    """

    def __init__(self, max_distance_threshold=3.0):
        self.vehicle_history = defaultdict(list)
        self.speeds = defaultdict(list)
        self.active_tracks = {}  # track_id -> last frame data
        self.next_track_id = 0
        self.max_distance_threshold = max_distance_threshold  # meters

    def calculate_centroid(self, corners):
        """Calculate centroid of the 4 bottom corners."""
        if not corners or len(corners) < 4:
            return None

        x_sum = sum(c['x'] for c in corners)
        y_sum = sum(c['y'] for c in corners)

        return {
            'x': x_sum / 4,
            'y': y_sum / 4
        }

    def calculate_corner_displacement(self, prev_corners, curr_corners):
        """
        Calculate average displacement of all 4 corners.
        This is more robust than using just the centroid.
        """
        if not prev_corners or not curr_corners:
            return None

        if len(prev_corners) < 4 or len(curr_corners) < 4:
            return None

        displacements = []
        for i in range(4):
            dx = curr_corners[i]['x'] - prev_corners[i]['x']
            dy = curr_corners[i]['y'] - prev_corners[i]['y']
            dist = np.sqrt(dx**2 + dy**2)
            displacements.append(dist)

        # Use median to reduce noise from corner matching issues
        return np.median(displacements)

    def match_to_existing_track(self, centroid, frame_num):
        """
        Match a detection to an existing track based on position proximity.

        Args:
            centroid: Current detection centroid
            frame_num: Current frame number

        Returns:
            track_id if matched, None otherwise
        """
        best_match = None
        best_distance = float('inf')

        for track_id, track_data in self.active_tracks.items():
            # Only match if track was seen recently (within 5 frames)
            if frame_num - track_data['frame'] > 5:
                continue

            prev_centroid = track_data['centroid']
            dx = centroid['x'] - prev_centroid['x']
            dy = centroid['y'] - prev_centroid['y']
            distance = np.sqrt(dx**2 + dy**2)

            if distance < best_distance and distance < self.max_distance_threshold:
                best_distance = distance
                best_match = track_id

        return best_match

    def process_frame(self, frame_num, detections, fps):
        """
        Process detections for a single frame.

        Args:
            frame_num: Current frame number
            detections: List of detections with bottom_corners
            fps: Frames per second

        Returns:
            List of speed estimates for each detection
        """
        frame_speeds = []
        matched_tracks = set()

        # Filter for vehicles only
        vehicle_detections = []
        for det in detections:
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
            if det.get('class') not in vehicle_classes:
                continue
            corners = det.get('bottom_corners', [])
            if len(corners) < 4:
                continue
            vehicle_detections.append(det)

        for det in vehicle_detections:
            corners = det.get('bottom_corners', [])
            centroid = self.calculate_centroid(corners)

            # Try to match to existing track
            track_id = self.match_to_existing_track(centroid, frame_num)

            if track_id is None:
                # Create new track
                track_id = f"vehicle_{self.next_track_id}"
                self.next_track_id += 1
            else:
                matched_tracks.add(track_id)

            # Store current frame data
            current_data = {
                'frame': frame_num,
                'corners': corners,
                'centroid': centroid,
                'depth': det.get('depth', 0),
                'class': det.get('class'),
                'confidence': det.get('confidence', 0)
            }

            # Calculate speed if we have previous data
            speed_kmh = 0
            if len(self.vehicle_history[track_id]) > 0:
                prev_data = self.vehicle_history[track_id][-1]

                # Calculate frame difference
                frame_diff = frame_num - prev_data['frame']

                if frame_diff > 0 and frame_diff < 10:  # Max 10 frame gap
                    # Corner-based displacement
                    displacement = self.calculate_corner_displacement(
                        prev_data['corners'],
                        corners
                    )

                    if displacement is not None:
                        # Convert to speed
                        time_diff = frame_diff / fps
                        speed_ms = displacement / time_diff
                        speed_kmh = speed_ms * 3.6

                        # Sanity check - cap at reasonable max
                        if speed_kmh > 200:
                            speed_kmh = 0  # Likely a detection issue

            current_data['speed_kmh'] = speed_kmh
            self.vehicle_history[track_id].append(current_data)
            self.active_tracks[track_id] = current_data

            if speed_kmh > 0:
                self.speeds[track_id].append(speed_kmh)

            frame_speeds.append({
                'track_id': track_id,
                'class': det.get('class'),
                'speed_kmh': speed_kmh,
                'depth': det.get('depth', 0),
                'corners': corners,
                'centroid': centroid
            })

        return frame_speeds

    def get_statistics(self):
        """Get speed statistics for all tracked vehicles."""
        stats = {}

        all_speeds = []
        for track_id, speeds in self.speeds.items():
            if speeds:
                track_stats = {
                    'count': len(speeds),
                    'mean': np.mean(speeds),
                    'std': np.std(speeds),
                    'min': np.min(speeds),
                    'max': np.max(speeds),
                    'median': np.median(speeds)
                }
                stats[track_id] = track_stats
                all_speeds.extend(speeds)

        # Overall statistics
        if all_speeds:
            stats['_overall'] = {
                'total_measurements': len(all_speeds),
                'vehicles_tracked': len(self.speeds),
                'mean_speed': np.mean(all_speeds),
                'std_speed': np.std(all_speeds),
                'median_speed': np.median(all_speeds),
                'min_speed': np.min(all_speeds),
                'max_speed': np.max(all_speeds)
            }

        return stats


def process_detection_file(json_path, output_path=None):
    """
    Process 3DBB detection results and calculate speeds.

    Args:
        json_path: Path to 3DBB results JSON
        output_path: Optional path to save speed results
    """
    print(f"Loading detections from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} frames")

    # Get FPS from first frame
    fps = data[0].get('fps', 25.0) if data else 25.0
    print(f"FPS: {fps}")

    estimator = CornerBasedSpeedEstimator()
    all_results = []

    for frame_data in data:
        frame_num = frame_data['frame']
        detections = frame_data.get('detections', [])

        frame_speeds = estimator.process_frame(frame_num, detections, fps)

        all_results.append({
            'frame': frame_num,
            'speeds': frame_speeds
        })

    # Get statistics
    stats = estimator.get_statistics()

    # Print summary
    print("\n" + "="*60)
    print("CORNER-BASED SPEED ESTIMATION RESULTS")
    print("="*60)

    if '_overall' in stats:
        overall = stats['_overall']
        print(f"\nOverall Statistics:")
        print(f"  Vehicles tracked: {overall['vehicles_tracked']}")
        print(f"  Total speed measurements: {overall['total_measurements']}")
        print(f"  Mean speed: {overall['mean_speed']:.2f} km/h")
        print(f"  Median speed: {overall['median_speed']:.2f} km/h")
        print(f"  Std deviation: {overall['std_speed']:.2f} km/h")
        print(f"  Speed range: {overall['min_speed']:.2f} - {overall['max_speed']:.2f} km/h")

    print("\nPer-Vehicle Statistics:")
    for track_id, track_stats in stats.items():
        if track_id == '_overall':
            continue
        print(f"  {track_id}: mean={track_stats['mean']:.1f}, median={track_stats['median']:.1f}, "
              f"range=[{track_stats['min']:.1f}-{track_stats['max']:.1f}] km/h ({track_stats['count']} samples)")

    # Save results
    if output_path:
        results = {
            'frame_speeds': all_results,
            'statistics': stats,
            'fps': fps
        }

        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        results = convert_numpy(results)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return stats, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate speed from 3D BBox corners")
    parser.add_argument('--input', type=str, required=True,
                       help='Path to 3DBB detection results JSON')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save speed results')
    args = parser.parse_args()

    process_detection_file(args.input, args.output)
