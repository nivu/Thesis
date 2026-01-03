"""
Comprehensive Comparison Script: All 4 Vehicle Localization Approaches

Compares:
1. BBox Pipeline - Bottom-center of bounding box
2. Keypoint Pipeline - Wheel keypoint detection
3. Segmentation Pipeline - Wheel segmentation masks
4. 3D BBox Pipeline - Depth estimation + 3D boxes

Usage:
    python compare_all_approaches.py
"""

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

# Paths to output files
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APPROACHES_DIR = os.path.join(PROJECT_ROOT, "approaches")

OUTPUT_FILES = {
    'bbox': os.path.join(APPROACHES_DIR, "bbox_pipeline/output/bbox_world_coordinates.csv"),
    'keypoint': os.path.join(APPROACHES_DIR, "keypoint_pipeline/output/keypoint_world_coordinates.csv"),
    'segmentation': os.path.join(APPROACHES_DIR, "seg_pipeline/output/segmentation_results.csv"),
    '3dbb': os.path.join(APPROACHES_DIR, "3dbb_pipeline/output/3dbb_world_coordinates.csv")
}

# Calibration bounds (filter unrealistic coordinates)
WORLD_X_BOUNDS = (-50, 50)  # meters
WORLD_Y_BOUNDS = (-20, 20)  # meters
MAX_SPEED = 150  # km/h - filter outliers


def load_results():
    """Load results from all 4 pipelines."""
    results = {}

    for approach, filepath in OUTPUT_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            # Standardize column names
            if 'id' in df.columns:
                df = df.rename(columns={'id': 'track_id'})

            # Filter invalid coordinates
            if 'world_x' in df.columns and 'world_y' in df.columns:
                df = df[
                    (df['world_x'] >= WORLD_X_BOUNDS[0]) & (df['world_x'] <= WORLD_X_BOUNDS[1]) &
                    (df['world_y'] >= WORLD_Y_BOUNDS[0]) & (df['world_y'] <= WORLD_Y_BOUNDS[1])
                ]

            # Filter invalid speeds
            if 'speed_kmh' in df.columns:
                df = df[df['speed_kmh'] <= MAX_SPEED]

            results[approach] = df
            print(f"Loaded {approach}: {len(df)} valid records")
        else:
            print(f"Warning: {approach} output not found at {filepath}")

    return results


def compare_positions(results):
    """Compare position estimates across approaches."""
    print("\n" + "=" * 70)
    print("POSITION COMPARISON")
    print("=" * 70)

    approaches = list(results.keys())
    comparisons = {}

    for i, approach1 in enumerate(approaches):
        for approach2 in approaches[i+1:]:
            df1 = results[approach1]
            df2 = results[approach2]

            # Merge on frame and track_id
            merged = pd.merge(
                df1[['frame', 'track_id', 'world_x', 'world_y']],
                df2[['frame', 'track_id', 'world_x', 'world_y']],
                on=['frame', 'track_id'],
                suffixes=('_1', '_2')
            )

            if len(merged) > 0:
                # Calculate Euclidean distance
                merged['pos_diff'] = np.sqrt(
                    (merged['world_x_1'] - merged['world_x_2'])**2 +
                    (merged['world_y_1'] - merged['world_y_2'])**2
                )

                mean_diff = merged['pos_diff'].mean()
                std_diff = merged['pos_diff'].std()
                max_diff = merged['pos_diff'].max()

                key = f"{approach1} vs {approach2}"
                comparisons[key] = {
                    'n_common': len(merged),
                    'mean_diff_m': mean_diff,
                    'std_diff_m': std_diff,
                    'max_diff_m': max_diff
                }

                print(f"\n{key}:")
                print(f"  Common detections: {len(merged)}")
                print(f"  Mean position diff: {mean_diff:.4f} m")
                print(f"  Std position diff:  {std_diff:.4f} m")
                print(f"  Max position diff:  {max_diff:.4f} m")

    return comparisons


def compare_speeds(results):
    """Compare speed estimates across approaches."""
    print("\n" + "=" * 70)
    print("SPEED COMPARISON")
    print("=" * 70)

    approaches = list(results.keys())
    comparisons = {}

    for i, approach1 in enumerate(approaches):
        for approach2 in approaches[i+1:]:
            df1 = results[approach1]
            df2 = results[approach2]

            # Merge on frame and track_id
            merged = pd.merge(
                df1[['frame', 'track_id', 'speed_kmh']],
                df2[['frame', 'track_id', 'speed_kmh']],
                on=['frame', 'track_id'],
                suffixes=('_1', '_2')
            )

            # Filter only non-zero speeds for meaningful comparison
            merged = merged[(merged['speed_kmh_1'] > 0) & (merged['speed_kmh_2'] > 0)]

            if len(merged) > 0:
                merged['speed_diff'] = abs(merged['speed_kmh_1'] - merged['speed_kmh_2'])

                mean_diff = merged['speed_diff'].mean()
                std_diff = merged['speed_diff'].std()

                key = f"{approach1} vs {approach2}"
                comparisons[key] = {
                    'n_common': len(merged),
                    'mean_diff_kmh': mean_diff,
                    'std_diff_kmh': std_diff
                }

                print(f"\n{key}:")
                print(f"  Common speed readings: {len(merged)}")
                print(f"  Mean speed diff: {mean_diff:.2f} km/h")
                print(f"  Std speed diff:  {std_diff:.2f} km/h")

    return comparisons


def analyze_per_vehicle(results):
    """Analyze results per vehicle across all approaches."""
    print("\n" + "=" * 70)
    print("PER-VEHICLE ANALYSIS")
    print("=" * 70)

    # Find vehicles present in all approaches
    all_track_ids = None
    for approach, df in results.items():
        track_ids = set(df['track_id'].unique())
        if all_track_ids is None:
            all_track_ids = track_ids
        else:
            all_track_ids = all_track_ids & track_ids

    print(f"\nVehicles detected by all 4 approaches: {len(all_track_ids)}")

    vehicle_stats = []

    for track_id in sorted(all_track_ids):
        stats = {'track_id': track_id}

        for approach, df in results.items():
            vehicle_data = df[df['track_id'] == track_id]
            speeds = vehicle_data[vehicle_data['speed_kmh'] > 0]['speed_kmh']

            if len(speeds) > 0:
                stats[f'{approach}_avg_speed'] = speeds.mean()
                stats[f'{approach}_n_frames'] = len(vehicle_data)

        vehicle_stats.append(stats)

    # Show top 5 vehicles with most data
    vehicle_df = pd.DataFrame(vehicle_stats)

    if 'bbox_n_frames' in vehicle_df.columns:
        vehicle_df = vehicle_df.sort_values('bbox_n_frames', ascending=False)

    print("\nTop 5 vehicles with most detections:")
    print("-" * 60)

    for _, row in vehicle_df.head(5).iterrows():
        print(f"\nVehicle ID {int(row['track_id'])}:")
        for approach in results.keys():
            speed_col = f'{approach}_avg_speed'
            frames_col = f'{approach}_n_frames'
            if speed_col in row and not pd.isna(row[speed_col]):
                print(f"  {approach:15s}: {row[speed_col]:.2f} km/h ({int(row[frames_col])} frames)")

    return vehicle_stats


def analyze_method_usage(results):
    """Analyze which localization method was used in each pipeline."""
    print("\n" + "=" * 70)
    print("METHOD USAGE ANALYSIS")
    print("=" * 70)

    for approach, df in results.items():
        print(f"\n{approach.upper()}:")

        if 'method' in df.columns:
            method_counts = df['method'].value_counts()
            total = len(df)
            for method, count in method_counts.items():
                pct = count / total * 100
                print(f"  {method}: {count} ({pct:.1f}%)")
        else:
            print(f"  Total records: {len(df)}")


def generate_summary(results, pos_comparisons, speed_comparisons):
    """Generate final summary and recommendations."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Detection counts
    print("\n1. Detection Counts:")
    for approach, df in results.items():
        n_records = len(df)
        n_vehicles = df['track_id'].nunique()
        print(f"   {approach:15s}: {n_records} records, {n_vehicles} unique vehicles")

    # Position accuracy ranking
    print("\n2. Position Consistency (smaller is better):")
    if pos_comparisons:
        # Calculate average deviation from bbox (baseline)
        for comp, stats in pos_comparisons.items():
            print(f"   {comp}: {stats['mean_diff_m']:.4f} m avg difference")

    # Speed accuracy ranking
    print("\n3. Speed Consistency (smaller is better):")
    if speed_comparisons:
        for comp, stats in speed_comparisons.items():
            print(f"   {comp}: {stats['mean_diff_kmh']:.2f} km/h avg difference")

    # Recommendations
    print("\n4. Key Observations:")

    # Compare keypoint to bbox
    kp_bbox_key = "bbox vs keypoint"
    if kp_bbox_key in pos_comparisons:
        diff = pos_comparisons[kp_bbox_key]['mean_diff_m']
        if diff < 0.5:
            print(f"   - Keypoint and BBox give similar results ({diff:.3f}m avg diff)")
        else:
            print(f"   - Keypoint differs from BBox by {diff:.3f}m on average")

    # Check if segmentation adds value
    if 'segmentation' in results:
        seg_df = results['segmentation']
        if 'method' in seg_df.columns:
            seg_used = (seg_df['method'] == 'wheel_seg').sum()
            total = len(seg_df)
            seg_rate = seg_used / total * 100 if total > 0 else 0
            print(f"   - Wheel segmentation successfully used in {seg_rate:.1f}% of cases")

    # Check 3D pipeline
    if '3dbb' in results:
        df_3d = results['3dbb']
        if 'method' in df_3d.columns:
            fused_count = (df_3d['method'] == 'fused').sum()
            print(f"   - 3D+2D fusion used in {fused_count} detections")

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("COMPREHENSIVE APPROACH COMPARISON")
    print("=" * 70)

    # Load all results
    results = load_results()

    if len(results) < 2:
        print("Error: Need at least 2 approaches to compare")
        return

    # Compare positions
    pos_comparisons = compare_positions(results)

    # Compare speeds
    speed_comparisons = compare_speeds(results)

    # Per-vehicle analysis
    vehicle_stats = analyze_per_vehicle(results)

    # Method usage
    analyze_method_usage(results)

    # Summary
    generate_summary(results, pos_comparisons, speed_comparisons)

    # Save detailed report
    report = {
        'position_comparisons': pos_comparisons,
        'speed_comparisons': speed_comparisons,
        'detection_counts': {
            approach: {
                'records': len(df),
                'vehicles': int(df['track_id'].nunique())
            }
            for approach, df in results.items()
        }
    }

    report_path = os.path.join(PROJECT_ROOT, "comparison/full_comparison_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
