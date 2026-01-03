import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import contextlib
import io

from pipeline.inference_pipeline import GeometryInferencePipeline, DepthInferencePipeline
from utils.kitti_utils import load_label, get_all_data, load_calib
from utils.evaluation import (
    match_predictions_to_gt,
    calculate_errors,
    get_metrics,
    plot_error_distributions,
    plot_metric_by_depth_range
)
from config import load_config

NB_IMAGES = 50
IOU_THRESHOLD = 0.5  # Fixed IoU for matching/evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D object detection on KITTI")
    parser.add_argument("--config", type=str, default='config/default.yaml', help="Path to config file")
    parser.add_argument("--conf_threshold", type=float, default=0.3, help="Confidence threshold for detections")
    parser.add_argument("--iou_threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--use_geometry", action="store_true", help="Use geometry-based inference pipeline (default False)")
    args = parser.parse_args()

    print("Loading configuration...")
    try:
        config = load_config(args.config)
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Updated to support new flat key 'kitti_path' with fallback
    kitti_base_dir = config.get('kitti_path') or config.get('paths', {}).get('kitti_base')
    if not kitti_base_dir or not os.path.exists(kitti_base_dir):
        print(f"Error: KITTI base directory not found: {kitti_base_dir}")
        return

    all_data = get_all_data(kitti_base_dir)[:NB_IMAGES]
    print(f"Processing {len(all_data)} images...")

    # Initialize pipeline
    try:
        if args.use_geometry:
            pipeline = GeometryInferencePipeline(config_path=args.config)
            print("✓ Using Geometry-based Inference Pipeline")
        else:
            pipeline = DepthInferencePipeline(config_path=args.config)
            print("✓ Using Depth-based Inference Pipeline")
    except Exception as e:
        print(f"Error initializing inference pipeline: {e}")
        return

    all_matches = []
    all_processing_times = []

    for i, (image_path, label_path, calib_path) in enumerate(tqdm(all_data, desc="Evaluating")):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            gt_objects = load_label(label_path)
            if not gt_objects:
                # Skip images without ground truth
                continue

            calib_data = load_calib(calib_path)
            proj_matrix = calib_data['P2']

            start_time = time.time()
            # Suppress pipeline print output
            with contextlib.redirect_stdout(io.StringIO()):
                if args.use_geometry:
                    boxes_3d, _ = pipeline.infer_3d_boxes(
                        image=image,
                        proj_matrix=proj_matrix,
                        conf_threshold=args.conf_threshold,
                        iou_threshold=args.iou_threshold
                    )
                else:
                    boxes_3d, _, _ = pipeline.infer_3d_boxes(
                        image=image,
                        proj_matrix=proj_matrix,
                        conf_threshold=args.conf_threshold,
                        iou_threshold=args.iou_threshold
                    )
            all_processing_times.append(time.time() - start_time)

            matches, _ = match_predictions_to_gt(boxes_3d, gt_objects, iou_threshold=IOU_THRESHOLD)
            all_matches.extend(matches)

        except Exception as e:
            print(f"Error processing image {i+1}: {e}")

    if not all_matches:
        print("No matches found during evaluation. Exiting.")
        return

    print("\nCalculating errors and metrics...")
    errors = calculate_errors(all_matches)
    metrics = get_metrics(boxes_3d, gt_objects, iou_threshold=0.5)

    print("\n=== Evaluation Summary ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    avg_time = np.mean(all_processing_times) if all_processing_times else 0
    print(f"Average processing time per image: {avg_time:.3f} seconds")

    # Plot error distributions and metrics by depth
    plot_error_distributions(errors)
    plot_metric_by_depth_range(errors, metric='z', stat='rmse', depth_interval=5.0)


if __name__ == "__main__":
    main()