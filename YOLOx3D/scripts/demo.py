import os
import time
import cv2
import matplotlib.pyplot as plt
import argparse

from pipeline.inference_pipeline import GeometryInferencePipeline, DepthInferencePipeline
from utils.depth import colorize_depth
from utils.bird_eye_view import plot_kitti_comparison_bev
from utils.kitti_utils import load_label, get_image_data, load_calib, get_random_image_data
from utils.visualization import draw_boxes_on_image, draw_comparison_boxes
from config import load_config

def main():
    parser = argparse.ArgumentParser(description="3D Object Detection Demo")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config yaml file')
    parser.add_argument('--image_index', type=str, default='random', help='KITTI image index or "random"')
    parser.add_argument('--use_geometry', action='store_true', help='Use geometry inference pipeline (default False)')
    parser.add_argument('--comparison', action='store_true', help='Enable ground truth comparison visualization')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--iou_threshold', type=float, default=0.45, help='IoU threshold for NMS')
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

    if args.image_index == "random":
        input_image_path, input_label_path, input_calib_path = get_random_image_data(kitti_base_dir)
    else:
        input_image_path, input_label_path, input_calib_path = get_image_data(kitti_base_dir, args.image_index)

    for pth in [input_image_path, input_calib_path]:
        if not os.path.exists(pth):
            print(f"Error: Required file not found: {pth}")
            return

    calib_data = load_calib(input_calib_path)
    proj_matrix = calib_data['P2']

    frame = cv2.imread(input_image_path)
    if frame is None:
        print(f"Error: Could not load image {input_image_path}")
        return
    height, width = frame.shape[:2]
    print(f"Image dimensions: {width}x{height}")

    gt_objects = []
    if os.path.exists(input_label_path):
        gt_objects = load_label(input_label_path)
        print(f"✓ Loaded {len(gt_objects)} ground truth objects")
    else:
        print(f"Warning: Ground truth file not found: {input_label_path}")
        print("Proceeding without ground truth comparison.")

    print("\nInitializing inference pipeline...")
    try:
        pipeline = GeometryInferencePipeline(config_path=args.config) if args.use_geometry else DepthInferencePipeline(config_path=args.config)
        print("✓ Inference pipeline initialized")
    except Exception as e:
        print(f"Error initializing inference pipeline: {e}")
        return

    print(f"\n{'='*50}")
    print("PROCESSING IMAGE")
    print(f"{'='*50}")
    start_time = time.time()

    if args.use_geometry:
        boxes_3d, detection_frame = pipeline.infer_3d_boxes(
            image=frame,
            proj_matrix=proj_matrix,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )
        depth_map = None
    else:
        boxes_3d, depth_map, detection_frame = pipeline.infer_3d_boxes(
            image=frame,
            proj_matrix=proj_matrix,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold
        )

    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f}s")

    print(f"\n{'='*50}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*50}")

    depth_colored = colorize_depth(depth_map) if depth_map is not None else None

    # Draw 3D boxes on image
    try:
        if args.comparison and gt_objects:
            image_with_comparison = draw_comparison_boxes(
                frame, boxes_3d, gt_objects, proj_matrix,
                pred_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=2
            )
        else:
            image_with_comparison = draw_boxes_on_image(
                frame, boxes_3d, proj_matrix,
                color=(0, 255, 0), thickness=2,
                draw_center=True, draw_labels=True
            )
        print("✓ 3D bounding box visualization created")
    except Exception as e:
        print(f"Warning: Could not create 3D box visualization: {e}")
        image_with_comparison = frame

    # Bird's eye view visualization
    try:
        plot_kitti_comparison_bev(
            boxes_3d,
            gt_objects,
            fx=proj_matrix[0, 0],
            image_shape=(height, width),
            title=f"Ground Truth vs Predictions - {len(boxes_3d)} Predicted, {len(gt_objects)} GT"
        )
        print("✓ Bird's eye view comparison displayed")
    except Exception as e:
        print(f"Warning: Could not create bird's eye view comparison: {e}")

    print("\nDisplaying results (close windows to exit)...")

    plt.figure(figsize=(15, 10))

    if args.use_geometry:
        gs = plt.GridSpec(2, 2)

        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Original Image - {args.image_index}')
        ax1.axis('off')

        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB))
        ax2.set_title('2D Detection')
        ax2.axis('off')

        ax3 = plt.subplot(gs[1, :])
        ax3.imshow(cv2.cvtColor(image_with_comparison, cv2.COLOR_BGR2RGB))
        ax3.set_title(f'3D Boxes - {args.image_index}')
        ax3.axis('off')

    else:
        gs = plt.GridSpec(2, 2)

        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Original Image - {args.image_index}')
        ax1.axis('off')

        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(image_with_comparison, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'3D Boxes (Prediction vs GT) - {args.image_index}')
        ax2.axis('off')

        if depth_colored is not None:
            ax3 = plt.subplot(gs[1, 0])
            ax3.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB))
            ax3.set_title('Depth Map')
            ax3.axis('off')

            ax4 = plt.subplot(gs[1, 1])
            ax4.imshow(cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB))
            ax4.set_title('2D Detection')
            ax4.axis('off')
        else:
            ax3 = plt.subplot(gs[1, 0])
            ax3.imshow(cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB))
            ax3.set_title('2D Detection')
            ax3.axis('off')
            plt.delaxes(plt.subplot(gs[1,1]))  # remove empty subplot

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()