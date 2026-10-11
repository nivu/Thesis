"""
Video inference script for YOLOx3D 3D object detection
Processes a video file and outputs a video with 3D bounding boxes
"""
import os
import sys
import cv2
import json
import numpy as np
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.inference_pipeline import GeometryInferencePipeline, DepthInferencePipeline
from utils.visualization import draw_boxes_on_image
from utils.depth import colorize_depth
from config import load_config


def create_projection_matrix(width, height, fov_degrees=70):
    """
    Create an approximate projection matrix for a camera without calibration.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        fov_degrees: Horizontal field of view in degrees (default 70 for GoPro)

    Returns:
        3x4 projection matrix
    """
    # Calculate focal length from FOV
    fov_rad = np.radians(fov_degrees)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels

    # Principal point at image center
    cx = width / 2
    cy = height / 2

    # Create 3x4 projection matrix (assuming no rotation/translation)
    proj_matrix = np.array([
        [fx, 0,  cx, 0],
        [0,  fy, cy, 0],
        [0,  0,  1,  0]
    ], dtype=np.float32)

    return proj_matrix


def process_video(input_path, output_path, config_path, use_geometry=True,
                  conf_threshold=0.25, iou_threshold=0.45, fov=70,
                  max_frames=None, skip_frames=1, show_depth=False,
                  save_json=True):
    """
    Process a video file and output with 3D bounding boxes.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        config_path: Path to config yaml
        use_geometry: Use geometry-based method (True) or depth-based (False)
        conf_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold
        fov: Camera field of view in degrees
        max_frames: Maximum frames to process (None for all)
        skip_frames: Process every Nth frame (1 = all frames)
        show_depth: Include depth map in output (only for depth-based method)
        save_json: Save 3D bounding box data to JSON file
    """
    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input video: {width}x{height} @ {fps} FPS, {total_frames} frames")

    # Create projection matrix
    proj_matrix = create_projection_matrix(width, height, fov)
    print(f"Created projection matrix with FOV={fov} degrees")
    print(f"Focal length: fx={proj_matrix[0,0]:.1f}, fy={proj_matrix[1,1]:.1f}")

    # Initialize pipeline
    print("\nInitializing inference pipeline...")
    try:
        if use_geometry:
            pipeline = GeometryInferencePipeline(config_path=config_path)
            print("Using GEOMETRY-based 3D estimation")
        else:
            pipeline = DepthInferencePipeline(config_path=config_path)
            print("Using DEPTH-based 3D estimation")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        cap.release()
        return

    # Calculate output dimensions
    if show_depth and not use_geometry:
        out_width = width * 2
        out_height = height
    else:
        out_width = width
        out_height = height

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps / skip_frames, (out_width, out_height))

    if not out.isOpened():
        print(f"Error: Could not create output video {output_path}")
        cap.release()
        return

    # Determine frames to process
    frames_to_process = min(total_frames, max_frames) if max_frames else total_frames

    print(f"\nProcessing {frames_to_process} frames (skip={skip_frames})...")

    frame_idx = 0
    processed = 0

    # Store all frame data for JSON export
    all_frames_data = []

    pbar = tqdm(total=frames_to_process // skip_frames, desc="Processing")

    while cap.isOpened() and (max_frames is None or processed < max_frames // skip_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if needed
        if frame_idx % skip_frames != 0:
            frame_idx += 1
            continue

        try:
            # Run inference
            if use_geometry:
                boxes_3d, detection_frame = pipeline.infer_3d_boxes(
                    image=frame,
                    proj_matrix=proj_matrix,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                depth_map = None
            else:
                boxes_3d, depth_map, detection_frame = pipeline.infer_3d_boxes(
                    image=frame,
                    proj_matrix=proj_matrix,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )

            # Store frame data for JSON
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'objects': []
            }
            for box in boxes_3d:
                obj_data = {
                    'class_name': box.get('class_name', 'unknown'),
                    'score': float(box.get('score', 0.0)),
                    'center_3d': box['center_3d'].tolist() if hasattr(box['center_3d'], 'tolist') else list(box['center_3d']),
                    'dimensions': box['dimensions'].tolist() if hasattr(box['dimensions'], 'tolist') else list(box['dimensions']),
                    'yaw': float(box.get('yaw', 0.0)),
                    'bbox_2d': box['bbox_2d'].tolist() if hasattr(box['bbox_2d'], 'tolist') else list(box['bbox_2d']),
                    'object_id': box.get('object_id')
                }
                frame_data['objects'].append(obj_data)
            all_frames_data.append(frame_data)

            # Draw 3D boxes on frame
            result_frame = draw_boxes_on_image(
                frame, boxes_3d, proj_matrix,
                color=(0, 255, 0), thickness=2,
                draw_center=True, draw_labels=True
            )

            # Add frame info
            info_text = f"Frame: {frame_idx} | Objects: {len(boxes_3d)}"
            cv2.putText(result_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Combine with depth if requested
            if show_depth and depth_map is not None:
                depth_colored = colorize_depth(depth_map)
                depth_colored = cv2.resize(depth_colored, (width, height))
                output_frame = np.hstack([result_frame, depth_colored])
            else:
                output_frame = result_frame

            out.write(output_frame)
            processed += 1
            pbar.update(1)

        except Exception as e:
            print(f"\nError processing frame {frame_idx}: {e}")
            # Write original frame on error
            out.write(frame)
            processed += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()
    out.release()

    # Save JSON data
    if save_json:
        json_path = os.path.splitext(output_path)[0] + '_data.json'
        json_output = {
            'video_info': {
                'input_path': input_path,
                'width': width,
                'height': height,
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed,
                'skip_frames': skip_frames,
                'fov': fov,
                'method': 'geometry' if use_geometry else 'depth'
            },
            'projection_matrix': proj_matrix.tolist(),
            'frames': all_frames_data
        }
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON data saved to: {json_path}")

    print(f"\nDone! Output saved to: {output_path}")
    print(f"Processed {processed} frames")


def main():
    parser = argparse.ArgumentParser(description="Video 3D Object Detection with YOLOx3D")
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Path to output video file (default: input_3dbb.mp4)')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to config yaml file')
    parser.add_argument('--use_geometry', action='store_true',
                       help='Use geometry-based method (default: depth-based)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Detection confidence threshold')
    parser.add_argument('--iou_threshold', type=float, default=0.45,
                       help='NMS IoU threshold')
    parser.add_argument('--fov', type=float, default=70,
                       help='Camera horizontal FOV in degrees (default: 70 for GoPro)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--show_depth', action='store_true',
                       help='Include depth map in output (side by side)')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_3dbb.mp4"

    print("=" * 60)
    print("YOLOx3D Video Inference")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {'Geometry' if args.use_geometry else 'Depth'}-based")
    print(f"FOV:    {args.fov} degrees")
    print("=" * 60)

    process_video(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
        use_geometry=args.use_geometry,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        fov=args.fov,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        show_depth=args.show_depth
    )


if __name__ == "__main__":
    main()
