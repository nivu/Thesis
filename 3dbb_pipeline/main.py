"""
3D Bounding Box Pipeline - Main Entry Point

Uses YOLOx3D for 3D vehicle detection and fuses with 2D homography-based estimates.
"""

import os
import sys
import cv2
import numpy as np
import json

# Add YOLOx3D to path
YOLOX3D_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "YOLOx3D")
sys.path.insert(0, os.path.join(YOLOX3D_PATH, "src"))

from pipeline.inference_pipeline import GeometryInferencePipeline, DepthInferencePipeline
from config import load_config as load_yolox3d_config

# Import local config
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PIPELINE_DIR)
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_config", os.path.join(PIPELINE_DIR, "config.py"))
pipeline_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_config)

# Import coordinate transformer from utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
try:
    from utils.coordinate_transformer import CoordinateTransformer
except ImportError:
    # Create a placeholder if not available
    class CoordinateTransformer:
        def __init__(self, mapping_data):
            self.mapping_data = mapping_data
        def pixel_to_world(self, x, y):
            return (0, 0)


class Pipeline3DBB:
    """
    3D Bounding Box Pipeline for vehicle localization.

    Combines YOLOx3D 3D detection with homography-based 2D estimation.
    """

    def __init__(self, use_geometry=True, use_depth=False):
        """
        Initialize the pipeline.

        Args:
            use_geometry: Use geometry-based 3D estimation
            use_depth: Use depth-based 3D estimation (requires depth model)
        """
        self.use_geometry = use_geometry
        self.use_depth = use_depth

        # Load YOLOx3D config
        yolox3d_config_path = os.path.join(YOLOX3D_PATH, "config", "default.yaml")

        # Initialize 3D detection pipeline
        print("Initializing 3D detection pipeline...")
        if use_geometry:
            self.detector_3d = GeometryInferencePipeline(config_path=yolox3d_config_path)
        elif use_depth:
            self.detector_3d = DepthInferencePipeline(config_path=yolox3d_config_path)
        else:
            raise ValueError("Must enable either use_geometry or use_depth")

        # Initialize coordinate transformer for 2D homography
        print("Initializing coordinate transformer...")
        self.transformer = None
        if os.path.exists(pipeline_config.MAPPING_FILE):
            with open(pipeline_config.MAPPING_FILE, 'r') as f:
                mapping_data = json.load(f)
            self.transformer = CoordinateTransformer(mapping_data)
            print(f"Loaded coordinate mapping from {pipeline_config.MAPPING_FILE}")
        else:
            print(f"Warning: Coordinate mapping file not found: {pipeline_config.MAPPING_FILE}")

        # Load camera calibration for projection matrix
        self.proj_matrix = None
        if os.path.exists(pipeline_config.CALIBRATION_FILE):
            calib_data = np.load(pipeline_config.CALIBRATION_FILE)
            # Create projection matrix from camera matrix
            # The file uses 'K' for camera matrix (fisheye calibration)
            K = calib_data.get('K', calib_data.get('camera_matrix', None))
            if K is not None:
                self.proj_matrix = np.hstack([K, np.zeros((3, 1))])  # 3x4 projection matrix
                print(f"Loaded camera calibration from {pipeline_config.CALIBRATION_FILE}")
            else:
                print(f"Warning: No camera matrix found in {pipeline_config.CALIBRATION_FILE}")
        else:
            print(f"Warning: Calibration file not found: {pipeline_config.CALIBRATION_FILE}")
            # Use default KITTI-like projection matrix
            self.proj_matrix = np.array([
                [721.5377, 0, 609.5593, 0],
                [0, 721.5377, 172.854, 0],
                [0, 0, 1, 0]
            ])

        print("Pipeline initialized successfully!")

    def _rotation_matrix_y(self, theta):
        """Create rotation matrix around Y-axis."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def get_3d_box_corners(self, center, dimensions, yaw):
        """
        Get all 8 corners of a 3D bounding box.

        Args:
            center: [x, y, z] center of the box
            dimensions: [h, w, l] height, width, length
            yaw: rotation around Y axis in radians

        Returns:
            8x3 array of corner coordinates
            Corner order: 0-3 are bottom face, 4-7 are top face
        """
        h, w, l = dimensions
        x, y, z = center

        # Corners in local frame (y=0 is bottom, y=-h is top in camera coords)
        # Bottom face corners (ground contact points)
        x_corners = [l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
        y_corners = [0,    0,    0,    0,   -h,   -h,   -h,   -h]
        z_corners = [w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

        corners = np.vstack([x_corners, y_corners, z_corners])

        # Apply rotation and translation
        R = self._rotation_matrix_y(yaw)
        rotated = R @ corners
        rotated[0, :] += x
        rotated[1, :] += y
        rotated[2, :] += z

        return rotated.T  # 8x3 array

    def get_bottom_corners(self, center, dimensions, yaw):
        """
        Get the 4 bottom corners of a 3D bounding box (ground contact points).

        Args:
            center: [x, y, z] center of the box
            dimensions: [h, w, l] height, width, length
            yaw: rotation around Y axis in radians

        Returns:
            4x3 array of bottom corner coordinates in camera frame
            Order: front-left, front-right, rear-right, rear-left
        """
        all_corners = self.get_3d_box_corners(center, dimensions, yaw)
        # Bottom corners are indices 0-3
        return all_corners[:4]

    def process_frame(self, frame, conf_threshold=0.25, iou_threshold=0.45):
        """
        Process a single frame and detect 3D bounding boxes.

        Args:
            frame: Input image (BGR format)
            conf_threshold: Detection confidence threshold
            iou_threshold: NMS IoU threshold

        Returns:
            dict with:
                - boxes_3d: List of 3D bounding box detections
                - detection_frame: Frame with 2D detections drawn
                - depth_map: Depth map (if using depth-based method)
        """
        result = {
            'boxes_3d': [],
            'detection_frame': frame.copy(),
            'depth_map': None
        }

        if self.use_geometry:
            boxes_3d, detection_frame = self.detector_3d.infer_3d_boxes(
                image=frame,
                proj_matrix=self.proj_matrix,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            result['boxes_3d'] = boxes_3d
            result['detection_frame'] = detection_frame

        elif self.use_depth:
            boxes_3d, depth_map, detection_frame = self.detector_3d.infer_3d_boxes(
                image=frame,
                proj_matrix=self.proj_matrix,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            result['boxes_3d'] = boxes_3d
            result['detection_frame'] = detection_frame
            result['depth_map'] = depth_map

        return result

    def get_world_coordinates(self, boxes_3d):
        """
        Convert 3D boxes to world coordinates including bottom 4 corners.

        Args:
            boxes_3d: List of 3D bounding box detections from YOLOx3D

        Returns:
            List of dicts with world coordinates and bottom corners for each detection
        """
        world_coords = []

        for box in boxes_3d:
            center_3d = box.get('center_3d', np.array([0, 0, 0]))
            dimensions = box.get('dimensions', np.array([0, 0, 0]))
            yaw = box.get('yaw', 0)

            # Extract world position (x, z are ground plane in camera frame)
            # In camera coordinates: x=right, y=down, z=forward
            world_x = center_3d[0]  # lateral position
            world_y = center_3d[2]  # distance forward (depth)

            # Get the 4 bottom corners of the 3D bounding box
            # These represent the ground contact points of the vehicle
            bottom_corners_3d = self.get_bottom_corners(center_3d, dimensions, yaw)

            # Convert bottom corners to ground plane coordinates (x, z)
            # In camera frame: x=lateral, y=vertical (down), z=forward
            # For ground plane, we use (x, z) as the 2D coordinates
            bottom_corners_ground = []
            for corner in bottom_corners_3d:
                # corner is [x, y, z] in camera frame
                # Ground plane uses x (lateral) and z (forward/depth)
                bottom_corners_ground.append({
                    'x': float(corner[0]),  # lateral position (camera x)
                    'y': float(corner[2]),  # forward position (camera z = depth)
                    'height': float(corner[1])  # vertical position (should be ~0 for ground)
                })

            world_coords.append({
                'track_id': box.get('object_id'),
                'class_name': box.get('class_name', 'unknown'),
                'confidence': box.get('score', 0),
                'world_x': world_x,
                'world_y': world_y,
                'depth': center_3d[2],
                'dimensions': {
                    'height': float(dimensions[0]) if len(dimensions) > 0 else 0,
                    'width': float(dimensions[1]) if len(dimensions) > 1 else 0,
                    'length': float(dimensions[2]) if len(dimensions) > 2 else 0,
                },
                'yaw': float(yaw),
                'center_3d': {
                    'x': float(center_3d[0]),
                    'y': float(center_3d[1]),
                    'z': float(center_3d[2])
                },
                # The 4 bottom corners representing vehicle ground footprint
                # Order: front-left, front-right, rear-right, rear-left
                'bottom_corners': bottom_corners_ground
            })

        return world_coords

    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process a video file and save results.

        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            max_frames: Maximum frames to process (optional)

        Returns:
            List of results per frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames")

        # Setup video writer if output path specified
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        all_results = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            # Process frame
            result = self.process_frame(frame)
            world_coords = self.get_world_coordinates(result['boxes_3d'])

            all_results.append({
                'frame': frame_count,
                'detections': world_coords
            })

            # Write annotated frame
            if writer:
                writer.write(result['detection_frame'])

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        cap.release()
        if writer:
            writer.release()

        print(f"Finished processing {frame_count} frames")
        return all_results


def main():
    """Main entry point for testing the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="3D Bounding Box Pipeline")
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output file')
    parser.add_argument('--use-depth', action='store_true', help='Use depth-based method')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--max-frames', type=int, help='Max frames to process')
    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline3DBB(
        use_geometry=not args.use_depth,
        use_depth=args.use_depth
    )

    if args.video:
        results = pipeline.process_video(
            args.video,
            output_path=args.output,
            max_frames=args.max_frames
        )

        # Save results to CSV
        output_csv = args.output.replace('.mp4', '_results.csv') if args.output else 'results.csv'
        with open(output_csv, 'w') as f:
            f.write('frame,track_id,class,confidence,world_x,world_y,depth,length,width,height,yaw\n')
            for frame_result in results:
                frame_num = frame_result['frame']
                for det in frame_result['detections']:
                    f.write(f"{frame_num},{det.get('track_id', '')},{det['class_name']},"
                            f"{det['confidence']:.3f},{det['world_x']:.3f},{det['world_y']:.3f},"
                            f"{det['depth']:.3f},{det['dimensions']['length']:.3f},"
                            f"{det['dimensions']['width']:.3f},{det['dimensions']['height']:.3f},"
                            f"{det['yaw']:.3f}\n")
        print(f"Results saved to {output_csv}")

    elif args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Error: Could not load image {args.image}")
            return

        result = pipeline.process_frame(frame, conf_threshold=args.conf)
        world_coords = pipeline.get_world_coordinates(result['boxes_3d'])

        print(f"\nDetected {len(world_coords)} vehicles:")
        for i, det in enumerate(world_coords):
            print(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f})")
            print(f"      Position: ({det['world_x']:.2f}, {det['world_y']:.2f}) m")
            print(f"      Depth: {det['depth']:.2f} m")
            print(f"      Dimensions: {det['dimensions']['length']:.2f} x {det['dimensions']['width']:.2f} x {det['dimensions']['height']:.2f} m")

        if args.output:
            cv2.imwrite(args.output, result['detection_frame'])
            print(f"\nAnnotated image saved to {args.output}")
    else:
        print("Please specify --video or --image")
        parser.print_help()


if __name__ == "__main__":
    main()
