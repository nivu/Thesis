import os
import cv2
import numpy as np
import torch

from models.yolo_v11 import YOLOv11
from models.depth_anything_v2 import DepthAnythingV2
from utils.bbox3d_estimators import BBox3DEstimatorGeometry, BBox3DEstimatorDepth
from config import load_config
from utils.depth_calibration import load_calibration, is_calibrated, calibrate_with_kitti_3D_objects_dataset
from utils.kitti_utils import load_calib
from utils.class_dimensions import load_class_dimensions, compute_class_dimensions_from_kitti, is_dimensions_loaded

def setup_device(config):
    config_device = config['device']
    if config_device == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = config_device
    print(f"Using device: {device}")
    return device

def setup_class_dimensions(config):
    # Adapted to new config structure: bbox_3d.class_dimensions_file and kitti_path
    bbox_3d_cfg = config.get('bbox_3d', {})
    class_dimensions_file = bbox_3d_cfg.get('class_dimensions_file')
    if not class_dimensions_file:
        # fallback to legacy location if present
        class_dimensions_file = config.get('models', {}).get('class_dimensions_file')
    if is_dimensions_loaded():
        print("✓ Class dimensions already loaded")
        return

    if class_dimensions_file and os.path.exists(class_dimensions_file):
        print(f"Loading class dimensions from: {class_dimensions_file}")
        if load_class_dimensions(class_dimensions_file):
            return

    print("Computing class dimensions from KITTI dataset...")
    kitti_base_dir = config.get('kitti_path') or config.get('paths', {}).get('kitti_base')
    if not kitti_base_dir or not os.path.exists(kitti_base_dir):
        raise ValueError(f"KITTI dataset not found at: {kitti_base_dir}. Cannot compute class dimensions.")

    compute_class_dimensions_from_kitti(
        kitti_base_dir=kitti_base_dir,
        save_file=class_dimensions_file,
        show_progress=True
    )
    print("✓ Class dimensions computed and saved")

def setup_calibration(config, device):
    # New structure: depth.calibration_file, depth.size, kitti_path
    depth_cfg = config.get('depth', {})
    depth_calibration_file = depth_cfg.get('calibration_file')
    if not depth_calibration_file:
        raise ValueError("depth.calibration_file missing in config")

    if not is_calibrated():
        if os.path.exists(depth_calibration_file):
            print(f"Loading existing calibration from: {depth_calibration_file}")
            load_calibration(depth_calibration_file)
        else:
            print("No existing calibration found, performing new calibration...")
            kitti_base_dir = config.get('kitti_path') or config.get('paths', {}).get('kitti_base')
            if os.path.exists(kitti_base_dir):
                max_images = config.get('calibration', {}).get('max_images', 500)
                calibrate_with_kitti_3D_objects_dataset(
                    kitti_base_dir,
                    depth_model_size=depth_cfg.get('size', 'small'),
                    device=device,
                    max_images=max_images,
                    show_progress=True,
                    save_file=depth_calibration_file
                )
            else:
                raise ValueError("Calibration impossible: KITTI base directory missing")

class GeometryInferencePipeline:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        self.device = setup_device(self.config)

        self.yolov11 = None
        self.bbox_3d_estimator = None

        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        self._initialize_models()
        setup_class_dimensions(self.config)

    def _initialize_models(self):
        try:
            yolo_cfg = self.config.get('yolo') or self.config.get('models', {}).get('yolo', {})
            self.yolov11 = YOLOv11(
                model_size=yolo_cfg.get('size', 'nano'),
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                classes=yolo_cfg.get('classes', []),
                device=self.device,
                weights_path=yolo_cfg.get('weights')
            )
            print("✓ Object detector initialized")
        except Exception as e:
            print(f"Error initializing YOLO: {e}")
            self.yolov11 = YOLOv11(
                model_size=yolo_cfg.get('size', 'nano'),
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                classes=yolo_cfg.get('classes', []),
                device='cpu',
                weights_path=yolo_cfg.get('weights')
            )
        bbox_3d_cfg = self.config.get('bbox_3d') or self.config.get('models', {}).get('bbox_3d', {})
        self.bbox_3d_estimator = BBox3DEstimatorGeometry(
            weights_path=bbox_3d_cfg.get('weights_path'),
            num_bins=bbox_3d_cfg.get('num_bins', 2)
        )

    def infer_3d_boxes(self, image, proj_matrix, conf_threshold=None, iou_threshold=None):

        if conf_threshold is not None:
            self.yolov11.conf_thres = conf_threshold
        if iou_threshold is not None:
            self.yolov11.iou_thres = iou_threshold

        detection_frame = image.copy()
        original_frame = image.copy()

        detection_frame, detections = self.yolov11.detect(detection_frame, track=False)
        print(f"Found {len(detections)} objects")

        boxes_3d = []
        for detection in detections:
            bbox, score, class_id, class_name, object_id = detection
            box_3d = self.bbox_3d_estimator.estimate_3d_box(
                image=original_frame,
                proj_matrix=proj_matrix,
                bbox_2d=bbox,
                class_name=class_name,
                object_id=object_id,
                score=score
            )
            boxes_3d.append(box_3d)

        print(f"Estimated {len(boxes_3d)} 3D boxes using geometry")
        return boxes_3d, detection_frame

class DepthInferencePipeline:
    def __init__(self, config_path="config/default.yaml"):
        self.config = load_config(config_path)
        self.device = setup_device(self.config)

        depth_cfg = self.config.get('depth', {})
        depth_calibration_file = depth_cfg.get('calibration_file')
        if not depth_calibration_file:
            raise ValueError("depth.calibration_file missing in config")

        if not os.path.exists(depth_calibration_file):
            kitti_base_dir = self.config.get('kitti_path') or self.config.get('paths', {}).get('kitti_base')
            if not kitti_base_dir or not os.path.exists(kitti_base_dir):
                raise ValueError("Missing kitti_path for depth calibration")
            max_images = self.config.get('calibration', {}).get('max_images', 500)
            calibrate_with_kitti_3D_objects_dataset(
                kitti_base_dir,
                depth_model_size=depth_cfg.get('size', 'small'),
                device=self.device,
                max_images=max_images,
                show_progress=True,
                save_file=depth_calibration_file
            )

        self.yolov11 = None
        self.depth_estimator = None
        self.bbox_3d_estimator = None

        self.conf_threshold = 0.25
        self.iou_threshold = 0.45

        self._initialize_models()
        setup_class_dimensions(self.config)

    def _initialize_models(self):
        yolo_cfg = self.config.get('yolo') or self.config.get('models', {}).get('yolo', {})
        try:
            self.yolov11 = YOLOv11(
                model_size=yolo_cfg.get('size', 'nano'),
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                classes=yolo_cfg.get('classes', []),
                device=self.device,
                weights_path=yolo_cfg.get('weights')
            )
            print("✓ Object detector initialized")
        except Exception as e:
            print(f"Error initializing YOLO: {e}")
            self.yolov11 = YOLOv11(
                model_size=yolo_cfg.get('size', 'nano'),
                conf_thres=self.conf_threshold,
                iou_thres=self.iou_threshold,
                classes=yolo_cfg.get('classes', []),
                device='cpu',
                weights_path=yolo_cfg.get('weights')
            )

        depth_cfg = self.config.get('depth', {})
        try:
            self.depth_estimator = DepthAnythingV2(
                model_size=depth_cfg.get('size', 'small'),
                device=self.device
            )
            print("✓ Depth estimator initialized")
        except Exception as e:
            print(f"Error initializing depth estimator: {e}")
            self.depth_estimator = DepthAnythingV2(
                model_size=depth_cfg.get('size', 'small'),
                device='cpu'
            )

        bbox_3d_cfg = self.config.get('bbox_3d') or self.config.get('models', {}).get('bbox_3d', {})
        self.bbox_3d_estimator = BBox3DEstimatorDepth(
            weights_path=bbox_3d_cfg.get('weights_path'),
            num_bins=bbox_3d_cfg.get('num_bins', 2)
        )

    def infer_3d_boxes(self, image, proj_matrix, conf_threshold=None, iou_threshold=None):
        if conf_threshold is not None:
            self.yolov11.conf_thres = conf_threshold
        if iou_threshold is not None:
            self.yolov11.iou_thres = iou_threshold

        setup_calibration(self.config, self.device)

        original_frame = image.copy()
        detection_frame = image.copy()

        detection_frame, detections = self.yolov11.detect(detection_frame, track=False)
        print(f"Found {len(detections)} objects")

        print("Estimating depth...")
        depth_map = self.depth_estimator.estimate_depth(original_frame)
        print("Depth estimation completed")

        boxes_3d = []
        for detection in detections:
            bbox, score, class_id, class_name, object_id = detection
            box_3d = self.bbox_3d_estimator.estimate_3d_box(
                image=original_frame,
                depth_map=depth_map,
                proj_matrix=proj_matrix,
                bbox_2d=bbox,
                class_name=class_name,
                object_id=object_id,
                score=score
            )
            boxes_3d.append(box_3d)

        print(f"Estimated {len(boxes_3d)} 3D boxes using depth")
        return boxes_3d, depth_map, detection_frame