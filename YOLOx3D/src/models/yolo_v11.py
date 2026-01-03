import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOv11:
    """
    Object detection using YOLOv11 from Ultralytics
    """

    MODELS = {
        'nano': 'yolo11n',
        'small': 'yolo11s',
        'medium': 'yolo11m',
        'large': 'yolo11l',
        'extra': 'yolo11x'
    }

    def __init__(self, model_size='small', conf_thres=0.25, iou_thres=0.45, classes=None, device=None, weights_path=None):
        """
        Initialize the object detector
        
        Args:
            model_size (str): Model size ('nano', 'small', 'medium', 'large', 'extra')
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            classes (list): List of classes to detect (None for all classes)
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
            weights_path (str): Path to custom weights file (if None, uses pretrained model)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model with custom weights or pretrained
        if weights_path is not None:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Custom weights file not found: {weights_path}")
            
            print(f"Loading custom weights from: {weights_path}")
            self.model = YOLO(weights_path)
            print(f"✓ Loaded custom YOLOv11 model from {weights_path}")
        else:
            if model_size not in self.MODELS:
                raise ValueError(f"Invalid model_size '{model_size}'. Choose from: {list(self.MODELS.keys())}")
            
            model_name = self.MODELS[model_size]
            self.model = YOLO(model_name)
            print(f"✓ Loaded YOLOv11 {model_size} pretrained model")

        # Device is handled during inference in newer ultralytics versions
        print(f"Model will use device: {self.device}")
        
        # Set model parameters
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = 1000
        
        if classes is not None and len(classes) > 0:
            self.model.overrides['classes'] = classes
    
    def detect(self, image, track=True):
        """
        Detect objects in an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format as it's a numpy array)
            track (bool): Whether to use tracking or just detection
            
        Returns:
            tuple: (annotated_image, detections)
                - annotated_image (numpy.ndarray): Image with detections drawn
                - detections (list): List of detections [bbox, score, class_id, class_name, object_id]
        """
        detections = []
        
        if track:
            results = self.model.track(image, verbose=False, device=self.device, persist=True)
        else:
            results = self.model.predict(image, verbose=False, device=self.device)
        
        annotated_image = results[0].plot()
        
        for predictions in results: # Loop through batch of images (here just one image)
            if predictions is None or predictions.boxes is None:
                continue
            
            for bbox in predictions.boxes:
                score = float(bbox.conf)
                class_id = int(bbox.cls)
                class_name = self.model.names[class_id]
                xmin, ymin, xmax, ymax = bbox.xyxy[0].cpu().numpy().tolist()
                id = int(bbox.id) if (track and hasattr(bbox, 'id') and bbox.id is not None) else None

                detections.append([
                    [xmin, ymin, xmax, ymax],
                    score,
                    class_id,
                    class_name,
                    id
                ])
        
        return annotated_image, detections
    
    def get_class_names(self):
        """
        Get the names of the classes that the model can detect
        
        Returns:
            list: List of class names
        """
        return self.model.names