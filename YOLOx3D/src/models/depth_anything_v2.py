import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

MODEL_MAP = {
    'small': 'depth-anything/Depth-Anything-V2-Small-hf',
    'base': 'depth-anything/Depth-Anything-V2-Base-hf',
    'large': 'depth-anything/Depth-Anything-V2-Large-hf'
}

class DepthAnythingV2:
    """
    Depth estimation using Depth Anything v2
    """

    MODELS = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf'
    }

    def __init__(self, model_size='small', device=None):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        
        if model_size not in self.MODELS:
            raise ValueError(f"Invalid model_size '{model_size}'. Choose from: {list(self.MODELS.keys())}")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        model_name = self.MODELS[model_size]
        
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
            print(f"Loaded Depth Anything v2 {model_size} model on {self.device}")
        except Exception as e:
            print(f"Error loading model on {self.device}: {e}")
            print("Falling back to CPU for depth estimation (Depth Anything v2)")
            self.device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
            print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image_rgb)
        
        depth_result = self.pipe(pil_image)
        depth_map = depth_result["depth"]
        
        # Convert PIL Image to numpy array if needed
        if isinstance(depth_map, Image.Image):
            depth_map = np.array(depth_map)
        elif isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        return depth_map