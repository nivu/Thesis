import numpy as np
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from models.orientation_dimension_resnet18 import ResNet18, estimate_orientation_and_dimensions
from utils.location_estimators import compute_location_with_geometry, compute_location_with_depth

class BBox3DEstimatorGeometry:
    """BBox3D estimator using geometric location estimation"""

    def __init__(self, weights_path='/home/nimdaba/Documents/test-repo-YOLO3D/YOLO3D/weights/resnet18.pkl', num_bins=2):
        self.weights_path = weights_path
        self.num_bins = num_bins
        self._init_model()

    def _init_model(self):
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.regressor = ResNet18(model=backbone, bins=self.num_bins)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.regressor = self.regressor.to(device)

        try:
            checkpoint = torch.load(self.weights_path, map_location=device, weights_only=False)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.regressor.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                # Handle ultralytics-style checkpoint
                model_state = checkpoint['model']
                if hasattr(model_state, 'state_dict'):
                    self.regressor.load_state_dict(model_state.state_dict())
                else:
                    self.regressor.load_state_dict(model_state)
            else:
                self.regressor.load_state_dict(checkpoint)
            print(f"✓ Loaded weights from {self.weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {self.weights_path}: {e}")
            print("Using randomly initialized weights")

        self.regressor.eval()

    def estimate_3d_box(self, image, proj_matrix, bbox_2d, class_name, object_id, score):
        rotation_y, dimensions, alpha = estimate_orientation_and_dimensions(
            self.regressor, image, bbox_2d, class_name, proj_matrix, self.num_bins
        )
        location, _ = compute_location_with_geometry(
            dimensions, proj_matrix, bbox_2d, rotation_y, alpha
        )
        center_3d = np.array(location)

        box_3d = {
            'bbox_2d': bbox_2d,
            'center_3d': center_3d,
            'dimensions': dimensions,
            'yaw': rotation_y,
            'class_name': class_name,
            'object_id': object_id,
            'score': score
        }
        return box_3d


class BBox3DEstimatorDepth:
    """BBox3D estimator using depth map based location estimation"""

    def __init__(self, weights_path='/home/nimdaba/Documents/test-repo-YOLO3D/YOLO3D/weights/resnet18.pkl', num_bins=2):
        self.weights_path = weights_path
        self.num_bins = num_bins
        self._init_model()

    def _init_model(self):
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.regressor = ResNet18(model=backbone, bins=self.num_bins)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.regressor = self.regressor.to(device)

        try:
            checkpoint = torch.load(self.weights_path, map_location=device, weights_only=False)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.regressor.load_state_dict(checkpoint['model_state_dict'])
            elif 'model' in checkpoint:
                # Handle ultralytics-style checkpoint
                model_state = checkpoint['model']
                if hasattr(model_state, 'state_dict'):
                    self.regressor.load_state_dict(model_state.state_dict())
                else:
                    self.regressor.load_state_dict(model_state)
            else:
                self.regressor.load_state_dict(checkpoint)
            print(f"✓ Loaded weights from {self.weights_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {self.weights_path}: {e}")
            print("Using randomly initialized weights")

        self.regressor.eval()

    def estimate_3d_box(self, image, depth_map, proj_matrix, bbox_2d, class_name, object_id, score):
        rotation_y, dimensions, _ = estimate_orientation_and_dimensions(
            self.regressor, image, bbox_2d, class_name, proj_matrix, self.num_bins
        )
        center_3d = compute_location_with_depth(
            depth_map, bbox_2d, proj_matrix, class_name
        )

        box_3d = {
            'bbox_2d': bbox_2d,
            'center_3d': center_3d,
            'dimensions': dimensions,
            'yaw': rotation_y,
            'class_name': class_name,
            'object_id': object_id,
            'score': score
        }
        return box_3d