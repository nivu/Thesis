"""
Script for regressor model generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from data.dataset_multibin import format_img
from utils.class_dimensions import get_class_dimensions
from utils.angles import get_angle_from_bins
from utils.angles import calc_theta_ray

def orientationLoss(orient_batch, orientGT_batch, confGT_batch):
    """
    Orientation loss function
    """
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]

    # extract important bin
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]

    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])

    return -1 * torch.cos(theta_diff - estimated_theta_diff).mean()

class ResNet18(nn.Module):
    def __init__(self, model=None, bins=2, w=0.4):
        super(ResNet18, self).__init__()
        self.bins = bins
        self.w = w
        
        # Handle different model input types
        if isinstance(model, str):
            if model == 'resnet18':
                backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                raise ValueError(f"Unsupported model string: {model}")
        elif model is None:
            backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            backbone = model
            
        self.model = nn.Sequential(*(list(backbone.children())[:-2]))

        # orientation head, for orientation estimation
        self.orientation = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins*2) # 4 bins
        )

        # confident head, for orientation estimation
        self.confidence = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, bins) # 2 bins   
        )

        # dimension head
        self.dimension = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 3) # x, y, z
        )

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512 * 7 * 7)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)
        
        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence, dimension

def estimate_orientation_and_dimensions(regressor, image, bbox_2d, class_name, proj_matrix, num_bins):
    """
    Estimate orientation and dimensions using the ResNet18 model
    
    Args:
        image (numpy.ndarray): Input image
        bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
        class_name (str): Class name of the object
        proj_matrix (numpy.ndarray): Camera projection matrix
        num_bins (int): Number of orientation bins
        
    Returns:
        tuple: rotation_y, dimensions, alpha
    """
    fx = proj_matrix[0, 0]
    theta_ray = calc_theta_ray(image.shape[1], bbox_2d, fx)

    # Prepare input for the model
    input_img = format_img(image, bbox_2d)

    device = next(regressor.parameters()).device
    input_tensor = input_img.unsqueeze(0).to(device)

    # Predict orientation, confidence, and dimension adjustments
    with torch.no_grad():
        orient, conf, dim = regressor(input_tensor)
        orient = orient.cpu().data.numpy()[0, :, :]
        conf = conf.cpu().data.numpy()[0, :]
        dim = dim.cpu().data.numpy()[0, :]

    dimensions = dim.copy()
    dimensions += get_class_dimensions(class_name.lower())

    # Calculate orientation
    alpha = get_angle_from_bins(conf, orient, num_bins)

    rotation_y = theta_ray + alpha

    return rotation_y, dimensions, alpha