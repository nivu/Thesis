"""
Visualization utilities for 3D bounding boxes and plotting
"""
import cv2
import numpy as np
from .kitti_utils import rotation_matrix_y, project_3d_box_corners_to_2d, project_3d_to_2d

def draw_box_3d(image, corners, color=(0, 255, 0), thickness=2):
    """
    Draw enhanced 3D bounding box on image with better depth perception
    
    Args:
        image (numpy.ndarray): Image to draw on
        corners (numpy.ndarray): 8x2 array of corner coordinates
        color (tuple): Line color (BGR format)
        thickness (int): Line thickness
    """
    # Create overlay for transparency effects
    overlay = image.copy()
    
    # Define edges for wireframe
    edges = [
        (0,1), (1,2), (2,3), (3,0),  # bottom face
        (4,5), (5,6), (6,7), (7,4),  # top face
        (0,4), (1,5), (2,6), (3,7)   # vertical edges
    ]
    
    # Draw wireframe edges
    for i, j in edges:
        pt1 = tuple(map(int, corners[i]))
        pt2 = tuple(map(int, corners[j]))
        cv2.line(image, pt1, pt2, color, thickness)
    
    # Fill only the front face with semi-transparent color for 3D effect
    pts_front = np.array([corners[0], corners[1], corners[5], corners[4]], np.int32)
    pts_front = pts_front.reshape((-1, 1, 2))
    cv2.fillPoly(overlay, [pts_front], color)
    
    # Apply transparency
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def draw_boxes_on_image(image, objects_3d, proj_matrix, color=(0, 255, 0), thickness=2, 
                        draw_center=True, draw_labels=True):
    """
    Draw multiple enhanced 3D bounding boxes on image
    
    Args:
        image (numpy.ndarray): Image to draw on (will be modified)
        objects_3d (list): List of 3D object dictionaries from estimate_3d_box
        proj_matrix (numpy.ndarray): 3x3 camera projection matrix
        color (tuple): Line color (BGR format)
        thickness (int): Line thickness
        draw_center (bool): Whether to draw center points
        draw_labels (bool): Whether to draw class labels and scores
        
    Returns:
        numpy.ndarray: Modified image with 3D boxes drawn
    """
    result_image = image.copy()
    
    for i, obj in enumerate(objects_3d):
        try:
            # Compute 3D box corners
            corners_2d = project_3d_box_corners_to_2d(obj, proj_matrix)
            
            # Draw enhanced 3D box
            draw_box_3d(result_image, corners_2d, color, thickness)
            
            # Draw center point with enhanced style
            if draw_center:
                center_2d = project_3d_to_2d(obj['center_3d'].reshape(3, 1), proj_matrix)[0]
                cv2.circle(result_image, tuple(center_2d), thickness * 2, color, -1)
            
            # Draw enhanced labels
            if draw_labels:
                # Find top-left corner for label position
                min_x = int(min(corners_2d[:, 0]))
                min_y = int(min(corners_2d[:, 1]))
                label_pos = (min_x, min_y - 10)
                
                # Create enhanced label text
                class_name = obj.get('class_name', 'unknown')
                score = obj.get('score', 0.0)
                depth = obj['center_3d'][2]
                obj_id = obj.get('object_id', None)
                
                # Draw multiple lines of text
                text_y = label_pos[1]
                
                # Class name
                cv2.putText(result_image, class_name, (label_pos[0], text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                text_y -= 15
                
        except Exception as e:
            print(f"Warning: Could not draw 3D box for object {i}: {e}")
            continue
    
    return result_image

def draw_comparison_boxes(image, predicted_objects, ground_truth_objects, proj_matrix, 
                         pred_color=(0, 255, 0), gt_color=(0, 0, 255), thickness=2):
    """
    Draw both predicted and ground truth 3D boxes for comparison
    
    Args:
        image (numpy.ndarray): Image to draw on
        predicted_objects (list): List of predicted 3D objects
        ground_truth_objects (list): List of ground truth objects (KITTI format)
        proj_matrix (numpy.ndarray): 3x3 camera projection matrix
        pred_color (tuple): Color for predicted boxes (BGR)
        gt_color (tuple): Color for ground truth boxes (BGR)
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with both predicted and GT boxes drawn
    """
    result_image = image.copy()
    
    # Draw ground truth boxes
    if ground_truth_objects:
        result_image = draw_boxes_on_image(
            result_image, ground_truth_objects, proj_matrix, 
            color=gt_color, thickness=thickness, draw_labels=True
        )
    
    # Draw predicted boxes
    if predicted_objects:
        result_image = draw_boxes_on_image(
            result_image, predicted_objects, proj_matrix, 
            color=pred_color, thickness=thickness, draw_labels=True
        )
    
    # Add legend
    legend_y = 30
    cv2.putText(result_image, "Green: Predicted", (10, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
    cv2.putText(result_image, "Red: Ground Truth", (10, legend_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, gt_color, 2)
    
    return result_image