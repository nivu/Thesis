"""
Utilities for KITTI dataset loading and processing
"""
import os
import numpy as np

def load_calib(calib_file):
    """
    Load calibration matrices from KITTI calibration file
    :param calib_file: Path to calibration file
    :return: Dictionary containing calibration matrices
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            if ':' not in line:
                continue
            key, value = line.strip().split(':', 1)
            values = [float(x) for x in value.strip().split()]
            if key.startswith('P'):
                calib[key] = np.array(values).reshape(3, 4)
            else:
                calib[key] = np.array(values)  # leave as is for others
    return calib

def load_label(label_file):
    """
    Load KITTI label file and return objects formatted as box_3d dicts.
    :param label_file: Path to label file
    :return: List of box_3d dictionaries
    """
    objects = []
    if not os.path.isfile(label_file):
        return objects

    with open(label_file) as f:
        for idx, line in enumerate(f.readlines()):
            data = line.strip().split()

            class_name = data[0]

            if class_name == "DontCare" or class_name == "Misc":
                continue

            bbox_2d = np.array([float(x) for x in data[4:8]])  # (xmin, ymin, xmax, ymax)
            dimensions = np.array([float(x) for x in data[8:11]])  # h, w, l
            location = np.array([float(x) for x in data[11:14]])         # x, y, z
            rotation_y = float(data[14])
            alpha = float(data[3]) # orientation in camera coordinates

            center_3d = location.copy()
            center_3d[1] -= dimensions[0] / 2  # y -= height / 2

            box_3d = {
                'bbox_2d': bbox_2d,
                'center_3d': center_3d,
                'dimensions': dimensions,
                'yaw': rotation_y,
                'class_name': class_name,
                'score': 1.0,
                'alpha': alpha
            }   

            objects.append(box_3d)

    return objects

def get_kitti_directories(base_path):
    """
    Get standard KITTI dataset directory paths
    :param base_path: Base path to KITTI dataset
    :return: Dictionary containing paths to images, labels, and calibration files
    """
    return {
        'images': os.path.join(base_path, 'data_object_image_2', 'training', 'image_2'),
        'labels': os.path.join(base_path, 'data_object_label_2', 'training', 'label_2'),
        'calib': os.path.join(base_path, 'data_object_calib', 'training', 'calib')
    }

def get_image_data(base_path, idx):
    """
    Get file paths for image, label, and calibration files for a given index
    :param base_path: Base path to KITTI dataset
    :param idx: Image index (e.g., '000080')
    :return: Tuple of (image_path, label_path, calib_path)
    """
    paths = get_kitti_directories(base_path)
    image_path = os.path.join(paths['images'], f'{idx}.png')
    label_path = os.path.join(paths['labels'], f'{idx}.txt')
    calib_path = os.path.join(paths['calib'], f'{idx}.txt')
    return image_path, label_path, calib_path

def get_random_image_data(base_path):
    """
    Get random image, label, and calibration data from KITTI dataset
    :param base_path: Base path to KITTI dataset
    :return: Tuple of (image_path, label_path, calib_path)
    """
    dirs = get_kitti_directories(base_path)
    image_files = [f[:-4] for f in os.listdir(dirs['images']) if f.endswith('.png')]
    
    if not image_files:
        raise ValueError("No images found in the specified directory")
    
    idx = np.random.choice(image_files)
    return get_image_data(base_path, idx)

def get_all_data(base_path):
    """
    Get all image, label, and calibration data for the KITTI dataset
    :param base_path: Base path to KITTI dataset
    :return: List of tuples (image_path, label_path, calib_path)
    """
    dirs = get_kitti_directories(base_path)
    image_files = [f[:-4] for f in os.listdir(dirs['images']) if f.endswith('.png')]
    
    data = []
    for idx in image_files:
        image_path, label_path, calib_path = get_image_data(base_path, idx)
        data.append((image_path, label_path, calib_path))
    
    return data

def rotation_matrix_y(theta):
    """
    Create rotation matrix around Y-axis
    :param theta: Rotation angle in radians
    :return: 3x3 rotation matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def project_3d_to_2d(points_3d, proj_matrix):
    """
    Project 3D points to 2D image coordinates
    
    Args:
        points_3d (numpy.ndarray): 3xN array of 3D points
        proj_matrix (numpy.ndarray): 3x4 camera projection matrix
        
    Returns:
        numpy.ndarray: Nx2 array of 2D coordinates
    """
    # Convert to homogeneous coordinates if needed
    if points_3d.shape[0] == 3:
        points_3d_hom = np.vstack([points_3d, np.ones((1, points_3d.shape[1]))])
    else:
        points_3d_hom = points_3d
    
    # Project points using projection matrix
    points_2d_hom = np.dot(proj_matrix, points_3d_hom)
    
    # Convert from homogeneous coordinates
    points_2d = points_2d_hom[:2] / points_2d_hom[2]
    
    return points_2d.T.astype(np.int32)

def get_3d_box_corners(center, dimensions, yaw):
    """
    Retourne les 8 coins d'une box 3D orientée autour de Y
    center: [x, y, z]
    dimensions: [h, w, l]  (h=vertical, w=axe X latéral, l=axe Z longitudinal)
    yaw: rotation autour de Y en rad
    """
    h, w, l = dimensions
    x, y, z = center
    
    # Coins dans repère local
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [  0,    0,    0,    0,   -h,   -h,   -h,   -h ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners])
    
    # Rotation + translation
    R = rotation_matrix_y(yaw)
    rotated = R @ corners
    rotated[0, :] += x
    rotated[1, :] += y
    rotated[2, :] += z
    return rotated.T

def project_3d_box_corners_to_2d(obj, proj_matrix):
    """
    Compute 3D bounding box corners projected to 2D image coordinates
    
    Args:
        obj (dict): Object dictionary containing center_3d, dimensions, and yaw
        proj_matrix (numpy.ndarray): 3x4 camera projection matrix
        
    Returns:
        numpy.ndarray: 8x2 array of 2D corner coordinates
    """
    # Extract object properties
    center_3d = obj['center_3d']
    dimensions = obj['dimensions']  # [w, h, l]
    yaw = obj['yaw']
    
    h, w, l = dimensions
    x, y, z = center_3d
    
    # Define 3D corners in object coordinate system (centered at origin)
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
    y_corners = np.array([-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2])
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
    
    # Rotate corners by yaw angle
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_rotated = np.dot(rotation_matrix_y(yaw), corners_3d)
    
    # Translate to object center
    corners_world = corners_rotated + np.array([[x], [y], [z]])
    
    # Project to image coordinates using projection matrix
    corners_2d = project_3d_to_2d(corners_world, proj_matrix)
    
    return corners_2d