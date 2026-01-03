import numpy as np
from utils.kitti_utils import rotation_matrix_y

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def compute_location_with_geometry(dimension, proj_matrix, bbox_2d, rotation_y, alpha):
    #global orientation
    R = rotation_matrix_y(rotation_y)

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3], dtype=float)
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(proj_matrix, M)

            A[row, :] = M[index,:3] - bbox_2d[row] * M[2,:3]
            b[row] = bbox_2d[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X

def compute_location_with_depth(depth_map, box_2d, proj_matrix, class_name):
    """
    Compute 3D location using depth map and camera intrinsics
    
    Args:
        depth_map (numpy.ndarray): Depth map corresponding to the image
        box_2d (list): 2D bounding box [x1, y1, x2, y2]
        proj_matrix (numpy.ndarray): Camera projection matrix

    Returns:
        numpy.ndarray: 3D point (x, y, z) in camera coordinates
    """
    x1, y1, x2, y2 = box_2d
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Get depth value at center (this would need depth estimation logic)
    # For now, assuming depth is provided or estimated elsewhere
    depth = estimate_depth(depth_map, box_2d, class_name, center_x, center_y)

    return backproject_pixel_to_3d(proj_matrix, center_x, center_y, depth)

def backproject_pixel_to_3d(proj_matrix, x, y, depth):
    """
    Backproject a 2D pixel coordinate to 3D camera space
    
    Args:
        proj_matrix (numpy.ndarray): Camera projection matrix (3x4)
        x (float): X coordinate in image space
        y (float): Y coordinate in image space
        depth (float): Depth value in meters

    Returns:
        numpy.ndarray: 3D point (x, y, z) in camera coordinates
    """
    K = proj_matrix[:3, :3]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_real = (x - cx) * depth / fx
    y_real = (y - cy) * depth / fy

    return np.array([x_real, y_real, depth])

def estimate_depth(depth_map, bbox_2d, class_name, center_x, center_y):
    """
    Estimate depth for the object based on class-specific strategies
    
    Args:
        depth_map (numpy.ndarray): Depth map corresponding to the image
        bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
        class_name (str): Class name of the object
        center_x (float): Center X coordinate
        center_y (float): Center Y coordinate
        
    Returns:
        float: Estimated depth value
    """
    from utils.depth import get_depth_in_region
    from utils.depth_calibration import convert_depth_to_absolute
    
    x1, y1, x2, y2 = bbox_2d
    
    bbox_int = [int(x1), int(y1), int(x2), int(y2)]
    rev_depth = get_depth_in_region(depth_map, bbox_int, method='median')

    return convert_depth_to_absolute(rev_depth)