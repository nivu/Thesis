#This script handles image preprocessing for fisheye camera footage. 
#It includes functions to undistort images using previously computed calibration parameters, resize frames for recognition and display, and rescale coordinates between different resolutions. 
#The load_calibration_data function retrieves the camera matrix and distortion coefficients from a saved .npz file. 

import cv2
import numpy as np

def undistort(img, K, D, DIM, scale=0.6):
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:  # The scale is to resize the final undistorted image to zoom in
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def preprocess_frame(frame, K, D, DIM, recognition_size=(640, 640), display_size=None):
    # Step 1: Undistort the frame
    undistorted_frame = undistort(frame, K, D, DIM)
    
    # Step 2: Resize the frame to the recognition size
    recognition_frame = cv2.resize(undistorted_frame, recognition_size, interpolation=cv2.INTER_AREA)
    
    # Step 3: If display_size is provided and different from recognition_size, resize for display
    if display_size and display_size != recognition_size:
        display_frame = cv2.resize(undistorted_frame, display_size, interpolation=cv2.INTER_AREA)
    else:
        display_frame = recognition_frame
    
    return recognition_frame, display_frame

def rescale_coordinates(coords, from_size, to_size):
    fx = to_size[0] / from_size[0]
    fy = to_size[1] / from_size[1]
    return [coord * fx if i % 2 == 0 else coord * fy for i, coord in enumerate(coords)]

# Load calibration data
def load_calibration_data(file_path='gopro_calibration_fisheye.npz'):
    try:
        with np.load(file_path) as X:
            K, D, DIM = [X[i] for i in ('K', 'D', 'DIM')]
        print("Calibration data loaded successfully.")
        return K, D, DIM
    except FileNotFoundError:
        print(f"Error: Calibration file '{file_path}' not found.")
        return None, None, None
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None, None, None