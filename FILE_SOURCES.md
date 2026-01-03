# File Sources and Origins

This document tracks the source and origin of all files and folders in the project.

---

## Folders

### `fahrradstrase-main/`

- **Source:** Shared by Professor/Research Guide
- **Original Location:** `/Users/navaneethmalingan/Downloads/germany/fahrradstrase-main.zip`
- **Purpose:** Camera calibration toolkit for generating homography transformation matrix
- **Modifications:**
  - `Beispieldaten/Max_Pla.txt` - Updated with 7 calibration points for Uni_west_1 location
  - `main.py` - Fixed array syntax bug, added validation
  - `utils/selectPoints.py` - Replaced tkinter popup with keyboard-based selection (Mac compatibility)

### `traffic_analyis_data/`

- **Source:** Shared by Professor/Research Guide
- **Original Location:** `/Users/navaneethmalingan/Downloads/germany/traffic_analyis_data.zip`
- **Purpose:** Original GoPro video footage for vehicle tracking
- **Contents:**
  - `Uni_west_1/` - **Has calibration points** (Max_Pla.txt) - use this for testing
  - Other video folders may not have calibration data

### `Dataset/`

- **Source:** Roboflow - "car bb - v5 2024-03-13"
- **Purpose:** Vehicle bounding box training dataset (900 images)
- **Format:** YOLOv8 detection format

### `Wheel_seg-6/`

- **Source:** Roboflow Universe - Wheel segmentation dataset
- **Purpose:** Wheel instance segmentation training (1,420 images)
- **Classes:** frontwheel, backwheel, middlewheel

### `approaches/`

- **Source:** Created during thesis work
- **Purpose:** Contains all 4 vehicle localization approaches
- **Contents:**
  - `bbox_pipeline/` - Bounding box bottom-center approach
  - `keypoint_pipeline/` - Wheel keypoint detection approach
  - `seg_pipeline/` - Wheel segmentation approach
  - `3dbb_pipeline/` - 3D bounding box with depth estimation

### `comparison/`

- **Source:** Created during thesis work
- **Purpose:** Tools for comparing approach results

### `utils/`

- **Source:** Created during thesis work
- **Purpose:** Shared utility functions

---

## Calibration Files

### `gopro_calibration_fisheye.npz`

- **Source:** Generated using `GoPro_fisheye_calibration.py`
- **Purpose:** GoPro fisheye lens distortion correction parameters (K, D, DIM)
- **Generated From:** Checkerboard calibration images

### `coordinate_mapping_2030.json`

- **Source:** Generated using `fahrradstrase-main` calibration toolkit
- **Purpose:** Homography matrix for pixel-to-world coordinate transformation
- **Calibration Points:** 7 points from `Max_Pla.txt` for Uni_west_1 location

---

## Reference Documents

### `Thesis__Oleg_Porokhniak_ ver4 (2) (1).pdf`

- **Source:** Previous thesis by Oleg Porokhniak
- **Original Location:** `/Users/navaneethmalingan/Downloads/germany/Thesis/`
- **Purpose:** Reference implementation and methodology

### `Ground Truth Data.docx`

- **Source:** Shared by Professor/Research Guide
- **Original Location:** `/Users/navaneethmalingan/Downloads/germany/`
- **Purpose:** Ground truth documentation for validation

---

## Model Files

### `approaches/*/models/best.pt`

- **Source:** Downloaded from Oleg's GitHub: <https://github.com/Olegja89/Thesis>
- **Purpose:** YOLOv8 Pose model with 10 keypoints per vehicle
- **Classes:** vehicle_2_wheels, vehicle_4_wheels, vehicle_6_wheels, vehicle_8_wheels, vehicle_10_wheels

### `approaches/seg_pipeline/models/wheel_seg_best.pt`

- **Source:** Trained on Wheel_seg-6 dataset
- **Purpose:** Wheel instance segmentation model

### `approaches/seg_pipeline/models/vehicle_best.pt`

- **Source:** Trained on Dataset (Roboflow car bb)
- **Purpose:** Vehicle detection model

---

## Core Utility Files

| File | Purpose | Origin |
| ---- | ------- | ------ |
| `preprocess.py` | Fisheye undistortion, frame resizing | Created for thesis |
| `coordinate_transformer.py` | Pixel to world coordinate conversion | Created for thesis |
| `speed_utils.py` | SpeedTracker class for velocity calculation | Created for thesis |
| `data_export.py` | CSVExporter class | Created for thesis |
| `visualization_utils.py` | Annotation drawing | Created for thesis |
| `GoPro_fisheye_calibration.py` | Generate camera calibration | Created for thesis |

---

## Original Untouched Sources

The original files from the professor are preserved at:

```text
/Users/navaneethmalingan/Downloads/germany/
├── fahrradstrase-main.zip      # Original calibration toolkit
├── traffic_analyis_data.zip    # Original video footage
├── Ground Truth Data.docx      # Ground truth documentation
├── Max_Pla.txt                 # Original calibration points (different location)
├── Thesis/                     # Oleg's thesis materials
└── ...
```
