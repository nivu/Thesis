# Vehicle Speed & Dimension Estimation Demo

## Overview

This demo demonstrates a **wheel-based approach** for vehicle speed and dimension estimation using YOLOv8 instance segmentation. While the system is functional, **quantitative validation requires ground truth data** which is currently unavailable.

## What You Have ✓

### 1. **Trained Models**
- **Vehicle Detection Model**: `runs/detect/roboflow_train/weights/best.pt`
  - YOLOv8n detection
  - 3 classes: bus, cars, truck
  - mAP@50: 47.3% on test set

- **Wheel Segmentation Model**: `runs/segment/wheel_seg/weights/best.pt` (training in progress)
  - YOLOv8n-seg instance segmentation
  - 3 classes: frontwheel, backwheel, middlewheel
  - 1,420 training images

### 2. **Test Videos**
- **Reconstructed from Dataset**: `reconstructed_videos/`
  - GOPR0593.mp4 (128 frames)
  - GOPR0594.mp4 (3 frames)
  - GOPR0595.mp4 (105 frames)
  - GOPR0596.mp4 (300 frames)
  - GOPR0597.mp4 (364 frames)
  - all_videos_combined.mp4 (900 frames total)

### 3. **Pipeline Code**
- Camera calibration (fisheye correction)
- Homography transformation (pixel → real-world)
- Object tracking (multi-object tracking)
- Speed calculation (displacement over time)
- Wheel-based dimension estimation

---

## What's Missing ✗

### Critical for Validation

1. **Ground Truth Speed Data**
   - Videos with vehicles at known speeds
   - GPS or radar speed measurements
   - Synchronized with video timestamps

2. **Camera Calibration Parameters**
   - Intrinsic matrix (focal length, principal point)
   - Distortion coefficients
   - Camera height and angle

3. **Scene Reference Measurements**
   - Known distances in the scene (e.g., lane width = 3.5m)
   - Homography reference points with real-world coordinates
   - Scale validation objects

4. **Original Training Dataset**
   - The model `best.pt` was trained on: `D:/Thesis/training_4/YOLO/`
   - Contains keypoint annotations (10 keypoints per vehicle)
   - Not available in current repository

---

## Demo Scripts

### 1. Quick Test (Run First)

```bash
python quick_test_wheels.py
```

**What it does:**
- Tests wheel detection on 5 random dataset images
- Tests on a frame from reconstructed video
- Generates `test_output_*.jpg` images
- Verifies model works correctly

**Expected output:**
```
Quick Wheel Detection Test
==================================================================
Loading model: runs/segment/wheel_seg/weights/best.pt
✓ Model loaded!

Testing on 5 random images from Wheel_seg-6/test/images
----------------------------------------------------------------------

1. Processing: bus_1.jpg
   ✓ Detected 3 wheels
      - frontwheel: confidence 0.87
      - backwheel: confidence 0.82
      - backwheel: confidence 0.79
   Saved: test_output_bus_1.jpg
```

### 2. Full Pipeline Demo

```bash
python demo_wheel_pipeline.py
```

**What it does:**
- Detects and segments wheels in video frames
- Groups wheels into vehicles
- Calculates wheelbase and track width
- Estimates speed via wheel tracking
- Generates annotated output videos
- Exports CSV data with measurements

**Features:**
- **Wheel Detection**: Instance segmentation masks for each wheel
- **Vehicle Grouping**: Clusters nearby wheels into vehicles
- **Dimension Calculation**: Wheelbase (front-to-rear) and track width (left-to-right)
- **Speed Tracking**: Frame-to-frame displacement with smoothing
- **Visualization**: Color-coded wheel masks, bounding boxes, metrics overlay

**Output files:**
```
demo_outputs/
├── demo_GOPR0596.mp4           # Annotated video
├── demo_GOPR0596_data.csv      # Vehicle measurements
├── demo_GOPR0597.mp4
└── demo_GOPR0597_data.csv
```

**CSV Format:**
```csv
frame,vehicle_id,num_wheels,wheelbase_m,track_width_m,center_x,center_y,speed_kmh
1,0,4,2.45,1.52,850.3,420.1,32.5
2,0,4,2.48,1.54,865.2,425.3,33.1
```

---

## Understanding the Results

### ⚠️ Important Limitations

1. **Speed Estimates are NOT VALIDATED**
   - No ground truth comparison available
   - Accuracy unknown without calibration data
   - For **demonstration purposes only**

2. **Dimension Estimates are APPROXIMATE**
   - Based on assumed wheel diameter (0.65m)
   - Homography transformation not calibrated
   - Pixel-to-meter conversion is estimated

3. **Suitable Uses:**
   - ✓ Demonstrating the pipeline works
   - ✓ Showing wheel detection capability
   - ✓ Visualizing tracking and grouping
   - ✗ Claiming quantitative accuracy
   - ✗ Publishing as validated results

### What You CAN Say in Your Thesis

✅ **Acceptable Claims:**
- "Implemented a wheel-based vehicle detection and tracking system"
- "Developed pipeline for speed and dimension estimation"
- "Demonstrated wheel segmentation on 1,420 annotated images"
- "Achieved [X]% mAP on wheel detection test set"
- "System produces speed estimates pending ground truth validation"

❌ **Avoid Without Ground Truth:**
- "Achieved X% accuracy in speed estimation"
- "System accurately measures vehicle dimensions"
- "Validated against real-world speeds"
- Specific accuracy percentages or error metrics

---

## Next Steps for Full Validation

### Option 1: Contact Original Developer

Request the following from the thesis developer:

1. **Original Dataset**: `D:/Thesis/training_4/YOLO/`
   - Contains keypoint annotations
   - Matches the trained `best.pt` model

2. **Test Videos**: Videos with ground truth data
   - Known vehicle speeds
   - Camera calibration file
   - Reference measurements

3. **Validation Methodology**:
   - How they validated speed accuracy
   - Expected error margins
   - Calibration procedure

### Option 2: Create Your Own Validation Dataset

1. **Record Test Videos**:
   - Film vehicles at known speeds (use speedometer)
   - Record vehicle make/model for dimension lookup
   - Use same camera as original (GoPro with fisheye)

2. **Calibrate System**:
   - Measure reference distances (lane width, parking space)
   - Perform camera calibration with checkerboard
   - Create homography mapping with known points

3. **Validate**:
   - Run pipeline on test videos
   - Compare estimated vs actual speeds
   - Calculate error metrics (MAE, RMSE, percentage error)
   - Report results with confidence intervals

### Option 3: Qualitative Demonstration Only

1. **Document the Pipeline**:
   - Show system architecture
   - Explain each component (detection → tracking → speed)
   - Demonstrate on sample videos

2. **Report Limitations**:
   - Clearly state lack of ground truth
   - Note calibration requirements
   - Suggest future validation work

3. **Focus on Technical Implementation**:
   - Model training methodology
   - Algorithm design choices
   - Code architecture and modularity

---

## File Structure

```
Thesis/
├── best.pt                          # Original pose model (5 classes, 10 keypoints)
├── runs/
│   ├── detect/roboflow_train/       # Vehicle detection model
│   │   └── weights/best.pt
│   └── segment/wheel_seg/           # Wheel segmentation model (training)
│       └── weights/best.pt
├── reconstructed_videos/            # Test videos from dataset
│   ├── GOPR0593.mp4
│   ├── GOPR0596.mp4
│   └── all_videos_combined.mp4
├── demo_outputs/                    # Generated demo results
│   ├── demo_GOPR0596.mp4
│   └── demo_GOPR0596_data.csv
├── Dataset/                         # Roboflow vehicle dataset
│   ├── train/
│   ├── valid/
│   └── test/
├── Wheel_seg-6/                     # Wheel segmentation dataset
│   ├── train/
│   ├── valid/
│   └── test/
│
├── demo_wheel_pipeline.py          # Main demo script ⭐
├── quick_test_wheels.py            # Quick test script ⭐
├── test_model_compatibility.py     # Model compatibility check
│
├── main.py                         # Original pipeline (keypoint-based)
├── car_tracking.py                 # Vehicle tracking utilities
├── coordinates_mapping.py          # Homography setup
├── speed_utils.py                  # Speed calculation
├── visualization_utils.py          # Drawing utilities
├── preprocess.py                   # Camera calibration
└── config.py                       # Configuration
```

---

## Technical Details

### Wheel-Based Approach (New)

**Advantages:**
- ✓ Wheels are on ground plane (simplifies homography)
- ✓ Standard wheel diameter provides scale reference
- ✓ More stable tracking than vehicle centroid
- ✓ Enables accurate dimension estimation
- ✓ Can infer vehicle type from wheelbase/track width

**Pipeline Steps:**
1. **Detection**: YOLOv8-seg segments wheel masks
2. **Center Extraction**: Calculate centroid of each mask
3. **Grouping**: Cluster nearby wheels into vehicles
4. **Dimensions**:
   - Wheelbase = distance(front_wheel, rear_wheel)
   - Track width = distance(left_wheel, right_wheel)
5. **Speed**: Track wheel centers across frames, calculate displacement

### Keypoint-Based Approach (Original)

**Model**: `best.pt`
- 5 vehicle classes (by wheel count)
- 10 keypoints per vehicle
- Likely: 4 wheel centers + 6 reference points

**Issue**: Incompatible with current datasets
- Roboflow dataset: 3 classes, bounding boxes only
- Wheel_seg dataset: 3 wheel classes, segmentation masks
- Missing: Original training dataset with keypoints

---

## FAQ

**Q: Can I use the speed estimates in my thesis?**

A: You can report that the system *produces* speed estimates, but cannot claim accuracy without validation. State clearly: "Speed estimates pending ground truth validation."

**Q: How accurate are the dimension measurements?**

A: Unknown without calibration. The system uses an assumed wheel diameter (65cm) and pixel-based scaling. Actual accuracy depends on camera angle, distance, and homography calibration.

**Q: Why can't I use the original `best.pt` model?**

A: It's a pose detection model expecting keypoint annotations. Your datasets have bounding boxes (Roboflow) or segmentation masks (Wheel_seg), not keypoints. Training new models was necessary.

**Q: What should I tell my thesis committee?**

A: Focus on:
1. Technical implementation (you built a working system)
2. Novel wheel-based approach (innovative method)
3. Limitations and future work (honest about validation needs)
4. Demonstrate qualitative results (videos with annotations)

**Q: Can I get ground truth data somehow?**

A: Options:
1. Contact original developer for test dataset
2. Record your own test videos with known speeds
3. Use public datasets with speed annotations (e.g., KITTI, nuScenes)

---

## Running the Demo

### After Training Completes (~2.7 hours remaining)

1. **Quick Test First**:
   ```bash
   python quick_test_wheels.py
   ```
   - Verify wheel detection works
   - Check `test_output_*.jpg` files

2. **Run Full Demo on One Video**:
   ```bash
   python demo_wheel_pipeline.py
   # Select option 1 (single video)
   # Choose GOPR0596 (largest video, 300 frames)
   ```

3. **Review Results**:
   - Watch `demo_outputs/demo_GOPR0596.mp4`
   - Check `demo_outputs/demo_GOPR0596_data.csv`
   - Verify wheel detection and tracking quality

4. **Process All Videos** (if satisfied):
   ```bash
   python demo_wheel_pipeline.py
   # Select option 2 (all videos)
   ```

---

## Contact for Ground Truth

**Email Template for Original Developer:**

```
Subject: Request for Validation Dataset - Vehicle Speed Estimation Thesis

Hi [Developer Name],

I am working with your vehicle speed estimation code and have successfully
trained new models on Roboflow datasets. However, I need ground truth data
to validate the speed estimation pipeline.

Could you please share:
1. The original training dataset (D:/Thesis/training_4/YOLO/)
2. Test videos with known vehicle speeds
3. Camera calibration file
4. Your validation methodology and expected accuracy

This would help me properly validate the system for my thesis.

Thank you!
Best regards,
[Your Name]
```

---

## Summary

✅ **You have**: Working pipeline, trained models, test videos
❌ **You need**: Ground truth data for validation
🎯 **You can**: Demonstrate the system works qualitatively
⚠️ **You cannot**: Claim quantitative accuracy without validation

**Recommendation**: Run the demo, include it in your thesis as a "proof of concept," and clearly document the need for ground truth validation as future work.

---

## Training Status

Check training progress:
```bash
# View training output
tail -f [training_log]

# Or check in Python:
from ultralytics import YOLO
model = YOLO("runs/segment/wheel_seg/weights/last.pt")
# Review metrics: runs/segment/wheel_seg/results.csv
```

Expected completion: ~2.7 hours from training start

---

*Generated for thesis demonstration purposes*
*Last updated: 2025*
