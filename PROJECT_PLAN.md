# Vehicle Real-World Localization Project Plan

**Student**: Shalikha
**Supervisor**: Julian

---

## 1. Project Objective

Develop an improved method for converting vehicle positions from **pixel coordinates** to **real-world coordinates** (bird's-eye view) using a single calibrated static camera. The system will enable accurate distance measurement between vehicles on the street.

### Key Innovation
Combine **bounding box detection** with **tire contact point detection** to achieve more precise real-world vehicle localization than existing approaches.

---

## 2. Problem Statement

### Current Limitation
- The calibrated street plane allows pixel-to-real-world conversion only for points **on the street surface**
- Pixels on the vehicle body cannot be directly converted to real-world coordinates
- Previous approaches using only bounding boxes are "not perfectly precise"

### Proposed Solution
Detect **tire-street contact points** (where wheels touch the road) — these points lie on the calibrated street plane and can be accurately converted to real-world coordinates.

---

## 3. Background Research

### 3.1 Literature Review
- Read and summarize Oleg's thesis thoroughly
- Document Oleg's bounding box method and its limitations
- Review homography transformation mathematics
- Study YOLOv8 keypoint detection architecture

### 3.2 Existing Codebase Analysis
- Understand current calibration pipeline
- Review coordinate transformation code
- Analyze existing wheel segmentation model
- Document the current speed estimation pipeline

### Deliverable: Technical Summary Document
A written document summarizing:
- Oleg's approach and its limitations
- The mathematical foundation (homography, projection)
- Opportunities for improvement

---

## 4. Technical Approach

### Phase 1: Tire Contact Point Detection

#### 4.1 Dataset Preparation
- Extract frames from existing traffic videos
- Annotate tire-street contact points (keypoints) on vehicles:
  - Front-left wheel contact point
  - Front-right wheel contact point
  - Rear-left wheel contact point
  - Rear-right wheel contact point
- Note: Typically only 2 points visible from one camera angle
- Split data: 70% training, 20% validation, 10% test

#### 4.2 Keypoint Detection Model
- Train YOLOv8-pose model for tire contact point detection
- Alternative: Modify existing wheel segmentation to output contact points
- Evaluate detection precision (pixel error from ground truth)

#### 4.3 Integration with Bounding Box Detection
- Use existing vehicle detection to locate cars
- Apply keypoint detection within/around detected bounding boxes
- Handle occlusion cases (when only 1-2 wheels visible)

### Phase 2: Real-World Coordinate Conversion

#### 4.4 Homography Application
- Apply existing calibration to detected contact points
- Convert visible tire contact points from pixel to real-world coordinates
- Validate against known reference points

#### 4.5 Full Vehicle Boundary Estimation
- From 2 visible contact points + vehicle dimensions, estimate all 4 corners
- Use typical vehicle dimensions as priors (length: ~4.5m, width: ~1.8m)
- Mathematical modeling to extrapolate hidden corners

### Phase 3: Validation & Optimization

#### 4.6 Ground Truth Collection
- Measure real-world positions of parked vehicles
- Create test dataset with known vehicle boundaries
- Compare estimated vs. actual positions

#### 4.7 Error Analysis & Improvement
- Quantify localization error (in meters)
- Compare against Oleg's bounding box method
- Identify and address failure cases

---

## 5. Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT: Video Frame                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: Fisheye Distortion Correction               │
│                   (Using camera calibration data)                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                STEP 2: Vehicle Detection (YOLOv8)                │
│                    Output: Bounding boxes + IDs                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: Tire Contact Point Detection (NEW)               │
│             Output: 2-4 keypoints per vehicle                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 4: Homography Transformation                     │
│       Pixel coordinates → Real-world coordinates (meters)        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 5: Full Boundary Estimation                       │
│      From visible points → Estimate all 4 vehicle corners        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 OUTPUT: Real-World Vehicle Position              │
│               (x, y coordinates in meters + orientation)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Tire contact point detection accuracy | < 5 pixel error | Compare with manual annotations |
| Real-world position error | < 0.5 meters | Compare with laser measurements |
| Improvement over Oleg's method | > 20% reduction in error | Side-by-side comparison |
| Processing speed | > 10 FPS | Runtime benchmarking |

---

## 7. Tools & Resources

### Existing Resources
- Camera calibration data
- Coordinate mapping files
- YOLOv8 detection model
- Wheel segmentation model
- Traffic video dataset

### Required Resources
- Oleg's Python calibration script (from Julian)
- Additional annotated keypoint training data
- Ground truth measurements for validation

### Software Stack
- Python 3.x
- OpenCV (camera calibration, image processing)
- Ultralytics YOLOv8 (detection, segmentation, keypoints)
- NumPy/SciPy (mathematical transformations)

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Tire contact points occluded by vehicle body | High | Use mathematical estimation from visible wheels |
| Keypoint detection not precise enough | High | Iterative model improvement, data augmentation |
| Camera angle limits visible wheels to 1-2 | Medium | Combine with bounding box dimensions |
| Calibration drift over time | Low | Periodic re-calibration verification |

---

## 9. Expected Contribution

This project will improve upon existing work by:

1. **Precision**: Using tire contact points instead of bounding box centers for more accurate ground-plane localization

2. **Methodology**: Combining detection (bounding boxes) + keypoints (tire contacts) for robust estimation

3. **Validation**: Quantitative comparison against previous methods with measured ground truth

4. **Applicability**: Practical solution for single-camera vehicle localization without stereo setup

---

## 10. Questions for Supervisor

1. Can you share Oleg's calibration script for marking reference points?
2. What was the measured precision of Oleg's bounding box method?
3. Are there existing keypoint annotations for tire contact points?
4. What is the acceptable error threshold for thesis approval?
5. Should I focus on a specific subset of the traffic videos?

---

*This plan will be updated based on supervisor feedback and project progress.*
