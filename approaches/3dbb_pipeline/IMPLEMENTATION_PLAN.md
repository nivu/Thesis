# 3D Bounding Box Integration Plan

## Objective

Integrate YOLO-based 3D bounding box detection into the existing vehicle localization pipeline to potentially improve accuracy over the current 2D keypoint + homography approach.

---

## Current Approach (Baseline)

| Component | Method | Accuracy |
|-----------|--------|----------|
| Detection | YOLOv8 pose (2D bbox + 10 keypoints) | 100% detection rate |
| Ground Contact | Bottom-center of bbox OR keypoints | ~0.31m position error |
| Coordinate Transform | Homography (street plane only) | RMS error: 0.73 |

**Limitation**: Only points ON the calibrated street plane can be converted to real-world coordinates.

---

## Proposed 3D Approach

### Why 3D Bounding Boxes Could Improve Accuracy

1. **Direct 3D Position Estimation**: Eliminates need to infer ground contact from 2D features
2. **Vehicle Orientation**: Yaw angle enables precise boundary estimation
3. **Known Dimensions**: 3D boxes provide L×W×H, allowing full vehicle extent calculation
4. **Ground Plane Projection**: 3D box bottom corners project directly to street plane

### Expected Improvements

| Metric | Current | Expected with 3D |
|--------|---------|------------------|
| Position accuracy | ~0.31m | <0.20m |
| Orientation | Not available | Direct estimation |
| Vehicle boundaries | Approximated from 2D | Precise 3D projection |

---

## Recommended Model: YOLOx3D

Based on research, **YOLOx3D** is the most suitable framework:

### Architecture
```
┌─────────────────┐
│   Input Frame   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ YOLOv11 │ ──→ 2D Bounding Boxes
    └────┬────┘
         │
┌────────▼────────┐
│ Depth Anything  │ ──→ Depth Map
│      v2         │
└────────┬────────┘
         │
┌────────▼────────┐
│ Multi-bin CNN   │ ──→ Orientation + Dimensions
│   Regressor     │
└────────┬────────┘
         │
┌────────▼────────┐
│ 3D Localization │ ──→ 3D Bounding Box (x, y, z, l, w, h, yaw)
└────────┬────────┘
         │
┌────────▼────────┐
│ Ground Plane    │ ──→ 4 Bottom Corners (real-world coords)
│   Projection    │
└─────────────────┘
```

### Why YOLOx3D
- Uses latest YOLOv11 for detection
- Depth Anything v2 provides robust monocular depth
- Modular architecture allows integration with existing homography validation
- Open source with pretrained weights available

---

## Integration Architecture

### Hybrid Pipeline (Recommended)

```
                    ┌──────────────────────────────────────┐
                    │          Input Frame                 │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────▼───────────────────┐
                    │     Fisheye Undistortion             │
                    │   (gopro_calibration_fisheye.npz)    │
                    └──────────────────┬───────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │  YOLOx3D Pipeline │   │ Current Keypoint  │   │   Depth Map       │
    │  (3D Detection)   │   │ Pipeline (2D+KP)  │   │   Generation      │
    └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
              │                        │                        │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐              │
    │ 3D Box → Ground   │   │ Keypoint →        │              │
    │ Corner Projection │   │ Homography        │              │
    └─────────┬─────────┘   └─────────┬─────────┘              │
              │                        │                        │
              └────────────┬───────────┘                        │
                           │                                    │
              ┌────────────▼────────────┐                       │
              │   Fusion & Validation   │◄──────────────────────┘
              │  - Cross-validate       │
              │  - Select best estimate │
              │  - Confidence weighting │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Final Real-World      │
              │   Position & Boundaries │
              └─────────────────────────┘
```

---

## Implementation Steps

### Phase 1: Environment Setup (Week 1)

**Step 1.1: Install YOLOx3D Dependencies**
```bash
# Create new virtual environment
python -m venv 3dbb_env
source 3dbb_env/bin/activate

# Clone YOLOx3D
git clone https://github.com/baptdes/YOLOx3D.git
cd YOLOx3D
pip install -e .

# Additional dependencies
pip install depth-anything-v2 ultralytics torch torchvision
```

**Step 1.2: Download Pretrained Weights**
- YOLOv11 weights → `weights/yolov11.pt`
- Multi-bin regressor → `weights/multibin_model.pt`
- Depth Anything v2 → auto-downloaded

**Step 1.3: Verify Installation**
```python
# Test script
from yolox3d import YOLOX3D
model = YOLOX3D()
result = model.predict("test_frame.jpg")
print(result.boxes_3d)
```

### Phase 2: Camera Calibration Adaptation (Week 2)

**Step 2.1: Extract Camera Intrinsics**
```python
import numpy as np

# Load existing calibration
calib = np.load('gopro_calibration_fisheye.npz')
K = calib['K']  # Camera matrix
D = calib['D']  # Distortion coefficients

# Convert to format required by YOLOx3D
camera_params = {
    'fx': K[0, 0],
    'fy': K[1, 1],
    'cx': K[0, 2],
    'cy': K[1, 2]
}
```

**Step 2.2: Calibrate Depth Scale**
- YOLOx3D outputs relative depth
- Use known street calibration points to compute absolute scale factor
- Validation: compare 3D-projected distances vs homography distances

### Phase 3: 3D Detector Module (Week 3-4)

**Step 3.1: Create 3D Detector Wrapper**

File: `3dbb_pipeline/detector_3d.py`
```python
class Detector3D:
    """3D bounding box detector using YOLOx3D."""

    def __init__(self, camera_params, depth_scale):
        self.camera_params = camera_params
        self.depth_scale = depth_scale
        self.model = self._load_model()

    def detect(self, frame):
        """
        Detect vehicles and estimate 3D bounding boxes.

        Returns:
            List of Box3D objects with:
            - center: (x, y, z) in camera frame
            - dimensions: (length, width, height) in meters
            - orientation: yaw angle in radians
            - corners_3d: 8 corner points of the 3D box
            - corners_ground: 4 bottom corners projected to ground
        """
        pass

    def project_to_ground(self, box_3d):
        """Project 3D box bottom corners to ground plane."""
        pass
```

**Step 3.2: Ground Plane Projection**
```python
def project_to_ground(self, box_3d):
    """
    Project 3D bounding box to ground plane.

    The 4 bottom corners of the 3D box lie on the ground plane,
    providing precise vehicle footprint in real-world coordinates.
    """
    # Get bottom 4 corners of 3D box
    corners_3d = box_3d.get_corners()  # 8 corners
    bottom_corners = corners_3d[4:8]   # Bottom 4 corners

    # Transform to real-world coordinates
    # Using camera extrinsics + known ground plane
    ground_corners = self.camera_to_world(bottom_corners)

    return ground_corners
```

### Phase 4: Fusion Module (Week 5)

**Step 4.1: Create Fusion Strategy**

File: `3dbb_pipeline/fusion.py`
```python
class CoordinateFusion:
    """Fuse 3D detection with homography-based estimation."""

    def __init__(self, transformer, depth_validator):
        self.transformer = transformer  # Existing homography
        self.depth_validator = depth_validator

    def fuse(self, detection_3d, detection_2d):
        """
        Fuse 3D and 2D estimates with confidence weighting.

        Strategy:
        1. If 3D depth is within calibrated range → weight 3D higher
        2. If vehicle near calibration points → weight homography higher
        3. Cross-validate: if estimates differ >1m → flag for review
        """
        # Get 3D estimate
        pos_3d = detection_3d.ground_center
        conf_3d = detection_3d.confidence

        # Get 2D homography estimate
        pos_2d = self.transformer.pixel_to_world(
            detection_2d.contact_point[0],
            detection_2d.contact_point[1]
        )
        conf_2d = self._homography_confidence(detection_2d.contact_point)

        # Weighted fusion
        total_conf = conf_3d + conf_2d
        fused_x = (pos_3d[0] * conf_3d + pos_2d[0] * conf_2d) / total_conf
        fused_y = (pos_3d[1] * conf_3d + pos_2d[1] * conf_2d) / total_conf

        return (fused_x, fused_y), total_conf / 2
```

### Phase 5: Main Pipeline (Week 6)

**Step 5.1: Create 3D Pipeline**

File: `3dbb_pipeline/main.py`
```python
def main():
    # Load models
    model_2d = YOLO(MODEL_PATH)  # Existing
    model_3d = Detector3D(camera_params, depth_scale)  # New

    # Load transformers
    transformer = CoordinateTransformer(MAPPING_FILE)
    fusion = CoordinateFusion(transformer, depth_validator)

    for frame in video:
        # Preprocess
        frame_undistorted = preprocess_frame(frame, K, D, DIM)

        # Run both detection methods
        results_2d = model_2d.track(frame_undistorted, persist=True)
        results_3d = model_3d.detect(frame_undistorted)

        # Associate 3D detections with 2D tracks
        matched = associate_detections(results_2d, results_3d)

        # Fuse coordinates
        for det_2d, det_3d in matched:
            position, confidence = fusion.fuse(det_3d, det_2d)
            boundaries = det_3d.ground_corners  # Full vehicle extent

            # Calculate speed
            speed = speed_tracker.update(det_2d.track_id, position)

            # Export
            export_results(position, boundaries, speed)
```

### Phase 6: Validation & Comparison (Week 7-8)

**Step 6.1: Create Comparison Framework**

File: `3dbb_pipeline/evaluate.py`
```python
def evaluate_accuracy():
    """
    Compare 3D vs 2D approaches on known calibration points.

    Metrics:
    - Position MAE (mean absolute error in meters)
    - Position RMSE
    - Boundary IoU (if ground truth available)
    - Speed estimation error
    """
    pass
```

**Step 6.2: Test Cases**
1. Vehicles at known calibration points
2. Vehicles at varying distances (near/mid/far)
3. Multiple vehicles simultaneously
4. Occluded vehicles (partial visibility)

---

## File Structure

```
3dbb_pipeline/
├── IMPLEMENTATION_PLAN.md     # This document
├── main.py                    # Main 3D pipeline
├── config.py                  # Configuration
├── detector_3d.py             # 3D detection wrapper
├── fusion.py                  # Coordinate fusion module
├── depth_calibration.py       # Depth scale calibration
├── evaluate.py                # Accuracy evaluation
├── utils/
│   ├── camera_utils.py        # Camera intrinsics handling
│   ├── box3d.py               # 3D bounding box class
│   └── projection.py          # 3D → ground projection
└── weights/                   # Model weights (gitignored)
    ├── yolov11.pt
    └── multibin_model.pt
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Domain gap (KITTI → surveillance) | High | High | Fine-tune on similar data or use domain adaptation |
| Depth estimation inaccuracy | Medium | High | Calibrate with known distances, fuse with homography |
| Computational overhead | Medium | Low | Use efficient models, batch processing |
| Integration complexity | Low | Medium | Modular design, extensive testing |

---

## Success Criteria

1. **Position accuracy improvement**: Reduce error from ~0.31m to <0.20m
2. **Boundary estimation**: Provide full vehicle footprint (4 corners)
3. **Orientation detection**: Yaw angle estimation within ±10°
4. **Robustness**: Maintain 100% detection rate
5. **Real-time capability**: Process at >15 FPS

---

## References

- [YOLOx3D GitHub](https://github.com/baptdes/YOLOx3D) - Monocular 3D detection with depth estimation
- [YOLO3D GitHub](https://github.com/ruhyadi/YOLO3D) - YOLOv5 + 3D regression approach
- [YOLOv7-3D Paper](https://www.mdpi.com/2076-3417/13/20/11402) - Roadside perspective 3D detection

---

## Next Steps

1. [ ] Set up 3D detection environment
2. [ ] Test YOLOx3D on sample frames from current dataset
3. [ ] Calibrate depth scale using known street measurements
4. [ ] Implement basic 3D detector wrapper
5. [ ] Compare initial 3D results with current 2D approach
