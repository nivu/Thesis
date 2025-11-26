# Calibration Data Summary

## ✅ YES - You Have Real-World Coordinates!

You have **two complete calibration datasets** with real-world coordinates already in your project.

---

## Dataset 1: coordinate_mapping_2030.json

**7 Calibration Points:**

| Point # | Pixel (x, y) | Real-world (X, Y) meters | Notes |
|---------|--------------|--------------------------|-------|
| 1 | (939, 549) | (0.0, 0.0) | **Origin point** |
| 2 | (929, 494) | (-0.2587, 4.8672) | ~4.87m forward |
| 3 | (945, 819) | (-0.1187, -4.3256) | ~4.33m backward |
| 4 | (1625, 453) | (20.7026, 4.9524) | ~20.7m right, 5m forward |
| 5 | (1800, 617) | (11.3906, -4.247) | ~11.4m right, 4.2m back |
| 6 | (1412, 500) | (10.8631, 1.5942) | ~10.9m right, 1.6m forward |
| 7 | (172, 529) | (-20.268, 4.8034) | ~20.3m left, 4.8m forward |

**Coverage:** Approximately **40 meters width** (from -20m to +20m) × **10 meters depth**

**Status:** ✅ Currently used by your pipeline (in `config.py` and `validate_speed_estimation.py`)

---

## Dataset 2: coordinate_mapping_4050.json

**6 Calibration Points:**

| Point # | Pixel (x, y) | Real-world (X, Y) meters | Notes |
|---------|--------------|--------------------------|-------|
| 1 | (702, 848) | (0.0, 0.0) | **Origin point** |
| 2 | (855, 577) | (0.0964, 4.4748) | ~4.5m forward |
| 3 | (1608, 553) | (15.0, 4.4748) | 15m right, 4.5m forward |
| 4 | (964, 842) | (2.0, 0.0) | 2m right |
| 5 | (601, 586) | (-5.0, 4.4748) | 5m left, 4.5m forward |
| 6 | (1556, 502) | (20.0371, 9.1933) | ~20m right, ~9m forward |

**Coverage:** Approximately **25 meters width** (from -5m to +20m) × **9 meters depth**

**Status:** ✅ Available but not currently used

---

## Ground Truth Data Context

From `Ground Truth Data.docx`:

### Camera Setup
- **Camera model:** GoPro Hero 5 Session
- **Height:** 3 meters above ground
- **Orientation:** Perpendicular to vehicle path
- **Resolution:** 1920 x 1080
- **Calibration RMS error:** 0.7324

### Test Conditions
- **Environment:** Straight road, normal daytime lighting
- **Speeds tested:** 20, 30, 40, 50 km/h
- **Reference:** Dashboard speedometer

### Calibration Files
- **Camera intrinsics:** `gopro_calibration_fisheye.npz` (for lens distortion)
- **Homography:** `coordinate_mapping_2030.json` (pixel → world)

---

## Comparison with Professor's Example

Your data matches the structure mentioned in the conversation with Julian:

| Source | Points | Coverage | Format |
|--------|--------|----------|--------|
| **Your data** | 7 points | 40m × 10m | ✅ JSON with homography |
| **Professor's example** | 7 points | Similar | ✅ Mentioned in chat |
| **Oleg's example** | 35 points | Larger area | ✅ CSV format |

---

## How These Were Created

Using `coordinates_mapping.py`:
1. Load a frame from the video
2. Click points in the image (pixel coordinates)
3. Manually enter real-world coordinates for each point
4. Compute homography transformation matrix using RANSAC
5. Save to JSON file

---

## What This Means for Your Project

### ✅ You Can Already Do:
1. **Convert pixel coordinates to real-world coordinates**
   - Load the homography matrix from JSON
   - Apply transformation to any pixel on the street plane

2. **Calculate real distances**
   - Distance between cars in meters
   - Accurate speed estimation

3. **Integrate with wheel detection**
   - Detect tire-street contact points (pixels)
   - Transform to real-world coordinates
   - Calculate actual vehicle positions

### 🎯 Your Advantage:
- You have **working calibration data**
- You have **wheel keypoint detection**
- You just need to **combine them** properly!

---

## Next Steps

### 1. Verify Calibration Quality
- Test if these coordinates match the new traffic videos
- If videos are from different locations, you may need new calibration

### 2. Check Video-to-Calibration Mapping
Which video corresponds to which calibration file?
- `coordinate_mapping_2030.json` → ?
- `coordinate_mapping_4050.json` → ?

### 3. Integrate with New Traffic Data
Do the new videos in `traffic_analyis_data/` match these calibrations?
- `Uni_west_1/GOPR0574.MP4` → ?
- `Uni_ost_2023-10-18/Test-*.MP4` → ?
- `circle_2023_12_15/GOPR*.MP4` → ?

### 4. Validate with Oleg's Calibration Tool
- Use `camera_calibration/main.py` to verify calibration
- Compare with your existing JSON data
- Potentially create more precise calibrations for new videos

---

## Questions to Ask Julian

1. Do the videos in `traffic_analyis_data/` correspond to the existing calibrations?
2. Were these calibration measurements (coordinate_mapping_*.json) from his previous work?
3. Should you create new calibrations for the new video locations?
4. Can you use the existing calibrations or need fresh measurements?

---

## Key Insight

**You're further along than you thought!**

You already have:
- ✅ Real-world coordinate calibration
- ✅ Working homography transformations
- ✅ Wheel detection model
- ✅ Speed estimation pipeline

The missing piece is just **verification and integration** with the new video data.
