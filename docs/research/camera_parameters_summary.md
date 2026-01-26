# Summary: Camera Parameters and Depth Estimation

## Your Questions Answered

### 1. What camera parameters affect depth?

**Ranked by impact:**

| Rank | Parameter | Impact | What It Does |
|------|-----------|--------|--------------|
| ⭐⭐⭐⭐⭐ | **Focal Length** | CRITICAL | Determines FOV and pixel-to-meter ratio |
| ⭐⭐⭐⭐ | **Camera Type** | HIGH | Perspective vs fisheye vs spherical |
| ⭐⭐⭐ | **Image Width/Height** | MEDIUM | Affects depth resolution |
| ⭐⭐ | **k1, k2 (Distortion)** | LOW-MED | Affects edge pixels |
| ⭐⭐ | **Compass Angle** | MEDIAN | Critical for GPS projection |
| ⭐ | **Pitch** | LOW | Affects POI visibility |

**For your project:** Focus on **focal length** - it's the primary factor.

### 2. How many camera categories?

Based on FOV (Field of View) categories for **perspective cameras only**:

| Category | FOV Range | Focal (pixels) | Typical Use |
|----------|-----------|----------------|-------------|
| **Narrow** | < 60° | > 1000px | Telephoto, distant objects |
| **Normal** | 60-90° | 500-1000px | Standard street view |
| **Wide** | > 90° | < 500px | Wide angle, close objects |

**Why only 3 categories?**
- You want to verify depth accuracy manually
- Too many categories = too much manual verification work
- These 3 cover the typical POI range (2-50m)

### 3. How to get percentages?

Run the camera analyzer:
```bash
uv run python scripts/analyze_camera_parameters.py
```

Output example:
```
Perspective Camera Categories:
Category                        Count    Percentage
----------------------------------------------------
Normal FOV (60-90°)            45       62.5%
Wide FOV (>90°)                  25       34.7%
Narrow FOV (<60°)                 2        2.8%

Summary:
  Total perspective: 92.3% of 72 images
```

This tells you:
- **62.5%** are "Normal" - your primary category
- **34.7%** are "Wide" - good for close POIs
- **2.8%** are "Narrow" - few images, may skip or handle separately

## Your Simplified Pipeline (Recommended)

```
SVI Image → Object Detection → ML Depth → Geometric Projection → POI GPS
```

**No stereo. No SfM. No image pairing.**

## Documentation Created

1. **`docs/research/mapillary_sfm_cluster_guide.md`**
   - How to access SfM cluster (documented but not recommended for POI)
   - When to use it (3D reconstruction, not single POI)
   - Code examples for fetching and parsing

2. **`docs/tutorial/understanding_depth_maps.md`**
   - What depth map gradients mean
   - Sequential stereo explanation
   - Why 77km result occurred (baseline too large)

3. **`scripts/analyze_camera_parameters.py`**
   - Categorizes cameras by FOV
   - Shows percentages for each category
   - Generates interactive HTML map
   - Filters to perspective cameras only

## Next Steps

1. **Run the analyzer** to see your camera distribution:
   ```bash
   uv run python scripts/analyze_camera_parameters.py
   ```

2. **Open the HTML** to explore categories:
   ```
   data/debug/camera_categories.html
   ```

3. **Manually verify** 2-3 images per category on Google Maps

4. **If categories are too many**, adjust the FOV ranges in `categorize_camera()` function

---

**Key Takeaway:** For HCMC POI extraction with 60-90% "Normal" cameras, you can optimize for that FOV range and achieve good depth accuracy.
