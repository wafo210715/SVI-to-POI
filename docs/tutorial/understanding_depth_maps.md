# Understanding Depth Maps and Stereo Depth Calculation

## What is a Depth Map?

A depth map is an image where **each pixel value represents the distance** from the camera to that point in the scene.

### Visual Representation

```
Original Scene:     Depth Map (grayscale):     Depth Map (color):
┌─────────────┐      ┌─────────────┐           ┌─────────────┐
│    ___      │      │    ░░░      │           │    ▓▓▓      │  ← Near (dark/warm)
│   /   \     │      │   ▒▒▒▒▒     │           │   ▔▔▔▔▔     │
│  | person|  │  →   │  ▓▓▓███▓▓   │     →     │  ▒▒▒XXX▒▒▒  │  ← Medium
│   \   /     │      │   ▒▒▒▒▒     │           │   ▓▓▓▓▓▓▓   │
│    ───      │      │    ▒▒▒      │           │    ▒▒▒      │  ← Far (light/cool)
└─────────────┘      └─────────────┘           └─────────────┘
```

### Color Interpretation (jet colormap commonly used)

| Color Range | Depth Meaning | Example Distance |
|-------------|---------------|------------------|
| **Red/Orange** | Very Near | 0-2m (sidewalk, immediate foreground) |
| **Yellow** | Near | 2-5m (close storefronts) |
| **Green** | Medium | 5-15m (typical POI distance) |
| **Cyan/Blue** | Far | 15-50m (distant buildings) |
| **Purple/Dark Blue** | Very Far | 50m+ (background, skyline) |

### Grayscale Interpretation

| Value | Depth Meaning |
|-------|---------------|
| **0 (Black)** | Very Near or Invalid |
| **128 (Gray)** | Medium distance |
| **255 (White)** | Far or infinite |

## Sequential Stereo Depth: Explained

### The Formula

```
depth = (baseline × focal_length) / disparity
```

### What Each Term Means

**1. Baseline (meters)**
- The distance between the two camera positions
- For driving sequences: typically 1-10 meters
- Calculated from GPS coordinates using Haversine formula

**2. Focal Length (pixels)**
- The camera's "zoom" level
- Higher focal = narrower field of view
- From Mapillary `camera_parameters[0]`
- Typical values: 500-2000 pixels

**3. Disparity (pixels)**
- How much the POI shifts between two images
- Larger disparity = closer object
- Smaller disparity = farther object
- Calculated as: `pixel_x_in_image1 - pixel_x_in_image2`

### Geometric Intuition

```
Top-down view:

    Image 1          Image 2
    (Camera 1)       (Camera 2)
       ◄──────────────────►
       baseline
       |     |
       |     | depth
       |     |
       ↓     ↓
    ╔═══════════════════╗
    ║                   ║
    ║      POI          ║
    ║                   ║
    ╚═══════════════════╝

In Image 1:        In Image 2:
POI at pixel 300   POI at pixel 500
Disparity = 500 - 300 = 200 pixels
```

### Why Your Result Was 77,672m

Let's break down the calculation:

```
baseline = 932.07m  ← Cameras are 932m apart! (almost 1km)
focal   = 1000      ← Default focal length we used
disparity = 12      ← Assumed POI position difference

depth = (932.07 × 1000) / 12
      = 932,070 / 12
      ≈ 77,672m
      ≈ 77.7 km  ← This is correct but unrealistic!
```

**The Problem:**
1. The two images are **932 meters apart** - way too far for stereo
2. With such a large baseline, even small disparities give huge depths
3. For realistic POI depths (5-50m), we need:
   - Baseline: 5-20m
   - Disparities: 50-500 pixels

### Realistic Example

For a POI at **20 meters**:

```
baseline = 10m      (cameras 10m apart)
focal   = 1000      (typical focal length)
depth   = 20m       (desired depth)

disparity = (baseline × focal) / depth
         = (10 × 1000) / 20
         = 500 pixels

So the POI should appear 500 pixels apart in the two images.
```

## Requirements for Sequential Stereo to Work

1. **Same POI visible in both images**
   - Need feature matching or object detection
   - POI shouldn't be occluded in either view

2. **Reasonable baseline**
   - **Too small (<1m)**: Depth errors become large
   - **Too large (>50m)**: POI often not visible in both
   - **Optimal**: 5-20m for street scenes

3. **Accurate pixel matching**
   - Need to find the same POI feature in both images
   - Use SIFT, ORB, or deep learning feature matching
   - Or use detected bounding boxes (simpler but less accurate)

4. **Camera calibration**
   - Need accurate focal length from Mapillary
   - Need accurate GPS for baseline calculation

## When Sequential Stereo Fails

| Scenario | Issue | Fallback |
|----------|-------|----------|
| Images too far apart | Baseline too large | Use geometric projection |
| POI not in both images | Can't match features | Use ML monocular depth |
| Poor texture/featureless | Feature matching fails | Use ML monocular depth |
| Moving objects | Parallax breaks | Use single-image methods |

## Practical Tips

1. **Always check baseline first**
   ```python
   baseline = haversine_distance(cam1_lat, cam1_lon, cam2_lat, cam2_lon)
   if baseline > 50:
       print("Warning: Baseline too large for stereo")
   ```

2. **Validate disparity is reasonable**
   ```python
   if disparity < 10:
       print("Warning: Very small disparity - depth uncertain")
   if disparity > image_width / 2:
       print("Warning: Very large disparity - check matching")
   ```

3. **Cross-validate with other methods**
   - Compare stereo depth with ML depth
   - Compare with geometric projection
   - Use consensus for robustness

---

**References:**
- [Stereo Vision - Wikipedia](https://en.wikipedia.org/wiki/Stereo_vision)
- [Haversine Formula](https://en.wikipedia.org/wiki/Haversine_formula)
- Mapillary API: `sfm_cluster` field for pre-computed depth
