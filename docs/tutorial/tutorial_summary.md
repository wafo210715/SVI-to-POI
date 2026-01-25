# HCMC POI Reconstruction - Tutorial Summary

## Overview
This tutorial teaches the complete data flow for extracting street-level imagery from Mapillary and analyzing camera parameters for monocular depth estimation.

---

## Pipeline Steps

| Block | Action | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| **0** | Import libraries | - | Ready environment | Load required packages |
| **1** | Define HCMC bounding boxes | Coordinates | `HCMC_AREAS` dict | Set geographic area to analyze |
| **2** | Define `MapillaryClient` class | - | Class with API methods | Inline class for tile fetching |
| **3** | Initialize client | API token | `client` object | Connect to Mapillary API |
| **4** | Get tile coordinates | Bounding box | `tiles` list (z, x, y) | Convert bbox to tile grid |
| **5** | Fetch tile data | Tile coordinates | `tiles_with_data` + JSON files | Download image metadata per tile |
| **6** | Load all images | Tile JSON files | `all_images` list | Parse images from tile data |
| **7** | Sample images | All images | `sampled_images` (100) | Random subset for analysis |
| **8** | Fetch entity data | Image IDs | `entities` with camera params | Get detailed camera parameters |
| **9** | Analyze cameras | Entity data | `ANALYSIS_RESULTS` dict | Calculate feasibility metrics |
| **10** | Save & visualize | Analysis results | JSON + HTML files | Output summary and charts |

---

## Data Flow Diagram

```
Bounding Box (min_lon, min_lat, max_lon, max_lat)
              │
              ▼
    Mapillary Tiles (z, x, y)
              │
              ▼
    Tile JSON Files (GeoJSON)
              │
              ▼
    all_images List
    - image_id
    - camera_loc
    - heading
    - sequence_id
              │
              ▼ (sample 100)
    entities List
    - camera_type
    - camera_parameters
    - make/model
    - altitude
              │
              ▼
    ANALYSIS_RESULTS
    - feasibility
    - camera_counts
    - intrinsics_%
```

---

## Key Data Structures

### HCMC_AREAS (Block 1)
```python
{
    "small_test": {
        "min_lon": 106.69,
        "min_lat": 10.76,
        "max_lon": 106.71,
        "max_lat": 10.78,
        "name": "Small Test Area (Ben Thanh)"
    }
}
```

### Tile (Block 4)
```python
{"z": 14, "x": 12791, "y": 7527}
```

### Image Metadata (Block 6)
```python
{
    "image_id": "123456789",
    "camera_loc": {"lon": 106.70, "lat": 10.77},
    "heading": 45.0,
    "is_pano": false,
    "sequence_id": "abc123"
}
```

### Entity with Camera Params (Block 8)
```python
{
    "camera_type": "perspective",
    "camera_parameters": [0.0833, 0, 0],  # [focal, k1, k2]
    "make": "Apple",
    "model": "iPhone 12",
    "altitude": 0,
    "computed_altitude": 2.0
}
```

---

## Key Concepts

### Tile System
- **Zoom 14** = Image layer
- Each tile = Geographic grid cell
- Multiple tiles cover a bounding box
- Tiles contain images captured in that area

### Camera Parameters
| Parameter | Example | Purpose |
|-----------|---------|---------|
| `focal` (normalized) | 0.0833 | For pixel-to-ray conversion |
| `k1, k2` | 0, 0 | Radial distortion (0 = none) |

### Camera Types
| Type | Meaning | Depth Suitability |
|------|---------|-------------------|
| `perspective` | Standard flat image | ✅ Best |
| `fisheye` | Wide-angle lens | ⚠️ Needs correction |
| `spherical` | 360° panorama | ⚠️ Different projection |

### is_pano Field
```python
"is_pano": false  # Regular flat image (FOV = 60°)
"is_pano": true   # 360° panorama (FOV = 90°)
```

### altitude vs computed_altitude
| Field | Value | Source | Reliability |
|-------|-------|--------|-------------|
| `altitude` | 0 | EXIF/GPS | ❌ Often missing |
| `computed_altitude` | 2.0 | SfM | ✅ More reliable |

**Recommendation:** Use `computed_altitude` as it's calculated from 3D reconstruction.

### Feasibility Classification
```python
if intrinsic_ratio >= 0.8 and perspective_ratio >= 0.7:
    feasibility = "MONOCULAR DEPTH FEASIBLE"
elif intrinsic_ratio >= 0.5 and perspective_ratio >= 0.5:
    feasibility = "PARTIAL"
else:
    feasibility = "NOT FEASIBLE"
```

| Tier | Condition | Meaning |
|------|-----------|---------|
| **MONOCULAR DEPTH FEASIBLE** | 80%+ with focal, 70%+ perspective | ✅ Ready for depth estimation |
| **PARTIAL** | 50%+ meet criteria | ⚠️ Some images will work |
| **NOT FEASIBLE** | Below 50% | ❌ Need more data |

---

## Output Files

| File | Location | Content |
|------|----------|---------|
| Area map | `data/debug/tutorial_hcmc_areas.html` | HCMC area rectangles (colored) |
| Tile grid | `data/debug/tutorial_tiles_small_test.html` | Tiles (blue) vs area bbox (green) |
| Tile JSON | `data/raw/tiles/small_test/tile_*.json` | Raw tile data from Mapillary |
| Summary | `data/debug/tutorial_depth_analysis_summary.json` | Analysis results |
| Charts | `data/debug/tutorial_camera_analysis.html` | Camera type & intrinsics charts |

---

## Using Your Own District Boundaries

The bounding boxes in this tutorial are **simplified rectangles**. For accurate district boundaries:

1. **Download HCMC district SHP file** from:
   - GADM: https://gadm.org/download_country_v3.html
   - OpenStreetMap: https://download.geofabrik.de/asia/vietnam.html

2. **Load in ArcGIS/QGIS:**
   ```
   - Add SHP layer → Select your district
   - Properties → Source tab → Extent shows bbox
   - Or: Data Management Tools → Features → Bounding Box
   ```

3. **Get coordinates and replace in Block 1:**
   ```python
   "district_1_full": {
       "min_lon": YOUR_MIN_LON,
       "min_lat": YOUR_MIN_LAT,
       "max_lon": YOUR_MAX_LON,
       "max_lat": YOUR_MAX_LAT,
       "name": "District 1 (from SHP)",
   }
   ```

---

## Practical Takeaways

1. **Bounding boxes are simplified** - Use official SHP files from ArcGIS for accurate boundaries
2. **Tiles organize images** - Zoom 14 tiles = manageable chunks of geographic space
3. **Camera parameters are critical** - Focal length required for monocular depth
4. **Computed altitude > EXIF altitude** - SfM-derived height is more reliable
5. **Sampling saves API calls** - 100 images enough to understand distribution

---

## What This Enables

After completing this tutorial, you can:
- ✅ Fetch Mapillary imagery for any HCMC area
- ✅ Access camera parameters for depth estimation
- ✅ Understand which images are suitable for monocular depth
- ✅ Prepare data for POI localization pipeline
- ✅ Replace sample areas with your own SHP-derived boundaries

---

## How to Run

```bash
# Run entire script
uv run python scripts/tutorial/00_pipeline_tutorial.py

# Or in VS Code: Shift+Enter to run each block sequentially
```

**Required:** Set `MAPILLARY_ACCESS_TOKEN` in your `.env` file.
