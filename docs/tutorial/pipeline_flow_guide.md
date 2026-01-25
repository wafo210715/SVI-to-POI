# HCMC POI Reconstruction - Pipeline Tutorial Guide

## Overview

This tutorial teaches the complete data flow for the HCMC Spatial-Semantic POI Reconstruction project. It's designed for **teaching students** how the pipeline works, block by block.

## Tutorial Script Location

```
scripts/tutorial/00_pipeline_tutorial.py
```

## How to Use This Tutorial

### Option 1: Jupyter Notebook (Recommended)

```bash
# Convert to notebook
pip install jupyter

# In VS Code, right-click the file and select "Open with Jupyter"
# Or run:
jupyter notebook scripts/tutorial/00_pipeline_tutorial.py
```

Then use **Shift+Enter** to execute each block sequentially.

### Option 2: IPython

```bash
# In VS Code, open the file and select "Run Cell" from the command palette
# Or use IPython terminal:
ipython
%run scripts/tutorial/00_pipeline_tutorial.py
```

### Option 3: Python Interactive

```bash
# Run the entire script at once
uv run python scripts/tutorial/00_pipeline_tutorial.py
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  BLOCK 1        │  Define HCMC bounding boxes
│  Input: None    │  Output: Dict of areas with coordinates
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 2        │  Initialize Mapillary client
│  Input: API key │  Output: MapillaryClient object
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 3        │  Get tile coordinates for bbox
│  Input: bbox    │  Output: List of tiles with z, x, y
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 4        │  Fetch tile data (JSON files)
│  Input: tiles   │  Output: Tiles with image counts
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 5        │  Parse images from tiles
│  Input: JSON    │  Output: List of image metadata
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 6        │  Sample random images
│  Input: all img │  Output: Sampled images (100)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 7        │  Fetch detailed entity data
│  Input: sample  │  Output: Full camera parameters
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 8        │  Analyze camera parameters
│  Input: entities│  Output: Statistics + feasibility
└────────┬────────┘
         │
         v
┌─────────────────┐
│  BLOCK 9        │  Save and visualize results
│  Input: analysis│  Output: JSON + HTML files
└─────────────────┘
```

---

## Data Structures Explained

### 1. HCMC_AREAS (Block 1)

```python
{
    "area_key": {
        "min_lon": float,  # West boundary (longitude)
        "min_lat": float,  # South boundary (latitude)
        "max_lon": float,  # East boundary (longitude)
        "max_lat": float,  # North boundary (latitude)
        "name": str        # Human-readable name
    }
}
```

**Example:**
```python
"small_test": {
    "min_lon": 106.69,  # West edge of Ben Thanh area
    "min_lat": 10.76,   # South edge
    "max_lon": 106.71,  # East edge
    "max_lat": 10.78,   # North edge
    "name": "Small Test Area (Ben Thanh)"
}
```

---

### 2. Tile Coordinates (Block 3)

```python
{
    "z": int,  # Zoom level (14 = image layer)
    "x": int,  # X coordinate (column)
    "y": int   # Y coordinate (row)
}
```

**Example:**
```python
{
    "z": 14,
    "x": 12785,
    "y": 7523
}
```

---

### 3. Tile JSON Structure (Block 4)

Each tile JSON file contains:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [106.70, 10.77]  # [lon, lat]
      },
      "properties": {
        "id": "123456789012345",
        "compass_angle": 45.0,
        "is_pano": false,
        "sequence_id": "abc123",
        "captured_at": "2024-01-15T10:30:00Z",
        "creator_id": "user_456"
      }
    }
  ]
}
```

---

### 4. Image Metadata (Block 5)

```python
{
    "image_id": str,          # Unique image identifier
    "camera_loc": {
        "lon": float,         # Longitude
        "lat": float          # Latitude
    },
    "heading": float,         # Camera direction (0-360°)
    "is_pano": bool,          # Is this a panorama?
    "sequence_id": str,       # Sequence this image belongs to
    "creator_id": str,        # User who uploaded
    "captured_at": str        # Timestamp
}
```

---

### 5. Entity Data (Block 7)

```python
{
    "id": str,                    # Image ID
    "camera_type": str,           # "perspective", "fisheye", "spherical"
    "camera_parameters": list,    # [focal, k1, k2]
    "make": str,                  # Camera manufacturer
    "model": str,                 # Camera model
    "sequence": str,              # Sequence ID
    "altitude": float,            # Altitude in meters
    "computed_altitude": float    # Computed altitude
}
```

---

## Key Concepts for Students

### What is a Tile?

Mapillary uses a **tile system** to organize street-level images:

- **Zoom level 14** is used for the image layer
- Each tile is a small geographic square
- Multiple tiles cover a bounding box
- Each tile contains images captured in that area

**Analogy:** Think of tiles like a grid overlay on a map. Each grid cell contains all the photos taken in that geographic area.

---

### What is a Sequence?

A **sequence** is a series of images captured by the same device in chronological order:

- Same camera = same focal length and distortion
- Enables consistent depth estimation
- Useful for 3D reconstruction

**Analogy:** Like a video frame-by-frame. Each frame is an image, and the entire video is a sequence.

---

### What are Camera Parameters?

**Camera parameters** describe the optical properties of the camera:

```python
camera_parameters = [focal, k1, k2]
```

- **focal**: Focal length in pixels (principal distance)
- **k1, k2**: Radial distortion coefficients

**Why it matters:** These parameters are essential for:
1. **Monocular depth estimation** - estimating distance from a single image
2. **Geometric projection** - converting pixels to real-world coordinates

---

## Visualization Files Generated

After running the tutorial, you'll find:

| File | Location | Description |
|------|----------|-------------|
| HCMC Areas Map | `data/debug/tutorial_hcmc_areas.html` | Shows where each area is located |
| Tile Grid Map | `data/debug/tutorial_tiles_{area}.html` | Shows tile coverage for selected area |
| Camera Analysis | `data/debug/tutorial_camera_analysis.html` | Interactive charts of camera stats |
| Analysis Summary | `data/debug/tutorial_depth_analysis_summary.json` | Raw analysis data |

---

## Common Questions

### Q: Why do we sample only 100 images?

**A:** Fetching full entity data is API-intensive. 100 images is enough to:
- Understand camera distribution
- Check depth estimation feasibility
- Determine most common camera types

You can increase `sample_size` in Block 6 if needed.

---

### Q: What's the difference between tile data and entity data?

**A:**

| Data Type | Source | Contains | Used For |
|-----------|--------|----------|----------|
| Tile Data | Tiles API | Basic location, heading | Finding images in an area |
| Entity Data | Graph API | Camera parameters, make/model | Depth estimation |

---

### Q: Why check for "intrinsics availability"?

**A:** Camera intrinsics (focal length, distortion) are required for:
1. **Monocular depth estimation** - estimating pixel depth
2. **Geometric projection** - converting pixels to GPS

Without intrinsics, we can't accurately estimate POI locations.

---

## Next Steps After Tutorial

### 1. Implement `geolocator.py`

Convert pixel coordinates to GPS using geometric projection.

### 2. Implement `vlm_extractor.py`

Extract semantic POI data using GLM-4V.

### 3. Integrate the Pipeline

Combine all modules into the full POI reconstruction pipeline.

---

## File Structure

```
SVI-to-POI/
├── scripts/
│   └── tutorial/
│       └── 00_pipeline_tutorial.py    # Main tutorial script
├── docs/
│   └── tutorial/
│       └── pipeline_flow_guide.md     # This file
├── data/
│   ├── raw/
│   │   └── tiles/
│   │       └── small_test/            # Tile JSON files
│   └── debug/
│       ├── tutorial_hcmc_areas.html
│       ├── tutorial_tiles_small_test.html
│       ├── tutorial_camera_analysis.html
│       └── tutorial_depth_analysis_summary.json
└── src/
    ├── crawler.py                     # Mapillary/GSV client
    ├── geolocator.py                  # (To be implemented)
    └── vlm_extractor.py               # (To be implemented)
```

---

## Teaching Tips

### Before Running the Tutorial

1. **Review the data flow diagram** above
2. **Explain bounding boxes** - what they represent geographically
3. **Show the HCMC map** - where District 1, Ben Thanh are located

### During the Tutorial

1. **Pause after each block** - check understanding
2. **Show output files** - open HTML visualizations in browser
3. **Compare data structures** - print and compare JSON outputs
4. **Discuss edge cases** - what if no images? what if API fails?

### After the Tutorial

1. **Quiz students** on data flow
2. **Have them modify** the area selection
3. **Discuss next modules** - geolocator, vlm_extractor

---

## Glossary

| Term | Definition |
|------|------------|
| **BBox** | Bounding Box - rectangular area defined by min/max lat/lon |
| **Tile** | Geographic grid cell containing image data |
| **Sequence** | Series of images from same device in chronological order |
| **Entity** | Full image metadata from Mapillary Graph API |
| **Intrinsics** | Camera internal parameters (focal length, distortion) |
| **SfM** | Structure from Motion - 3D reconstruction technique |
| **SVI** | Street View Imagery |
| **POI** | Point of Interest |

---

**Last Updated:** 2026-01-25
**Author:** Claude (Lead Engineer)
