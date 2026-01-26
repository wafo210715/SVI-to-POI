# Mapillary SfM Cluster - Complete Guide

## Overview

Mapillary's `sfm_cluster` field contains **Structure-from-Motion (SfM) reconstruction data** for sequences of images. It's a powerful but complex resource.

## When to Use SfM Cluster

✅ **Use SfM Cluster when:**
- Building complete 3D reconstruction of scene
- Need accurate camera poses for multiple views
- Doing photogrammetry/NeRF/3D Gaussian Splatting
- Need sub-centimeter camera pose accuracy

❌ **Don't Use SfM Cluster when:**
- Only need distance to single object (signboard/POI)
- Working with single image
- Need quick processing
- Building simple POI database

**For POI extraction:** SfM is "杀鸡用宰牛刀" (overkill) - use ML depth instead.

## API Access

### Check if SfM Cluster Exists

```python
import requests

def check_sfm_cluster(image_id: str, access_token: str) -> bool:
    """Check if image has SfM cluster data."""
    url = "https://graph.mapillary.com"
    params = {
        "ids": image_id,
        "fields": "id,sfm_cluster",
        "access_token": access_token
    }

    response = requests.get(url, params=params)
    data = response.json()

    entity = data[image_id]
    return "sfm_cluster" in entity
```

### Fetch SfM Cluster Data

```python
def fetch_sfm_cluster(image_id: str, access_token: str) -> dict:
    """Download and parse SfM cluster data."""
    import zlib
    import requests

    # Get sfm_cluster URL
    url = "https://graph.mapillary.com"
    params = {
        "ids": image_id,
        "fields": "id,sfm_cluster",
        "access_token": access_token
    }

    response = requests.get(url, params=params)
    data = response.json()

    sfm_url = data[image_id]["sfm_cluster"]["url"]

    # Download point cloud
    sfm_response = requests.get(sfm_url, timeout=30)

    # Decompress (zlib-compressed JSON)
    decompressed = zlib.decompress(sfm_response.content)
    point_cloud = json.loads(decompressed)

    return point_cloud
```

## SfM Cluster Structure

```json
{
  "cameras": {
    "<camera_id>": {
      "projection_type": "perspective",
      "width": 4160,
      "height": 3120,
      "focal": 0.08265333127856844,
      "k1": -0.002518278651149906,
      "k2": 2.0046742717307887e-05
    }
  },
  "shots": {
    "<image_id>": {
      "rotation": [0.909, 1.585, -1.569],
      "translation": [-290.86, 3.81, -1742.76],
      "camera": "<camera_id>",
      "orientation": 1,
      "capture_time": 1735915586.9,
      "gps_dop": 15.0,
      "gps_position": [-1660.51, -601.89, 1.75],
      "compass": {
        "angle": 237.0,
        "accuracy": -1.0
      },
      "vertices": [],
      "faces": [],
      "scale": 1.0148097633400803,
      "covariance": [],
      "merge_cc": 545490967
    }
  }
}
```

## Key Fields

### Camera Parameters
- `focal`: Focal length in **normalized coordinates** (0-1 range)
- `k1`, `k2`: Radial distortion coefficients
- `width`, `height`: Image dimensions

### Shot Parameters
- `rotation`: Camera rotation as quaternion [x, y, z, w]
- `translation`: Camera position in 3D scene coordinates
- `gps_position`: GPS position in scene coordinates
- `compass`: Compass heading (degrees)
- `scale`: Scene scale factor

## Converting Focal Length

**Important:** SfM `focal` is **normalized** (0-1), not in pixels!

```python
def convert_focal_to_pixels(focal_normalized: float, image_width: int) -> float:
    """Convert normalized focal to pixels.

    Mapillary SfM uses normalized focal (0-1).
    Multiply by image width to get pixel value.
    """
    return focal_normalized * image_width

# Example
focal_norm = 0.08265333127856844
image_width = 4160
focal_pixels = convert_focal_to_pixels(focal_norm, image_width)
print(f"Focal length: {focal_pixels:.1f} pixels")
# Output: Focal length: 343.8 pixels
```

## Comparison: SfM vs ML Depth

| Aspect | SfM Cluster | ML Depth (ZoeDepth) |
|--------|-------------|---------------------|
| **Data Source** | Pre-computed by Mapillary | On-demand prediction |
| **Accuracy** | Very High (cm-level) | Good (~20% error) |
| **Processing Time** | Fast (download only) | Medium (30-100ms) |
| **Dependencies** | Network request | GPU + model |
| **Output** | Full 3D reconstruction | Single depth map |
| **Use Case** | 3D rendering/NeRF | POI distance |
| **Availability** | Most sequences | All images |

## Example: Your Image

For image `616035544433751`:

```bash
# Check availability
curl "https://graph.mapillary.com?ids=616035544433751&fields=id,sfm_cluster&access_token=$TOKEN"
```

Response shows sfm_cluster exists with URL to download 24KB zlib-compressed JSON data containing camera poses and intrinsics.

## File Format

- **Compression:** zlib (RFC 1950)
- **Content:** JSON array of point cloud data
- **Size:** Typically 10-50 KB per cluster
- **Encoding:** UTF-8

---

**Last Updated:** 2025-01-26
**Status:** Documented for future reference
**Location:** `docs/research/mapillary_sfm_cluster_guide.md`
