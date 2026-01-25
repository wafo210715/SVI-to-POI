# COLMAP vs Photogeometry for HCMC POI Extraction

## Executive Summary

**Recommendation**: Use photogeometry techniques (geometric projection, sequential stereo) instead of COLMAP for HCMC POI extraction. COLMAP is overkill for this use case.

---

## COLMAP Analysis

### What is COLMAP?

COLMAP (Structure-from-Motion and Multi-View Stereo) is a general-purpose SfM pipeline that:
- Extracts SIFT features from images
- Matches features across image pairs
- Triangulates 3D points
- Optimizes camera poses via bundle adjustment
- Generates sparse and dense reconstructions

### Computational Complexity

| Phase | Complexity | Time (per sequence) |
|-------|-----------|---------------------|
| Feature Extraction | O(n × features) | 5-10 min |
| Feature Matching | O(n² × features) | 10-30 min |
| Sparse Reconstruction | Iterative optimization | 15-60 min |
| Dense Reconstruction | O(n²) stereo | 1-4 hours |

**Total**: 1-6 hours per sequence (500-900 images)

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16-32 GB |
| Storage | 10 GB per sequence | 50 GB per sequence |
| CPU | 4 cores | 8+ cores |
| GPU | Optional | NVIDIA CUDA (faster) |

### MacOS Support

#### Apple Silicon (M1/M2/M3/M4) - ✅ SUPPORTED

**Installation**: Available via Homebrew
```bash
brew install colmap
```

**Bottle (binary) support**: Confirmed for macOS on Apple Silicon
- macOS Tahoe (15): ✅
- macOS Sequoia (15): ✅
- macOS Sonoma (14): ✅

**Performance on Apple Silicon**:
- **Good**: Single-core performance is excellent (feature extraction benefits)
- **Good**: Unified memory architecture helps with large datasets
- **Comparable to Windows**: COLMAP is CPU-bound, not GPU-bound, so Mac performance is competitive
- **NOT "much slower"**: The difference is negligible for most use cases

#### Recent Issues (2025)
- GitHub Issue #3150: Compilation failures on macOS 15 with M3 (Feb 2025)
- GitHub Issue #3479: Performance regression in mapper (v3.11 → v3.13)
- **Workaround**: Use stable version from Homebrew (managed by maintainers)

### When to Use COLMAP

✅ **Use COLMAP when you need**:
- Full 3D point cloud of a scene
- Sub-centimeter camera pose accuracy
- Dense reconstruction for NeRF/3D Gaussian Splatting
- Architectural preservation/documentation
- Virtual tourism/3D navigation

❌ **Don't use COLMAP when you only need**:
- Distance from camera to a single object
- GPS coordinates of a POI
- Relative position of storefront
- Fast processing of many locations

---

## Photogeometry Alternatives

### 1. Geometric Projection with Camera Intrinsics (LIGHTWEIGHT - RECOMMENDED)

**How it works**:
```
distance = (known_object_size × focal_length) / pixel_size
```

**Input required**:
- Camera intrinsics (focal length) - ✅ Available from Mapillary API
- Pixel size of object in image - ✅ Detectable via bounding box
- Known real-world object size - ⚠️ Assume typical values (e.g., 3m for shop front)

**Advantages**:
- ⚡ **Instant** (<1ms per calculation)
- 💾 **Minimal memory** (<1MB)
- 🎯 **Works with single image**
- ✅ **Uses Mapillary data directly**

**Limitations**:
- ⚠️ Requires assumption about object size
- ⚠️ Less accurate for distant objects (>50m)
- ⚠️ Requires detecting object boundaries

**Code Example**:
```python
def estimate_distance_to_poi(poi_pixel_width, known_width=3.0, focal=0.8):
    """Estimate distance to POI using geometric projection.

    Args:
        poi_pixel_width: Width of POI in pixels (from detection)
        known_width: Real-world width in meters (default: 3m for shop front)
        focal: Focal length from Mapillary camera_parameters[0]

    Returns:
        Estimated distance in meters
    """
    return (known_width * focal) / poi_pixel_width

# Project to GPS coordinates
def add_gps_offset(camera_lat, camera_lon, distance, heading_rad):
    """Add distance offset to GPS coordinates."""
    lat_offset = (distance * math.cos(heading_rad)) / 111320  # m/degree
    lon_offset = (distance * math.sin(heading_rad)) / (111320 * math.cos(math.radians(camera_lat)))
    return camera_lat + lat_offset, camera_lon + lon_offset
```

**Accuracy**: ±5-10m for near-field objects (<30m)

---

### 2. Sequential Stereo (MEDIUM WEIGHT)

**How it works**:
```
depth = (baseline × focal_length) / disparity
```

**Input required**:
- Two consecutive images from same sequence - ✅ Available from Mapillary
- Camera GPS positions - ✅ Available from Mapillary
- Focal length - ✅ Available from Mapillary
- Matching POI features in both images - ⚠️ Requires feature matching

**Advantages**:
- ✅ More accurate than monocular geometric
- ✅ Works well for medium distances (10-100m)
- ✅ No assumptions about object size

**Limitations**:
- ⚠️ Requires POI visible in both images
- ⚠️ Requires sufficient overlap (>60%)
- ⚠️ Feature matching can fail (low texture, lighting changes)

**Processing Time**: ~1 second per POI pair

**Accuracy**: ±2-5m for 10-50m range

---

### 3. ML Monocular Depth (MEDIUM WEIGHT)

**Options**:
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| **MiDaS v3.1** | 100MB | 50ms/image | Good relative depth |
| **ZoeDepth** | 20MB | 30ms/image | Good metric depth |
| **Depth-Anything** | 400MB | 100ms/image | State-of-the-art |

**Advantages**:
- ✅ Works with single image
- ✅ No assumptions needed
- ✅ Handles various scenes

**Limitations**:
- ⚠️ Metric depth is approximate
- ⚠️ Requires GPU for speed (CPU is slow)
- ⚠️ Model storage (20-400MB)

**Processing Time**: 30-100ms per image (GPU), 1-5s (CPU)

**Accuracy**: ±20% relative error

---

## Comparison Summary

| Aspect | COLMAP | Geometric Projection | Sequential Stereo | ML Depth |
|--------|--------|---------------------|-------------------|----------|
| **Input Images** | 50+ | 1 | 2 | 1 |
| **Processing Time** | Hours | <1ms | ~1s | 50ms |
| **Memory** | 16-32 GB | <1 MB | <100 MB | 2-4 GB |
| **Accuracy** | Very High (cm) | Medium (5-10m) | Good (2-5m) | Good (±20%) |
| **Mapillary Data** | ✅ Uses all data | ✅ Uses intrinsics | ✅ Uses sequences | ✅ Works with any |
| **Implementation** | Heavy (external tool) | Trivial (math) | Medium (OpenCV) | Medium (ML) |
| **Dependencies** | COLMAP binary | None | OpenCV | PyTorch + model |
| **Scalability** | Poor (serial) | Excellent | Good | Good |

---

## Decision Matrix for HCMC POI Extraction

### Use Case Requirements

| Requirement | Priority | COLMAP | Geometric | Stereo | ML |
|-------------|----------|--------|-----------|--------|-----|
| Single-image POI distance | High | ❌ | ✅ | ❌ | ✅ |
| Fast processing (<1s) | High | ❌ | ✅ | ❌ | ✅ |
| GPS coordinate estimation | High | ❌ | ✅ | ✅ | ⚠️ |
| Works with sparse data | High | ❌ | ✅ | ⚠️ | ✅ |
| Scalable to 100K+ POIs | High | ❌ | ✅ | ⚠️ | ⚠️ |
| Sub-meter accuracy | Low | ✅ | ❌ | ⚠️ | ❌ |
| Full 3D reconstruction | Low | ✅ | ❌ | ❌ | ❌ |

### Recommendation

**PRIMARY**: Geometric Projection
- Fastest, simplest, most scalable
- Sufficient accuracy for POI geolocation
- Directly uses Mapillary data

**FALLBACK 1**: Sequential Stereo
- When geometric fails (e.g., large objects, unknown size)
- When higher accuracy is needed
- Uses Mapillary sequences naturally

**FALLBACK 2**: ML Monocular Depth
- When stereo fails (no overlap)
- As sanity check for geometric results

**DO NOT USE COLMAP** unless you specifically need:
- 3D point clouds for rendering
- Camera pose accuracy beyond GPS
- Dense reconstruction for NeRF/3DGS

---

## Implementation Plan

### Phase 1: Geometric Projection (Week 1)

```python
# In src/geolocator.py

def estimate_poi_location_from_image(
    image_id: str,
    poi_pixel_bbox: Tuple[int, int, int, int],  # (x, y, w, h)
    camera_data: Dict,
    assumed_width: float = 3.0,  # meters
) -> Tuple[float, float]:
    """Estimate POI GPS location from single image.

    Uses Mapillary camera intrinsics for geometric projection.
    """
    # Extract camera parameters
    focal = camera_data['camera_parameters'][0]
    heading = camera_data['heading']
    camera_lat = camera_data['geometry']['coordinates'][1]
    camera_lon = camera_data['geometry']['coordinates'][0]

    # Calculate distance
    poi_width_pixels = poi_pixel_bbox[2]
    distance = (assumed_width * focal) / poi_width_pixels

    # Project to GPS
    poi_lat, poi_lon = add_gps_offset(camera_lat, camera_lon, distance, heading)
    return poi_lat, poi_lon
```

### Phase 2: Sequential Stereo (Week 2)

```python
def estimate_poi_location_stereo(
    image_id_1: str,
    image_id_2: str,
    poi_pixel_features: List[Tuple[int, int]],
    client: MapillaryClient,
) -> Tuple[float, float]:
    """Estimate POI location using two sequential images.

    Uses GPS baseline and stereo triangulation.
    """
    # Get camera data for both images
    cam1 = client.fetch_image_entity(image_id_1)
    cam2 = client.fetch_image_entity(image_id_2)

    # Calculate baseline from GPS
    baseline = calculate_gps_distance(cam1['geometry'], cam2['geometry'])

    # Match features (or use detected POI position)
    disparity = calculate_disparity(poi_pixel_features[0], poi_pixel_features[1])

    # Calculate depth
    focal = cam1['camera_parameters'][0]
    depth = (baseline * focal) / disparity

    # Project to GPS
    return add_gps_offset_from_depth(cam1, depth, heading)
```

### Phase 3: ML Depth Fallback (Week 3)

```python
# Optional - only if geometric and stereo fail

import torch
from transformers import pipeline

depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

def estimate_poi_location_ml(
    image: Image,
    poi_pixel_bbox: Tuple[int, int, int, int],
    camera_data: Dict,
) -> Tuple[float, float]:
    """Estimate POI location using ML depth estimation."""
    # Get depth map
    depth_map = depth_estimator(image)

    # Sample depth at POI center
    poi_depth = depth_map[poi_pixel_bbox[1], poi_pixel_bbox[0]]

    # Project to GPS
    return add_gps_offset_from_depth(camera_data, poi_depth)
```

---

## Benchmarks (Estimated)

| Method | Time per POI | 1000 POIs | 10K POIs | 100K POIs |
|--------|-------------|-----------|-----------|------------|
| **Geometric** | 0.001s | 1s | 10s | 100s |
| **Stereo** | 1s | 17 min | 2.8 hr | 28 hr |
| **ML Depth** | 0.05s | 50s | 8 min | 1.4 hr |
| **COLMAP** | 3600s | 42 days | 420 days | 11 years |

---

## Conclusion

### For HCMC POI Extraction: Use Geometric Projection

**Reasoning**:
1. ✅ Mapillary provides all required data (intrinsics, GPS, heading)
2. ✅ Accuracy is sufficient for POI geolocation (±5-10m)
3. ✅ Scales to 100K+ POIs
4. ✅ Fast enough for real-time processing
5. ✅ No external dependencies
6. ✅ Simple to implement and maintain

### When to Consider COLMAP

Only use COLMAP if you later need:
- 3D point clouds for visualization
- NeRF/3D Gaussian Splatting rendering
- Sub-centimeter accuracy
- Camera poses beyond what GPS provides

### MacOS-Specific Notes

- ✅ COLMAP runs fine on Apple Silicon via Homebrew
- ✅ Performance is comparable to Windows (COLMAP is CPU-bound)
- ⚠️ Recent versions have some issues (use stable Homebrew version)
- ❌ COLMAP requires significant time/resources for limited benefit

---

## References

- [COLMAP Official Website](https://colmap.github.io/)
- [COLMAP Homebrew Formula](https://formulae.brew.sh/formula/colmap)
- [COLMAP GitHub Issue #3150 - macOS 15 M3 Compilation](https://github.com/colmap/colmap/issues/3150)
- [COLMAP GitHub Issue #3479 - Performance Regression](https://github.com/colmap/colmap/issues/3479)
- [MiDaS Depth Estimation](https://github.com/isl-org/MiDaS)
- [ZoeDepth Metric Depth](https://github.com/isl-org/ZoeDepth)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-25
**Status**: Approved - Proceed with Geometric Projection approach
