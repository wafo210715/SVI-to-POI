# COLMAP Quality Assessment - Status and Instructions

## Current Status

COLMAP is not installed on this system. The COLMAP quality assessment script (`scripts/06_colmap_quality_assessment.py`) is ready but cannot run without COLMAP.

## What the COLMAP Assessment Would Do

The script (`06_colmap_quality_assessment.py`) would:

1. **Select diverse sequences** from tile data (10 sequences with different creators/locations)
2. **Download images** from each sequence (up to 50 images per sequence for speed)
3. **Run COLMAP reconstruction**:
   - Feature extraction (SIFT features)
   - Feature matching (exhaustive matching)
   - Sparse reconstruction (mapper)
4. **Analyze reconstruction quality**:
   - Registration rate (registered_images / total_images)
   - Point cloud density
   - Reconstruction success rate
5. **Generate feasibility report** comparing against Amsterdam baseline (~10% qualified)

## Installing COLMAP

### Option 1: Homebrew (Intel Mac)
```bash
brew install colmap
```

### Option 2: Build from Source (Apple Silicon / ARM64)

COLMAP doesn't have pre-built binaries for Apple Silicon. You'll need to build from source:

```bash
# Install dependencies
brew install cmake boost eigen libjpeg-turbo sqlite3

# Clone COLMAP
git clone https://github.com/colmap/colmap.git
cd colmap

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

# Install
sudo make install
```

### Option 3: Docker (Universal)

```bash
docker pull colmap/colmap
# Then run with mount points for data access
```

### Option 4: Python Wrapper (colmap-py)

```bash
pip install colmap
# Note: This provides Python bindings but still requires COLMAP binaries
```

## Alternative: Use Existing Results

Given the investigation so far, we can make an informed assessment:

### Positive Indicators for 3D Reconstruction:
1. **Very large sequences** (500-900 images) - excellent for SfM
2. **High image density** per sequence
3. **100% camera consistency** within sequences (from depth investigation)
4. **92% with valid intrinsics** - good for geometric reconstruction

### Concerns:
1. Amsterdam dataset had only ~10% qualify for NeRF
2. HCMC sequences may have similar issues (motion blur, varying lighting)
3. Need actual COLMAP run to confirm

## Recommendation

**Skip COLMAP for now** and proceed with the monocular depth approach since:
- 92% of images have valid intrinsics
- Sequences are camera-consistent
- Monocular depth is simpler and more scalable

If 3D reconstruction becomes critical later, COLMAP can be installed and run on specific high-value sequences.

## Current Investigation Results Summary

| Investigation | Result | File |
|--------------|--------|------|
| Depth & Camera | 92% with intrinsics, monocular_depth_feasible | `data/debug/depth_investigation_summary.json` |
| Sequences | 3,670 sequences, 500-900 images/sequence (top 100) | `data/debug/sequence_profile_summary.json` |
| COLMAP | Not run (COLMAP not installed) | N/A |

## Next Steps

1. Use monocular depth estimation for initial POI geolocation
2. Implement geometric projection using camera intrinsics from Mapillary
3. If results are insufficient, consider installing COLMAP for 3D reconstruction
