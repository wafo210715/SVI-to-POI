#!/usr/bin/env python3
"""
HCMC POI Reconstruction - Stress Test (All Images)
===================================================

This stress test runs the complete POI reconstruction pipeline on ALL images
instead of the 100-image sample used in the tutorial.

Usage:
    ipython
    %run scripts/tutorial/stress_test.py

Key Differences from 00_pipeline_tutorial.py:
- Block 7: Uses ALL_IMAGES instead of sampling 100 images
- Block 8+: Processes all images through the complete pipeline
- Longer runtime expected (proportional to dataset size)

SHIFT OF MIND (Key Learning):
==============================
Original Plan: Compare three depth approaches - Mapillary SfM, Sequential Stereo, ML
Reality After Analysis:
- SfM cluster is overkill for single POI
- Sequential stereo requires image pairing + feature matching - too complex
- Simplified approach wins: Detect POI -> Sample depth -> Project to GPS
- 95% perspective cameras with 69.5% Normal FOV -> geometric projection is reliable

STRESS TEST MODE:
=================
This script processes ALL available images in the selected area.
Expect significantly longer runtime:
- 100 images: ~30-70 minutes (GPU) or ~1.5-3 hours (CPU)
- All images: Proportionally longer based on dataset size
"""

# =============================================================================
# BLOCK 0: IMPORTS (Same as tutorial)
# =============================================================================

import json
import math
import random
import sys
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
import folium
from tqdm import tqdm

# Optional dependencies
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

print("=" * 70)
print("STRESS TEST MODE: Processing ALL Images")
print("=" * 70)
print("BLOCK 0: Libraries imported")
print("Next: Run Block 1")


# =============================================================================
# BLOCK 1: DEFINE HCMC BOUNDING BOX AREAS (Same as tutorial)
# =============================================================================

HCMC_AREAS = {
    "district_1_full": {
        "min_lon": 106.65,
        "min_lat": 10.75,
        "max_lon": 106.75,
        "max_lat": 10.82,
        "name": "District 1 (Full)",
    },
    "central_hcmc": {
        "min_lon": 106.62,
        "min_lat": 10.72,
        "max_lon": 106.72,
        "max_lat": 10.82,
        "name": "Central HCMC",
    },
    "small_test": {
        "min_lon": 106.69,
        "min_lat": 10.76,
        "max_lon": 106.71,
        "max_lat": 10.78,
        "name": "Small Test Area (Ben Thanh) - ~4.8km²",
    },
    "micro_test": {
        "min_lon": 106.695,
        "min_lat": 10.765,
        "max_lon": 106.705,
        "max_lat": 10.775,
        "name": "Micro Test Area (1 block) - ~1.2km²",
    },
    "tiny_test": {
        "min_lon": 106.698,
        "min_lat": 10.768,
        "max_lon": 106.702,
        "max_lat": 10.772,
        "name": "Tiny Test Area - ~0.4km²",
    },
}

print("BLOCK 1: HCMC Bounding Box Areas Defined")
for key, area in HCMC_AREAS.items():
    print(f"  {key}: {area['name']}")

# Create visualization
m = folium.Map(location=[10.77, 106.69], zoom_start=12)
for key, area in HCMC_AREAS.items():
    folium.Rectangle(
        bounds=[[area['min_lat'], area['min_lon']], [area['max_lat'], area['max_lon']]],
        popup=area['name'],
        color='red',  # Red to indicate stress test mode
        fill=True,
        fillOpacity=0.2,
    ).add_to(m)

Path("data/debug").mkdir(parents=True, exist_ok=True)
m.save(Path("data/debug/stress_test_hcmc_areas.html"))

SELECTED_AREA = "tiny_test"  # Options: tiny_test, micro_test, small_test, central_hcmc, district_1_full
print(f"Selected: {SELECTED_AREA} - {HCMC_AREAS[SELECTED_AREA]['name']}")
print("")
print("Available areas:")
for key, area_info in HCMC_AREAS.items():
    print(f"  - {key}: {area_info['name']}")
print("")
print("To change area, edit SELECTED_AREA above and re-run from Block 1")
print("Next: Run Block 2")


# =============================================================================
# BLOCK 2: DEFINE MAPILLARY CLIENT CLASS (Same as tutorial)
# =============================================================================

@dataclass
class StreetViewImage:
    """Data contract for Street View image with required metadata."""
    image_id: str
    camera_loc: Dict[str, float]
    heading: float
    pitch: float
    fov: float
    depth_map: Optional[str] = None
    capture_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MapillaryClient:
    """Mapillary API client - handles tile fetching and coordinate conversion."""

    GRAPH_API_URL = "https://graph.mapillary.com"
    TILES_URL = "https://tiles.mapillary.com"

    def __init__(self, access_token: str):
        self.access_token = access_token

    def _build_tile_url(self, z: int, x: int, y: int) -> str:
        """Build URL for Mapillary vector tile."""
        return f"{self.TILES_URL}/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={self.access_token}"

    def _tile_to_bbox(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
        """Convert tile coordinates to bounding box."""
        import mercantile
        n = 2 ** zoom
        min_lon = tile_x / n * 360 - 180
        max_lon = (tile_x + 1) / n * 360 - 180
        lat_rad1 = math.asin(math.tanh(math.pi * (1 - 2 * tile_y / n)))
        lat_rad2 = math.asin(math.tanh(math.pi * (1 - 2 * (tile_y + 1) / n)))
        min_lat = math.degrees(lat_rad2)
        max_lat = math.degrees(lat_rad1)
        return min_lon, max_lon, min_lat, max_lat

    def get_tiles_for_bbox(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = 14) -> List[Dict[str, int]]:
        """Get list of tiles covering the bounding box."""
        import mercantile
        tiles = []
        for tile in mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom):
            tiles.append({"z": tile.z, "x": tile.x, "y": tile.y})
        return tiles


print("BLOCK 2: MapillaryClient class defined")
print("Next: Run Block 3")


# =============================================================================
# BLOCK 3: INITIALIZE CLIENT (Same as tutorial)
# =============================================================================

load_dotenv()

access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
if not access_token:
    print("ERROR: MAPILLARY_ACCESS_TOKEN not found in .env file")
    print("Please create a .env file with your Mapillary access token.")
    sys.exit(1)

client = MapillaryClient(access_token)

area = HCMC_AREAS[SELECTED_AREA]
print(f"BLOCK 3: Client initialized for {area['name']}")
print(f"  Bounding box: {area['min_lon']}, {area['min_lat']} -> {area['max_lon']}, {area['max_lat']}")
print("Next: Run Block 4")


# =============================================================================
# BLOCK 4: GET TILE COORDINATES (Same as tutorial)
# =============================================================================

tiles = client.get_tiles_for_bbox(
    area['min_lon'], area['min_lat'],
    area['max_lon'], area['max_lat'],
    zoom=14
)

print(f"BLOCK 4: Tile Coordinates")
print(f"  Tiles needed: {len(tiles)}")

m = folium.Map(location=[(area['min_lat'] + area['max_lat'])/2,
                        (area['min_lon'] + area['max_lon'])/2], zoom_start=15)

for tile in tiles:
    min_lon, max_lon, min_lat, max_lat = client._tile_to_bbox(tile['x'], tile['y'], tile['z'])
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        popup=f"Tile {tile['z']}/{tile['x']}/{tile['y']}",
        color='red',  # Red for stress test mode
        fill=True,
        fillOpacity=0.4,
    ).add_to(m)

m.save(Path("data/debug/stress_test_tiles_small_test.html"))

CURRENT_TILES = tiles
print("Next: Run Block 5")


# =============================================================================
# BLOCK 5: FETCH TILE DATA (Same as tutorial)
# =============================================================================

from vt2geojson.tools import vt_bytes_to_geojson

output_dir = Path("data/raw/tiles") / SELECTED_AREA
output_dir.mkdir(parents=True, exist_ok=True)

tiles_with_data = []
total_images = 0

for tile in CURRENT_TILES:
    try:
        url = client._build_tile_url(tile['z'], tile['x'], tile['y'])
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        tile_data = vt_bytes_to_geojson(response.content, tile['x'], tile['y'], tile['z'], layer="image")
        feature_count = len(tile_data.get("features", []))
        tiles_with_data.append({**tile, "image_count": feature_count})
        total_images += feature_count

        tile_file = output_dir / f"tile_{tile['z']}_{tile['x']}_{tile['y']}.json"
        with open(tile_file, "w") as f:
            json.dump(tile_data, f, indent=2)

        print(f"  Tile {tile['z']}/{tile['x']}/{tile['y']}: {feature_count} images")
    except Exception as e:
        print(f"  Warning: Failed to fetch tile {tile['z']}/{tile['x']}/{tile['y']}: {e}")
        tiles_with_data.append({**tile, "image_count": 0})

print(f"BLOCK 5: Tile Data Fetched")
print(f"  Total images: {total_images}")

CURRENT_TILES_WITH_DATA = tiles_with_data
print("Next: Run Block 6")


# =============================================================================
# BLOCK 6: LOAD ALL IMAGES FROM TILES (Same as tutorial)
# =============================================================================

all_images = []

for tile_file in sorted(output_dir.glob("*.json")):
    with open(tile_file, "r") as f:
        tile_data = json.load(f)

    for feature in tile_data.get("features", []):
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})

        if geometry.get("type") != "Point":
            continue

        coords = geometry.get("coordinates", [0, 0])
        if len(coords) < 2:
            continue

        lon, lat = coords[0], coords[1]

        # Check if within bounding box
        if not (area['min_lon'] <= lon <= area['max_lon'] and area['min_lat'] <= lat <= area['max_lat']):
            continue

        all_images.append({
            "image_id": properties.get("id"),
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": properties,
        })

print(f"BLOCK 6: Images Loaded")
print(f"  Total images: {len(all_images)}")

ALL_IMAGES = all_images
print("Next: Run Block 7")


# =============================================================================
# BLOCK 7: STRESS TEST CONFIGURATION (DIFFERENT FROM TUTORIAL)
# =============================================================================

# =============================================================================
# STRESS TEST MODE: USE ALL IMAGES
# =============================================================================
# Unlike the tutorial which samples 100 images, this stress test processes
# ALL available images. Be prepared for significantly longer runtime!
#
# Runtime Estimates (proportional to dataset size):
# - 100 images: ~30-70 min (GPU) or ~1.5-3 hours (CPU)
# - 500 images: ~2.5-6 hours (GPU) or ~7.5-15 hours (CPU)
# - 1000 images: ~5-12 hours (GPU) or ~15-30 hours (CPU)
# =============================================================================

# NO SAMPLING - Use all images for stress testing
sample_size = len(ALL_IMAGES)  # Use ALL images
test_images = ALL_IMAGES  # No sampling

print("=" * 70)
print("BLOCK 7: STRESS TEST CONFIGURATION")
print("=" * 70)
print(f"  Mode: STRESS TEST (All Images)")
print(f"  Total images: {len(ALL_IMAGES)}")
print(f"  Sample size: {sample_size} (100% of dataset)")
print(f"  Expected runtime: {sample_size // 100 * 30}-{sample_size // 100 * 70} minutes (GPU)")
print(f"                   or {sample_size // 100 * 1.5}-{sample_size // 100 * 3} hours (CPU)")
print("=" * 70)

TEST_IMAGES = test_images
print("Next: Run Block 8")


# =============================================================================
# BLOCK 8: FETCH DETAILED ENTITY DATA (Same as tutorial, with progress bar)
# =============================================================================

print(f"BLOCK 8: Fetching detailed entity data for {len(TEST_IMAGES)} images...")
print("  This will take some time...")

entities = []
batch_size = 100
total_batches = (len(TEST_IMAGES) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(TEST_IMAGES), batch_size), desc="  Fetching batches", unit="batch", total=total_batches):
    batch = TEST_IMAGES[i:i + batch_size]
    image_ids = [str(img["image_id"]) for img in batch]

    try:
        ids_str = ",".join(image_ids)
        fields = "id,camera_type,camera_parameters,make,model,sequence"
        params = {"ids": ids_str, "fields": fields, "access_token": client.access_token}

        response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        # API returns dict with image IDs as keys: {'id1': {...}, 'id2': {...}}
        if isinstance(data, dict):
            entities.extend(list(data.values()))
        elif isinstance(data, list):
            entities.extend(data)

    except Exception as e:
        tqdm.write(f"  Error fetching batch {i // batch_size + 1}: {e}")
        continue

print(f"BLOCK 8: Entity Data Fetched")
print(f"  Entities: {len(entities)}")

ENTITIES = entities
print("Next: Run Block 9")


# =============================================================================
# BLOCK 9: ANALYZE CAMERA PARAMETERS (Same as tutorial)
# =============================================================================

camera_types = []
has_intrinsics = []

for entity in ENTITIES:
    camera_type = entity.get("camera_type")
    if camera_type:
        camera_types.append(camera_type)

    camera_params = entity.get("camera_parameters", [])
    has_focal = camera_params and len(camera_params) >= 1 and camera_params[0] is not None
    has_intrinsics.append(has_focal)

camera_type_counts = dict(Counter(camera_types))
has_intrinsics_count = sum(1 for x in has_intrinsics if x)
perspective_ratio = camera_type_counts.get("perspective", 0) / len(ENTITIES) if ENTITIES else 0

ANALYSIS_RESULTS = {
    "total_images_tested": len(ENTITIES),
    "camera_params": {
        "perspective": camera_type_counts.get("perspective", 0),
        "fisheye": camera_type_counts.get("fisheye", 0),
        "spherical": camera_type_counts.get("spherical", 0),
        "has_valid_intrinsics": has_intrinsics_count,
        "perspective_ratio": perspective_ratio,
    },
}

print("BLOCK 9: Camera Parameter Analysis")
print("  Camera types:")
for camera_type, count in sorted(camera_type_counts.items(), key=lambda x: -x[1]):
    total = len(camera_types)
    pct = (count / total) * 100 if total > 0 else 0
    print(f"    {camera_type}: {count} ({pct:.1f}%)")
print(f"  Has valid intrinsics: {has_intrinsics_count}/{len(ENTITIES)} ({has_intrinsics_count/len(ENTITIES)*100 if ENTITIES else 0:.1f}%)")

print("Next: Run Block 10")


# =============================================================================
# BLOCK 10: CREATE ENTITY MAP (Same as tutorial)
# =============================================================================

print("BLOCK 10: Creating entity visualization map...")

m = folium.Map(location=[10.77, 106.69], zoom_start=14)

entity_dict = {e.get("id"): e for e in entities if isinstance(e, dict)}

for img in TEST_IMAGES:
    img_id = str(img.get("image_id"))
    entity = entity_dict.get(img_id)

    if not entity:
        continue

    camera_type = entity.get("camera_type", "unknown")

    color = "blue"
    if camera_type == "perspective":
        color = "green"
    elif camera_type == "fisheye":
        color = "red"
    elif camera_type == "spherical":
        color = "orange"

    # Extract coordinates from geometry.coordinates [lon, lat]
    geometry = img.get("geometry", {})
    coords = geometry.get("coordinates", [0, 0])
    if len(coords) >= 2:
        lon, lat = coords[0], coords[1]
    else:
        continue

    if lat and lon:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=f"ID: {img_id}<br>Type: {camera_type}",
        ).add_to(m)

Path("data/debug").mkdir(parents=True, exist_ok=True)
m.save(Path("data/debug/stress_test_entities.html"))
# the visualization here is a bit unnecessary?

print("  Map saved to data/debug/stress_test_entities.html")
print("Next: Run Block 10.5: Filter by Camera Type")
print("=" * 70)
print("NOTE: Blocks 11-19 will process PERSPECTIVE cameras only")
print("=" * 70)


# =============================================================================
# BLOCK 10.5: FILTER BY CAMERA TYPE
# =============================================================================
# IMPORTANT: Filter to only perspective cameras before expensive operations
#
# Why filter here:
# - Non-perspective cameras (fisheye, spherical) cannot be processed with
#   the simple geometric projection approach used in this pipeline
# - Fisheye/spherical require complex undistortion that's not implemented
# - Filtering here saves time and money on:
#   * Downloading images we can't process
#   * Running VLM on incompatible images
#   * Computing depth maps that won't be used
#
# From Block 9 analysis:
# - Perspective cameras: ~95% of data (these we keep)
# - Fisheye/Spherical: ~5% of data (these we filter out)
# =============================================================================

print(f"\nBLOCK 10.5: Filter by Camera Type")
print("=" * 70)

# Filter TEST_IMAGES to only include perspective cameras
perspective_images = []
filtered_out_counts = defaultdict(int)

entity_dict = {str(e.get("id")): e for e in entities if isinstance(e, dict)}

for img in TEST_IMAGES:
    img_id = str(img.get("image_id"))
    entity = entity_dict.get(img_id)

    if not entity:
        # No camera data available - skip these
        filtered_out_counts["no_camera_data"] += 1
        continue

    camera_type = entity.get("camera_type", "unknown")

    if camera_type == "perspective":
        perspective_images.append(img)
    else:
        filtered_out_counts[camera_type] += 1

print(f"  Original images: {len(TEST_IMAGES)}")
print(f"  Perspective cameras (kept): {len(perspective_images)}")
print(f"  Filtered out:")
for camera_type, count in sorted(filtered_out_counts.items(), key=lambda x: -x[1]):
    pct = (count / len(TEST_IMAGES)) * 100 if TEST_IMAGES else 0
    print(f"    - {camera_type}: {count} ({pct:.1f}%)")

# Update TEST_IMAGES to only include perspective cameras
TEST_IMAGES = perspective_images

print(f"\n  Updated TEST_IMAGES: {len(TEST_IMAGES)} perspective cameras")
print("=" * 70)
print("Next: Run Block 11 (Download images)")


# =============================================================================
# BLOCK 11: DOWNLOAD SVI IMAGES
# =============================================================================

print(f"BLOCK 11: Downloading {len(TEST_IMAGES)} SVI images...")
print("  This will take some time...")
print("  Supports resume: Will skip already downloaded images")


def load_already_downloaded(output_dir):
    """Load list of already downloaded images from output directory.

    This enables resuming interrupted downloads.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return set()

    # Find all .jpg files and extract image IDs
    downloaded_ids = set()
    for jpg_file in output_dir.glob("*.jpg"):
        # Image ID is the filename without .jpg extension
        downloaded_ids.add(jpg_file.stem)

    return downloaded_ids


def download_image(client, image_id, output_dir, size=1024):
    """Download a single image from Mapillary.

    Returns (image_id, Path) tuple or (image_id, None) on failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{image_id}.jpg"

    # Skip if already downloaded
    if output_path.exists():
        return image_id, output_path

    # Try thumb_1024_url first, then thumb_2048_url
    url = f"https://graph.mapillary.com/{image_id}?fields=thumb_2048_url,thumb_1024_url&access_token={client.access_token}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        image_url = data.get("thumb_1024_url") or data.get("thumb_2048_url")

        if image_url:
            img_response = requests.get(image_url, timeout=60)
            img_response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(img_response.content)

            return image_id, output_path
        else:
            return image_id, None

    except Exception:
        return image_id, None


output_dir = Path("data/raw/images")

# Check for already downloaded images (resume support)
already_downloaded = load_already_downloaded(output_dir)
print(f"  Found {len(already_downloaded)} already downloaded images")

# Filter out already downloaded images
images_to_download = [
    img for img in TEST_IMAGES
    if str(img.get("image_id")) not in already_downloaded
]

print(f"  Need to download {len(images_to_download)} new images")
print(f"  Total after completion: {len(already_downloaded) + len(images_to_download)}")

downloaded = []
failed = []

# Download new images
for img in tqdm(images_to_download, desc="  Downloading images", unit="img"):
    img_id = str(img.get("image_id"))
    result_id, path = download_image(client, img_id, output_dir)

    if path:
        downloaded.append((img_id, path))
    else:
        failed.append(img_id)

# Add already downloaded images to the downloaded list
for existing_id in already_downloaded:
    existing_path = output_dir / f"{existing_id}.jpg"
    if existing_path.exists():
        downloaded.append((existing_id, existing_path))

print(f"\nBLOCK 11: Download complete")
print(f"  Newly downloaded: {len([d for d in downloaded if d[0] not in already_downloaded])}")
print(f"  Already downloaded: {len(already_downloaded)}")
print(f"  Total success: {len(downloaded)}")
print(f"  Failed: {len(failed)}")

if failed:
    print(f"  Failed image IDs saved for retry:")
    for fid in failed[:10]:  # Show first 10
        print(f"    - {fid}")
    if len(failed) > 10:
        print(f"    ... and {len(failed) - 10} more")

DOWNLOADED_IMAGES = downloaded
print("Next: Run Block 12")


# =============================================================================
# BLOCKS 12-19: COMPLETE POI RECONSTRUCTION PIPELINE
# =============================================================================
# STRESS TEST MODE: Processing ALL images instead of 100-sample
# =============================================================================

print("=" * 70)
print("STRESS TEST: Blocks 12-19 - Complete POI Pipeline")
print("=" * 70)
print(f"Processing {len(DOWNLOADED_IMAGES)} downloaded images...")
print("WARNING: This will take a long time with 226K+ images!")
print("=" * 70)


# =============================================================================
# BLOCK 12: VLM VALIDATION EXPERIMENT
# =============================================================================
# TEACHING NOTE: ONE-TIME validation to prove VLM approach works
# For stress test, we skip this and assume VLM works (run tutorial first to validate)
# =============================================================================

@dataclass
class VLMValidationResult:
    """Result from VLM validation experiment."""
    image_id: str
    vietnamese_text_found: bool
    vietnamese_text: List[str]
    signboard_count: int
    detection_confidence: float
    api_response_time_ms: float
    success: bool
    error_message: Optional[str] = None


class VLMClient:
    """Client for VLM APIs (GLM-4V)."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.model = "glm-4.6v"

    def call(self, image_path: Path, prompt: str) -> str:
        """Call VLM API with image and prompt."""
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                    ],
                }
            ],
            "temperature": 0.2,
        }

        max_retries = 3
        base_timeout = 90

        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    tqdm.write(f"    Timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API timeout after {max_retries} attempts")

            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API request failed after {max_retries} attempts")


print(f"\nBLOCK 12: VLM Validation Experiment")
print("  STRESS TEST: Skipping validation (run tutorial first to validate)")
print("  Assuming VLM approach works for all images...")


# =============================================================================
# BLOCK 12.5: SVI SAFETY CHECK (PRE-FILTERING)
# =============================================================================

@dataclass
class SafetyCheckResult:
    """Result from safety check."""
    image_path: str
    is_valid: bool
    skip_reason: Optional[str] = None
    blur_score: Optional[float] = None
    brightness_score: Optional[float] = None
    edge_density: Optional[float] = None


def has_valid_poi_potential(image_path: Path):
    """Quick check if image likely contains POI (signboard/storefront)."""
    if not HAS_CV2:
        return SafetyCheckResult(
            image_path=str(image_path),
            is_valid=True,
            skip_reason="OpenCV not available",
        )

    image = cv2.imread(str(image_path))
    if image is None:
        return SafetyCheckResult(
            image_path=str(image_path),
            is_valid=False,
            skip_reason="Failed to load image",
        )

    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Check 1: Blur detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = float(laplacian_var)

    if blur_score < 100.0:
        return SafetyCheckResult(
            image_path=str(image_path),
            is_valid=False,
            skip_reason=f"Blurry (variance={blur_score:.1f})",
            blur_score=blur_score,
        )

    # Check 2: Brightness check
    brightness = float(np.mean(gray))
    if brightness < 30.0 or brightness > 220.0:
        return SafetyCheckResult(
            image_path=str(image_path),
            is_valid=False,
            skip_reason=f"Poor lighting (brightness={brightness:.1f})",
            blur_score=blur_score,
            brightness_score=brightness,
        )

    # Check 3: Edge density in upper 60%
    upper_region = gray[:int(H * 0.6), :]
    edges = cv2.Canny(upper_region, 50, 150)
    edge_density = float(np.sum(edges > 0) / edges.size)

    if edge_density < 0.1:
        return SafetyCheckResult(
            image_path=str(image_path),
            is_valid=False,
            skip_reason=f"Low edge density (edge_density={edge_density:.3f})",
            blur_score=blur_score,
            brightness_score=brightness,
            edge_density=edge_density,
        )

    return SafetyCheckResult(
        image_path=str(image_path),
        is_valid=True,
        blur_score=blur_score,
        brightness_score=brightness,
        edge_density=edge_density,
    )


def batch_safety_check(images_with_paths):
    """Filter images by safety check before expensive processing."""
    results = []
    skip_reasons = defaultdict(int)
    skipped_images = []
    valid_images = []

    for image_dict, img_path in tqdm(images_with_paths, desc="  Safety check", unit="img"):
        result = has_valid_poi_potential(img_path)
        results.append(result)

        if not result.is_valid:
            skip_reasons[result.skip_reason or "unknown"] += 1
            # Extract coordinates for visualization
            geometry = image_dict.get("geometry", {})
            coords = geometry.get("coordinates", [0, 0])
            if len(coords) >= 2:
                lon, lat = coords[0], coords[1]
            else:
                lon, lat = None, None

            skipped_images.append({
                "image_id": img_path.stem,
                "image_path": str(img_path),
                "skip_reason": result.skip_reason,
                "blur_score": result.blur_score,
                "brightness_score": result.brightness_score,
                "edge_density": result.edge_density,
                "latitude": lat,
                "longitude": lon,
            })
        else:
            valid_images.append((image_dict, img_path))

    valid_paths = [img_path for _, img_path in valid_images]
    valid_image_dicts = [img_dict for img_dict, _ in valid_images]

    skip_report = {
        "total_images": len(images_with_paths),
        "valid_images": len(valid_paths),
        "filtered_images": len(images_with_paths) - len(valid_paths),
        "filter_rate": (len(images_with_paths) - len(valid_paths)) / len(images_with_paths) if images_with_paths else 0,
        "skip_reasons": dict(skip_reasons),
        "skipped_images": skipped_images,
    }

    report_path = Path("data/debug") / "stress_test_safety_check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(skip_report, f, indent=2, ensure_ascii=False)

    print(f"  Safety check report: {report_path}")
    print(f"  Valid: {len(valid_paths)}/{len(images_with_paths)}")
    print(f"  Filtered: {len(images_with_paths) - len(valid_paths)} images")

    return valid_paths, valid_image_dicts, skip_report


print(f"\nBLOCK 12.5: SVI Safety Check")

# Prepare downloaded images dict for mapping
downloaded_paths = {img_id: Path(path) for img_id, path in DOWNLOADED_IMAGES}

# Prepare images with paths
downloaded_images = [
    (img, downloaded_paths.get(str(img.get("image_id"))))
    for img in TEST_IMAGES
    if str(img.get("image_id")) in downloaded_paths
]

print(f"  Checking {len(downloaded_images)} images...")

valid_paths, valid_image_dicts, skip_report = batch_safety_check(downloaded_images)

print(f"  Valid images: {len(valid_paths)}/{len(downloaded_images)}")
print(f"  Filter rate: {skip_report['filter_rate']*100:.1f}%")

# Store valid images for downstream blocks
VALID_IMAGES = valid_image_dicts
VALID_IMAGE_PATHS = valid_paths


# =============================================================================
# BLOCK 13: CAMERA PARAMETER ANALYSIS
# =============================================================================

def fetch_camera_parameters_batch(client, image_ids):
    """Batch fetch camera parameters from Mapillary API."""
    batch_size = 100
    camera_data = {}

    for i in tqdm(range(0, len(image_ids), batch_size), desc="  Fetching camera params", unit="batch"):
        batch = image_ids[i:i + batch_size]

        try:
            ids_str = ",".join(str(img_id) for img_id in batch)
            fields = "id,camera_type,camera_parameters,width,height"
            params = {"ids": ids_str, "fields": fields, "access_token": client.access_token}

            response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            entities = list(data.values()) if isinstance(data, dict) else data

            for entity in entities:
                image_id = entity.get("id")
                if image_id:
                    camera_data[image_id] = entity

        except Exception as e:
            tqdm.write(f"    Warning: Failed to fetch camera batch: {e}")

    return camera_data


def categorize_camera(camera_data):
    """Categorize camera based on FOV calculated from focal length."""
    camera_type = camera_data.get("camera_type", "perspective")

    if camera_type in ["fisheye", "spherical"]:
        return {"category": None, "fov": None, "camera_type": camera_type}

    params = camera_data.get("camera_parameters", [])
    if params and len(params) >= 1 and params[0]:
        focal = params[0]
        width = camera_data.get("width", 1024)
        fov_rad = 2 * math.atan(width / (2 * focal)) if focal > 0 else math.radians(60)
        fov = math.degrees(fov_rad)
    else:
        fov = 60.0

    if fov < 60:
        category = "Narrow FOV"
    elif fov <= 90:
        category = "Normal FOV"
    else:
        category = "Wide FOV"

    return {"category": category, "fov": fov, "camera_type": camera_type}


def analyze_camera_categories(images, camera_data_map):
    """Calculate percentages for each camera category."""
    categories = []

    for img in images:
        image_id = str(img.get("image_id"))
        cam_data = camera_data_map.get(image_id, {})
        cat = categorize_camera(cam_data)
        if cat["category"]:
            categories.append(cat["category"])

    category_counts = dict(Counter(categories))
    total = sum(category_counts.values())

    percentages = {
        k: (v / total * 100) if total > 0 else 0
        for k, v in category_counts.items()
    }

    return category_counts, percentages


print(f"\nBLOCK 13: Camera Parameter Analysis")

valid_image_ids = [img.get("image_id") for img in VALID_IMAGES]
camera_data_map = fetch_camera_parameters_batch(client, valid_image_ids)

category_counts, percentages = analyze_camera_categories(VALID_IMAGES, camera_data_map)

print(f"  Camera categories:")
for cat, pct in percentages.items():
    count = category_counts.get(cat, 0)
    print(f"    {cat}: {count} ({pct:.1f}%)")


# =============================================================================
# BLOCK 14: ML MONOCULAR DEPTH ESTIMATION
# =============================================================================

def estimate_depth_ml(image_path, model_name="dpt"):
    """Use ML for monocular depth prediction."""
    if not HAS_TRANSFORMERS:
        return None

    try:
        if model_name == "dpt":
            estimator = pipeline("depth-estimation", model="Intel/dpt-large")

        if HAS_PIL:
            image = Image.open(image_path)
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        result = estimator(image)

        if isinstance(result, dict):
            depth_map = result.get("depth")
        elif hasattr(result, "depth"):
            depth_map = result.depth
        else:
            depth_map = result

        return np.array(depth_map)

    except Exception as e:
        tqdm.write(f"    Warning: ML depth estimation failed: {e}")
        return None


def batch_estimate_depth(valid_image_paths, output_dir, model_name="dpt"):
    """Process all valid images through depth model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_maps = {}

    print(f"  Estimating depth for {len(valid_image_paths)} images...")

    for img_path in tqdm(valid_image_paths, desc="  Depth estimation", unit="img"):
        image_id = img_path.stem

        depth_map = estimate_depth_ml(img_path, model_name)

        if depth_map is not None:
            depth_path = output_path / f"{image_id}_depth.npy"
            np.save(depth_path, depth_map)
            depth_maps[image_id] = depth_path

    print(f"  Generated {len(depth_maps)} depth maps")

    return depth_maps


print(f"\nBLOCK 14: ML Monocular Depth Estimation")

if HAS_TRANSFORMERS and valid_paths:
    # Process all valid images (or limit for testing)
    depth_maps = batch_estimate_depth(
        valid_paths,  # ALL valid images for stress test
        "data/interim/depth_maps",
        model_name="dpt"
    )
else:
    print(f"  Skipped: Install transformers with 'uv add transformers'")
    depth_maps = {}


# =============================================================================
# BLOCK 15: DEPTH MAP VISUALIZATION
# =============================================================================

def visualize_depth_map(image, depth_map, output_path, title="Depth Map"):
    """Generate side-by-side visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="jet")
    axes[1].set_title(f"Depth (min={depth_map.min():.2f}, max={depth_map.max():.2f})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].hist(depth_map.flatten(), bins=50, edgecolor="black")
    axes[2].set_title("Depth Distribution")
    axes[2].set_xlabel("Depth")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


print(f"\nBLOCK 15: Depth Map Visualization")
print("  STRESS TEST: Skipping visualization for all images")
print("  Run tutorial Blocks 14-15 to see depth map examples")


# =============================================================================
# BLOCK 16: GEOLOCATOR IMPLEMENTATION
# =============================================================================

def pixel_to_ray(u, v, W, H, fov=60.0, heading=0.0, pitch=0.0):
    """Convert pixel coordinates to 3D ray direction."""
    x_norm = (u - W / 2) / (W / 2)
    y_norm = (v - H / 2) / (H / 2)

    fov_rad = math.radians(fov)
    focal = 1.0 / math.tan(fov_rad / 2)

    ray_camera = np.array([x_norm, y_norm, focal])
    ray_camera = ray_camera / np.linalg.norm(ray_camera)

    pitch_rad = math.radians(pitch)
    rotation_pitch = np.array([
        [1, 0, 0],
        [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
        [0, math.sin(pitch_rad), math.cos(pitch_rad)]
    ])

    heading_rad = math.radians(heading - 90)
    rotation_heading = np.array([
        [math.cos(heading_rad), -math.sin(heading_rad), 0],
        [math.sin(heading_rad), math.cos(heading_rad), 0],
        [0, 0, 1]
    ])

    ray_world = rotation_heading @ (rotation_pitch @ ray_camera)

    return ray_world


def offset_from_camera(depth, ray_direction, camera_lat, camera_lon):
    """Compute GPS offset from camera using Haversine formula."""
    dx, dy, dz = ray_direction

    horizontal_distance = depth * math.sqrt(dx**2 + dz**2)

    bearing = math.degrees(math.atan2(dx, dz))
    if bearing < 0:
        bearing += 360

    bearing_rad = math.radians(bearing)
    R = 6371000

    lat_offset_rad = (horizontal_distance * math.cos(bearing_rad)) / R
    lon_offset_rad = (horizontal_distance * math.sin(bearing_rad)) / (R * math.cos(math.radians(camera_lat)))

    target_lat = camera_lat + math.degrees(lat_offset_rad)
    target_lon = camera_lon + math.degrees(lon_offset_rad)

    return target_lat, target_lon


def pixel_to_gps(u, v, depth, W, H, camera_lat, camera_lon, heading=0.0, pitch=0.0, fov=60.0):
    """Complete pipeline: convert pixel coordinates to GPS."""
    ray = pixel_to_ray(u, v, W, H, fov, heading, pitch)
    target_lat, target_lon = offset_from_camera(depth, ray, camera_lat, camera_lon)
    return round(target_lat, 6), round(target_lon, 6)


class Geolocator:
    """High-level interface for pixel-to-GPS conversion."""

    def __init__(self, camera_data):
        self.camera_data = camera_data
        self.focal = None

        params = camera_data.get("camera_parameters", [])
        if params and len(params) >= 1 and params[0] is not None:
            self.focal = params[0]

    def pixel_to_gps(self, u, v, depth, heading, pitch=0.0, fov=60.0):
        # Extract coordinates from geometry
        geometry = self.camera_data.get("geometry", {})
        coords = geometry.get("coordinates", [0, 0])
        if len(coords) >= 2:
            camera_lon, camera_lat = float(coords[0]), float(coords[1])
        else:
            camera_lon, camera_lat = 0.0, 0.0

        W = self.camera_data.get("width", 1024)
        H = self.camera_data.get("height", 1024)

        if self.focal:
            fov_rad = 2 * math.atan(W / (2 * self.focal)) if self.focal > 0 else math.radians(fov)
            actual_fov = math.degrees(fov_rad)
        else:
            actual_fov = fov

        return pixel_to_gps(u, v, depth, W, H, camera_lat, camera_lon, heading, pitch, actual_fov)


print(f"\nBLOCK 16: Geolocator Implementation")
print("  Geolocator class defined for pixel-to-GPS conversion")


# =============================================================================
# BLOCK 17: VLM SIGNBOARD DETECTION AND POI EXTRACTION
# =============================================================================

@dataclass
class POIDetection:
    """Detection of a signboard with POI data."""
    bbox: Tuple[int, int, int, int]
    center_pixel: Tuple[int, int]
    poi_data: Dict[str, Any]
    confidence: float


def detect_and_extract_pois(vlm_client, image_path: Path):
    """Detect signboards and extract POI data in one call."""
    prompt = """Analyze this street view image and identify all storefront/business signs.

For each signboard you find:
1. Extract the bounding box as [x1, y1, x2, y2] (pixel coordinates)
2. Extract the center pixel as [u, v]
3. Extract structured POI data:
   - poi_name_vietnamese: Vietnamese name on sign
   - poi_name_english: English translation
   - business_category: Restaurant, Retail, Service, etc.
   - sub_category: More specific type
   - address_text: Address if visible

Return your answer as a JSON array of objects, each with:
{
  "bbox": [x1, y1, x2, y2],
  "center_pixel": [u, v],
  "poi_data": { ... },
  "confidence": 0.0-1.0
}

If no signboards are found, return an empty array [].

IMPORTANT: Return ONLY valid JSON, no other text."""

    try:
        response = vlm_client.call(image_path, prompt)

        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]

        data = json.loads(response)

        if not isinstance(data, list):
            return []

        detections = []
        for item in data:
            try:
                detections.append(POIDetection(
                    bbox=tuple(item.get("bbox", [0, 0, 0, 0])),
                    center_pixel=tuple(item.get("center_pixel", [0, 0])),
                    poi_data=item.get("poi_data", {}),
                    confidence=item.get("confidence", 0.0),
                ))
            except (ValueError, KeyError, TypeError):
                continue

        return detections

    except Exception as e:
        tqdm.write(f"    Warning: VLM detection failed: {e}")
        return []


print(f"\nBLOCK 17: VLM Signboard Detection + POI Extraction")

glm_key = os.getenv("GLM_KEY")

if glm_key and VALID_IMAGE_PATHS:
    print(f"  Running VLM detection on {len(VALID_IMAGE_PATHS)} images...")
    print(f"  STRESS TEST: Processing ALL images - this will take a long time!")

    vlm_client = VLMClient(api_key=glm_key)

    vlm_results = {}
    for img_path in tqdm(VALID_IMAGE_PATHS, desc="  VLM detection", unit="img"):
        image_id = img_path.stem

        try:
            detections = detect_and_extract_pois(vlm_client, img_path)
            vlm_results[image_id] = detections

        except Exception as e:
            tqdm.write(f"    Error processing {image_id}: {e}")
            vlm_results[image_id] = []

    # Save results
    results_path = Path("data/interim/stress_test_vlm_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_results = {}
    for img_id, detections in vlm_results.items():
        serializable_results[img_id] = [
            {
                "bbox": det.bbox,
                "center_pixel": det.center_pixel,
                "poi_data": det.poi_data,
                "confidence": det.confidence,
            }
            for det in detections
        ]

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")
    total_detections = sum(len(d) for d in vlm_results.values())
    print(f"  Total detections: {total_detections}")

else:
    print(f"  Skipped: GLM_KEY not found in .env file")
    vlm_results = {}


# =============================================================================
# BLOCK 18: ASSIGN POI INFORMATION TO CORRESPONDING DATA POINT
# =============================================================================

def assign_poi_to_location(vlm_detections, geolocator, depth_map, camera_data):
    """Combine POI data with GPS coordinates."""
    heading = camera_data.get("compass_angle", 0.0)
    pitch = 0.0

    complete_pois = []

    for detection in vlm_detections:
        u, v = detection.center_pixel
        poi_data = detection.poi_data

        if depth_map is not None:
            depth_relative = float(depth_map[int(v), int(u)])
            depth_meters = 2.0 + (depth_relative / 255.0) * 98.0
        else:
            depth_meters = 15.0

        lat, lon = geolocator.pixel_to_gps(u, v, depth_meters, heading, pitch)

        complete_poi = {
            "poi_id": f"poi_{uuid.uuid4().hex[:8]}",
            "latitude": lat,
            "longitude": lon,
            "depth_meters": depth_meters,
            "depth_relative": depth_map[int(v), int(u)] if depth_map is not None else None,
            "center_pixel": [u, v],
            "bbox": list(detection.bbox),
            "confidence": detection.confidence,
            **poi_data,
        }

        complete_pois.append(complete_poi)

    return complete_pois


print(f"\nBLOCK 18: Assign POI to Location")

if vlm_results and depth_maps:
    print(f"  Processing {len(vlm_results)} images with detections...")

    all_pois = []

    for image_id, detections_data in tqdm(vlm_results.items(), desc="  Assigning GPS", unit="img"):
        if not detections_data:
            continue

        # Get image path
        image_path = downloaded_paths.get(image_id)
        if not image_path or not image_path.exists():
            continue

        # Get depth map
        depth_path = depth_maps.get(image_id)
        if depth_path:
            depth_map = np.load(depth_path)
        else:
            depth_map = None

        # Get camera data
        camera_data = None
        for img in VALID_IMAGES:
            if str(img.get("image_id")) == image_id:
                camera_data = img
                break

        if not camera_data:
            continue

        # Convert to POIDetection objects
        detections = []
        for det_data in detections_data:
            detections.append(POIDetection(
                bbox=tuple(det_data["bbox"]),
                center_pixel=tuple(det_data["center_pixel"]),
                poi_data=det_data["poi_data"],
                confidence=det_data["confidence"],
            ))

        # Create geolocator
        geolocator = Geolocator(camera_data)

        # Assign POI to location
        complete_pois = assign_poi_to_location(detections, geolocator, depth_map, camera_data)
        all_pois.extend(complete_pois)

    print(f"  Total POIs created: {len(all_pois)}")

    # Save all POIs
    pois_path = Path("data/processed/stress_test_pois.json")
    pois_path.parent.mkdir(parents=True, exist_ok=True)

    with open(pois_path, "w", encoding="utf-8") as f:
        json.dump(all_pois, f, indent=2, ensure_ascii=False)

    print(f"  Saved to: {pois_path}")

else:
    print(f"  Skipped: No VLM results or depth maps available")


# =============================================================================
# BLOCK 19: END-TO-END PIPELINE ORCHESTRATION
# =============================================================================

def save_pois_geojson(pois, output_path):
    """Save POI results as GeoJSON."""
    features = []

    for poi in pois:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [poi["longitude"], poi["latitude"]],
            },
            "properties": {
                "poi_id": poi.get("poi_id"),
                "poi_name_vietnamese": poi.get("poi_name_vietnamese"),
                "poi_name_english": poi.get("poi_name_english"),
                "business_category": poi.get("business_category"),
                "sub_category": poi.get("sub_category"),
                "address_text": poi.get("address_text"),
                "depth_meters": poi.get("depth_meters"),
                "confidence": poi.get("confidence"),
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"  Saved {len(features)} POIs to {output_path}")


print(f"\nBLOCK 19: End-to-End Pipeline Orchestration")

# Load POIs from Block 18 if available
pois_path = Path("data/processed/stress_test_pois.json")
if pois_path.exists():
    with open(pois_path, "r", encoding="utf-8") as f:
        all_pois = json.load(f)

    # Save as GeoJSON
    geojson_path = Path("data/processed/stress_test_pois.geojson")
    save_pois_geojson(all_pois, geojson_path)

    print(f"\n{'='*70}")
    print(f"STRESS TEST COMPLETE!")
    print(f"{'='*70}")
    print(f"Total POIs reconstructed: {len(all_pois)}")
    print(f"Output files:")
    print(f"  - JSON: {pois_path}")
    print(f"  - GeoJSON: {geojson_path}")
    print(f"{'='*70}")
else:
    print(f"  No POIs found. Run Blocks 17-18 first.")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"STRESS TEST: Blocks 0-19 COMPLETE!")
print(f"{'='*70}")
print(f"""
Stress Test Summary:
- Processed {len(TEST_IMAGES)} images (vs 100 in tutorial)
- Valid images after safety check: {len(VALID_IMAGES) if VALID_IMAGES else 0}
- Total POIs reconstructed: Check output files

Output Files:
- data/processed/stress_test_pois.json - All POIs with GPS
- data/processed/stress_test_pois.geojson - GeoJSON format
- data/debug/stress_test_safety_check_report.json - Safety check results
- data/interim/stress_test_vlm_results.json - VLM detection results

Key Differences from Tutorial:
- Uses ALL images instead of 100-sample
- Progress bars with tqdm for long operations
- Skips visualization for all images (too many)
- Results prefixed with "stress_test_"

Next Steps:
1. Review the POI results in data/processed/
2. Visualize on a map using the GeoJSON file
3. Analyze the quality of reconstructed POIs
""")
print(f"{'='*70}")
