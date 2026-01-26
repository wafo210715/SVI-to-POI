#!/usr/bin/env python3
"""
HCMC POI Reconstruction - Pipeline Tutorial
===========================================

SHIFT+ENTER to run each block sequentially.

This tutorial implements the complete POI reconstruction pipeline:
- Blocks 0-10: Data acquisition and camera analysis
- Blocks 11-19: POI detection, geolocation, and extraction

SHIFT OF MIND (Key Learning):
==============================
Original Plan: Compare three depth approaches - Mapillary SfM, Sequential Stereo, ML
Reality After Analysis:
- SfM cluster is "杀鸡用宰牛刀" (overkill for single POI)
- Sequential stereo requires image pairing + feature matching - too complex
- Simplified approach wins: Detect POI -> Sample depth -> Project to GPS
- 95% perspective cameras with 69.5% Normal FOV -> geometric projection is reliable
"""

# =============================================================================
# BLOCK 0: IMPORTS
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

print("BLOCK 0: Libraries imported")
print("Next: Run Block 1")


# =============================================================================
# BLOCK 1: DEFINE HCMC BOUNDING BOX AREAS
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
        "name": "Small Test Area (Ben Thanh)",
    },
}

print("BLOCK 1: HCMC Bounding Box Areas Defined")
for key, area in HCMC_AREAS.items():
    print(f"  {key}: {area['name']}")

m = folium.Map(location=[10.77, 106.69], zoom_start=12)
for key, area in HCMC_AREAS.items():
    folium.Rectangle(
        bounds=[[area['min_lat'], area['min_lon']], [area['max_lat'], area['max_lon']]],
        popup=area['name'],
        color='blue',
        fill=True,
        fillOpacity=0.2,
    ).add_to(m)

Path("data/debug").mkdir(parents=True, exist_ok=True)
m.save(Path("data/debug/tutorial_hcmc_areas.html"))

SELECTED_AREA = "small_test"
print(f"Selected: {SELECTED_AREA}")
print("Next: Run Block 2")


# =============================================================================
# BLOCK 2: DEFINE MAPILLARY CLIENT CLASS
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
        return f"{self.TILES_URL}/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={self.access_token}"

    def _tile_to_bbox(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
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
        import mercantile
        tiles = []
        for tile in mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom):
            tiles.append({"z": tile.z, "x": tile.x, "y": tile.y})
        return tiles

    def fetch_image_entity(self, image_id: str) -> Dict[str, Any]:
        fields = [
            "id", "geometry", "camera_parameters", "camera_type",
            "compass_angle", "width", "height", "sequence"
        ]
        fields_str = ",".join(fields)
        params = {"fields": fields_str, "access_token": self.access_token}
        url = f"{self.GRAPH_API_URL}/images/{image_id}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

print("BLOCK 2: MapillaryClient class defined")
print("Next: Run Block 3")


# =============================================================================
# BLOCK 3: INITIALIZE MAPILLARY CLIENT
# =============================================================================

load_dotenv()
access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")

if not access_token:
    print("ERROR: Set MAPILLARY_ACCESS_TOKEN in .env file")
    sys.exit(1)

client = MapillaryClient(access_token=access_token)
CURRENT_AREA = HCMC_AREAS[SELECTED_AREA]

print(f"BLOCK 3: Client initialized")
print(f"  Area: {CURRENT_AREA['name']}")
print("Next: Run Block 4")


# =============================================================================
# BLOCK 4: GET TILE COORDINATES
# =============================================================================

tiles = client.get_tiles_for_bbox(
    CURRENT_AREA['min_lon'], CURRENT_AREA['min_lat'],
    CURRENT_AREA['max_lon'], CURRENT_AREA['max_lat'],
    zoom=14
)

print(f"BLOCK 4: Tile Coordinates")
print(f"  Tiles needed: {len(tiles)}")

m = folium.Map(location=[(CURRENT_AREA['min_lat'] + CURRENT_AREA['max_lat'])/2,
                        (CURRENT_AREA['min_lon'] + CURRENT_AREA['max_lon'])/2], zoom_start=15)

for tile in tiles:
    min_lon, max_lon, min_lat, max_lat = client._tile_to_bbox(tile['x'], tile['y'], tile['z'])
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        popup=f"Tile {tile['z']}/{tile['x']}/{tile['y']}",
        color='blue',
        fill=True,
        fillOpacity=0.4,
    ).add_to(m)

m.save(Path("data/debug/tutorial_tiles_small_test.html"))

CURRENT_TILES = tiles
print("Next: Run Block 5")


# =============================================================================
# BLOCK 5: FETCH TILE DATA
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
    except Exception as e:
        tiles_with_data.append({**tile, "image_count": 0})

print(f"BLOCK 5: Tile Data Fetched")
print(f"  Total images: {total_images}")

CURRENT_TILES_WITH_DATA = tiles_with_data
print("Next: Run Block 6")


# =============================================================================
# BLOCK 6: LOAD ALL IMAGES FROM TILES
# =============================================================================

all_images = []

for tile_file in sorted(output_dir.glob("*.json")):
    with open(tile_file, "r") as f:
        tile_data = json.load(f)

    for feature in tile_data.get("features", []):
        props = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        all_images.append({
            "image_id": props.get("id"),
            "camera_loc": {
                "lon": geometry.get("coordinates", [0, 0])[0],
                "lat": geometry.get("coordinates", [0, 0])[1],
            },
            "heading": props.get("compass_angle", 0),
            "is_pano": props.get("is_pano", False),
            "sequence_id": props.get("sequence_id"),
        })

print(f"BLOCK 6: Images Loaded")
print(f"  Total images: {len(all_images)}")

ALL_IMAGES = all_images
print("Next: Run Block 7")


# =============================================================================
# BLOCK 7: SAMPLE IMAGES FOR ANALYSIS
# =============================================================================

sample_size = min(100, len(ALL_IMAGES))
random.seed(42)
sampled_images = random.sample(ALL_IMAGES, sample_size)

print(f"BLOCK 7: Images Sampled")
print(f"  Sampled: {len(sampled_images)} from {len(ALL_IMAGES)}")

SAMPLED_IMAGES = sampled_images
print("Next: Run Block 8")


# =============================================================================
# BLOCK 8: FETCH DETAILED ENTITY DATA
# =============================================================================

entities = []
batch_size = 100

for i in range(0, len(SAMPLED_IMAGES), batch_size):
    batch = SAMPLED_IMAGES[i:i + batch_size]
    image_ids = [str(img["image_id"]) for img in batch]

    try:
        ids_str = ",".join(image_ids)
        fields = "id,camera_type,camera_parameters,make,model,sequence"
        params = {"ids": ids_str, "fields": fields, "access_token": client.access_token}

        response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            entities.extend(list(data.values()))
        elif isinstance(data, list):
            entities.extend(data)
    except Exception:
        pass

print(f"BLOCK 8: Entity Data Fetched")
print(f"  Entities: {len(entities)}")

ENTITIES = entities
print("Next: Run Block 9")


# =============================================================================
# BLOCK 9: ANALYZE CAMERA PARAMETERS
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
    },
}

print(f"BLOCK 9: Camera Analysis")
print(f"  Camera types: {camera_type_counts}")
print(f"  With intrinsics: {has_intrinsics_count}/{len(ENTITIES)}")
print("Next: Run Block 10")


# =============================================================================
# BLOCK 10: SAVE AND VISUALIZE RESULTS
# =============================================================================

Path("data/debug").mkdir(parents=True, exist_ok=True)
with open(Path("data/debug/tutorial_depth_analysis_summary.json"), "w") as f:
    json.dump(ANALYSIS_RESULTS, f, indent=2)

print(f"BLOCK 10: Results Saved")
print("\nTUTORIAL PART 1 COMPLETE (Blocks 0-10)")
print("=" * 50)
print("Next: Blocks 11-19 for complete POI pipeline")
print("=" * 50)


# =============================================================================
# =============================================================================
# BLOCKS 11-19: COMPLETE POI RECONSTRUCTION PIPELINE
# =============================================================================
# SHIFT OF MIND (Key Learning):
# ===============================
# Original Plan: Compare three depth approaches - Mapillary SfM, Sequential Stereo, ML
# Reality After Analysis:
# - SfM cluster is overkill for single POI
# - Sequential stereo requires image pairing + feature matching - too complex
# - Simplified approach wins: Detect POI -> Sample depth -> Project to GPS
# - 95% perspective cameras with 69.5% Normal FOV -> geometric projection is reliable
# =============================================================================


# =============================================================================
# BLOCK 11: DOWNLOAD SVI IMAGES LOCALLY
# =============================================================================

def download_svi_images(client, image_list, output_dir, max_size=1024):
    """Download street view images from Mapillary to local disk.

    Why Download First:
    - VLM APIs require image URLs or base64-encoded images
    - Downloading locally enables faster batch processing
    - Can cache images to avoid re-downloading
    - VLM can read local files directly
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    downloaded_paths = {}
    failed_downloads = []
    batch_size = 100

    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        image_ids = [str(img["image_id"]) for img in batch]

        try:
            ids_str = ",".join(image_ids)
            fields = "id,thumb_1024_url,thumb_2048_url,thumb_original_url,width,height"
            params = {"ids": ids_str, "fields": fields, "access_token": client.access_token}

            response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            entities = list(data.values()) if isinstance(data, dict) else data

            for entity in entities:
                image_id = entity.get("id")
                if not image_id:
                    continue

                # Choose appropriate URL
                if max_size <= 1024:
                    url = entity.get("thumb_1024_url") or entity.get("thumb_2048_url")
                else:
                    url = entity.get("thumb_2048_url") or entity.get("thumb_original_url")

                if not url:
                    failed_downloads.append(image_id)
                    continue

                # Download image
                try:
                    img_response = requests.get(url, timeout=30)
                    img_response.raise_for_status()

                    img_path = output_path / f"{image_id}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(img_response.content)

                    # Optional resize
                    if HAS_PIL and max_size < 2048:
                        try:
                            img = Image.open(img_path)
                            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                            img.save(img_path, "JPEG", quality=85)
                        except Exception:
                            pass

                    downloaded_paths[image_id] = img_path

                except Exception:
                    failed_downloads.append(image_id)

        except Exception:
            pass

    # Generate report
    report = {
        "total_images": len(image_list),
        "downloaded": len(downloaded_paths),
        "failed": len(failed_downloads),
        "success_rate": len(downloaded_paths) / len(image_list) if image_list else 0,
    }

    report_path = Path("data/debug") / "download_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  Download report: {report_path}")
    print(f"  Downloaded: {len(downloaded_paths)}/{len(image_list)}")

    return downloaded_paths, failed_downloads


print(f"\nBLOCK 11: Download SVI Images Locally")
print(f"  Downloading {len(SAMPLED_IMAGES)} images...")

downloaded_paths, failed_downloads = download_svi_images(
    client, SAMPLED_IMAGES, "data/raw/images", max_size=1024
)

print(f"  Downloaded: {len(downloaded_paths)}/{len(SAMPLED_IMAGES)}")
print("Next: Run Block 12 (VLM Validation)")


# =============================================================================
# BLOCK 12: VLM VALIDATION EXPERIMENT
# =============================================================================
# TEACHING NOTE: This is a ONE-TIME validation step to prove the VLM approach works.
#
# Why this block exists:
# - Before building the full pipeline, we need to verify GLM-4V can handle Vietnamese signboards
# - This is a "proof of concept" - if the VLM fails here, we stop and reconsider the approach
# - Once validation passes, you don't need to run this block again
#
# What it tests:
# 1. Can the VLM read Vietnamese text? (OCR accuracy)
# 2. Can the VLM detect storefront signs in street view context? (Detection recall)
# 3. What's the API response format and reliability? (Technical feasibility)
# 4. How much does it cost per image? (Budget planning)
#
# Output: vlm_validation_report.json with recommendation: PROCEED or REVIEW_NEEDED
#
# Contrast with Block 12.5:
# - Block 12: ONE-TIME validation (run once, then skip)
# - Block 12.5: EVERY-TIME filtering (run on every new batch of images)
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
        """Call VLM API with image and prompt.

        Includes retry logic for timeout errors.
        """
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

        # Retry logic with exponential backoff
        max_retries = 3
        base_timeout = 90  # Increased from 60 to 90 seconds

        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)  # 90, 180, 360 seconds
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"    Timeout on attempt {attempt + 1}/{max_retries}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API timeout after {max_retries} attempts: {e}")

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    Request failed on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"    Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"API request failed after {max_retries} attempts: {e}")


def run_vlm_validation_experiment(sample_image_paths, vlm_client, n=100):
    """Test VLM on sample images to validate capabilities."""
    results = []
    test_images = sample_image_paths[:n]

    print(f"Running VLM validation on {len(test_images)} images...")

    for img_path in test_images:
        start_time = time.time()

        try:
            prompt_ocr = (
                "List all Vietnamese text visible in this street view image. "
                "Return your answer as a JSON object with keys: "
                "'vietnamese_text_found' (boolean), 'texts' (list of strings)."
            )

            response = vlm_client.call(img_path, prompt_ocr)
            response_time = (time.time() - start_time) * 1000

            try:
                response_data = json.loads(response) if isinstance(response, str) else response
                vietnamese_found = response_data.get("vietnamese_text_found", False)
                texts = response_data.get("texts", [])
            except (json.JSONDecodeError, TypeError, KeyError):
                vietnamese_found = False
                texts = []

            prompt_detection = (
                "Identify all storefront/business signs in this street view image. "
                "Return your answer as a JSON object with keys: "
                "'signboard_count' (number), 'has_signboard' (boolean)."
            )

            response_det = vlm_client.call(img_path, prompt_detection)

            try:
                det_data = json.loads(response_det) if isinstance(response_det, str) else response_det
                signboard_count = det_data.get("signboard_count", 0)
                has_signboard = det_data.get("has_signboard", False)
            except (json.JSONDecodeError, TypeError, KeyError):
                signboard_count = 0
                has_signboard = False

            results.append(VLMValidationResult(
                image_id=img_path.stem,
                vietnamese_text_found=vietnamese_found,
                vietnamese_text=texts,
                signboard_count=signboard_count,
                detection_confidence=1.0 if has_signboard else 0.0,
                api_response_time_ms=response_time,
                success=True,
            ))

        except Exception as e:
            results.append(VLMValidationResult(
                image_id=img_path.stem,
                vietnamese_text_found=False,
                vietnamese_text=[],
                signboard_count=0,
                detection_confidence=0.0,
                api_response_time_ms=0.0,
                success=False,
                error_message=str(e),
            ))

    # Calculate metrics
    successful = [r for r in results if r.success]
    vietnamese_found_count = sum(1 for r in successful if r.vietnamese_text_found)
    signboard_found_count = sum(1 for r in successful if r.signboard_count > 0)

    avg_response_time = np.mean([r.api_response_time_ms for r in successful]) if successful else 0

    validation_report = {
        "total_tested": len(test_images),
        "successful": len(successful),
        "vietnamese_text_found_count": vietnamese_found_count,
        "vietnamese_text_accuracy": vietnamese_found_count / len(successful) if successful else 0,
        "signboard_detection_count": signboard_found_count,
        "signboard_detection_rate": signboard_found_count / len(successful) if successful else 0,
        "avg_response_time_ms": avg_response_time,
        "recommendation": "PROCEED" if (
            len(successful) >= len(test_images) * 0.8 and
            vietnamese_found_count >= len(successful) * 0.5
        ) else "REVIEW_NEEDED",
    }

    # Save report
    report_path = Path("data/debug") / "vlm_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "report": validation_report,
            "results": [asdict(r) for r in results],
        }, f, indent=2, ensure_ascii=False)

    print(f"VLM validation report: {report_path}")
    print(f"  Recommendation: {validation_report['recommendation']}")

    return validation_report, results


print(f"\nBLOCK 12: VLM Validation Experiment")
print(f"  CRITICAL: Testing VLM on {min(10, len(downloaded_paths))} images")

# Run validation if we have downloaded images and GLM_KEY
if downloaded_paths and len(downloaded_paths) > 0:
    glm_key = os.getenv("GLM_KEY")
    if glm_key:
        image_paths = list(downloaded_paths.values())[:10]
        vlm_client = VLMClient(api_key=glm_key)
        validation_report, results = run_vlm_validation_experiment(image_paths, vlm_client, n=10)
    else:
        print("  (Skipped: GLM_KEY not found in .env)")
else:
    print(f"  (Skipped: No images downloaded ({len(downloaded_paths)} images)")

print("Next: Run Block 12.5 (Safety Check)")


# =============================================================================
# BLOCK 12.5: SVI SAFETY CHECK (PRE-FILTERING)
# =============================================================================
# TEACHING NOTE: This runs EVERY TIME you process new images.
#
# Why this block exists:
# - Not all SVI images contain valid POI data (blurry, indoor, poor lighting, etc.)
# - Processing bad images through depth estimation + VLM is expensive and wasteful
# - Pre-filtering saves time and API costs by removing obvious failures upfront
#
# What it checks:
# 1. Blur detection: Laplacian variance (too blurry = can't read text)
# 2. Brightness check: Histogram analysis (too dark/bright = poor quality)
# 3. Edge density: Text regions have high edge density in upper 60% of image
# 4. Image loading: Skip corrupted files
#
# Output:
# - valid_paths: Images that pass all checks (proceed to Block 14)
# - safety_check_report.json: Detailed log of skipped images with reasons
# - safety_check_review.html: Interactive map for manual review
#
# Contrast with Block 12:
# - Block 12: ONE-TIME validation (proves VLM approach works)
# - Block 12.5: EVERY-TIME filtering (removes bad images from processing)
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
    """Filter images by safety check before expensive processing.

    Args:
        images_with_paths: List of (image_dict, local_path) tuples where
                          image_dict has camera_loc, and local_path is Path object
    """
    results = []
    skip_reasons = defaultdict(int)
    skipped_images = []  # Store detailed info about skipped images
    valid_images = []  # Store (image_dict, local_path) for valid images

    for image_dict, img_path in images_with_paths:
        result = has_valid_poi_potential(img_path)
        results.append(result)

        if not result.is_valid:
            skip_reasons[result.skip_reason or "unknown"] += 1
            # Store detailed info for manual review, including GPS location
            camera_loc = image_dict.get("camera_loc", {})
            skipped_images.append({
                "image_id": img_path.stem,
                "image_path": str(img_path),
                "skip_reason": result.skip_reason,
                "blur_score": result.blur_score,
                "brightness_score": result.brightness_score,
                "edge_density": result.edge_density,
                "latitude": camera_loc.get("lat"),
                "longitude": camera_loc.get("lon"),
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
        "skipped_images": skipped_images,  # Detailed list for manual review
    }

    # Also collect valid image data for visualization
    valid_images_data = []
    for image_dict, img_path in valid_images:
        camera_loc = image_dict.get("camera_loc", {})
        valid_images_data.append({
            "image_id": img_path.stem,
            "image_path": str(img_path),
            "latitude": camera_loc.get("lat"),
            "longitude": camera_loc.get("lon"),
        })

    report_path = Path("data/debug") / "safety_check_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(skip_report, f, indent=2, ensure_ascii=False)

    # Create HTML visualization for manual review (includes both valid and skipped)
    create_safety_check_visualization(
        skip_report,
        valid_images_data,
        output_path=Path("data/debug/safety_check_review.html")
    )

    print(f"Safety check report: {report_path}")
    print(f"  Valid: {len(valid_paths)}/{len(images_with_paths)}")
    print(f"  Filtered: {len(images_with_paths) - len(valid_paths)} images")
    print(f"  Skipped images saved to report for manual review")
    print(f"  Visualization: data/debug/safety_check_review.html")

    return valid_paths, valid_image_dicts, skip_report


def create_safety_check_visualization(skip_report, valid_images_data, output_path):
    """Create HTML visualization for manual review of both valid and skipped images."""
    import base64

    # Collect all coordinates for map center calculation
    skipped_images = skip_report.get("skipped_images", [])
    all_coords = [(s.get("latitude"), s.get("longitude"))
                  for s in skipped_images
                  if s.get("latitude") and s.get("longitude")]
    all_coords += [(v.get("latitude"), v.get("longitude"))
                   for v in valid_images_data
                   if v.get("latitude") and v.get("longitude")]

    if all_coords:
        # Use centroid of all image locations
        avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
        avg_lon = sum(c[1] for c in all_coords) / len(all_coords)
        center = [avg_lat, avg_lon]
    else:
        # Fallback to HCMC center
        center = [10.76, 106.69]

    m = folium.Map(location=center, zoom_start=13)

    # Add summary marker at center
    folium.Marker(
        location=center,
        popup=f"<b>Summary:</b><br>Valid: {skip_report['valid_images']}<br>Filtered: {skip_report['filtered_images']}",
        icon=folium.Icon(color="green", icon="ok-sign"),
    ).add_to(m)

    # Color map for skip reasons
    color_map = {
        "Blurry": "red",
        "Poor lighting": "orange",
        "Low edge density": "yellow",
        "Failed to load image": "gray",
        "OpenCV not available": "blue",
    }

    # Add VALID images with green checkmarks
    for valid_img in valid_images_data:
        image_id = valid_img["image_id"]
        lat = valid_img.get("latitude")
        lon = valid_img.get("longitude")
        img_path_str = valid_img["image_path"]

        if lat is None or lon is None:
            continue

        # Create popup with details
        popup_html = f"""
        <b>{image_id}</b><br>
        <b>Status:</b> <span style="color:green;">✓ PASSED SAFETY CHECK</span><br>
        <b>GPS:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>Image:</b> <a href="{img_path_str}" target="_blank">Open local file</a>
        """

        # Get image thumbnail if available
        img_path = Path(img_path_str)
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                encoded_image = f'<img src="data:image/jpeg;base64,{img_data}" width="200" style="max-height:150px;">'
                popup_html += f"<br><br>{encoded_image}"
            except Exception:
                pass

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="green", icon="ok-sign"),
        ).add_to(m)

    # Add SKIPPED images with detailed failure reasons
    for skipped in skipped_images:
        image_id = skipped["image_id"]
        reason = skipped["skip_reason"]
        blur_score = skipped.get("blur_score")
        brightness = skipped.get("brightness_score")
        edge_density = skipped.get("edge_density")
        lat = skipped.get("latitude")
        lon = skipped.get("longitude")

        if lat is None or lon is None:
            continue

        # Format metrics safely (handle None values)
        blur_str = f"{blur_score:.2f}" if blur_score is not None else "N/A"
        brightness_str = f"{brightness:.2f}" if brightness is not None else "N/A"
        edge_str = f"{edge_density:.3f}" if edge_density is not None else "N/A"

        # Create popup with details
        popup_html = f"""
        <b>{image_id}</b><br>
        <b>Status:</b> <span style="color:red;">✗ FILTERED OUT</span><br>
        <b>Reason:</b> {reason}<br>
        <b>Metrics:</b><br>
        - Blur score: {blur_str}<br>
        - Brightness: {brightness_str}<br>
        - Edge density: {edge_str}<br>
        <b>GPS:</b> ({lat:.6f}, {lon:.6f})<br>
        <b>Image:</b> <a href="{skipped['image_path']}" target="_blank">Open local file</a>
        """

        # Get image thumbnail if available
        img_path = Path(skipped["image_path"])
        if img_path.exists():
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                encoded_image = f'<img src="data:image/jpeg;base64,{img_data}" width="200" style="max-height:150px;">'
                popup_html += f"<br><br>{encoded_image}"
            except Exception:
                pass

        # Use actual GPS coordinates from camera_loc
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color_map.get(reason.split(" ")[0], "gray"), icon="info-sign"),
        ).add_to(m)

    m.save(str(output_path))


print(f"\nBLOCK 12.5: SVI Safety Check")

# Prepare images with paths: (image_dict, local_path) tuples
# Filter to only images that were successfully downloaded
downloaded_images = [
    (img, downloaded_paths.get(str(img["image_id"])))
    for img in SAMPLED_IMAGES
    if str(img["image_id"]) in downloaded_paths
]

valid_paths, valid_image_dicts, skip_report = batch_safety_check(downloaded_images)

print(f"  Valid images: {len(valid_paths)}/{len(downloaded_images)}")
print(f"  Filter rate: {skip_report['filter_rate']*100:.1f}%")
print("Next: Run Block 13 (Camera Analysis)")

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

    for i in range(0, len(image_ids), batch_size):
        batch = image_ids[i:i + batch_size]

        try:
            # Convert all image_ids to strings for the API call
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
            print(f"Warning: Failed to fetch camera batch: {e}")

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

# Use valid images from Block 12.5 safety check
valid_image_ids = [img.get("image_id") for img in VALID_IMAGES]
camera_data_map = fetch_camera_parameters_batch(client, valid_image_ids)

category_counts, percentages = analyze_camera_categories(VALID_IMAGES, camera_data_map)

print(f"  Camera categories:")
for cat, pct in percentages.items():
    count = category_counts.get(cat, 0)
    print(f"    {cat}: {count} ({pct:.1f}%)")
print("Next: Run Block 14 (ML Depth Estimation)")


# =============================================================================
# BLOCK 14: ML MONOCULAR DEPTH ESTIMATION
# =============================================================================

def estimate_depth_ml(image_path, model_name="dpt"):
    """Use ML for monocular depth prediction.

    Available models:
    - dpt: Intel/dpt-large (Dense Prediction Transformer)
    - glpn: vinvino02/glpn-kitti (Global-Local Path Networks)
    """
    if not HAS_TRANSFORMERS:
        print("Warning: transformers not installed")
        return None

    try:
        if model_name == "dpt":
            estimator = pipeline("depth-estimation", model="Intel/dpt-large")
        elif model_name == "glpn":
            estimator = pipeline("depth-estimation", model="vinvino02/glpn-kitti")
        else:
            raise ValueError(f"Unknown model: {model_name}")

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
        print(f"Warning: ML depth estimation failed: {e}")
        return None


def batch_estimate_depth(valid_image_paths, output_dir, model_name="dpt"):
    """Process all valid images through depth model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    depth_maps = {}

    print(f"Estimating depth for {len(valid_image_paths)} images using {model_name}...")

    for i, img_path in enumerate(valid_image_paths):
        image_id = img_path.stem

        depth_map = estimate_depth_ml(img_path, model_name)

        if depth_map is not None:
            depth_path = output_path / f"{image_id}_depth.npy"
            np.save(depth_path, depth_map)
            depth_maps[image_id] = depth_path

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(valid_image_paths)} images")

    print(f"  Generated {len(depth_maps)} depth maps")

    return depth_maps


if HAS_TRANSFORMERS and valid_paths:
    print(f"\nBLOCK 14: ML Monocular Depth Estimation")

    depth_maps = batch_estimate_depth(
        valid_paths[:20],  # Process first 20 for demo
        "data/interim/depth_maps",
        model_name="dpt"
    )
else:
    print(f"\nBLOCK 14: ML Monocular Depth Estimation")
    print(f"  Skipped: Install transformers with 'uv add transformers'")
    depth_maps = {}


# =============================================================================
# BLOCK 15: DEPTH MAP VISUALIZATION
# =============================================================================

def visualize_depth_map(image, depth_map, output_path, title="Depth Map Visualization"):
    """Generate side-by-side visualization of image and depth map."""
    # Check dynamically for matplotlib
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(depth_map, cmap="jet")
    axes[1].set_title(f"Depth Map (min={depth_map.min():.2f}, max={depth_map.max():.2f})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].hist(depth_map.flatten(), bins=50, edgecolor="black")
    axes[2].set_title("Depth Distribution")
    axes[2].set_xlabel("Depth (m)")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# Check for matplotlib dynamically (in case it was installed after script import)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB_NOW = True
except ImportError:
    HAS_MATPLOTLIB_NOW = False

if HAS_MATPLOTLIB_NOW and depth_maps:
    print(f"\nBLOCK 15: Depth Map Visualization")

    # Debug: Check what we have
    print(f"  Found {len(depth_maps)} depth maps")
    print(f"  Found {len(downloaded_paths)} downloaded images")

    # Create a mapping from image_id to image_path
    # downloaded_paths maps image_id (str) -> image_path (Path)
    # depth_maps maps image_id (str) -> depth_path (Path)

    for i, (image_id, depth_path) in enumerate(list(depth_maps.items())[:3]):
        print(f"\n  Processing image {i+1}/3: {image_id}")

        # Load the depth map
        depth_map = np.load(depth_path)
        print(f"    Depth map loaded: {depth_map.shape}")

        # Load the original image
        original_image_path = downloaded_paths.get(image_id)

        # Check if path exists, also try converting to string
        if original_image_path is None:
            print(f"    Warning: image_id not in downloaded_paths")
            # Try to find the image directly
            search_path = Path("data/raw/images") / f"{image_id}.jpg"
            if search_path.exists():
                original_image_path = search_path
                print(f"    Found image at: {search_path}")
            else:
                print(f"    Image not found at: {search_path}")

        if original_image_path and original_image_path.exists():
            print(f"    Loading image from: {original_image_path}")
            # Load the actual image
            if HAS_PIL:
                original_image = Image.open(original_image_path)
                original_image = np.array(original_image)
            else:
                original_image = cv2.imread(str(original_image_path))
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            print(f"    Image loaded: {original_image.shape}")
        else:
            # Fallback to dummy image if original not found
            print(f"    Warning: Original image not found, using dummy")
            original_image = np.random.randint(0, 255, (depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

        # Resize depth map to match image if needed
        if original_image.shape[:2] != depth_map.shape:
            import cv2
            depth_map = cv2.resize(depth_map, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            print(f"    Resized depth map to: {depth_map.shape}")

        # Create visualization
        viz_path = Path("data/debug") / f"depth_viz_{image_id}.png"
        viz_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_depth_map(original_image, depth_map, str(viz_path), title=f"Depth Map - {image_id}")

        print(f"    Saved: {viz_path}")

    print(f"\n  To view visualizations:")
    print(f"    open data/debug/depth_viz_*.png")
else:
    print(f"\nBLOCK 15: Depth Map Visualization")
    if not HAS_MATPLOTLIB:
        print(f"  Skipped: matplotlib not installed")
        print(f"  Install with: uv add matplotlib")
    if not depth_maps:
        print(f"  Skipped: no depth maps available")
        print(f"  Run Block 14 first to generate depth maps")


# =============================================================================
# BLOCK 16: GEOLOCATOR IMPLEMENTATION (CREATE POI DATA POINT)
# =============================================================================

def pixel_to_ray(u, v, W, H, fov=60.0, heading=0.0, pitch=0.0):
    """Convert pixel coordinates to 3D ray direction in camera frame."""
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
    R = 6371000  # Earth's radius in meters

    # Calculate offset in radians
    lat_offset_rad = (horizontal_distance * math.cos(bearing_rad)) / R
    lon_offset_rad = (horizontal_distance * math.sin(bearing_rad)) / (R * math.cos(math.radians(camera_lat)))

    # Convert to degrees and add to camera position
    target_lat = camera_lat + math.degrees(lat_offset_rad)
    target_lon = camera_lon + math.degrees(lon_offset_rad)

    return target_lat, target_lon


def pixel_to_gps(u, v, depth, W, H, camera_lat, camera_lon, heading=0.0, pitch=0.0, fov=60.0):
    """Complete pipeline: convert pixel coordinates to GPS."""
    ray = pixel_to_ray(u, v, W, H, fov, heading, pitch)
    target_lat, target_lon = offset_from_camera(depth, ray, camera_lat, camera_lon)
    return round(target_lat, 6), round(target_lon, 6)


class Geolocator:
    """High-level interface for pixel-to-GPS conversion.

    What this creates: A GPS coordinate (lat, lon) for each detected signboard.
    """

    def __init__(self, camera_data):
        self.camera_data = camera_data
        self.focal = None

        params = camera_data.get("camera_parameters", [])
        if params and len(params) >= 1 and params[0] is not None:
            self.focal = params[0]

    def _has_full_intrinsics(self):
        return self.focal is not None

    def pixel_to_gps(self, u, v, depth, heading, pitch=0.0, fov=60.0):
        # Handle both data structures:
        # 1. From VALID_IMAGES: has camera_loc with lat/lon
        # 2. From geometry field: GeoJSON format with coordinates
        camera_loc = self.camera_data.get("camera_loc", {})
        if camera_loc and "lat" in camera_loc and "lon" in camera_loc:
            camera_lat = float(camera_loc["lat"])
            camera_lon = float(camera_loc["lon"])
        else:
            # Fallback to geometry.coordinates format (GeoJSON: [lon, lat])
            geometry = self.camera_data.get("geometry", {})
            coords = geometry.get("coordinates", [0, 0])
            if len(coords) >= 2:
                camera_lon, camera_lat = float(coords[0]), float(coords[1])
            else:
                camera_lon, camera_lat = 0.0, 0.0

        # Debug: Check if camera coordinates are zero
        if abs(camera_lat) < 0.001 and abs(camera_lon) < 0.001:
            print(f"    WARNING: Camera coordinates near zero: ({camera_lat}, {camera_lon})")
            print(f"    camera_data keys: {list(self.camera_data.keys())}")
            print(f"    camera_loc: {camera_loc}")

        W = self.camera_data.get("width", 1024)
        H = self.camera_data.get("height", 1024)

        if self._has_full_intrinsics():
            fov_rad = 2 * math.atan(W / (2 * self.focal)) if self.focal > 0 else math.radians(fov)
            actual_fov = math.degrees(fov_rad)
        else:
            actual_fov = fov

        return pixel_to_gps(u, v, depth, W, H, camera_lat, camera_lon, heading, pitch, actual_fov)


print(f"\nBLOCK 16: Geolocator Implementation")
print(f"  What this creates: GPS coordinate (lat, lon) for each signboard")

# Mock camera data for demonstration
mock_camera_data = {
    "camera_parameters": [1000, -0.1, 0.001],
    "geometry": {"type": "Point", "coordinates": [106.69, 10.77]},
    "compass_angle": 45.0,
    "width": 1024,
    "height": 512,
}

geolocator = Geolocator(mock_camera_data)

poi_lat, poi_lon = geolocator.pixel_to_gps(u=600, v=200, depth=15.0, heading=45.0)

print(f"  Test: Pixel (600, 200) + Depth 15m -> GPS ({poi_lat:.6f}, {poi_lon:.6f})")
print(f"  Has full intrinsics: {geolocator._has_full_intrinsics()}")


# =============================================================================
# BLOCK 17: VLM SIGNBOARD DETECTION AND POI EXTRACTION (COMBINED)
# =============================================================================

@dataclass
class POIDetection:
    """Detection of a signboard with POI data."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center_pixel: Tuple[int, int]  # (u, v)
    poi_data: Dict[str, Any]
    confidence: float


# VLMClient class already defined in Block 12

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
        print(f"Warning: VLM detection failed: {e}")
        return []


print(f"\nBLOCK 17: VLM Signboard Detection + POI Extraction")
print(f"  Combined approach: One VLM call for detection AND extraction")

# Get API key from environment
glm_key = os.getenv("GLM_KEY")

if glm_key and VALID_IMAGE_PATHS:
    print(f"  Running VLM detection on {len(VALID_IMAGE_PATHS)} images...")

    # Initialize VLM client
    vlm_client = VLMClient(api_key=glm_key)

    # Process first 3 images as demo
    vlm_results = {}
    for i, img_path in enumerate(VALID_IMAGE_PATHS[:3]):
        image_id = img_path.stem
        print(f"\n  Processing {i+1}/3: {image_id}")

        try:
            detections = detect_and_extract_pois(vlm_client, img_path)
            vlm_results[image_id] = detections

            print(f"    Found {len(detections)} signboards")
            for j, det in enumerate(detections):
                name = det.poi_data.get("poi_name_vietnamese", "Unknown")
                print(f"      {j+1}. {name} (confidence: {det.confidence:.2f})")

        except Exception as e:
            print(f"    Error: {e}")
            vlm_results[image_id] = []

    # Save results
    results_path = Path("data/interim/vlm_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        # Convert to serializable format
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
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {results_path}")
    print(f"  Total detections: {sum(len(d) for d in vlm_results.values())}")
    print("  Next: Run Block 18 (Assign POI to Location)")

elif not glm_key:
    print("  Skipped: GLM_KEY not found in .env file")
    print("  Add your key: GLM_KEY=your_key_here")
else:
    print(f"  Skipped: No valid images available (run Blocks 11-12.5 first)")


# =============================================================================
# BLOCK 18: ASSIGN POI INFORMATION TO CORRESPONDING DATA POINT
# =============================================================================

def assign_poi_to_location(vlm_detections, geolocator, depth_map, camera_data):
    """Combine POI data with GPS coordinates.

    The Missing Link: Combining 'What' (POI data) + 'Where' (GPS)

    IMPORTANT: Depth map from DPT model outputs relative values (0-255), not meters.
    We need to convert to actual metric depth using a scaling factor.
    """
    heading = camera_data.get("compass_angle", 0.0)
    pitch = 0.0

    complete_pois = []

    for detection in vlm_detections:
        u, v = detection.center_pixel
        poi_data = detection.poi_data

        if depth_map is not None:
            # Depth map values are 0-255 (relative depth)
            # Convert to meters: typical street scene ranges from ~2m to ~100m
            # Use a nonlinear scaling: closer objects are more precise
            depth_relative = float(depth_map[int(v), int(u)])

            # Scale relative depth (0-255) to meters (2-100m range)
            # This is an approximation - for production, you'd want calibration
            # Formula: depth_m = 2 + (depth_relative / 255) * 98
            # This gives: 0 -> 2m, 255 -> 100m
            depth_meters = 2.0 + (depth_relative / 255.0) * 98.0
        else:
            depth_meters = 15.0  # Default fallback

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


def visualize_poi_detections(image_path, detections_with_pois, output_path):
    """Visualize POI detections on original image with annotations."""
    # Check for matplotlib dynamically
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Warning: matplotlib not installed")
        return

    # Load image
    if HAS_PIL:
        image = Image.open(image_path)
        image = np.array(image)
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image)

    # Colors for different POIs
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    # Draw each detection
    for idx, item in enumerate(detections_with_pois):
        detection, poi = item
        bbox = detection.bbox
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        color = colors[idx % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Create label text
        name_vn = poi.get('poi_name_vietnamese', 'Unknown')
        name_en = poi.get('poi_name_english', '')
        category = poi.get('business_category', '')
        confidence = detection.confidence
        lat = poi.get('latitude', 0)
        lon = poi.get('longitude', 0)
        depth = poi.get('depth_meters', 0)

        label = f"{idx+1}. {name_vn}"
        if name_en:
            label += f"\n   ({name_en})"
        label += f"\n   Category: {category}"
        label += f"\n   Conf: {confidence:.2f}"
        label += f"\n   GPS: ({lat:.6f}, {lon:.6f})"
        label += f"\n   Depth: {depth:.1f}m"

        # Add text label
        ax.text(
            x1, y1 - 5, label,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.7),
            fontsize=9, color='white', verticalalignment='top'
        )

        # Mark center pixel
        center = detection.center_pixel
        ax.plot(center[0], center[1], 'o', color=color, markersize=10)

    ax.axis('off')
    ax.set_title(f"POI Detections - {Path(image_path).stem}", fontsize=14)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_poi_map_visualization(image_data_with_pois, output_path):
    """Create interactive HTML map showing camera locations and derived POI locations.

    Args:
        image_data_with_pois: List of dicts with keys:
            - image_id: str
            - image_path: Path
            - camera_lat: float
            - camera_lon: float
            - pois: List of complete POI dicts with lat/lon
        output_path: Path to save the HTML file
    """
    import base64

    if not image_data_with_pois:
        print("    No data to visualize on map")
        return

    # Collect all coordinates for map center
    all_coords = []
    for item in image_data_with_pois:
        all_coords.append((item["camera_lat"], item["camera_lon"]))
        for poi in item.get("pois", []):
            all_coords.append((poi["latitude"], poi["longitude"]))

    # Calculate map center
    avg_lat = sum(c[0] for c in all_coords) / len(all_coords)
    avg_lon = sum(c[1] for c in all_coords) / len(all_coords)

    # Create map
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=18)

    # Add tile layers
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='Light Theme').add_to(m)

    # Add camera locations and POI locations for each image
    for item in image_data_with_pois:
        image_id = item["image_id"]
        camera_lat = item["camera_lat"]
        camera_lon = item["camera_lon"]
        pois = item.get("pois", [])

        # Camera marker (blue)
        mapillary_url = f"https://www.mapillary.com/app/?pKey={item['image_id']}"
        camera_popup = f"""
        <div style="width: 300px;">
            <h4>📷 Camera Location</h4>
            <b>Image ID:</b> <a href="{mapillary_url}" target="_blank">{image_id}</a><br>
            <b>GPS:</b> ({camera_lat:.6f}, {camera_lon:.6f})<br>
            <b>POIs detected:</b> {len(pois)}<br>
            <a href="{item['image_path']}" target="_blank">📁 Open local image</a>
        </div>
        """
        folium.Marker(
            location=[camera_lat, camera_lon],
            popup=folium.Popup(camera_popup, max_width=350),
            icon=folium.Icon(color="blue", icon="camera", prefix="fa"),
            tooltip="Camera"
        ).add_to(m)

        # Add lines from camera to each POI
        for poi in pois:
            poi_lat = poi["latitude"]
            poi_lon = poi["longitude"]

            # Draw line
            folium.PolyLine(
                locations=[[camera_lat, camera_lon], [poi_lat, poi_lon]],
                color="red",
                weight=2,
                opacity=0.6,
                popup=f"Distance: {poi.get('depth_meters', 0):.1f}m"
            ).add_to(m)

        # POI markers (red)
        for poi in pois:
            name_vn = poi.get("poi_name_vietnamese", "Unknown")
            name_en = poi.get("poi_name_english", "")
            category = poi.get("business_category", "")
            confidence = poi.get("confidence", 0)
            depth = poi.get("depth_meters", 0)

            poi_popup = f"""
            <div style="width: 350px;">
                <h4>📍 POI Location</h4>
                <b>Vietnamese:</b> {name_vn}<br>
                <b>English:</b> {name_en}<br>
                <b>Category:</b> {category}<br>
                <b>Confidence:</b> {confidence:.2f}<br>
                <b>Depth:</b> {depth:.1f}m<br>
                <b>GPS:</b> ({poi_lat:.6f}, {poi_lon:.6f})<br>
                <b>From camera:</b> ({camera_lat:.6f}, {camera_lon:.6f})
            </div>
            """
            folium.Marker(
                location=[poi_lat, poi_lon],
                popup=folium.Popup(poi_popup, max_width=400),
                icon=folium.Icon(color="red", icon="info-sign"),
                tooltip=name_vn
            ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    m.save(str(output_path))


print(f"\nBLOCK 18: Assign POI to Location")
print(f"  The Missing Link: Combining 'What' (POI data) + 'Where' (GPS)")

# Check if we have real VLM results from Block 17
vlm_results_path = Path("data/interim/vlm_results.json")
if vlm_results_path.exists():
    print(f"  Loading VLM results from Block 17...")

    with open(vlm_results_path, "r", encoding="utf-8") as f:
        vlm_results_data = json.load(f)

    print(f"  Processing {len(vlm_results_data)} images with detections...")

    # Collect all data for map visualization
    image_data_with_pois = []

    # Process each image that has detections
    for image_id, detections_data in vlm_results_data.items():
        if not detections_data:
            continue

        # Get the image path
        image_path = downloaded_paths.get(image_id)
        if not image_path or not image_path.exists():
            print(f"    Image not found: {image_id}")
            continue

        # Get corresponding depth map
        depth_path = depth_maps.get(image_id)
        if depth_path:
            depth_map = np.load(depth_path)
        else:
            print(f"    No depth map for {image_id}, using default depth")
            depth_map = None

        # Get camera data for this image
        camera_data = None
        for img in VALID_IMAGES:
            if str(img.get("image_id")) == image_id:
                camera_data = img
                break

        if not camera_data:
            print(f"    Camera data not found for {image_id}")
            continue

        # Debug: print camera_data structure
        camera_loc = camera_data.get("camera_loc", {})
        if camera_loc:
            print(f"    Camera: ({camera_loc.get('lat', 0):.6f}, {camera_loc.get('lon', 0):.6f})")

        # Convert detections back to POIDetection objects
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

        print(f"\n  {image_id}:")
        for poi in complete_pois:
            name = poi.get("poi_name_vietnamese", "Unknown")
            lat = poi.get("latitude", 0)
            lon = poi.get("longitude", 0)
            depth = poi.get("depth_meters", 0)
            print(f"    - {name}")
            print(f"      GPS: ({lat:.6f}, {lon:.6f}), Depth: {depth:.1f}m")

        # Visualize
        detections_with_pois = list(zip(detections, complete_pois))
        viz_path = Path("data/debug") / f"poi_viz_{image_id}.png"
        visualize_poi_detections(image_path, detections_with_pois, str(viz_path))
        print(f"    PNG Visualization: {viz_path}")

        # Collect data for map visualization
        camera_loc = camera_data.get("camera_loc", {})
        image_data_with_pois.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "camera_lat": camera_loc.get("lat", 0),
            "camera_lon": camera_loc.get("lon", 0),
            "pois": complete_pois,
        })

    # Create interactive map
    if image_data_with_pois:
        map_path = Path("data/debug/poi_map.html")
        create_poi_map_visualization(image_data_with_pois, str(map_path))
        print(f"\n  Interactive map saved to: {map_path}")
        print(f"  View with: open {map_path}")

    print(f"\n  All PNG visualizations: data/debug/poi_viz_*.png")
    print(f"  View with: open data/debug/poi_viz_*.png")

else:
    print(f"  No VLM results found from Block 17")
    print(f"  Running mock example...")

    # Mock example
    mock_detections = [
        POIDetection(
            bbox=(200, 100, 400, 200),
            center_pixel=(300, 150),
            poi_data={
                "poi_name_vietnamese": "Phở Hùng",
                "poi_name_english": "Hung Pho",
                "business_category": "Restaurant",
            },
            confidence=0.92
        )
    ]

    mock_depth_map = np.ones((512, 512)) * 15.0

    complete_pois = assign_poi_to_location(mock_detections, geolocator, mock_depth_map, mock_camera_data)

    print(f"  Example complete POI:")
    print(f"    Name: {complete_pois[0]['poi_name_vietnamese']}")
    print(f"    GPS: ({complete_pois[0]['latitude']:.6f}, {complete_pois[0]['longitude']:.6f})")
    print(f"    Depth: {complete_pois[0]['depth_meters']:.1f}m")


# =============================================================================
# BLOCK 19: END-TO-END PIPELINE ORCHESTRATION
# =============================================================================

def run_poi_pipeline(image_id, image_path, camera_data, vlm_client, depth_map=None):
    """Process single image through complete pipeline."""
    result = {
        "image_id": image_id,
        "success": False,
        "pois": [],
        "error": None,
    }

    try:
        detections = detect_and_extract_pois(vlm_client, image_path)

        if not detections:
            result["success"] = True
            result["message"] = "No POIs detected"
            return result

        if depth_map is None:
            depth_map = estimate_depth_ml(image_path)

        geolocator = Geolocator(camera_data)
        pois = assign_poi_to_location(detections, geolocator, depth_map, camera_data)

        result["success"] = True
        result["pois"] = pois
        result["poi_count"] = len(pois)

    except Exception as e:
        result["error"] = str(e)

    return result


def save_pois_geojson(results, output_path):
    """Save POI results as GeoJSON."""
    features = []

    for result in results:
        if not result.get("success"):
            continue

        for poi in result.get("pois", []):
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

    print(f"Saved {len(features)} POIs to {output_path}")


print(f"\nBLOCK 19: End-to-End Pipeline Orchestration")
print(f"  Complete pipeline: Download -> Validate -> Depth -> Detect -> GPS")
print("  (Skipped: Set GLM_KEY or OPENAI_API_KEY to run pipeline)")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*50}")
print(f"BLOCKS 0-19 IMPLEMENTATION COMPLETE!")
print(f"{'='*50}")
print(f"""
Simplified Pipeline Data Flow:
1. Download images (Block 11)
2. VLM validation (Block 12) - DECISION POINT
3. Safety check (Block 12.5)
4. Camera analysis (Block 13)
5. ML depth estimation (Block 14)
6. Depth visualization (Block 15)
7. Geolocator (Block 16) - Creates POI GPS point
8. VLM detection + extraction (Block 17)
9. Assign POI to location (Block 18)
10. End-to-end pipeline (Block 19)

Key Design Decisions:
- Simplified: ML depth + geometric projection (no SfM, no stereo)
- Hybrid: Pinhole (69.5% Normal FOV) + Full intrinsics (27.4% Wide FOV)
- Combined: VLM detection + extraction in one call
- Primary: Mapillary data source
- VLM: GLM-4V

Next Steps:
1. Set GLM_KEY in .env for VLM access
2. Run Block 12 validation first (CRITICAL)
3. If validation passes, run Block 19 full pipeline
4. Results saved to: data/processed/final_pois.geojson
""")
