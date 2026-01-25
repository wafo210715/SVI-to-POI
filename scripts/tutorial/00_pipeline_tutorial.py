#!/usr/bin/env python3
"""
HCMC POI Reconstruction - Pipeline Tutorial
===========================================

SHIFT+ENTER to run each block sequentially.

Data Flow:
1. Define HCMC bounding boxes
2. Define MapillaryClient class (inline)
3. Initialize client with API token
4. Get tile coordinates for bbox
5. Fetch tile data (JSON files)
6. Parse images from tiles
7. Sample images for analysis
8. Fetch detailed entity data
9. Analyze camera parameters
10. Save and visualize results
"""

# =============================================================================
# BLOCK 0: IMPORTS
# =============================================================================
# Returns: None (just imports)

import json
import random
import sys
import os
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import requests
from dotenv import load_dotenv
import folium

print("BLOCK 0: Libraries imported")
print("Next: Run Block 1")


# =============================================================================
# BLOCK 1: DEFINE HCMC BOUNDING BOX AREAS
# =============================================================================
# Returns: HCMC_AREAS dict
#
# NOTE: These are simplified rectangular bounding boxes.
#
# For official district boundaries:
# 1. Download HCMC district SHP file from GADM or OpenStreetMap
# 2. Load in ArcGIS/QGIS to get exact boundaries
# 3. Calculate bounding box: min_lon, min_lat, max_lon, max_lat
# 4. Replace the values below with your calculated coordinates
#
# Example workflow in ArcGIS:
#   - Add SHP layer -> Open attribute table -> Select district
#   - Right-click layer -> Properties -> Source tab -> Extent shows bbox
#   - Or use: Data Management Tools -> Features -> Bounding Box
#
# District 1 official boundary is irregular (not a perfect rectangle).
# The coordinates below are approximations covering the district area.

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
print("-" * 40)
for key, area in HCMC_AREAS.items():
    print(f"  {key}: {area['name']}")
    print(f"    BBox: ({area['min_lon']}, {area['min_lat']}) -> ({area['max_lon']}, {area['max_lat']})")

# Generate map visualization
m = folium.Map(location=[10.77, 106.69], zoom_start=12)
colors = {"small_test": "green", "district_1_full": "blue", "central_hcmc": "red"}

for key, area in HCMC_AREAS.items():
    folium.Rectangle(
        bounds=[[area['min_lat'], area['min_lon']], [area['max_lat'], area['max_lon']]],
        popup=area['name'],
        color=colors.get(key, 'gray'),
        fill=True,
        fillColor=colors.get(key, 'gray'),
        fillOpacity=0.2,
    ).add_to(m)

Path("data/debug").mkdir(parents=True, exist_ok=True)
map_path = Path("data/debug/tutorial_hcmc_areas.html")
m.save(str(map_path))
print(f"\nMap saved: {map_path}")

SELECTED_AREA = "small_test"
print(f"\nSelected: {SELECTED_AREA}")
print("Next: Run Block 2")


# =============================================================================
# BLOCK 2: DEFINE MAPILLARY CLIENT CLASS
# =============================================================================
# Returns: MapillaryClient class definition

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
        """Convert tile coordinates to bounding box."""
        import math
        n = 2 ** zoom
        min_lon = tile_x / n * 360 - 180
        max_lon = (tile_x + 1) / n * 360 - 180
        lat_rad1 = math.asin(math.tanh(math.pi * (1 - 2 * tile_y / n)))
        lat_rad2 = math.asin(math.tanh(math.pi * (1 - 2 * (tile_y + 1) / n)))
        min_lat = math.degrees(lat_rad2)
        max_lat = math.degrees(lat_rad1)
        return min_lon, max_lon, min_lat, max_lat

    def get_tiles_for_bbox(self, min_lon: float, min_lat: float, max_lon: float, max_lat: float, zoom: int = 14) -> List[Dict[str, int]]:
        """Get all tile coordinates covering a bounding box."""
        import mercantile
        tiles = []
        for tile in mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom):
            tiles.append({"z": tile.z, "x": tile.x, "y": tile.y})
        return tiles

    def _fetch_tile(self, z: int, x: int, y: int) -> Dict[str, Any]:
        """Fetch a single tile from Mapillary and convert to GeoJSON."""
        from vt2geojson.tools import vt_bytes_to_geojson
        url = self._build_tile_url(z, x, y)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return vt_bytes_to_geojson(response.content, x, y, z, layer="image")

print("BLOCK 2: MapillaryClient class defined")
print("  Methods: get_tiles_for_bbox(), _fetch_tile(), _tile_to_bbox()")
print("Next: Run Block 3")


# =============================================================================
# BLOCK 3: INITIALIZE MAPILLARY CLIENT
# =============================================================================
# Returns: MapillaryClient object

load_dotenv()
access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")

if not access_token:
    print("ERROR: Set MAPILLARY_ACCESS_TOKEN in .env file")
    sys.exit(1)

client = MapillaryClient(access_token=access_token)
CURRENT_AREA = HCMC_AREAS[SELECTED_AREA]

print(f"BLOCK 3: Client initialized")
print(f"  Area: {CURRENT_AREA['name']}")
print(f"  Token: {access_token[:20]}...")
print("Next: Run Block 4")


# =============================================================================
# BLOCK 4: GET TILE COORDINATES
# =============================================================================
# Returns: List of tile dicts with z, x, y

tiles = client.get_tiles_for_bbox(
    CURRENT_AREA['min_lon'],
    CURRENT_AREA['min_lat'],
    CURRENT_AREA['max_lon'],
    CURRENT_AREA['max_lat'],
    zoom=14
)

print(f"BLOCK 4: Tile Coordinates")
print(f"  Area: {CURRENT_AREA['name']}")
print(f"  Tiles needed: {len(tiles)}")
print(f"  All tiles: ", end="")
for t in tiles:
    print(f"{t['z']}/{t['x']}/{t['y']}", end=", ")
print()

# Visualize: Original area bbox (green) + Tile grid overlay (blue)
center_lat = (CURRENT_AREA['min_lat'] + CURRENT_AREA['max_lat']) / 2
center_lon = (CURRENT_AREA['min_lon'] + CURRENT_AREA['max_lon']) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

# Original area bbox (green)
folium.Rectangle(
    bounds=[[CURRENT_AREA['min_lat'], CURRENT_AREA['min_lon']],
            [CURRENT_AREA['max_lat'], CURRENT_AREA['max_lon']]],
    popup=CURRENT_AREA['name'],
    color='green',
    fill=True,
    fillColor='green',
    fillOpacity=0.2,
    weight=3,
).add_to(m)

# Tile grid (blue)
for i, tile in enumerate(tiles):
    min_lon, max_lon, min_lat, max_lat = client._tile_to_bbox(tile['x'], tile['y'], tile['z'])
    folium.Rectangle(
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        popup=f"Tile {i+1}: {tile['z']}/{tile['x']}/{tile['y']}",
        tooltip=f"Tile {i+1}",
        color='blue',
        fill=True,
        fillColor='lightblue',
        fillOpacity=0.4,
    ).add_to(m)

m.save(Path("data/debug") / f"tutorial_tiles_{SELECTED_AREA}.html")
print(f"  Tile grid saved: data/debug/tutorial_tiles_{SELECTED_AREA}.html")
print(f"  Green = Original area bbox")
print(f"  Blue = Tiles covering the area")

CURRENT_TILES = tiles
print("Next: Run Block 5")


# =============================================================================
# BLOCK 5: FETCH TILE DATA
# =============================================================================
# Returns: tiles_with_data + saves JSON files

output_dir = Path("data/raw/tiles") / SELECTED_AREA
output_dir.mkdir(parents=True, exist_ok=True)

tiles_with_data = []
total_images = 0

for i, tile in enumerate(CURRENT_TILES):
    try:
        tile_data = client._fetch_tile(tile['z'], tile['x'], tile['y'])
        feature_count = len(tile_data.get("features", []))
        tiles_with_data.append({**tile, "image_count": feature_count})
        total_images += feature_count

        # Save tile JSON
        tile_file = output_dir / f"tile_{tile['z']}_{tile['x']}_{tile['y']}.json"
        with open(tile_file, "w") as f:
            json.dump(tile_data, f, indent=2)

    except Exception as e:
        tiles_with_data.append({**tile, "image_count": 0})

print(f"BLOCK 5: Tile Data Fetched")
print(f"  Tiles processed: {len(tiles_with_data)}")
print(f"  Total images: {total_images}")

CURRENT_TILES_WITH_DATA = tiles_with_data
print("Next: Run Block 6")


# =============================================================================
# BLOCK 6: LOAD ALL IMAGES FROM TILES
# =============================================================================
# Returns: all_images list

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
            "creator_id": props.get("creator_id"),
            "captured_at": props.get("captured_at"),
        })

print(f"BLOCK 6: Images Loaded")
print(f"  Total images: {len(all_images)}")
print(f"  Unique sequences: {len(set(img['sequence_id'] for img in all_images if img['sequence_id']))}")

ALL_IMAGES = all_images
print("Next: Run Block 7")


# =============================================================================
# BLOCK 7: SAMPLE IMAGES FOR ANALYSIS
# =============================================================================
# Returns: sampled_images (100 random)

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
# Returns: entities with camera parameters

entities = []
batch_size = 100

for i in range(0, len(SAMPLED_IMAGES), batch_size):
    batch = SAMPLED_IMAGES[i:i + batch_size]
    image_ids = [str(img["image_id"]) for img in batch]

    try:
        ids_str = ",".join(image_ids)
        fields = "id,camera_type,camera_parameters,make,model,sequence,altitude,computed_altitude"
        params = {"ids": ids_str, "fields": fields, "access_token": client.access_token}

        response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            entities.extend(list(data.values()))
        elif isinstance(data, list):
            entities.extend(data)

    except Exception as e:
        pass

print(f"BLOCK 8: Entity Data Fetched")
print(f"  Entities: {len(entities)}")

ENTITIES = entities
print("Next: Run Block 9")


# =============================================================================
# BLOCK 9: ANALYZE CAMERA PARAMETERS
# =============================================================================
# Returns: ANALYSIS_RESULTS dict

camera_types = []
has_intrinsics = []
focal_lengths = []
sequence_cameras = defaultdict(set)

for entity in ENTITIES:
    camera_type = entity.get("camera_type")
    camera_params = entity.get("camera_parameters", [])
    sequence_id = entity.get("sequence")

    if camera_type:
        camera_types.append(camera_type)
    if sequence_id and camera_type:
        sequence_cameras[sequence_id].add(camera_type)

    has_focal = camera_params and len(camera_params) >= 1 and camera_params[0] is not None
    has_intrinsics.append(has_focal)
    if has_focal and isinstance(camera_params[0], (int, float)):
        focal_lengths.append(float(camera_params[0]))

camera_type_counts = dict(Counter(camera_types))
has_intrinsics_count = sum(1 for x in has_intrinsics if x)
intrinsic_ratio = has_intrinsics_count / len(ENTITIES) if ENTITIES else 0
perspective_ratio = camera_type_counts.get("perspective", 0) / len(ENTITIES) if ENTITIES else 0

if intrinsic_ratio >= 0.8 and perspective_ratio >= 0.7:
    feasibility = "MONOCULAR DEPTH FEASIBLE"
elif intrinsic_ratio >= 0.5 and perspective_ratio >= 0.5:
    feasibility = "PARTIAL"
else:
    feasibility = "NOT FEASIBLE"

ANALYSIS_RESULTS = {
    "total_images_tested": len(ENTITIES),
    "camera_params": {
        "perspective": camera_type_counts.get("perspective", 0),
        "fisheye": camera_type_counts.get("fisheye", 0),
        "spherical": camera_type_counts.get("spherical", 0),
        "has_valid_intrinsics": has_intrinsics_count,
    },
    "feasibility_recommendation": feasibility,
}

print(f"BLOCK 9: Camera Analysis")
print(f"  Total tested: {len(ENTITIES)}")
print(f"  Camera types: {camera_type_counts}")
print(f"  With intrinsics: {has_intrinsics_count}/{len(ENTITIES)} ({100*intrinsic_ratio:.1f}%)")
print(f"  Feasibility: {feasibility}")
print("Next: Run Block 10")


# =============================================================================
# BLOCK 10: SAVE AND VISUALIZE RESULTS
# =============================================================================
# Returns: Saves JSON + HTML files

output_path = Path("data/debug/tutorial_depth_analysis_summary.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(ANALYSIS_RESULTS, f, indent=2)

print(f"BLOCK 10: Results Saved")
print(f"  Summary: {output_path}")

# Optional visualization
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Camera Types", "Intrinsics"),
                        specs=[[{"type": "pie"}, {"type": "bar"}]])

    type_labels, type_values = [], []
    for k, v in ANALYSIS_RESULTS["camera_params"].items():
        if k != "has_valid_intrinsics" and v > 0:
            type_labels.append(k.capitalize())
            type_values.append(v)

    fig.add_trace(go.Pie(labels=type_labels, values=type_values, hole=0.3), row=1, col=1)

    has_intrinsics = ANALYSIS_RESULTS["camera_params"]["has_valid_intrinsics"]
    fig.add_trace(go.Bar(x=["With", "Without"], y=[has_intrinsics, len(ENTITIES)-has_intrinsics],
                        marker_color=['green', 'red']), row=1, col=2)

    fig.update_layout(title_text=f"Camera Analysis - {feasibility}")
    fig.write_html(Path("data/debug/tutorial_camera_analysis.html"))
    print(f"  Visualization: data/debug/tutorial_camera_analysis.html")

except ImportError:
    print("  (Install plotly for visualization)")

print("\nTUTORIAL COMPLETE")
