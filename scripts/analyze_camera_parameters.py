#!/usr/bin/env python3
"""
Camera Parameter Analyzer - Categorize and visualize camera settings.

Creates HTML visualization showing:
- Camera categories by key parameters
- Camera positions and headings on map
- Sample images for each category
- Interactive verification workflow

Usage:
    uv run python scripts/analyze_camera_parameters.py
"""

import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import requests
from jinja2 import Template

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class CameraCategory:
    """A category of cameras with similar parameters."""
    name: str
    focal_range: Tuple[float, float]
    fov_range: Tuple[float, float]
    count: int = 0
    sample_images: List[str] = None
    color: str = "#95a5a6"
    color_class: str = ""

    def __post_init__(self):
        if self.sample_images is None:
            self.sample_images = []


def categorize_camera(camera_data: Dict[str, Any]) -> str:
    """Categorize camera based on key parameters affecting depth.

    Only processes PERSPECTIVE cameras (not fisheye or spherical).
    Categorizes by FOV derived from focal length.

    Key parameters that affect depth:
    1. focal_length - Determines field of view and depth scale
    2. camera_type - Must be perspective (skip fisheye/spherical)
    3. width, height - Image dimensions affect pixel-to-metric conversion

    Args:
        camera_data: Camera entity from Mapillary

    Returns:
        Category name string, or None if not perspective
    """
    params = camera_data.get("camera_parameters", [])
    camera_type = camera_data.get("camera_type", "perspective")
    width = camera_data.get("width", 1024)

    # Skip non-perspective cameras
    if camera_type != "perspective":
        return None

    if not params or params[0] is None:
        return "Unknown (no intrinsics)"

    focal = params[0]

    # Categorize by focal length (normalized or pixels)
    # Assuming pixels if > 1.0, normalized if <= 1.0
    if focal <= 1.0:
        # Normalized focal - convert to approximate FOV
        # FOV = 2 * arctan(1 / (2 * focal))
        import math
        fov_rad = 2 * math.atan(1 / (2 * focal))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < 60:
            return "Narrow FOV (<60°)"
        elif fov_deg < 90:
            return "Normal FOV (60-90°)"
        else:
            return "Wide FOV (>90°)"
    else:
        # Pixel focal - estimate FOV from width
        # FOV ≈ 2 * arctan(width / (2 * focal))
        import math
        fov_rad = 2 * math.atan(width / (2 * focal))
        fov_deg = math.degrees(fov_rad)

        if fov_deg < 60:
            return "Narrow FOV (<60°)"
        elif fov_deg < 90:
            return "Normal FOV (60-90°)"
        else:
            return "Wide FOV (>90°)"


def calculate_fov_from_focal(focal: float, width: int, height: int) -> float:
    """Calculate field of view from focal length.

    Args:
        focal: Focal length in pixels
        width, height: Image dimensions

    Returns:
        Horizontal FOV in degrees
    """
    import math
    fov_rad = 2 * math.atan(width / (2 * focal))
    return math.degrees(fov_rad)


def analyze_camera_categories(images: List[Dict[str, Any]]) -> Tuple[Dict[str, CameraCategory], Dict[str, float]]:
    """Analyze images and group by camera category.

    Only processes perspective cameras. Calculates percentages.

    Args:
        images: List of image entities with camera data

    Returns:
        Tuple of (categories dict, percentages dict)
    """
    categories = defaultdict(lambda: CameraCategory(
        name="", focal_range=(float('inf'), 0), fov_range=(float('inf'), 0)
    ))

    total_perspective = 0
    total_images = 0

    for img in images:
        category_name = categorize_camera(img)
        total_images += 1

        # Skip non-perspective and unknown cameras
        if category_name is None or category_name == "Unknown (no intrinsics)":
            continue

        total_perspective += 1

        category = categories[category_name]
        category.name = category_name
        category.count += 1

        # Store sample image (first 3 per category)
        if len(category.sample_images) < 3:
            image_id = img.get("id")
            if image_id:
                category.sample_images.append(image_id)

        # Track focal and FOV ranges
        params = img.get("camera_parameters", [])
        if params and params[0]:
            focal = params[0]
            category.focal_range = (
                min(category.focal_range[0], focal),
                max(category.focal_range[1], focal)
            )

            # Calculate FOV
            width = img.get("width", 1024)
            fov = calculate_fov_from_focal(focal, width, img.get("height", 512))
            category.fov_range = (
                min(category.fov_range[0], fov),
                max(category.fov_range[1], fov)
            )

    # Calculate percentages
    percentages = {}
    if total_perspective > 0:
        for name, category in categories.items():
            percentages[name] = (category.count / total_perspective) * 100
        percentages["total_perspective"] = (total_perspective / total_images) * 100
        percentages["total_images"] = total_images

    return dict(categories), percentages


def generate_html_visualization(
    categories: Dict[str, CameraCategory],
    percentages: Dict[str, float],
    images: List[Dict[str, Any]],
    output_path: str = "data/debug/camera_categories.html"
) -> None:
    """Generate interactive HTML visualization.

    Args:
        categories: Camera categories from analyze_camera_categories
        percentages: Percentage breakdown of categories
        images: Full list of images with GPS data
        output_path: Where to save HTML file
    """

    html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Camera Parameter Analysis</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary h2 { margin-top: 0; color: #4CAF50; }
        .category-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .category-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .category-card h3 { margin-top: 0; color: #333; }
        .category-card .count { font-size: 2em; font-weight: bold; color: #4CAF50; }
        .category-card .percentage { font-size: 1.2em; font-weight: bold; color: #2196F3; }
        .category-card .range { color: #666; font-size: 0.9em; }
        .map-container { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        #map { height: 500px; border-radius: 4px; }
        .legend { background: white; padding: 15px; border-radius: 8px; margin-top: 10px; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color { width: 20px; height: 20px; border-radius: 50%; margin-right: 10px; }
        .sample-images { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin-top: 10px; }
        .sample-image { background: #f0f0f0; border-radius: 4px; padding: 10px; text-align: center; font-size: 0.8em; }
        .params-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .params-table th, .params-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .params-table th { background: #f5f5f5; }
        .color-narrow { background: #e74c3c; }
        .color-normal { background: #3498db; }
        .color-wide { background: #2ecc71; }
        .info-box { background: #e3f2fd; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px; }
        .filter-info { background: #fff3e0; border-left: 4px solid #FF9800; padding: 15px; margin: 20px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📷 Camera Parameter Analysis (Perspective Only)</h1>

        <div class="filter-info">
            <strong>Filtered:</strong> Only perspective cameras included (fisheye and spherical excluded)
            <br><strong>Total Perspective:</strong> {{ "%.1f"|format(percentages.get("total_perspective", 0)) }}% of {{ percentages.get("total_images", 0) }} total images
        </div>

        <div class="info-box">
            <strong>Purpose:</strong> Verify depth estimation accuracy by comparing camera categories.
            <br><br>
            <strong>How to use:</strong>
            <ol>
                <li>Review camera categories below (grouped by FOV)</li>
                <li>For each category, open sample images on the map</li>
                <li>On Google Maps, measure distance from camera to a visible landmark</li>
                <li>Compare with our depth estimation to validate accuracy</li>
            </ol>
        </div>

        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Total Categories:</strong> {{ categories|length }}</p>
            <p><strong>Total Perspective Images:</strong> {{ percentages.get("total_perspective", 0) }}%</p>
        </div>

        <div class="category-grid">
            {% for name, category in categories.items() %}
            <div class="category-card">
                <h3>{{ name }}</h3>
                <div class="count">{{ category.count }}</div>
                <div class="percentage">{{ "%.1f"|format(percentages[name]) }}%</div>
                <p class="range">FOV Range: {{ "%.1f"|format(category.fov_range[0]) }}° - {{ "%.1f"|format(category.fov_range[1]) }}°</p>
                <p class="range">Focal Range: {{ "%.3f"|format(category.focal_range[0]) }} - {{ "%.3f"|format(category.focal_range[1]) }}</p>

                {% if category.sample_images %}
                <p><strong>Sample Images:</strong></p>
                <div class="sample-images">
                    {% for img_id in category.sample_images[:3] %}
                    <div class="sample-image">
                        <a href="https://www.mapillary.com/app/?pKey={{ img_id }}" target="_blank">
                            {{ img_id[:12] }}...
                        </a>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>

        <div class="map-container">
            <h2>Camera Positions by Category (Perspective Only)</h2>
            <div id="map"></div>
            <div class="legend">
                <strong>Legend:</strong><br>
                {% for name, category in categories.items() %}
                <div class="legend-item">
                    <div class="legend-color {{ category.color_class }}"></div>
                    <span>{{ name }} ({{ category.count }}, {{ "%.1f"|format(percentages[name]) }}%)</span>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="summary">
            <h2>Key Camera Parameters Affecting Depth</h2>
            <table class="params-table">
                <tr>
                    <th>Parameter</th>
                    <th>Affects</th>
                    <th>Why It Matters</th>
                </tr>
                <tr>
                    <td><strong>Focal Length</strong></td>
                    <td>Depth scale, FOV</td>
                    <td>Primary factor for pixel-to-metric conversion. Shorter focal = wider FOV = depth appears compressed</td>
                </tr>
                <tr>
                    <td><strong>Camera Type</strong></td>
                    <td>Projection model</td>
                    <td>This analysis: Perspective only (standard). Fisheye/spherical excluded.</td>
                </tr>
                <tr>
                    <td><strong>Image Width/Height</strong></td>
                    <td>Pixel-to-meter ratio</td>
                    <td>Larger images = more pixels = better depth resolution</td>
                </tr>
                <tr>
                    <td><strong>k1, k2 (Distortion)</strong></td>
                    <td>Radial distortion</td>
                    <td>Lens barrel/pincushion distortion. Affects pixels at image edges</td>
                </tr>
                <tr>
                    <td><strong>Compass Angle</strong></td>
                    <td>Direction of view</td>
                    <td>Camera heading. Critical for projecting depth to GPS</td>
                </tr>
                <tr>
                    <td><strong>Pitch</strong></td>
                    <td>Vertical angle</td>
                    <td>Camera tilt. Affects whether POI is visible and depth accuracy</td>
                </tr>
            </table>
        </div>
    </div>

    <script>
        // Initialize map
        const map = L.map('map').setView([10.77, 106.69], 14);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
        }).addTo(map);

        // Category colors
        const categoryColors = {
            {% for name, category in categories.items() %}
            '{{ name }}': '{{ category.color }}',
            {% endfor %}
        };

        // Add camera markers
        {% for img in images %}
        L.circleMarker([{{ img.lat }}, {{ img.lon }}], {
            radius: 5,
            fillColor: '{{ img.color }}',
            color: '#000',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        }).addTo(map)
            .bindPopup(`
                <strong>{{ img.category }}</strong><br>
                Image ID: {{ img.id }}<br>
                <a href="https://www.mapillary.com/app/?pKey={{ img.id }}" target="_blank">View on Mapillary</a><br>
                <br>
                <strong>Parameters:</strong><br>
                Focal: {{ img.focal }}<br>
                FOV: {{ "%.1f"|format(img.fov) }}°<br>
                Heading: {{ img.heading }}°<br>
                Type: {{ img.camera_type }}
            `);
        {% endfor %}
    </script>
</body>
</html>
    """)

    # Prepare data for template
    category_colors = {
        "Narrow FOV (<60°)": "#e74c3c",
        "Normal FOV (60-90°)": "#3498db",
        "Wide FOV (>90°)": "#2ecc71",
    }

    for name, category in categories.items():
        category.color = category_colors.get(name, "#95a5a6")
        # Assign color class for legend
        if "Narrow" in name:
            category.color_class = "color-narrow"
        elif "Normal" in name:
            category.color_class = "color-normal"
        elif "Wide" in name:
            category.color_class = "color-wide"

    # Prepare images with metadata
    template_data = []
    for img in images:
        category_name = categorize_camera(img)
        params = img.get("camera_parameters", [])
        focal = params[0] if params else None

        geometry = img.get("geometry", {})
        coords = geometry.get("coordinates", [0, 0])

        width = img.get("width", 1024)
        fov = calculate_fov_from_focal(focal, width, img.get("height", 512)) if focal else 60

        template_data.append({
            "id": img.get("id"),
            "lat": coords[1],
            "lon": coords[0],
            "category": category_name,
            "color": category_colors.get(category_name, "#95a5a6"),
            "focal": focal,
            "fov": fov,
            "heading": img.get("compass_angle", 0),
            "camera_type": img.get("camera_type", "unknown")
        })

    # Render template
    html = html_template.render(
        categories=categories,
        percentages=percentages,
        images=template_data,
        total_images=len(images)
    )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)

    print(f"Camera analysis saved to: {output_path}")


def fetch_camera_parameters(client, image_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch detailed camera parameters from Mapillary API.

    Args:
        client: MapillaryClient instance
        image_ids: List of image IDs to fetch

    Returns:
        List of image entities with camera parameters
    """
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not access_token:
        print("Error: MAPILLARY_ACCESS_TOKEN not found")
        return []

    BATCH_SIZE = 100
    entities = []

    for i in range(0, len(image_ids), BATCH_SIZE):
        batch = image_ids[i:i + BATCH_SIZE]
        ids_str = ",".join(str(img_id) for img_id in batch)

        # Request camera_parameters, camera_type, width, height
        fields = "id,camera_parameters,camera_type,width,height,geometry,compass_angle"

        url = "https://graph.mapillary.com"
        params = {
            "ids": ids_str,
            "fields": fields,
            "access_token": access_token
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Convert dict values to list
            batch_entities = list(data.values())
            entities.extend(batch_entities)

            print(f"  Fetched batch {i//BATCH_SIZE + 1}: {len(batch_entities)} entities")

        except Exception as e:
            print(f"  Warning: Batch fetch failed: {e}")
            continue

    return entities


def main():
    """Main entry point."""
    from dotenv import load_dotenv

    load_dotenv()

    # Load sampled images from tutorial data
    tile_dir = Path("data/raw/tiles/small_test")
    if not tile_dir.exists():
        print("Error: Run tutorial Blocks 1-7 first to generate tile data")
        sys.exit(1)

    # Load image IDs from tiles
    image_ids = []
    image_id_to_geometry = {}
    for tile_file in sorted(tile_dir.glob("*.json")):
        with open(tile_file, "r") as f:
            tile_data = json.load(f)

        for feature in tile_data.get("features", []):
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            img_id = props.get("id")
            if img_id:
                image_ids.append(img_id)
                # Store geometry for later
                image_id_to_geometry[img_id] = {
                    "geometry": geometry,
                    "compass_angle": props.get("compass_angle", 0),
                }

    print(f"Loaded {len(image_ids)} image IDs from tiles")

    # Sample first 100 images (to avoid too many API calls)
    SAMPLE_SIZE = min(100, len(image_ids))
    import random
    random.seed(42)
    sampled_ids = random.sample(image_ids, SAMPLE_SIZE)

    print(f"Sampling {SAMPLE_SIZE} images for detailed analysis")
    print("Fetching camera parameters from Mapillary API...")

    # Import MapillaryClient
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from crawler import MapillaryClient

    load_dotenv()
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    client = MapillaryClient(access_token)

    # Fetch camera parameters
    entities = fetch_camera_parameters(client, sampled_ids)

    if not entities:
        print("Error: No entities fetched. Check your MAPILLARY_ACCESS_TOKEN")
        sys.exit(1)

    # Merge geometry data
    for entity in entities:
        img_id = entity.get("id")
        if img_id and img_id in image_id_to_geometry:
            entity.update(image_id_to_geometry[img_id])

    print(f"\nAnalyzing {len(entities)} perspective cameras...")

    # Analyze categories and get percentages
    categories, percentages = analyze_camera_categories(entities)

    print(f"\nPerspective Camera Categories:")
    print(f"{'Category':<30} {'Count':>8} {'Percentage':>12}")
    print("-" * 52)
    for name, category in sorted(categories.items(), key=lambda x: x[1].count, reverse=True):
        print(f"{name:<30} {category.count:>8} {percentages[name]:>8.1f}%")

    print(f"\nSummary:")
    print(f"  Total perspective: {percentages.get('total_perspective', 0):.1f}%")
    print(f"  Non-perspective excluded: {percentages.get('total_images', 0) - percentages.get('total_perspective', 0):.1f}%")

    # Generate visualization
    generate_html_visualization(
        categories,
        percentages,
        entities,
        "data/debug/camera_categories.html"
    )

    print("\nOpen data/debug/camera_categories.html in your browser to view")
    print("Click on markers to see camera details and link to Mapillary")


if __name__ == "__main__":
    main()
