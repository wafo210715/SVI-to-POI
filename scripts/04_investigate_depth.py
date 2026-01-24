#!/usr/bin/env python3
"""Investigate camera parameters and depth availability from Mapillary tiles.

Fetches 100 random images from existing tile data and analyzes:
- Camera parameters (focal, k1, k2) for monocular depth estimation
- Camera type distribution (perspective/fisheye/spherical)
- Camera make/model distribution
- Per-sequence camera consistency

Outputs:
- data/debug/depth_investigation_summary.json - Investigation results
- data/debug/camera_analysis.html - Interactive visualization
"""

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crawler import MapillaryClient
from dotenv import load_dotenv
from tqdm import tqdm


def load_all_tile_images(tile_dir: Path) -> List[Dict[str, Any]]:
    """Load all images from tile JSON files.

    Args:
        tile_dir: Directory containing tile JSON files

    Returns:
        List of all image metadata dicts from tiles
    """
    all_images = []

    for tile_file in tile_dir.glob("*.json"):
        with open(tile_file, "r") as f:
            tile_data = json.load(f)

        for feature in tile_data.get("features", []):
            props = feature.get("properties", {})
            geometry = feature.get("geometry", {})

            img = {
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
            }
            all_images.append(img)

    return all_images


def sample_images(images: List[Dict[str, Any]], n: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample random images from the list.

    Args:
        images: List of image dicts
        n: Number of samples to draw
        seed: Random seed for reproducibility

    Returns:
        List of sampled image dicts
    """
    random.seed(seed)
    return random.sample(images, min(n, len(images)))


def fetch_image_entities(
    client: MapillaryClient,
    images: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fetch full entity metadata for each image using batch API.

    Uses Mapillary's batch entity endpoint to fetch multiple images at once.

    Args:
        client: MapillaryClient instance
        images: List of image dicts with at least 'image_id'

    Returns:
        List of full entity responses
    """
    entities = []
    batch_size = 100  # Max batch size for Mapillary API

    # Process in batches
    for i in tqdm(range(0, len(images), batch_size), desc="Fetching image entities"):
        batch = images[i:i + batch_size]
        image_ids = [str(img["image_id"]) for img in batch]

        try:
            # Use batch endpoint - returns dict keyed by image ID
            ids_str = ",".join(image_ids)
            fields = "id,camera_type,camera_parameters,make,model,sequence,altitude,computed_altitude"

            params = {
                "ids": ids_str,
                "fields": fields,
                "access_token": client.access_token,
            }

            response = requests.get(client.GRAPH_API_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()

            # Mapillary batch API returns a dict keyed by image ID
            if isinstance(data, dict):
                # Convert dict values to list
                batch_entities = list(data.values())
                entities.extend(batch_entities)
                print(f"  Fetched {len(batch_entities)} entities from batch")
            elif isinstance(data, list):
                entities.extend(data)

        except Exception as e:
            print(f"Warning: Batch fetch failed for {len(image_ids)} images: {e}")
            # Continue to next batch on error
            continue

    return entities


def analyze_camera_parameters(entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze camera parameters from fetched entities.

    Args:
        entities: List of image entity responses

    Returns:
        Analysis results dict
    """
    camera_types = []
    camera_makes = []
    camera_models = []
    has_intrinsics = []
    focal_lengths = []
    sequences = set()
    sequence_cameras = defaultdict(set)

    for entity in entities:
        camera_type = entity.get("camera_type")
        camera_make = entity.get("make")
        camera_model = entity.get("model")
        camera_params = entity.get("camera_parameters", [])
        sequence_id = entity.get("sequence")

        if camera_type:
            camera_types.append(camera_type)
        if camera_make:
            camera_makes.append(camera_make)
        if camera_model:
            camera_models.append(camera_model)

        # Check for intrinsics - Mapillary returns [focal, k1, k2] array
        has_focal = camera_params and len(camera_params) >= 1 and camera_params[0] is not None
        has_intrinsics.append(has_focal)

        if has_focal:
            focal = camera_params[0]  # First element is focal length
            if isinstance(focal, (int, float)):
                focal_lengths.append(float(focal))

        if sequence_id:
            sequences.add(sequence_id)
            if camera_type:
                sequence_cameras[sequence_id].add(camera_type)

    # Count camera types
    camera_type_counts = dict(Counter(camera_types))
    camera_make_counts = dict(Counter(camera_makes))
    camera_model_counts = dict(Counter(camera_models))

    # Sequence consistency
    sequences_with_consistent_camera = sum(
        1 for cams in sequence_cameras.values() if len(cams) == 1
    )

    return {
        "total_images_tested": len(entities),
        "camera_params": {
            "perspective": camera_type_counts.get("perspective", 0),
            "fisheye": camera_type_counts.get("fisheye", 0),
            "spherical": camera_type_counts.get("spherical", 0),
            "equirectangular": camera_type_counts.get("equirectangular", 0),
            "unknown": camera_type_counts.get("unknown", 0),
            "has_valid_intrinsics": sum(1 for x in has_intrinsics if x),
        },
        "camera_makes": camera_make_counts,
        "camera_models": dict(list(camera_model_counts.items())[:10]),  # Top 10
        "focal_length_stats": {
            "min": min(focal_lengths) if focal_lengths else None,
            "max": max(focal_lengths) if focal_lengths else None,
            "mean": sum(focal_lengths) / len(focal_lengths) if focal_lengths else None,
        },
        "sequences_analyzed": len(sequences),
        "sequences_with_consistent_camera": sequences_with_consistent_camera,
    }


def determine_feasibility(analysis: Dict[str, Any]) -> str:
    """Determine if monocular depth estimation is feasible.

    Args:
        analysis: Analysis results from analyze_camera_parameters

    Returns:
        Feasibility recommendation: "monocular_depth_feasible", "partial", or "not_feasible"
    """
    total = analysis["total_images_tested"]
    has_intrinsics = analysis["camera_params"]["has_valid_intrinsics"]
    perspective = analysis["camera_params"]["perspective"]

    # Need at least 80% with intrinsics and mostly perspective cameras
    intrinsic_ratio = has_intrinsics / total if total > 0 else 0
    perspective_ratio = perspective / total if total > 0 else 0

    if intrinsic_ratio >= 0.8 and perspective_ratio >= 0.7:
        return "monocular_depth_feasible"
    elif intrinsic_ratio >= 0.5 and perspective_ratio >= 0.5:
        return "partial"
    else:
        return "not_feasible"


def generate_html_visualization(
    analysis: Dict[str, Any],
    output_path: str = "data/debug/camera_analysis.html",
) -> None:
    """Generate HTML visualization of camera analysis.

    Args:
        analysis: Analysis results from analyze_camera_parameters
        output_path: Where to save the HTML file
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Prepare data
    camera_types = analysis["camera_params"]
    total = analysis["total_images_tested"]

    # 1. Camera Type Pie Chart
    type_labels = []
    type_values = []
    for key in ["perspective", "fisheye", "spherical", "equirectangular"]:
        val = camera_types.get(key, 0)
        if val > 0:
            type_labels.append(key.capitalize())
            type_values.append(val)

    fig_pie = go.Figure(data=[go.Pie(
        labels=type_labels,
        values=type_values,
        title="Camera Type Distribution",
        hole=0.3,
    )])

    # 2. Camera Make Bar Chart
    makes = analysis["camera_makes"]
    if makes:
        make_fig = go.Figure(data=[go.Bar(
            x=list(makes.keys()),
            y=list(makes.values()),
            marker_color='steelblue',
        )])
        make_fig.update_layout(
            title="Camera Make Distribution",
            xaxis_title="Make",
            yaxis_title="Count",
        )
    else:
        make_fig = go.Figure()

    # 3. Intrinsics Availability
    has_intrinsics = camera_types["has_valid_intrinsics"]

    fig_intrinsics = go.Figure(data=[go.Bar(
        x=["With Intrinsics", "Without Intrinsics"],
        y=[has_intrinsics, total - has_intrinsics],
        marker_color=['green', 'red'],
    )])
    fig_intrinsics.update_layout(
        title="Camera Intrinsics Availability",
        yaxis_title="Count",
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Camera Type Distribution",
            "Intrinsics Availability",
            "Camera Make Distribution",
            "Sequence Consistency",
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
    )

    # Add traces
    fig.add_trace(fig_pie.data[0], row=1, col=1)
    fig.add_trace(fig_intrinsics.data[0], row=1, col=2)

    if makes:
        fig.add_trace(go.Bar(
            x=list(makes.keys()),
            y=list(makes.values()),
            marker_color='steelblue',
        ), row=2, col=1)

    # Sequence consistency indicator
    seq_ratio = analysis["sequences_with_consistent_camera"] / analysis["sequences_analyzed"]
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=seq_ratio * 100,
        title={"text": "% Sequences with Consistent Camera"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightgray"},
                {"range": [50, 80], "color": "gray"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ), row=2, col=2)

    fig.update_layout(
        title_text="Mapillary Camera Analysis - HCMC POI Extraction",
        showlegend=False,
        height=800,
    )

    # Save as HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))


def save_summary_json(
    analysis: Dict[str, Any],
    feasibility: str,
    output_path: str = "data/debug/depth_investigation_summary.json",
) -> None:
    """Save investigation summary to JSON.

    Args:
        analysis: Analysis results from analyze_camera_parameters
        feasibility: Feasibility recommendation
        output_path: Where to save the JSON file
    """
    output = {
        **analysis,
        "feasibility_recommendation": feasibility,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    """Main entry point for depth investigation."""
    load_dotenv()

    # Check for MAPILLARY_ACCESS_TOKEN
    import os
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not access_token:
        print("Error: MAPILLARY_ACCESS_TOKEN not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)

    print("=" * 60)
    print("Mapillary Depth & Camera Investigation")
    print("=" * 60)

    # Initialize client
    client = MapillaryClient(access_token)

    # Load tile data
    tile_dir = Path("data/raw/tiles/small_test")
    if not tile_dir.exists():
        print(f"Error: Tile directory not found: {tile_dir}")
        print("Please run scripts/02_scrape_mapillary_tiles.py first")
        sys.exit(1)

    print(f"\nLoading tile data from {tile_dir}...")
    all_images = load_all_tile_images(tile_dir)
    print(f"Found {len(all_images)} total images in tiles")

    # Sample images
    sample_size = 100
    print(f"\nSampling {sample_size} random images...")
    sampled = sample_images(all_images, n=sample_size)

    # Fetch entities
    print(f"\nFetching detailed metadata for {len(sampled)} images...")
    entities = fetch_image_entities(client, sampled)
    print(f"Successfully fetched {len(entities)} entities")

    if len(entities) == 0:
        print("Error: No entities fetched. Please check your API token")
        sys.exit(1)

    # Analyze
    print("\nAnalyzing camera parameters...")
    analysis = analyze_camera_parameters(entities)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total images tested: {analysis['total_images_tested']}")
    print(f"\nCamera Types:")
    for k, v in analysis["camera_params"].items():
        if k != "has_valid_intrinsics":
            print(f"  - {k}: {v}")
    print(f"\nWith valid intrinsics: {analysis['camera_params']['has_valid_intrinsics']}")
    print(f"\nCamera Makes (top 5):")
    for make, count in list(analysis["camera_makes"].items())[:5]:
        print(f"  - {make}: {count}")
    print(f"\nSequences analyzed: {analysis['sequences_analyzed']}")
    print(f"Sequences with consistent camera: {analysis['sequences_with_consistent_camera']}")

    # Determine feasibility
    feasibility = determine_feasibility(analysis)
    print(f"\nFeasibility Recommendation: {feasibility}")

    # Save results
    print("\nSaving results...")
    save_summary_json(analysis, feasibility)
    print(f"  - Saved summary to: data/debug/depth_investigation_summary.json")

    generate_html_visualization(analysis)
    print(f"  - Saved visualization to: data/debug/camera_analysis.html")

    print("\n" + "=" * 60)
    print("Investigation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
