#!/usr/bin/env python3
"""Investigate sequence metadata from Mapillary tiles.

Analyzes sequences in existing tile data:
- Groups images by sequence_id
- For each unique sequence, fetches ONE image entity to get camera info
- Builds sequence profile: device type, image count, time span
- Generates HTML visualization

Outputs:
- data/debug/sequence_profile_summary.json - Sequence analysis results
- data/debug/sequence_analysis.html - Interactive visualization
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

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


def group_and_summarize_sequences(images: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group images by sequence and generate initial summaries.

    Args:
        images: List of image dicts from tiles

    Returns:
        Dict with 'grouped' and 'summary' keys
    """
    # Import after adding to path
    from crawler import MapillaryClient

    grouped = MapillaryClient(access_token="dummy").group_tiles_by_sequence(images)

    # Get summaries without API calls
    summaries = []
    for seq_id, seq_images in grouped.items():
        if not seq_images:
            continue

        creator_id = seq_images[0].get("creator_id", "unknown")
        capture_times = [img.get("captured_at") for img in seq_images if img.get("captured_at")]

        summaries.append({
            "sequence_id": seq_id,
            "image_count": len(seq_images),
            "creator_id": creator_id,
            "first_capture_time": min(capture_times) if capture_times else None,
            "last_capture_time": max(capture_times) if capture_times else None,
            "sample_image_id": seq_images[0].get("image_id"),
        })

    # Sort by image count
    summaries.sort(key=lambda x: x["image_count"], reverse=True)

    return {
        "grouped": grouped,
        "summaries": summaries,
    }


def fetch_sequence_entities(
    client: MapillaryClient,
    summaries: List[Dict[str, Any]],
    max_sequences: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch entity metadata for one image per sequence.

    NOTE ON CAMERA METADATA AVAILABILITY:
    --------------------------------------
    This function attempts to fetch camera metadata (camera_type, make, model)
    by sampling one image from each sequence. However, the sample image IDs
    from tile data may be invalid/deleted, causing the API to return errors.

    IMPACT:
    - If camera metadata fetch fails, the sequence is still included in results
      but without camera fields (they will be None/empty)
    - Empty camera distributions in final output indicate API fetch failures
    - Sequence grouping and image count statistics remain valid

    WHY THIS HAPPENS:
    - Tile data contains image IDs that may be deleted/invalid
    - Mapillary's individual image endpoint returns 404 for invalid IDs
    - The tenacity retry mechanism exhausts retries before continuing

    ALTERNATIVE APPROACH:
    - The depth investigation script (04_investigate_depth.py) uses random
      sampling which yields higher success rates for valid image IDs
    - For camera metadata, refer to depth_investigation_summary.json results

    Args:
        client: MapillaryClient instance
        summaries: List of sequence summaries
        max_sequences: Maximum number of sequences to analyze

    Returns:
        List of enriched sequence summaries with camera info (if available)
    """
    # Limit to top sequences by image count
    top_summaries = summaries[:max_sequences]

    enriched = []

    for seq in tqdm(top_summaries, desc="Fetching sequence metadata"):
        try:
            entity = client.fetch_image_entity(seq["sample_image_id"])

            enriched_seq = {
                **seq,
                "camera_type": entity.get("camera_type"),
                "camera_make": entity.get("make"),
                "camera_model": entity.get("model"),
                "camera_parameters": entity.get("camera_parameters", {}),
                "altitude": entity.get("altitude"),
                "computed_altitude": entity.get("computed_altitude"),
            }

            enriched.append(enriched_seq)

        except Exception as e:
            print(f"Warning: Failed to fetch entity for sequence {seq['sequence_id']}: {e}")
            # Still add the summary without camera info
            # The sequence data (image_count, creator_id, etc.) remains valid
            enriched.append(seq)

    return enriched


def analyze_sequences(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze sequence metadata.

    Args:
        sequences: List of enriched sequence summaries

    Returns:
        Analysis results dict
    """
    camera_types = []
    camera_makes = []
    camera_models = []
    image_counts = []

    for seq in sequences:
        camera_type = seq.get("camera_type")
        camera_make = seq.get("camera_make")
        camera_model = seq.get("camera_model")

        if camera_type:
            camera_types.append(camera_type)
        if camera_make:
            camera_makes.append(camera_make)
        if camera_model:
            camera_models.append(camera_model)

        image_counts.append(seq["image_count"])

    # Count distributions
    camera_type_counts = dict(Counter(camera_types))
    camera_make_counts = dict(Counter(camera_makes))
    camera_model_counts = dict(Counter(camera_models))

    # Image count stats
    image_counts.sort()
    image_count_stats = {
        "min": image_counts[0] if image_counts else 0,
        "max": image_counts[-1] if image_counts else 0,
        "median": image_counts[len(image_counts) // 2] if image_counts else 0,
        "mean": sum(image_counts) / len(image_counts) if image_counts else 0,
        "total": sum(image_counts),
    }

    # Group by image count ranges
    count_ranges = {
        "1-10": sum(1 for c in image_counts if 1 <= c <= 10),
        "11-50": sum(1 for c in image_counts if 11 <= c <= 50),
        "51-100": sum(1 for c in image_counts if 51 <= c <= 100),
        "100+": sum(1 for c in image_counts if c > 100),
    }

    return {
        "total_sequences_analyzed": len(sequences),
        "total_images_in_sequences": image_count_stats["total"],
        "camera_type_distribution": camera_type_counts,
        "camera_make_distribution": camera_make_counts,
        "camera_model_distribution": dict(list(camera_model_counts.items())[:10]),
        "image_count_stats": image_count_stats,
        "image_count_ranges": count_ranges,
    }


def generate_html_visualization(
    sequences: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    output_path: str = "data/debug/sequence_analysis.html",
) -> None:
    """Generate HTML visualization of sequence analysis.

    Args:
        sequences: List of enriched sequence summaries
        analysis: Analysis results
        output_path: Where to save the HTML file
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed, skipping HTML visualization")
        print("Install with: pip install plotly")
        return

    # 1. Image Count Distribution
    count_ranges = analysis["image_count_ranges"]
    fig_counts = go.Figure(data=[go.Bar(
        x=list(count_ranges.keys()),
        y=list(count_ranges.values()),
        marker_color='steelblue',
    )])
    fig_counts.update_layout(
        title="Sequence Image Count Distribution",
        xaxis_title="Images per Sequence",
        yaxis_title="Number of Sequences",
    )

    # 2. Camera Type Distribution
    type_dist = analysis["camera_type_distribution"]

    # 3. Top Sequences by Image Count
    top_sequences = sorted(sequences, key=lambda x: x["image_count"], reverse=True)[:20]
    fig_top = go.Figure(data=[go.Bar(
        x=[s["sequence_id"][:8] + "..." for s in top_sequences],
        y=[s["image_count"] for s in top_sequences],
        marker_color='green',
    )])
    fig_top.update_layout(
        title="Top 20 Sequences by Image Count",
        xaxis_title="Sequence ID",
        yaxis_title="Image Count",
    )

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Sequence Image Count Distribution",
            "Camera Type Distribution",
            "Top 20 Sequences",
            "Summary Stats",
        ),
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "bar"}, {"type": "indicator"}],
        ],
    )

    # Add traces
    fig.add_trace(go.Bar(
        x=list(count_ranges.keys()),
        y=list(count_ranges.values()),
        marker_color='steelblue',
    ), row=1, col=1)

    if type_dist:
        fig.add_trace(go.Pie(
            labels=list(type_dist.keys()),
            values=list(type_dist.values()),
        ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=[s["sequence_id"][:8] + "..." for s in top_sequences],
        y=[s["image_count"] for s in top_sequences],
        marker_color='green',
    ), row=2, col=1)

    # Summary indicator
    total_images = analysis["total_images_in_sequences"]
    fig.add_trace(go.Indicator(
        mode="number",
        value=total_images,
        title={"text": f"Total Images in {len(sequences)} Sequences"},
    ), row=2, col=2)

    fig.update_layout(
        title_text="Mapillary Sequence Analysis - HCMC POI Extraction",
        height=800,
    )

    # Save as HTML
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))


def save_summary_json(
    sequences: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    output_path: str = "data/debug/sequence_profile_summary.json",
) -> None:
    """Save sequence summary to JSON.

    Args:
        sequences: List of enriched sequence summaries
        analysis: Analysis results
        output_path: Where to save the JSON file
    """
    output = {
        "analysis": analysis,
        "top_sequences": sequences[:50],  # Top 50 sequences
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    """Main entry point for sequence investigation."""
    load_dotenv()

    # Check for MAPILLARY_ACCESS_TOKEN
    import os
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not access_token:
        print("Error: MAPILLARY_ACCESS_TOKEN not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)

    print("=" * 60)
    print("Mapillary Sequence Metadata Investigation")
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

    # Group by sequence
    print("\nGrouping images by sequence...")
    result = group_and_summarize_sequences(all_images)
    grouped = result["grouped"]
    summaries = result["summaries"]

    print(f"Found {len(grouped)} unique sequences")
    print(f"Total images in sequences: {sum(s['image_count'] for s in summaries)}")

    # Print some stats before fetching
    print("\nTop 10 sequences by image count:")
    for i, seq in enumerate(summaries[:10], 1):
        print(f"  {i}. {seq['sequence_id']}: {seq['image_count']} images")

    # Fetch sequence metadata (limit to top 100)
    max_sequences = 100
    print(f"\nFetching metadata for top {max_sequences} sequences...")
    enriched = fetch_sequence_entities(client, summaries, max_sequences=max_sequences)
    print(f"Successfully enriched {len(enriched)} sequences")

    # Analyze
    print("\nAnalyzing sequence metadata...")
    analysis = analyze_sequences(enriched)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total sequences analyzed: {analysis['total_sequences_analyzed']}")
    print(f"Total images in sequences: {analysis['total_images_in_sequences']}")
    print(f"\nImage Count Stats:")
    stats = analysis['image_count_stats']
    print(f"  - Min: {stats['min']}")
    print(f"  - Max: {stats['max']}")
    print(f"  - Median: {stats['median']}")
    print(f"  - Mean: {stats['mean']:.1f}")

    # Camera metadata may be empty if API fetches failed
    print(f"\nCamera Types:")
    if analysis['camera_type_distribution']:
        for k, v in analysis['camera_type_distribution'].items():
            print(f"  - {k}: {v}")
    else:
        print("  (No camera type data available - see notes in fetch_sequence_entities())")

    print(f"\nCamera Makes (top 5):")
    if analysis['camera_make_distribution']:
        for make, count in list(analysis['camera_make_distribution'].items())[:5]:
            print(f"  - {make}: {count}")
    else:
        print("  (No camera make data available - see notes in fetch_sequence_entities())")
        print("  For camera metadata, refer to: data/debug/depth_investigation_summary.json")

    # Save results
    print("\nSaving results...")
    save_summary_json(enriched, analysis)
    print(f"  - Saved summary to: data/debug/sequence_profile_summary.json")

    generate_html_visualization(enriched, analysis)
    print(f"  - Saved visualization to: data/debug/sequence_analysis.html")

    print("\n" + "=" * 60)
    print("Investigation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
