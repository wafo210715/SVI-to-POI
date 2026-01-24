#!/usr/bin/env python3
"""Scrape Mapillary tiles to visualize HCMC coverage.

This script:
1. Gets all tile coordinates for HCMC at zoom 14
2. Fetches tile data to see image counts
3. Generates HTML visualization
4. Saves raw tile data for analysis
"""

import json
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from src.crawler import MapillaryClient

# HCMC bounding boxes for different areas
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


def analyze_area(
    client: MapillaryClient,
    area_key: str,
    area_config: Dict,
) -> Dict:
    """Analyze Mapillary coverage for a specific area.

    Returns:
        Summary dict with tile info and image counts
    """
    print(f"\n{'=' * 50}")
    print(f"Analyzing: {area_config['name']}")
    print(f"{'=' * 50}")

    min_lon = area_config["min_lon"]
    min_lat = area_config["min_lat"]
    max_lon = area_config["max_lon"]
    max_lat = area_config["max_lat"]

    # Get tile coordinates
    tiles = client.get_tiles_for_bbox(min_lon, min_lat, max_lon, max_lat, zoom=14)
    print(f"Total tiles: {len(tiles)}")

    # Fetch each tile
    tiles_with_data = []
    total_images = 0

    for tile in tqdm(tiles, desc=f"Fetching tiles ({area_key})"):
        try:
            tile_data = client._fetch_tile(tile["z"], tile["x"], tile["y"])
            feature_count = len(tile_data.get("features", []))

            tiles_with_data.append({
                **tile,
                "image_count": feature_count,
                "feature_count": feature_count,
            })

            total_images += feature_count

            # Save raw tile data
            output_dir = Path("data/raw/tiles") / area_key
            output_dir.mkdir(parents=True, exist_ok=True)
            tile_file = output_dir / f"tile_{tile['z']}_{tile['x']}_{tile['y']}.json"
            with open(tile_file, "w") as f:
                json.dump(tile_data, f, indent=2)

        except Exception as e:
            print(f"\n  Warning: Failed tile {tile['z']}/{tile['x']}/{tile['y']}: {e}")
            tiles_with_data.append({**tile, "image_count": 0})

    # Generate visualization
    viz_path = Path(f"data/debug/tiles_{area_key}_viz.html")
    client._generate_tile_visualization_html(
        tiles=tiles_with_data,
        bbox=area_config,
        output_path=str(viz_path),
    )
    print(f"Visualization saved to: {viz_path}")

    return {
        "area_key": area_key,
        "area_name": area_config["name"],
        "total_tiles": len(tiles),
        "tiles_with_images": sum(1 for t in tiles_with_data if t["image_count"] > 0),
        "total_images": total_images,
        "tiles": tiles_with_data,
    }


def print_summary(all_results: List[Dict]) -> None:
    """Print overall summary."""
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY - Mapillary HCMC Coverage")
    print("=" * 60)

    for result in all_results:
        area = result["area_name"]
        tiles = result["total_tiles"]
        with_images = result["tiles_with_images"]
        total_imgs = result["total_images"]

        print(f"\n{area}:")
        print(f"  Tiles: {tiles}")
        print(f"  Tiles with images: {with_images} ({100 * with_images / tiles if tiles > 0 else 0:.1f}%)")
        print(f"  Total images: {total_imgs}")

    # Grand totals
    grand_tiles = sum(r["total_tiles"] for r in all_results)
    grand_with_images = sum(r["tiles_with_images"] for r in all_results)
    grand_total_images = sum(r["total_images"] for r in all_results)

    print(f"\n{'=' * 60}")
    print(f"TOTALS:")
    print(f"  Total tiles analyzed: {grand_tiles}")
    print(f"  Tiles with images: {grand_with_images}")
    print(f"  Total images found: {grand_total_images}")
    print(f"{'=' * 60}")

    # Save summary
    summary_file = Path("data/raw/tile_coverage_summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "total_areas": len(all_results),
        "total_tiles": grand_tiles,
        "tiles_with_images": grand_with_images,
        "total_images": grand_total_images,
        "areas": [
            {
                "key": r["area_key"],
                "name": r["area_name"],
                "total_tiles": r["total_tiles"],
                "tiles_with_images": r["tiles_with_images"],
                "total_images": r["total_images"],
            }
            for r in all_results
        ],
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Print visualizations
    print(f"\nHTML Visualizations:")
    for result in all_results:
        viz_path = Path(f"data/debug/tiles_{result['area_key']}_viz.html")
        if viz_path.exists():
            print(f"  file://{viz_path.absolute()}")


def main():
    """Main entry point."""
    load_dotenv()

    access_token = Path(".env").read_text().strip().split("=")[-1]
    if "MAPILLARY" not in access_token and not access_token.startswith("MLY"):
        # Try to get from env
        import os
        access_token = os.getenv("MAPILLARY_ACCESS_TOKEN", "")

    if not access_token or not access_token.startswith("MLY"):
        print("ERROR: MAPILLARY_ACCESS_TOKEN not found in .env")
        print("Please add: MAPILLARY_ACCESS_TOKEN=MLY|...")
        return

    print("=== Mapillary Tile Coverage Analysis - HCMC ===")

    client = MapillaryClient(access_token=access_token)

    # Analyze each area
    all_results = []

    # Start with small test area first
    print("\nStarting with small test area...")
    result = analyze_area(client, "small_test", HCMC_AREAS["small_test"])
    all_results.append(result)

    # Ask if user wants to continue
    print("\n" + "=" * 60)
    print("Small area complete!")
    print("Check the visualization: data/debug/tiles_small_test_viz.html")
    print("\nTo continue with larger areas, run this script again")
    print("and modify the main() function to process more areas.")
    print("=" * 60)

    print_summary(all_results)


if __name__ == "__main__":
    main()
