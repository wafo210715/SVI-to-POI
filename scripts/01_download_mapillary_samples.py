#!/usr/bin/env python3
"""Download sample Street View images from Mapillary.

Mapillary advantages:
- No API key required for public images (client token needed)
- Provides depth maps via SfM pipeline
- Higher resolution images available

This script tests:
1. Mapillary API response structure
2. Depth availability
3. Image download capability
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm

from src.crawler import MapillaryClient, StreetViewImage

# HCMC bounding boxes for testing
# Format: [min_lon, min_lat, max_lon, max_lat]
HCMC_TEST_BBOXES = {
    "ben_thanh": "106.690,10.765,106.705,10.780",  # Ben Thanh Market
    "nguyen_hue": "106.695,10.770,106.710,10.785",  # Nguyen Hue Walking Street
    "opera_house": "106.698,10.772,106.708,10.782",  # Opera House
    "pham_ngu_lao": "106.685,10.770,106.700,10.785",  # Backpacker area
    "district_1": "106.680,10.765,106.720,10.790",  # Larger District 1 area
}


def analyze_mapillary_response(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze Mapillary image metadata response."""
    return {
        "image_id": image_data.get("id"),
        "has_geometry": "geometry" in image_data,
        "has_compass_angle": "compass_angle" in image_data,
        "has_is_pano": "is_pano" in image_data,
        "compass_angle": image_data.get("compass_angle"),
        "is_pano": image_data.get("is_pano", False),
        "captured_at": image_data.get("captured_at"),
        "all_keys": list(image_data.keys()),
    }


def save_raw_response(
    output_dir: Path,
    bbox_name: str,
    response_data: List[Dict[str, Any]],
) -> None:
    """Save raw Mapillary API response."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"mapillary_{bbox_name}_raw_response.json"
    with open(filename, "w") as f:
        json.dump(response_data, f, indent=2)

    print(f"  Saved raw response to: {filename}")


def main():
    """Download and analyze sample Mapillary images."""
    load_dotenv()

    # Mapillary requires a client access token
    # Get from: https://mapillary.com/dashboard/developers
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")

    if not access_token:
        print("ERROR: MAPILLARY_ACCESS_TOKEN not found in .env file")
        print("\nTo get a Mapillary access token:")
        print("  1. Go to https://mapillary.com/dashboard/developers")
        print("  2. Register your application")
        print("  3. Copy the client access token")
        print("  4. Add to .env: MAPILLARY_ACCESS_TOKEN=your_token")
        return

    print("=== Mapillary SVI Analysis for HCMC ===\n")
    print(f"Testing {len(HCMC_TEST_BBOXES)} areas...\n")

    client = MapillaryClient(access_token=access_token)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for bbox_name, bbox in tqdm(HCMC_TEST_BBOXES.items(), desc="Fetching areas"):
        print(f"\n[{bbox_name.upper()}] BBOX: {bbox}")

        try:
            # Search for images in this area
            images = client.search_images(bbox=bbox, limit=5)

            if not images:
                print(f"  ❌ No images found")
                all_results[bbox_name] = {"status": "NO_DATA", "count": 0}
                continue

            print(f"  ✓ Found {len(images)} images")

            # Analyze first image in detail
            sample = images[0]
            analysis = analyze_mapillary_response(sample)

            print(f"  Sample image ID: {analysis['image_id']}")
            print(f"  Compass angle: {analysis['compass_angle']}")
            print(f"  Is panorama: {analysis['is_pano']}")
            print(f"  Captured at: {analysis['captured_at']}")

            # Check depth availability
            image_id = analysis["image_id"]
            has_depth = client.check_depth_available(image_id)
            print(f"  {'✓' if has_depth else '✗'} Depth available: {has_depth}")

            # Save raw response
            save_raw_response(output_dir, bbox_name, images)

            # Build URLs for reference
            image_url = client._build_image_url(image_id)
            depth_url = client._build_depth_url(image_id)

            print(f"  Image URL: {image_url}")
            print(f"  Depth URL: {depth_url}")

            all_results[bbox_name] = {
                "status": "OK",
                "count": len(images),
                "sample_image_id": image_id,
                "sample_compass_angle": analysis["compass_angle"],
                "sample_is_pano": analysis["is_pano"],
                "depth_available": has_depth,
                "image_url": image_url,
                "depth_url": depth_url,
                "all_metadata_keys": analysis["all_keys"],
            }

        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_results[bbox_name] = {"status": "ERROR", "error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    total_images = sum(r.get("count", 0) for r in all_results.values())
    with_depth = sum(1 for r in all_results.values() if r.get("depth_available"))

    print(f"Areas tested: {len(all_results)}")
    print(f"Total images found: {total_images}")
    print(f"Areas with depth support: {with_depth}/{len(all_results)}")

    # Save summary
    summary_file = output_dir / "mapillary_analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Print all unique keys found
    all_keys = set()
    for r in all_results.values():
        if "all_metadata_keys" in r:
            all_keys.update(r["all_metadata_keys"])

    print("\nAll unique metadata keys found:")
    for key in sorted(all_keys):
        print(f"  - {key}")

    print("\n" + "=" * 50)
    print("Next: Run scripts/02_test_depth_projection.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
