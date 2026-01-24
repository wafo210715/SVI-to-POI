#!/usr/bin/env python3
"""Download sample Street View images from Google Street View.

Note: Consider using Mapillary instead (01_download_mapillary_samples.py)
- Mapillary provides depth maps via SfM pipeline
- No API key required for public images
- Higher resolution images available

This GSV script tests:
1. GSV Metadata API response structure
2. Depth information availability (heuristic based on date)
3. Camera parameters (heading, pitch, FOV)
"""

import os
import json
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm

from src.crawler import GSVClient, StreetViewImage

# HCMC test locations (District 1 - busy commercial area)
HCMC_TEST_LOCATIONS = [
    # Ben Thanh Market area
    {"lat": 10.7720, "lon": 106.6960, "name": "Ben Thanh Market"},
    # Nguyen Hue Walking Street
    {"lat": 10.7742, "lon": 106.7015, "name": "Nguyen Hue Street"},
    # Opera House area
    {"lat": 10.7748, "lon": 106.7025, "name": "Opera House"},
    # Pham Ngu Lao street (backpacker area)
    {"lat": 10.7740, "lon": 106.6920, "name": "Pham Ngu Lao"},
    # A residential street in District 1
    {"lat": 10.7780, "lon": 106.6880, "name": "District 1 Residential"},
]


def analyze_metadata_structure(metadata: dict) -> dict:
    """Analyze GSV metadata response structure.

    Returns what we found and what's missing.
    """
    analysis = {
        "has_pano_id": "pano_id" in metadata,
        "has_location": "location" in metadata,
        "has_heading": "heading" in metadata,
        "has_pitch": "pitch" in metadata,
        "has_date": "date" in metadata,
        "pano_id": metadata.get("pano_id"),
        "date": metadata.get("date"),
        # Check for any depth-related fields
        "has_any_depth_field": any(
            k for k in metadata.keys()
            if "depth" in k.lower() or "geometry" in k.lower()
        ),
        "all_keys": list(metadata.keys()),
    }

    # Check camera parameters
    if "location" in metadata:
        analysis["camera_lat"] = metadata["location"].get("lat")
        analysis["camera_lon"] = metadata["location"].get("lng")

    return analysis


def save_raw_metadata(
    output_dir: Path,
    location_name: str,
    metadata: dict,
) -> None:
    """Save raw metadata response for inspection."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"{location_name.replace(' ', '_')}_raw_metadata.json"
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved raw metadata to: {filename}")


def main():
    """Download and analyze sample GSV metadata."""
    load_dotenv()

    api_key = os.getenv("GOOGLE_KEY")
    if not api_key:
        print("ERROR: GOOGLE_KEY not found in .env file")
        print("Please create a .env file with: GOOGLE_KEY=your_api_key")
        return

    print("=== GSV Metadata Analysis for HCMC ===\n")
    print(f"Testing {len(HCMC_TEST_LOCATIONS)} locations...\n")

    client = GSVClient(api_key=api_key)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for loc in tqdm(HCMC_TEST_LOCATIONS, desc="Fetching metadata"):
        lat, lon, name = loc["lat"], loc["lon"], loc["name"]

        print(f"\n[{name}] ({lat:.4f}, {lon:.4f})")

        # Fetch metadata
        metadata = client.get_metadata(lat, lon)

        if not metadata:
            print(f"  ❌ No panorama found")
            results.append({
                "name": name,
                "lat": lat,
                "lon": lon,
                "status": metadata.get("status") if metadata else "NO_DATA",
            })
            continue

        # Analyze structure
        analysis = analyze_metadata_structure(metadata)

        print(f"  ✓ Pano ID: {analysis['pano_id']}")
        print(f"  ✓ Date: {analysis['date']}")
        print(f"  ✓ Location: {analysis.get('camera_lat', 'N/A')}, {analysis.get('camera_lon', 'N/A')}")

        # Save raw response
        save_raw_metadata(output_dir, name, metadata)

        # Check depth support
        has_depth = client._check_depth_support(metadata)
        print(f"  {'✓' if has_depth else '✗'} Likely depth support: {has_depth}")

        results.append({
            "name": name,
            "lat": lat,
            "lon": lon,
            "status": metadata.get("status"),
            "pano_id": analysis["pano_id"],
            "date": analysis["date"],
            "likely_has_depth": has_depth,
            "all_metadata_keys": analysis["all_keys"],
        })

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    with_depth = sum(1 for r in results if r.get("likely_has_depth"))
    print(f"Locations tested: {len(results)}")
    print(f"Likely have depth: {with_depth}/{len(results)}")

    # Save summary
    summary_file = output_dir / "metadata_analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Print all unique keys found across all responses
    all_keys = set()
    for r in results:
        if "all_metadata_keys" in r:
            all_keys.update(r["all_metadata_keys"])

    print("\nAll unique metadata keys found:")
    for key in sorted(all_keys):
        print(f"  - {key}")


if __name__ == "__main__":
    main()
