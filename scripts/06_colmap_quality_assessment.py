#!/usr/bin/env python3
"""COLMAP quality assessment for Mapillary sequences.

Tests COLMAP reconstruction quality on sample sequences:
- Selects 10 diverse sequences from tile data
- Downloads images from each sequence
- Runs COLMAP to assess:
  - Feature matching success rate
  - Number of registered images per sequence
  - Point cloud density
  - Reconstruction quality metrics

Outputs:
- data/debug/colmap_quality_report.json - COLMAP assessment results
- data/colmap_temp/ - Temporary COLMAP workspace

Note: This script requires COLMAP to be installed.
Install via: brew install colmap (macOS) or apt-get install colmap (Linux)
"""

import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def group_by_sequence(images: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group images by sequence_id.

    Args:
        images: List of image dicts

    Returns:
        Dict mapping sequence_id to list of images
    """
    sequences = defaultdict(list)

    for img in images:
        seq_id = img.get("sequence_id")
        if seq_id:
            sequences[seq_id].append(img)

    return sequences


def select_diverse_sequences(
    grouped: Dict[str, List[Dict[str, Any]]],
    n: int = 10,
    min_images: int = 20,
    seed: int = 42,
) -> List[str]:
    """Select diverse sequences for COLMAP testing.

    Prioritizes:
    1. Different creator_ids (different devices)
    2. Different camera types (if available)
    3. Different geographic locations

    Args:
        grouped: Dict mapping sequence_id to images
        n: Number of sequences to select
        min_images: Minimum images per sequence
        seed: Random seed

    Returns:
        List of selected sequence_ids
    """
    random.seed(seed)

    # Filter sequences by minimum image count
    valid_sequences = {
        seq_id: images
        for seq_id, images in grouped.items()
        if len(images) >= min_images
    }

    if len(valid_sequences) < n:
        print(f"Warning: Only {len(valid_sequences)} sequences with >= {min_images} images")
        n = len(valid_sequences)

    # Sort by image count (prefer larger sequences)
    sorted_sequences = sorted(
        valid_sequences.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Select diverse sequences (different creators)
    selected = []
    seen_creators = set()

    for seq_id, images in sorted_sequences:
        if len(selected) >= n:
            break

        creator_id = images[0].get("creator_id")

        # Prioritize new creators
        if creator_id not in seen_creators or len(selected) < n // 2:
            selected.append(seq_id)
            seen_creators.add(creator_id)

    # If we still need more, add remaining
    for seq_id, _ in sorted_sequences:
        if len(selected) >= n:
            break
        if seq_id not in selected:
            selected.append(seq_id)

    return selected


def download_sequence_images(
    client: MapillaryClient,
    sequence_id: str,
    images: List[Dict[str, Any]],
    output_dir: Path,
    max_images: int = 50,
) -> List[str]:
    """Download images for a sequence.

    Args:
        client: MapillaryClient instance
        sequence_id: Sequence ID
        images: List of image dicts in this sequence
        output_dir: Directory to save images
        max_images: Maximum images to download (for speed)

    Returns:
        List of downloaded image paths
    """
    seq_dir = output_dir / sequence_id
    seq_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    # Limit images for speed
    sample_images = images[:max_images]

    for i, img in enumerate(tqdm(sample_images, desc=f"Downloading {sequence_id[:8]}", leave=False)):
        try:
            image_data = client.fetch_image(img["image_id"], size=(640, 640))

            # Save with sequential filename
            filename = f"{i:04d}.jpg"
            filepath = seq_dir / filename

            with open(filepath, "wb") as f:
                f.write(image_data)

            downloaded.append(str(filepath))

        except Exception as e:
            print(f"Warning: Failed to download {img['image_id']}: {e}")
            continue

    return downloaded


def check_colmap_installed() -> bool:
    """Check if COLMAP is installed.

    Returns:
        True if COLMAP is available
    """
    try:
        result = subprocess.run(
            ["colmap", "-h"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_colmap_reconstruction(
    image_dir: Path,
    workspace_dir: Path,
    sequence_id: str,
) -> Dict[str, Any]:
    """Run COLMAP reconstruction on a sequence.

    Args:
        image_dir: Directory containing images
        workspace_dir: COLMAP workspace directory
        sequence_id: Sequence ID for naming

    Returns:
        Reconstruction results dict
    """
    # Create workspace
    seq_workspace = workspace_dir / sequence_id
    seq_workspace.mkdir(parents=True, exist_ok=True)

    database_path = seq_workspace / "database.db"
    image_path = image_dir
    sparse_path = seq_workspace / "sparse"

    try:
        # Step 1: Feature extraction
        print(f"  [{sequence_id[:8]}] Extracting features...")
        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "PINHOLE",
        ], check=True, capture_output=True, timeout=300)

        # Step 2: Feature matching
        print(f"  [{sequence_id[:8]}] Matching features...")
        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(database_path),
        ], check=True, capture_output=True, timeout=300)

        # Step 3: Reconstruction
        print(f"  [{sequence_id[:8]}] Running reconstruction...")
        subprocess.run([
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_path),
            "--output_path", str(sparse_path),
        ], check=True, capture_output=True, timeout=600)

        # Parse results
        # Count registered images
        images_txt = sparse_path / "0" / "images.txt"
        points_txt = sparse_path / "0" / "points3D.txt"

        registered_images = 0
        num_points = 0

        if images_txt.exists():
            with open(images_txt, "r") as f:
                for line in f:
                    if not line.startswith("#") and line.strip():
                        registered_images += 1
                # Adjust for header (every 4th line is data)
                registered_images = (registered_images - 3) // 4

        if points_txt.exists():
            with open(points_txt, "r") as f:
                for line in f:
                    if not line.startswith("#") and line.strip():
                        num_points += 1
                # Adjust for header
                num_points = num_points - 3

        # Get total image count
        total_images = len(list(image_dir.glob("*.jpg")))

        return {
            "sequence_id": sequence_id,
            "total_images": total_images,
            "registered_images": max(0, registered_images),
            "num_3d_points": max(0, num_points),
            "registration_rate": max(0, registered_images) / total_images if total_images > 0 else 0,
            "success": registered_images > 0,
        }

    except subprocess.CalledProcessError as e:
        return {
            "sequence_id": sequence_id,
            "total_images": len(list(image_dir.glob("*.jpg"))),
            "registered_images": 0,
            "num_3d_points": 0,
            "registration_rate": 0,
            "success": False,
            "error": str(e),
        }
    except subprocess.TimeoutExpired:
        return {
            "sequence_id": sequence_id,
            "total_images": len(list(image_dir.glob("*.jpg"))),
            "registered_images": 0,
            "num_3d_points": 0,
            "registration_rate": 0,
            "success": False,
            "error": "timeout",
        }


def generate_quality_report(
    results: List[Dict[str, Any]],
    output_path: str = "data/debug/colmap_quality_report.json",
) -> Dict[str, Any]:
    """Generate COLMAP quality report.

    Args:
        results: List of reconstruction results
        output_path: Where to save the report

    Returns:
        Quality report dict
    """
    total_sequences = len(results)
    successful = [r for r in results if r.get("success", False)]

    total_images = sum(r["total_images"] for r in results)
    registered_images = sum(r["registered_images"] for r in results)
    total_points = sum(r["num_3d_points"] for r in successful)

    # Registration stats
    registration_rates = [r["registration_rate"] for r in results if r["registration_rate"] > 0]
    avg_registration_rate = sum(registration_rates) / len(registration_rates) if registration_rates else 0

    # Determine feasibility
    if avg_registration_rate >= 0.5:
        feasibility = "high"
        recommendation = "HCMC sequences are suitable for 3D reconstruction"
    elif avg_registration_rate >= 0.2:
        feasibility = "moderate"
        recommendation = "Some sequences suitable for 3D reconstruction, consider filtering"
    elif avg_registration_rate >= 0.1:
        feasibility = "low"
        recommendation = "Limited 3D reconstruction potential (similar to Amsterdam dataset)"
    else:
        feasibility = "very_low"
        recommendation = "Not suitable for 3D reconstruction, use alternative approach"

    report = {
        "sequences_tested": total_sequences,
        "sequences_reconstructed": len(successful),
        "reconstruction_rate": len(successful) / total_sequences if total_sequences > 0 else 0,
        "total_images": total_images,
        "images_registered": registered_images,
        "overall_registration_rate": registered_images / total_images if total_images > 0 else 0,
        "avg_registration_rate_per_sequence": avg_registration_rate,
        "avg_points_per_reconstruction": total_points / len(successful) if successful else 0,
        "total_3d_points": total_points,
        "feasibility": feasibility,
        "recommendation": recommendation,
        "individual_results": results,
    }

    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    """Main entry point for COLMAP quality assessment."""
    load_dotenv()

    # Check for MAPILLARY_ACCESS_TOKEN
    import os
    access_token = os.getenv("MAPILLARY_ACCESS_TOKEN")
    if not access_token:
        print("Error: MAPILLARY_ACCESS_TOKEN not found in environment")
        print("Please set it in your .env file")
        sys.exit(1)

    # Check for COLMAP
    if not check_colmap_installed():
        print("Error: COLMAP is not installed")
        print("\nInstall via:")
        print("  macOS: brew install colmap")
        print("  Linux: sudo apt-get install colmap")
        sys.exit(1)

    print("=" * 60)
    print("COLMAP Quality Assessment")
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
    grouped = group_by_sequence(all_images)
    print(f"Found {len(grouped)} unique sequences")

    # Select diverse sequences
    n_sequences = 10
    print(f"\nSelecting {n_sequences} diverse sequences...")
    selected = select_diverse_sequences(grouped, n=n_sequences)
    print(f"Selected {len(selected)} sequences:")
    for seq_id in selected:
        print(f"  - {seq_id}: {len(grouped[seq_id])} images")

    # Download images
    download_dir = Path("data/colmap_temp/images")
    print(f"\nDownloading images to {download_dir}...")

    for seq_id in selected:
        images = grouped[seq_id]
        downloaded = download_sequence_images(client, seq_id, images, download_dir)
        print(f"  {seq_id[:8]}: Downloaded {len(downloaded)} images")

    # Run COLMAP
    print("\nRunning COLMAP reconstruction...")
    workspace = Path("data/colmap_temp/workspace")

    results = []
    for seq_id in selected:
        image_dir = download_dir / seq_id
        if not image_dir.exists():
            print(f"Warning: No images found for {seq_id}")
            continue

        print(f"\n[{seq_id[:8]}] Running COLMAP...")
        result = run_colmap_reconstruction(image_dir, workspace, seq_id)
        results.append(result)

        print(f"  [{seq_id[:8]}] Result: {result['registered_images']}/{result['total_images']} registered ({result['registration_rate']:.1%})")

    # Generate report
    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)

    report = generate_quality_report(results)

    print(f"Sequences tested: {report['sequences_tested']}")
    print(f"Sequences reconstructed: {report['sequences_reconstructed']}")
    print(f"Total images: {report['total_images']}")
    print(f"Images registered: {report['images_registered']}")
    print(f"Overall registration rate: {report['overall_registration_rate']:.1%}")
    print(f"Average registration rate per sequence: {report['avg_registration_rate_per_sequence']:.1%}")
    print(f"Average 3D points per reconstruction: {report['avg_points_per_reconstruction']:.0f}")
    print(f"\nFeasibility: {report['feasibility']}")
    print(f"Recommendation: {report['recommendation']}")

    print(f"\nReport saved to: data/debug/colmap_quality_report.json")

    # Cleanup prompt
    print("\n" + "=" * 60)
    print("Assessment complete!")
    print("=" * 60)
    print(f"\nTemporary files saved in: data/colmap_temp/")
    print("Cleanup with: rm -rf data/colmap_temp/")


if __name__ == "__main__":
    main()
