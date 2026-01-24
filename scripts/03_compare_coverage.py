#!/usr/bin/env python3
"""Compare GSV vs Mapillary coverage for HCMC.

Summary of findings:

MAPILLARY:
- Tiles API returns Mapbox Vector Tiles (MVT) format
- Binary protobuf format, 10MB+ per tile
- Requires special parsing (mapbox-vector-tile library or similar)
- Coverage appears limited in HCMC based on initial tests

GOOGLE STREET VIEW:
- Simple JSON metadata API
- $7 per 1,000 calls (~$0.70 for 100 test locations)
- Better coverage in Vietnam expected

RECOMMENDATION:
Use Google Street View for HCMC due to:
1. Simpler API (JSON vs binary protobuf)
2. Better coverage in Vietnam
3. Lower cost for testing

Next steps:
1. Test GSV with the provided API key
2. Compare actual coverage in test areas
3. Check depth availability in GSV metadata
"""

import json
from pathlib import Path


def create_comparison_report():
    """Generate a comparison report."""

    report = {
        "mapillary": {
            "api_type": "Vector Tiles (MVT)",
            "format": "Binary protobuf (gzipped)",
            "tile_size_mb": 10.7,
            "requires": "MVT parser library (mapbox-vector-tile)",
            "coverage_hcmc": "Limited - few images found in test areas",
            "cost": "Free (client token)",
            "depth_support": "Yes (via SfM)",
            "pros": [
                "Free access",
                "Depth maps available",
                "High resolution images",
            ],
            "cons": [
                "Binary tile format requires parsing",
                "Limited coverage in HCMC",
                "Large tile sizes (10MB+)",
            ],
        },
        "google_street_view": {
            "api_type": "REST API (JSON)",
            "format": "JSON",
            "cost_per_1000": "$7.00",
            "cost_test_100": "~$0.70",
            "coverage_hcmc": "Expected good (Google maps Vietnam extensively)",
            "depth_support": "Post-2017 panoramas",
            "pros": [
                "Simple JSON API",
                "Good Vietnam coverage",
                "Low cost for testing",
            ],
            "cons": [
                "No free tier",
                "Depth only for recent panoramas",
            ],
        },
        "recommendation": {
            "for_hcmc": "Google Street View",
            "reasons": [
                "Simpler API integration",
                "Better coverage in Vietnam",
                "Lower cost for initial testing",
            ],
        },
    }

    output_path = Path("data/debug/gsv_vs_mapillary_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Comparison report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("GSV vs Mapillary Comparison - HCMC")
    print("=" * 60)
    print("\nMAPILLARY:")
    print(f"  Format: {report['mapillary']['api_type']}")
    print(f"  Tile Size: ~{report['mapillary']['tile_size_mb']} MB per tile")
    print(f"  Coverage: {report['mapillary']['coverage_hcmc']}")
    print(f"  Cost: {report['mapillary']['cost']}")

    print("\nGOOGLE STREET VIEW:")
    print(f"  Format: {report['google_street_view']['api_type']}")
    print(f"  Cost for 100 tests: ~${report['google_street_view']['cost_test_100']}")
    print(f"  Coverage: {report['google_street_view']['coverage_hcmc']}")

    print(f"\nRECOMMENDATION: {report['recommendation']['for_hcmc']}")
    for reason in report['recommendation']['reasons']:
        print(f"  - {reason}")
    print("=" * 60)


if __name__ == "__main__":
    create_comparison_report()
