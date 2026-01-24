#!/usr/bin/env python3
"""Quick environment check and setup."""

import os
import sys
from pathlib import Path

def check_env():
    """Check if environment is properly configured."""
    print("=== HCMC POI Extraction - Environment Check ===\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("  ⚠️  Warning: Python 3.9+ recommended")
    else:
        print("  ✓ Python version OK")

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("  ✓ .env file exists")

        # Check for API keys
        from dotenv import load_dotenv
        load_dotenv()

        keys = {
            "GOOGLE_KEY": os.getenv("GOOGLE_KEY"),
            "MAPILLARY_ACCESS_TOKEN": os.getenv("MAPILLARY_ACCESS_TOKEN"),
            "GLM_KEY": os.getenv("GLM_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        }

        print("\nAPI Keys configured:")
        for key, value in keys.items():
            if value and value != f"your_{key.lower()}_here":
                print(f"  ✓ {key}: Set (length: {len(value)})")
            else:
                print(f"  ✗ {key}: Not set")
    else:
        print("  ✗ .env file not found")
        print("    Copy .env.example to .env and add your API keys")

    # Check directories
    print("\nDirectory structure:")
    dirs = ["src", "scripts", "tests", "data/raw", "data/interim", "data/processed", "data/debug"]
    for d in dirs:
        p = Path(d)
        if p.exists():
            print(f"  ✓ {d}/")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ {d}/ (created)")

    # Check modules
    print("\nRequired packages:")
    packages = ["requests", "pytest"]
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (not installed)")

    print("\n" + "=" * 50)
    print("Next steps:")
    print("  Option 1 - Google Street View:")
    print("    1. Add your GOOGLE_KEY to .env file")
    print("    2. Run: python3 scripts/01_download_samples.py")
    print("")
    print("  Option 2 - Mapillary (recommended):")
    print("    1. Get token from https://mapillary.com/dashboard/developers")
    print("    2. Add MAPILLARY_ACCESS_TOKEN to .env")
    print("    3. Run: python3 scripts/01_download_mapillary_samples.py")
    print("=" * 50)

if __name__ == "__main__":
    check_env()
