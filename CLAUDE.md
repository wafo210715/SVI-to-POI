Project: HCMC Spatial-Semantic POI Reconstruction (2026)

0. Role & Behavior Protocol
You are the Lead Engineer for this project.

Do not ask for permission on library choices if they are listed in Section 1.

Do not ask about folder structure; create the structure defined in Section 2 if it doesn't exist.

Do not hallucinate APIs; use the specific APIs defined for VLM.

Language: The target city is Ho Chi Minh City (Vietnamese), but all code comments, documentation, and output JSON keys/values must be in English.

---

## Quick Start

### Initial Setup

1. **Clone and navigate to project:**
   ```bash
   cd /path/to/SVI-to-POI
   ```

2. **Install dependencies using uv (recommended):**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

   Alternative: Using traditional pip
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment:**
   ```bash
   # Copy .env.example to .env (if .env.example exists)
   # Then add your API keys to .env:
   # - GOOGLE_KEY (for Google Street View)
   # - MAPILLARY_ACCESS_TOKEN (for Mapillary)
   # - GLM_KEY (for GLM-4V VLM)
   # - OPENAI_API_KEY (for GPT-4o fallback)
   ```

4. **Run environment check:**
   ```bash
   uv run python scripts/00_setup_env.py
   ```

### Common Workflows

**Download Mapillary samples (recommended over GSV):**
```bash
uv run python scripts/01_download_mapillary_samples.py
```

**Download GSV samples:**
```bash
uv run python scripts/01_download_samples.py
```

**Run tests:**
```bash
uv run pytest
```

**Run tests with coverage:**
```bash
uv run pytest --cov=src --cov-report=html
```

**Investigate depth/camera parameters:**
```bash
uv run python scripts/04_investigate_depth.py
```

---

## Development Commands

### Package Management (uv)

```bash
# Install all dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Add a dev dependency
uv add --dev <package-name>

# Remove a dependency
uv remove <package-name>

# Run scripts with uv
uv run <command>

# Run Python with uv
uv run python <script.py>
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_crawler.py

# Run specific test
uv run pytest tests/test_crawler.py::test_street_view_image_contract

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run pytest --cov=src

# Run with coverage HTML report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Run with coverage for specific module
uv run pytest --cov=src.crawler
```

### Environment Validation

```bash
# Check environment setup
uv run python scripts/00_setup_env.py

# Verify Python version
uv run python --version

# List installed packages
uv pip list
```

### Script Execution Patterns

```bash
# All scripts should be run with uv run
uv run python scripts/<script_name>.py

# Examples:
uv run python scripts/00_setup_env.py
uv run python scripts/01_download_mapillary_samples.py
uv run python scripts/02_scrape_mapillary_tiles.py
uv run python scripts/03_compare_coverage.py
uv run python scripts/04_investigate_depth.py
uv run python scripts/05_sequence_metadata_investigation.py
uv run python scripts/06_colmap_quality_assessment.py
```

---

1. Technology Stack (Strict)
Language: Python 3.10+

Environment Management: conda or venv (User preference: local).

Package Management: uv (recommended) or pip.

Key Libraries (Use these defaults):

Data Manipulation: pandas, numpy

Geospatial: geopandas, shapely, pyproj, folium (for viz)

Image Processing: opencv-python (cv2), Pillow

ML/DL: torch (for depth/localization models), transformers

Network/API: requests, httpx, tenacity (for retries)

CLI/Utils: click or typer, tqdm, python-dotenv

VLM Provider:

Primary: GLM-4V (via API).

Secondary: GPT-4o (via OpenAI API).

Configuration: Load API keys from .env.

---

2. Project Directory Structure
Enforce this structure. If a script needs to save a file, save it to the appropriate data/ subdirectory.

```
.
├── .env                    # API Keys (GLM_KEY, GOOGLE_KEY, etc.)
├── CLAUDE.md               # This file - project context and developer guide
├── pyproject.toml          # uv-compatible project configuration
├── requirements.txt        # Legacy requirements (for pip compatibility)
├── src/
│   ├── __init__.py
│   ├── crawler.py          # ✅ Mapillary/GSV API interaction
│   ├── geolocator.py       # ❌ Depth + Camera Params -> Lat/Lon (NOT IMPLEMENTED)
│   ├── vlm_extractor.py    # ❌ GLM-4V API wrapper & Prompting (NOT IMPLEMENTED)
│   └── utils.py            # ❌ Geometric math & helpers (NOT IMPLEMENTED)
├── scripts/
│   ├── 00_setup_env.py                    # Environment validation
│   ├── 01_download_samples.py             # GSV sample downloader (deprecated)
│   ├── 01_download_mapillary_samples.py   # Mapillary sample downloader (preferred)
│   ├── 02_scrape_mapillary_tiles.py       # Mapillary tile scraping
│   ├── 03_compare_coverage.py             # Coverage comparison tool
│   ├── 04_investigate_depth.py            # Camera parameters and depth investigation
│   ├── 05_sequence_metadata_investigation.py  # Sequence metadata analysis
│   └── 06_colmap_quality_assessment.py    # COLMAP quality assessment
├── tests/
│   ├── test_crawler.py        # Tests for StreetViewImage and crawler
│   └── test_mapillary_tiles.py # Tests for tile scraping
└── data/
    ├── raw/                # Original images & JSON metadata
    ├── interim/            # Cropped signboards, depth maps
    ├── processed/          # Final GeoJSON POI files
    └── debug/              # Analysis visualizations and summaries
```

---

## Architecture Overview

### Data Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mapillary/    │───>│   StreetView    │───>│   Depth &       │
│      GSV API    │    │    Image        │    │ Camera Params   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  POI GeoJSON    │<───│  Geo-localizer  │<───│  Pixel (u,v) +  │
│    Output       │    │   (geolocator)  │    │     Depth d     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VLM Extract   │<───│   VLM Service   │<───│  Cropped        │
│  (Structured    │    │  (vlm_extractor)│    │  Signboard      │
│     POI Data)   │    └─────────────────┘    └─────────────────┘
└─────────────────┘
```

### Module Connections

1. **crawler.py** -> Returns `StreetViewImage` data contract
2. **StreetViewImage.depth_map** -> Input to geolocator.py
3. **geolocator.py** -> Converts pixel coordinates to GPS coordinates
4. **vlm_extractor.py** -> Takes cropped image, returns structured POI JSON
5. **utils.py** -> Shared geometric functions

### GSV vs Mapillary Trade-offs

| Feature | Google Street View | Mapillary |
|---------|-------------------|-----------|
| Depth Availability | Post-2017 only, heuristic check | All images (SfM computed) |
| API Key Required | Yes | Yes (for search) |
| Coverage | Global, urban focus | Community-contributed |
| Image Quality | High | Variable (creator-dependent) |
| Rate Limits | Strict ($200 free credit) | More lenient |
| Depth Format | Separate endpoint | Tiles endpoint |

**Recommendation:** Use Mapillary as primary source due to reliable depth maps. Use GSV as fallback.

### Key Data Contracts

1. **StreetViewImage** (src/crawler.py:18-50)
   - Standard metadata format for all images
   - Used across all pipeline stages

2. **POI JSON Schema** (see Section 4)
   - Final output format for VLM extraction
   - Must be strictly followed for consistency

---

## Implementation Status

### Core Modules

| Module | File | Status | Notes |
|--------|------|--------|-------|
| Data Acquisition | `src/crawler.py` | ✅ Implemented | GSV and Mapillary clients fully functional |
| Geo-localization | `src/geolocator.py` | ❌ Not Implemented | Planned: Pixel-to-GPS geometric projection |
| VLM Extraction | `src/vlm_extractor.py` | ❌ Not Implemented | Planned: GLM-4V API wrapper |
| Utilities | `src/utils.py` | ❌ Not Implemented | Planned: Geometric math helpers |

### Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `00_setup_env.py` | Environment validation | ✅ Working |
| `01_download_samples.py` | GSV downloader | ⚠️ Deprecated |
| `01_download_mapillary_samples.py` | Mapillary downloader | ✅ Primary |
| `02_scrape_mapillary_tiles.py` | Tile scraping | ✅ Working |
| `03_compare_coverage.py` | Coverage analysis | ✅ Working |
| `04_investigate_depth.py` | Depth investigation | ✅ Working |
| `05_sequence_metadata_investigation.py` | Sequence analysis | ✅ Working |
| `06_colmap_quality_assessment.py` | Quality assessment | ✅ Working |

### Tests

| Test File | Coverage | Status |
|-----------|----------|--------|
| `tests/test_crawler.py` | crawler.py | ✅ Comprehensive |
| `tests/test_mapillary_tiles.py` | Tile scraping | ✅ Comprehensive |

### Next Priorities

1. **High Priority:** Implement `src/geolocator.py` (geometric projection doesn't require API keys)
2. **Medium Priority:** Implement `src/vlm_extractor.py` (requires GLM-4V API access)
3. **Low Priority:** Implement `src/utils.py` (can be done alongside geolocator)

---

3. Core Pipeline Specifications
Module A: Data Acquisition (The Input)
Source: Google Street View (GSV) or Mapillary.

Required Metadata per Image:

image_id: Unique ID.

camera_loc: {lat, lon}.

heading: Camera bearing (0-360).

pitch: Camera vertical angle.

fov: Field of view.

depth_map: Base64 encoded or separate file (Metric depth preferred).

Module B: Geo-localization (The Calculation)
Goal: Convert (pixel_u, pixel_v) of a signboard center to (poi_lat, poi_lon). Logic:

Input: Image W, H, Pixel (u, v), Depth d (meters), Camera (lat, lon, heading, pitch).

Algorithm (Geometric Projection):

Convert pixel (u,v) to spherical coordinates relative to camera center.

Adjust for heading and pitch.

Calculate offset vectors (delta_north, delta_east).

Apply offset to Camera GPS using geopy.distance or Haversine formula.

Supervised Learning Component (If geometric fails):

If implementing the supervised model, input is [image_crop, depth_crop], output is [relative_distance, relative_angle].

Module C: Semantic Extraction (The VLM)
Constraint: Zero-shot extraction. Prompting Strategy:

Input: Cropped Signboard Image + Context Image (Whole storefront).

System Prompt: "You are an urban surveyor in Ho Chi Minh City. Extract structured data from the signboard. Correct Vietnamese OCR errors based on context. Translate semantics to English."

Output Format: Strict JSON (see Section 4).

4. Data Contracts (JSON Schema)
The VLM must return this exact structure. Do not deviate.

```json
{
  "target_schema": {
    "poi_name_vietnamese": "Phở Hùng",
    "poi_name_english": "Hung Pho",
    "business_category": "Restaurant",
    "sub_category": "Noodle Shop",
    "address_text": "24 Nguyen Du",
    "is_temporary_stall": false,
    "storefront_attributes": {
        "signage_condition": "Good",
        "has_english_menu": true,
        "is_chain_brand": false
    },
    "accessibility": {
        "has_steps_at_entrance": true,
        "sidewalk_condition": "Crowded with motorbikes"
    }
  }
}
```

5. Implementation Rules
Error Handling:

VLM API calls must use tenacity for exponential backoff retries.

If an image has no depth metadata, log warning and skip (or fallback to monocular depth estimation).

Coordinate Precision:

Keep Latitude/Longitude to 6 decimal places.

Visualization:

Any debugging visualization should save to data/debug/ as .png (images) or .html (folium maps).

---

6. Immediate Action Items (Execution Plan)
When asked to "start" or "setup":

Check environment: Verify if .env exists (warn if not).

Skeleton: Create the directory structure defined in Section 2.

Prototype: Create src/geolocator.py first, as this is the most mathematical/logic-heavy part that doesn't require API keys to write.
