# Update CLAUDE.md for SVI-to-POI Project Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an improved CLAUDE.md file that provides comprehensive guidance for Claude Code working in this repository, along with uv-compatible pyproject.toml for modern Python package management.

**Architecture:** Preserve all existing project specifications while adding developer-focused sections: Quick Start, Development Commands, Architecture Overview, and Implementation Status. Add pyproject.toml for uv compatibility.

**Tech Stack:** Python 3.10+, uv (package manager), pytest, existing project dependencies

---

## Task 1: Create pyproject.toml with uv-compatible configuration

**Files:**
- Create: `pyproject.toml`

**Step 1: Write the pyproject.toml file**

The file should include:
- Project metadata (name, version, description)
- Python version requirement (3.10+)
- Dependencies from requirements.txt
- devDependencies (pytest, pytest-cov, etc.)
- Project configuration for uv

```toml
[project]
name = "svi-to-poi"
version = "0.1.0"
description = "HCMC Spatial-Semantic POI Reconstruction from Street View Images"
readme = "CLAUDE.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Project Team"}
]
keywords = ["poi", "street-view", "geospatial", "vlm", "ho-chi-minh-city"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    # Core Data
    "pandas>=2.0.0",
    "numpy>=1.24.0",

    # Geospatial
    "geopandas>=0.14.0",
    "shapely>=2.0.0",
    "pyproj>=3.6.0",
    "folium>=0.15.0",
    "mercantile>=1.2.0",
    "vt2geojson>=0.2.0",

    # Image Processing
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",

    # ML/DL
    "torch>=2.0.0",
    "transformers>=4.35.0",

    # Network/API
    "requests>=2.31.0",
    "httpx>=0.25.0",
    "tenacity>=8.2.0",

    # CLI/Utils
    "click>=8.1.0",
    "tqdm>=4.66.0",
    "python-dotenv>=1.0.0",

    # VLM Clients
    "openai>=1.0.0",

    # Visualization
    "plotly>=5.18.0",

    # Type Checking
    "pydantic>=2.5.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.12.0",
]

[project.scripts]
# CLI commands can be added here when implemented
# setup-env = "scripts.00_setup_env:check_env"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "pytest-mock>=3.12.0",
    "ruff>=0.1.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers --cov=src --cov-report=term-missing"

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]
```

**Step 2: Verify the file was created correctly**

Run: `cat pyproject.toml`
Expected: File content matches above

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add pyproject.toml for uv compatibility

- Add project metadata and configuration
- Migrate dependencies from requirements.txt
- Configure pytest and coverage settings
- Add uv dev dependencies"
```

---

## Task 2: Create comprehensive new CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Backup existing CLAUDE.md**

Run: `cp CLAUDE.md CLAUDE.md.backup`

**Step 2: Write the new CLAUDE.md**

The new CLAUDE.md should preserve all existing content (sections 0-6) and add the following new sections:

### New Section to Add after "0. Role & Behavior Protocol":

```markdown
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
```

### New Section to Add after "Implementation Rules" (before "6. Immediate Action Items"):

```markdown
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
```

### New Section to Add after "Project Directory Structure":

```markdown
**Note:** The `scripts/` directory contains more utilities than shown above:
- `00_setup_env.py` - Environment validation and setup
- `01_download_samples.py` - GSV sample downloader (deprecated)
- `01_download_mapillary_samples.py` - Mapillary sample downloader (preferred)
- `02_scrape_mapillary_tiles.py` - Mapillary tile scraping
- `03_compare_coverage.py` - Coverage comparison tool
- `04_investigate_depth.py` - Camera parameters and depth investigation
- `05_sequence_metadata_investigation.py` - Sequence metadata analysis
- `06_colmap_quality_assessment.py` - COLMAP quality assessment
```

### New Section to Add after "Development Commands":

```markdown
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

1. **StreetViewImage** (crawler.py:18-50)
   - Standard metadata format for all images
   - Used across all pipeline stages

2. **POI JSON Schema** (see Section 4)
   - Final output format for VLM extraction
   - Must be strictly followed for consistency
```

### New Section to Add before "6. Immediate Action Items":

```markdown
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
```

**Step 3: Verify the new CLAUDE.md contains all required sections**

Run: `grep -E "^##" CLAUDE.md`
Expected output should include:
- 0. Role & Behavior Protocol
- ## Quick Start
- 1. Technology Stack
- 2. Project Directory Structure
- ## Architecture Overview
- 3. Core Pipeline Specifications
- 4. Data Contracts
- 5. Implementation Rules
- ## Development Commands
- ## Implementation Status
- 6. Immediate Action Items

**Step 4: Run environment check to verify documentation accuracy**

Run: `uv run python scripts/00_setup_env.py`
Expected: Environment check runs successfully

**Step 5: Commit**

```bash
git add CLAUDE.md CLAUDE.md.backup
git commit -m "docs: enhance CLAUDE.md with developer guidance

- Add Quick Start section with setup instructions
- Add Development Commands section for uv and testing
- Add Architecture Overview explaining data flow
- Add Implementation Status documenting what exists
- Preserve all original project specifications
- Add script descriptions to directory structure"
```

---

## Task 3: Verification

**Files:**
- Test: `pyproject.toml`, `CLAUDE.md`

**Step 1: Verify uv sync works**

Run: `uv sync`
Expected: Dependencies install successfully without errors

**Step 2: Verify tests pass**

Run: `uv run pytest`
Expected: All tests pass

**Step 3: Verify test commands from CLAUDE.md**

Run: `uv run pytest --cov=src`
Expected: Coverage report generated successfully

**Step 4: Verify environment setup script works**

Run: `uv run python scripts/00_setup_env.py`
Expected: Environment check completes with clear output

**Step 5: Review CLAUDE.md completeness**

Run: `grep -c "##" CLAUDE.md`
Expected: At least 10 major sections (including original and new)

**Step 6: Final verification commit**

```bash
git add -A
git commit -m "chore: verify project setup with new configuration

- Confirmed uv sync works correctly
- Verified all tests pass
- Confirmed documentation commands are accurate
- Environment setup script functional"
```

---

## Summary

After completing this plan:

1. **pyproject.toml** will provide uv-compatible package management
2. **CLAUDE.md** will be a comprehensive developer guide with:
   - Quick Start instructions
   - All development commands
   - Architecture explanation
   - Current implementation status
   - All original preserved content

3. Developers can run `uv sync` and `uv run pytest` immediately
4. All test commands from the existing test suite are documented
5. Clear distinction between implemented and planned features
