# HCMC Spatial-Semantic POI Reconstruction

Automated extraction of Point of Interest (POI) data from street view imagery for Ho Chi Minh City, Vietnam.

## Overview

This project reconstructs POI data (business names, categories, addresses, GPS coordinates) from street view images using:
- **Computer Vision**: ML-based monocular depth estimation
- **Geometric Projection**: Converting pixel coordinates to GPS
- **Vision Language Models (VLM)**: Signboard detection and text extraction

### Key Features

- 🗺️ **GPS Precision**: Projects signboard locations to 6 decimal precision
- 📷 **Camera Agnostic**: Works with 95% of perspective cameras (69.5% Normal FOV)
- 🌏 **Vietnamese OCR**: Specialized for Vietnamese text recognition
- 🔄 **Complete Pipeline**: From raw Mapillary data to structured POI GeoJSON

## Quick Start

### Prerequisites

- Python 3.10+
- API keys (see [Configuration](#configuration))
- GPU recommended (for depth estimation)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd SVI-to-POI

# Install with uv (recommended)
uv sync

# Or with pip
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required: Mapillary access token
MAPILLARY_ACCESS_TOKEN=your_token_here

# Required: GLM-4V API key
GLM_KEY=your_glm_key_here

# Optional: Google Street View
GOOGLE_KEY=your_google_key_here
```

### Usage

#### 1. Download Mapillary Tiles

```bash
uv run python scripts/02_scrape_mapillary_tiles.py
```

#### 2. Run the Complete Pipeline (Tutorial)

```bash
# Launch IPython and run the tutorial
ipython

# In IPython:
%run scripts/tutorial/00_pipeline_tutorial.py
```

The tutorial guides you through:
- **Blocks 0-10**: Data acquisition and camera analysis
- **Blocks 11-19**: Complete POI reconstruction pipeline

#### 3. Run Individual Scripts

```bash
# Download sample images
uv run python scripts/01_download_mapillary_samples.py

# Investigate camera parameters
uv run python scripts/04_investigate_depth.py

# Analyze sequence metadata
uv run python scripts/05_sequence_metadata_investigation.py
```

## Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mapillary/    │───>│   StreetView    │───>│   Depth &       │
│      GSV API    │    │    Images        │    │ Camera Params   │
└─────────────────�    └─────────────────┘    └─────────────────�
                                                       │
                                                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  POI GeoJSON    │<───│  Geo-localizer  │<───│  Pixel (u,v) +  │
│    Output       │    │   (geolocator)  │    │     Depth d     │
└─────────────────�    └─────────────────�    └─────────────────�
                                                       │
                                                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   VLM Extract   │<───│   VLM Service   │<───│  Signboard      │
│  (Structured    │    │  (GLM-4V)       │    │  Detection      │
│     POI Data)   │    └─────────────────�    └─────────────────┘
└─────────────────�
```

### Data Flow

1. **Data Acquisition** (Blocks 1-6)
   - Fetch Mapillary tiles for HCMC regions
   - Extract image metadata (camera_loc, heading, sequence_id)
   - Download street view images locally

2. **Preprocessing** (Blocks 11-12.5)
   - Download SVI images (1024px for VLM efficiency)
   - VLM validation experiment (test Vietnamese OCR)
   - Safety check (filter blurry/indoor/poor-quality images)

3. **Depth Estimation** (Blocks 13-15)
   - Camera parameter analysis (FOV categorization)
   - ML monocular depth estimation (Intel DPT model)
   - Depth visualization with colormaps

4. **POI Detection** (Block 17)
   - VLM detects storefront signboards
   - Extracts bounding boxes and center pixels
   - Returns structured POI data (Vietnamese name, category, etc.)

5. **Geolocation** (Block 18)
   - Convert pixel coordinates to GPS using depth + camera params
   - Apply geometric projection with Haversine formula
   - Create interactive HTML maps for verification

6. **Output** (Block 19)
   - Final POI database as GeoJSON
   - Includes Vietnamese + English names, categories, GPS coordinates

## Project Structure

```
.
├── .env                    # API keys (not in git)
├── CLAUDE.md               # Developer guide
├── README.md               # This file
├── pyproject.toml          # uv project config
├── requirements.txt        # pip dependencies
├── src/                    # Production modules (to be extracted from tutorial)
├── scripts/
│   ├── 00_setup_env.py
│   ├── 01_download_mapillary_samples.py
│   ├── 02_scrape_mapillary_tiles.py
│   └── ...
│   └── tutorial/
│       └── 00_pipeline_tutorial.py   # Complete pipeline tutorial (Blocks 0-19)
├── tests/                   # Test suite
│   ├── test_crawler.py
│   └── test_mapillary_tiles.py
└── data/                    # Data files (not in git - see .gitignore)
    ├── raw/                # Downloaded images, API responses
    ├── interim/            # Depth maps, VLM results
    ├── processed/          # Final POI databases
    └── debug/              # Visualizations, reports
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Data Source** | Mapillary API | Street view images, depth metadata |
| **Depth Estimation** | Intel DPT | Monocular depth from single images |
| **Geolocation** | Geometric projection | Pixel → GPS conversion |
| **Text Extraction** | GLM-4V | Vietnamese OCR, POI data extraction |
| **Visualization** | Folium, Matplotlib | Interactive maps, depth colormaps |

## Data Contracts

### POI JSON Schema

```json
{
  "poi_id": "unique_id",
  "poi_name_vietnamese": "Phở Hùng",
  "poi_name_english": "Hung Pho",
  "business_category": "Restaurant",
  "sub_category": "Noodle Shop",
  "address_text": "24 Nguyen Du",
  "latitude": 10.771234,
  "longitude": 106.691234,
  "depth_meters": 15.5,
  "confidence": 0.92
}
```

### Camera Metadata

```python
{
  "image_id": "123456789",
  "camera_loc": {"lat": 10.77, "lon": 106.69},
  "heading": 45.0,
  "fov": 75.0,
  "camera_type": "perspective"
}
```

## Development

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Tutorial Workflow

The tutorial is designed to be run sequentially in IPython:

```bash
ipython
%run scripts/tutorial/00_pipeline_tutorial.py
```

Each block can be run independently:
- Blocks 0-10: Setup and data collection
- Blocks 11-19: POI reconstruction pipeline

## Design Decisions

### Simplified Approach

**Rejected Approaches:**
- ❌ Mapillary SfM: (overkill for single POI)
- ❌ Sequential Stereo: Too complex, requires image pairing
- ✅ **Chosen**: ML depth + geometric projection (works with single image)

### Depth Scaling

DPT model outputs relative depth (0-255). Converted to metric depth using:
```
depth_meters = 2.0 + (depth_relative / 255.0) * 98.0
```
- Maps 0 → 2m (close), 255 → 100m (far)
- Linear approximation (not production-ready, needs calibration)

### Camera Focus

- **Perspective cameras**: 95% of data
  - Normal FOV (60-90°): 69.5%
  - Wide FOV (>90°): 27.4%
- **Fisheye/Spherical**: 5% (excluded)

## License

[Add your license here]

## Contributors

[Add contributors here]

## Acknowledgments

- Mapillary community for street view imagery
- Intel for DPT depth estimation model
- GLM-4V for Vietnamese text recognition capabilities
