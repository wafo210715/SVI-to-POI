"""Tests for Street View crawler (GSV/Mapillary)."""

import json
import os
from pathlib import Path

import pytest

from src.crawler import GSVClient, MapillaryClient, StreetViewImage


class TestStreetViewImage:
    """Test the StreetViewImage data contract."""

    def test_street_view_image_has_required_fields(self):
        """Test that StreetViewImage contains all required metadata fields."""
        # Sample metadata from GSV API
        metadata = {
            "image_id": "test_pano_123",
            "camera_loc": {"lat": 10.7769, "lon": 106.7009},  # HCMC coordinates
            "heading": 180.0,
            "pitch": 0.0,
            "fov": 90.0,
            "depth_map": None,  # Will test depth availability separately
            "capture_date": "2024-01-15",
        }

        image = StreetViewImage(**metadata)

        assert image.image_id == "test_pano_123"
        assert image.camera_loc["lat"] == 10.7769
        assert image.camera_loc["lon"] == 106.7009
        assert image.heading == 180.0
        assert image.pitch == 0.0
        assert image.fov == 90.0
        assert image.depth_map is None

    def test_street_view_image_to_dict(self):
        """Test conversion to dictionary for JSON serialization."""
        metadata = {
            "image_id": "test_pano_456",
            "camera_loc": {"lat": 10.7820, "lon": 106.6950},
            "heading": 270.0,
            "pitch": -5.0,
            "fov": 120.0,
            "depth_map": "base64_encoded_depth_data",
            "capture_date": "2024-02-01",
        }

        image = StreetViewImage(**metadata)
        result = image.to_dict()

        assert result["image_id"] == "test_pano_456"
        assert result["camera_loc"]["lat"] == 10.7820
        assert result["heading"] == 270.0
        assert result["depth_map"] == "base64_encoded_depth_data"


class TestGSVMetadataExtraction:
    """Test GSV metadata extraction - specifically for depth and camera params."""

    @pytest.fixture
    def gsv_client(self):
        """Create a GSV client instance."""
        # Use env var for API key, but allow mock for testing
        api_key = os.getenv("GOOGLE_KEY", "test_key_for_unit_tests")
        return GSVClient(api_key=api_key)

    def test_gsv_client_initialization(self, gsv_client):
        """Test that GSV client initializes with API key."""
        assert gsv_client.api_key == "test_key_for_unit_tests"
        assert gsv_client.base_url == "https://maps.googleapis.com/maps/api/streetview"

    def test_construct_metadata_url(self, gsv_client):
        """Test GSV metadata URL construction with all parameters."""
        location = (10.7769, 106.7009)  # HCMC
        url = gsv_client._build_metadata_url(
            lat=location[0],
            lon=location[1],
            radius=50,
        )

        assert "maps.googleapis.com" in url
        assert "metadata" in url
        assert "location=" in url
        assert "radius=50" in url
        assert "key=" in url
        assert "return_error_code=true" in url

    def test_construct_panorama_url(self, gsv_client):
        """Test GSV panorama image URL construction."""
        pano_id = "test_pano_xyz"
        url = gsv_client._build_panorama_url(
            pano_id=pano_id,
            size=(640, 640),
            fov=90,
            heading=180,
            pitch=0,
        )

        assert "maps.googleapis.com" in url
        assert "size=640x640" in url
        assert "fov=90" in url
        assert "heading=180" in url
        assert "pitch=0" in url
        assert f"pano={pano_id}" in url


class TestGSVDepthAvailability:
    """Test depth map availability detection - critical for the project."""

    @pytest.fixture
    def gsv_client(self):
        """GSV client fixture for depth tests."""
        api_key = os.getenv("GOOGLE_KEY", "test_key_for_unit_tests")
        return GSVClient(api_key=api_key)

    @pytest.fixture
    def mock_gsv_metadata_with_depth(self):
        """Mock GSV metadata response that includes depth information."""
        return {
            "status": "OK",
            "pano_id": "depth_enabled_pano",
            "location": {"lat": 10.7769, "lng": 106.7009},
            "heading": 180.0,
            "pitch": 0.0,
            # Note: GSV doesn't directly provide depth in metadata API
            # This tests our parsing logic when depth data is available
            "date": "2024-01-15",
        }

    @pytest.fixture
    def mock_gsv_metadata_no_depth(self):
        """Mock GSV metadata response without depth information."""
        return {
            "status": "OK",
            "pano_id": "no_depth_pano",
            "location": {"lat": 10.7769, "lng": 106.7009},
            "heading": 180.0,
            "pitch": 0.0,
            "date": "2020-01-15",  # Older panoramas likely don't have depth
        }

    def test_check_depth_support_in_metadata(self, gsv_client, mock_gsv_metadata_with_depth):
        """Test checking if a panorama supports depth data."""
        has_depth = gsv_client._check_depth_support(mock_gsv_metadata_with_depth)

        # This will fail initially - we need to implement the logic
        # For now, test that our method exists and returns a boolean
        assert isinstance(has_depth, bool)


class TestMapillaryMetadataExtraction:
    """Test Mapillary metadata extraction - alternative to GSV."""

    @pytest.fixture
    def mapillary_client(self):
        """Create a Mapillary client instance."""
        access_token = os.getenv("MAPILLARY_ACCESS_TOKEN", "test_token")
        return MapillaryClient(access_token=access_token)

    def test_mapillary_client_initialization(self, mapillary_client):
        """Test Mapillary client initialization."""
        assert mapillary_client.access_token == "test_token"
        assert "graph.mapillary.com" in mapillary_client.base_url

    def test_construct_mapillary_search_url(self, mapillary_client):
        """Test Mapillary image search URL construction."""
        bbox = "106.65,10.75,106.75,10.82"  # HCMC bounding box
        url = mapillary_client._build_search_url(
            bbox=bbox,
            limit=10,
        )

        assert "graph.mapillary.com" in url
        assert "images" in url
        assert "bbox=" in url
        assert "limit=10" in url

    def test_construct_mapillary_image_url(self, mapillary_client):
        """Test Mapillary original image URL construction."""
        image_id = "test_image_xyz"
        url = mapillary_client._build_image_url(image_id, size=(1024, 1024))

        assert "mapillary.com" in url
        assert image_id in url
        assert url.startswith("https://")
        assert "thumb-1024x1024" in url

    def test_construct_mapillary_depth_url(self, mapillary_client):
        """Test Mapillary depth tiles URL construction.

        Mapillary provides depth via tiles (similar to map tiles).
        Format: https://tiles.mapillary.com/...
        """
        image_id = "depth_test_image"
        url = mapillary_client._build_depth_url(image_id, zoom=17)

        assert "mapillary.com" in url
        assert "depth" in url.lower() or "planar" in url.lower()
        assert image_id in url

    def test_parse_mapillary_image_data(self, mapillary_client):
        """Test parsing Mapillary image response for required metadata."""
        mock_response = {
            "data": [
                {
                    "id": "mapillary_image_123",
                    "geometry": {
                        "coordinates": [106.7009, 10.7769]  # [lon, lat]
                    },
                    "compass_angle": 180.0,
                    "is_pano": True,
                    # Mapillary provides computed depth via their SfM pipeline
                    "original_image_hash": "abc123",
                }
            ]
        }

        images = mapillary_client._parse_image_response(mock_response)

        assert len(images) == 1
        assert images[0]["image_id"] == "mapillary_image_123"
        assert images[0]["camera_loc"]["lat"] == 10.7769
        assert images[0]["camera_loc"]["lon"] == 106.7009
        assert images[0]["heading"] == 180.0

    def test_parse_mapillary_compass_angle(self, mapillary_client):
        """Test compass_angle is correctly parsed as heading."""
        mock_response = {
            "data": [
                {
                    "id": "heading_test",
                    "geometry": {"coordinates": [106.7, 10.77]},
                    "compass_angle": 273.5,  # North is 0, degrees clockwise
                    "is_pano": False,
                }
            ]
        }

        images = mapillary_client._parse_image_response(mock_response)
        assert images[0]["heading"] == 273.5

    def test_parse_mapillary_pano_detection(self, mapillary_client):
        """Test FOV is set based on is_pano flag."""
        mock_response_pano = {
            "data": [
                {"id": "pano", "geometry": {"coordinates": [106.7, 10.77]}, "is_pano": True}
            ]
        }
        mock_response_flat = {
            "data": [
                {"id": "flat", "geometry": {"coordinates": [106.7, 10.77]}, "is_pano": False}
            ]
        }

        pano_images = mapillary_client._parse_image_response(mock_response_pano)
        flat_images = mapillary_client._parse_image_response(mock_response_flat)

        assert pano_images[0]["fov"] == 90.0
        assert flat_images[0]["fov"] == 60.0


class TestCrawlerSaveLoad:
    """Test saving and loading raw metadata."""

    def test_save_metadata_to_file(self, tmp_path):
        """Test saving StreetViewImage metadata to JSON file."""
        metadata = {
            "image_id": "save_test_123",
            "camera_loc": {"lat": 10.7769, "lon": 106.7009},
            "heading": 90.0,
            "pitch": 0.0,
            "fov": 90.0,
            "depth_map": None,
            "capture_date": "2024-01-15",
        }

        image = StreetViewImage(**metadata)

        # Save to temp file
        output_file = tmp_path / "raw" / "test_metadata.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        image.save_to_file(output_file)

        # Verify file exists and contains correct data
        assert output_file.exists()

        with open(output_file, "r") as f:
            saved_data = json.load(f)

        assert saved_data["image_id"] == "save_test_123"
        assert saved_data["camera_loc"]["lat"] == 10.7769

    def test_load_metadata_from_file(self, tmp_path):
        """Test loading StreetViewImage from JSON file."""
        # First save a file
        metadata = {
            "image_id": "load_test_456",
            "camera_loc": {"lat": 10.7800, "lon": 106.7100},
            "heading": 45.0,
            "pitch": 5.0,
            "fov": 120.0,
            "depth_map": "test_depth",
            "capture_date": "2024-03-01",
        }

        input_file = tmp_path / "raw" / "test_load.json"
        input_file.parent.mkdir(parents=True, exist_ok=True)

        with open(input_file, "w") as f:
            json.dump(metadata, f)

        # Load it back
        image = StreetViewImage.load_from_file(input_file)

        assert image.image_id == "load_test_456"
        assert image.heading == 45.0
        assert image.depth_map == "test_depth"
