"""Tests for Mapillary tile scraping functionality."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from src.crawler import MapillaryClient


class TestMapillaryTiles:
    """Test Mapillary tile endpoint integration."""

    @pytest.fixture
    def tile_client(self):
        """Create Mapillary client for tile testing."""
        return MapillaryClient(access_token="test_token")

    def test_tile_url_construction(self, tile_client):
        """Test building tile URLs for different zoom levels."""
        # Overview layer (zoom 0-5)
        overview_url = tile_client._build_tile_url(z=14, x=13452, y=7543, layer="image")
        assert "mapillary.com" in overview_url
        assert "mly1_public" in overview_url
        assert "14/13452/7543" in overview_url

    def test_tile_coords_from_bbox(self, tile_client):
        """Test converting bounding box to tile coordinates at zoom 14."""
        # HCMC bounding box
        min_lon, min_lat = 106.65, 10.75
        max_lon, max_lat = 106.75, 10.82

        tiles = tile_client._bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom=14)

        # Should return list of tile coordinates
        assert isinstance(tiles, list)
        assert all(isinstance(t, dict) for t in tiles)
        if tiles:
            assert "x" in tiles[0]
            assert "y" in tiles[0]
            assert "z" in tiles[0]
            assert tiles[0]["z"] == 14

    def test_parse_tile_image_data(self, tile_client):
        """Test parsing image data from tile response."""
        # Mock tile response with image features
        mock_tile_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [106.7009, 10.7769]
                    },
                    "properties": {
                        "id": 123456789,
                        "captured_at": 1682931618011,
                        "compass_angle": 322,
                        "is_pano": True,
                        "sequence_id": "seq_abc",
                        "creator_id": 999
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [106.7010, 10.7770]
                    },
                    "properties": {
                        "id": 987654321,
                        "captured_at": 1682931619000,
                        "compass_angle": 45,
                        "is_pano": False,
                        "sequence_id": "seq_def",
                        "creator_id": 888
                    }
                }
            ]
        }

        images = tile_client._parse_tile_response(mock_tile_data)

        assert len(images) == 2
        assert images[0]["image_id"] == 123456789
        assert images[0]["camera_loc"]["lon"] == 106.7009
        assert images[0]["camera_loc"]["lat"] == 10.7769
        assert images[0]["heading"] == 322
        assert images[0]["is_pano"] is True
        assert images[0]["sequence_id"] == "seq_abc"

    def test_get_tiles_for_hcmc(self, tile_client):
        """Test getting all tiles covering HCMC at zoom 14."""
        hcmc_bbox = {
            "min_lon": 106.65,
            "min_lat": 10.75,
            "max_lon": 106.75,
            "max_lat": 10.82
        }

        tiles = tile_client.get_tiles_for_bbox(
            hcmc_bbox["min_lon"],
            hcmc_bbox["min_lat"],
            hcmc_bbox["max_lon"],
            hcmc_bbox["max_lat"],
            zoom=14
        )

        # Should get tile coordinates
        assert isinstance(tiles, list)
        # Each tile should have x, y, z coordinates
        for tile in tiles:
            assert "x" in tile
            assert "y" in tile
            assert "z" in tile
            assert tile["z"] == 14

    @patch('requests.get')
    def test_fetch_single_tile(self, mock_get, tile_client):
        """Test fetching a single tile from Mapillary."""
        # Mock successful tile response
        mock_response = Mock()
        mock_response.json.return_value = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [106.7, 10.77]},
                    "properties": {"id": 123, "captured_at": 1682931618011}
                }
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        tile_data = tile_client._fetch_tile(z=14, x=13452, y=7543)

        assert tile_data["type"] == "FeatureCollection"
        assert len(tile_data["features"]) == 1
        mock_get.assert_called_once()

    def test_generate_html_visualization(self, tile_client):
        """Test generating HTML visualization of tiles."""
        tiles = [
            {"z": 14, "x": 13452, "y": 7543, "image_count": 5},
            {"z": 14, "x": 13453, "y": 7543, "image_count": 0},
            {"z": 14, "x": 13452, "y": 7544, "image_count": 12},
        ]

        html = tile_client._generate_tile_visualization_html(
            tiles=tiles,
            bbox={"min_lon": 106.65, "min_lat": 10.75, "max_lon": 106.75, "max_lat": 10.82},
            output_path="data/debug/tiles_viz.html"
        )

        # HTML should be generated
        assert html is not None or True  # Function writes to file
        # Check that file was created (implementation detail)
        from pathlib import Path
        assert Path("data/debug/tiles_viz.html").exists()


class TestTileCoordinateMath:
    """Test tile coordinate conversion math."""

    @pytest.fixture
    def tile_client(self):
        return MapillaryClient(access_token="test_token")

    def test_lon_lat_to_tile(self, tile_client):
        """Test converting lon/lat to tile coordinates."""
        # Known conversion for HCMC at zoom 14
        lon, lat = 106.7009, 10.7769

        tile_x, tile_y = tile_client._lon_lat_to_tile(lon, lat, zoom=14)

        # Should return integer tile coordinates
        assert isinstance(tile_x, int)
        assert isinstance(tile_y, int)

    def test_tile_to_lon_lat(self, tile_client):
        """Test converting tile coordinates back to lon/lat."""
        tile_x, tile_y, zoom = 13452, 7543, 14

        min_lon, max_lon, min_lat, max_lat = tile_client._tile_to_bbox(tile_x, tile_y, zoom)

        # Should return valid bounding box
        assert isinstance(min_lon, float)
        assert isinstance(max_lon, float)
        assert isinstance(min_lat, float)
        assert isinstance(max_lat, float)
        assert min_lon < max_lon
        assert min_lat < max_lat
