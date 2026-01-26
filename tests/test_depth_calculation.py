"""Tests for depth calculation methods (Block 11).

Tests three approaches for depth estimation:
1. Mapillary SfM depth (pre-computed via sfm_cluster)
2. Sequential stereo triangulation
3. ML monocular depth estimation
"""

import json
import zlib
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

import numpy as np

import sys

# Add scripts/tutorial to path for importing blocks_11_20
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tutorial"))

# Test data constants
SAMPLE_IMAGE_ID = "123456789012345"
SAMPLE_IMAGE_ID_2 = "123456789012346"


class TestFetchMapillaryDepth:
    """Test fetching Mapillary pre-computed depth from sfm_cluster."""

    def test_fetch_mapillary_depth_saves_npy_file(self, tmp_path):
        """Test that Mapillary depth fetching saves to .npy file."""
        import blocks_11_20

        client = Mock()
        output_dir = tmp_path / "depth_maps"

        # Mock API response with sfm_cluster data
        mock_entity = {
            "id": SAMPLE_IMAGE_ID,
            "sfm_cluster": {
                "id": "cluster_123",
                "url": "https://example.com/sfm_cluster.json"
            }
        }
        client.fetch_image_entity.return_value = mock_entity

        # Mock the HTTP response for downloading sfm_cluster
        mock_response = Mock()
        mock_response.content = zlib.compress(json.dumps([{"position": [1, 2, 3]}]).encode())
        mock_response.raise_for_status = Mock()

        with patch("blocks_11_20.requests.get", return_value=mock_response):
            result_path = blocks_11_20.fetch_mapillary_depth(client, SAMPLE_IMAGE_ID, str(output_dir))

            # Verify it was saved with correct name
            expected_path = output_dir / f"{SAMPLE_IMAGE_ID}_mapillary_depth.npy"
            assert result_path == expected_path

            # Verify file was actually created
            assert result_path.exists()

    def test_fetch_mapillary_depth_handles_missing_sfm_cluster(self, tmp_path):
        """Test graceful handling when sfm_cluster is not available."""
        import blocks_11_20

        client = Mock()
        output_dir = tmp_path / "depth_maps"

        # Mock response without sfm_cluster
        mock_entity = {
            "id": SAMPLE_IMAGE_ID,
            # No sfm_cluster field
        }
        client.fetch_image_entity.return_value = mock_entity

        result = blocks_11_20.fetch_mapillary_depth(client, SAMPLE_IMAGE_ID, str(output_dir))

        assert result is None


class TestSequentialStereoDepth:
    """Test sequential stereo triangulation for depth estimation."""

    def test_calculates_baseline_from_gps_distance(self):
        """Test that baseline is calculated as GPS distance between cameras."""
        import blocks_11_20

        # Two cameras 10 meters apart (approximately)
        camera1_data = {
            "lat": 10.770,
            "lon": 106.690,
            "camera_parameters": [1000, 0, 0],  # focal, k1, k2
        }
        camera2_data = {
            "lat": 10.77009,  # ~10m north
            "lon": 106.690,
            "camera_parameters": [1000, 0, 0],
        }

        # Same POI at different pixel positions (disparity)
        poi_pixel_1 = (500, 300)
        poi_pixel_2 = (450, 300)  # 50 pixels disparity

        depth = blocks_11_20.estimate_depth_sequential_stereo(
            None, None,  # images not needed for basic calculation
            poi_pixel_1, poi_pixel_2,
            camera1_data, camera2_data
        )

        # depth should be positive and reasonable (baseline * focal / disparity)
        # baseline ~10m, focal=1000, disparity=50 => depth ~200m
        assert depth > 0
        assert depth < 500  # Should be reasonable

    def test_depth_formula_correct(self):
        """Test the stereo depth formula: depth = (baseline * focal) / disparity."""
        import blocks_11_20

        # Simplified test case
        camera1_data = {
            "lat": 0.0,
            "lon": 0.0,
            "camera_parameters": [100, 0, 0],
        }
        camera2_data = {
            "lat": 0.0,
            "lon": 0.001,  # ~111m apart at equator
            "camera_parameters": [100, 0, 0],
        }

        poi_pixel_1 = (100, 100)
        poi_pixel_2 = (90, 100)  # 10 pixels disparity

        depth = blocks_11_20.estimate_depth_sequential_stereo(
            None, None, poi_pixel_1, poi_pixel_2, camera1_data, camera2_data
        )

        # With baseline ~111m, focal=100, disparity=10 => depth ~1110m
        assert depth > 1000
        assert depth < 1500


class TestMLMonocularDepth:
    """Test ML-based monocular depth estimation."""

    @patch("blocks_11_20.pipeline")
    def test_estimates_depth_from_single_image(self, mock_pipeline):
        """Test that ML depth estimator returns depth map."""
        import blocks_11_20

        # Mock the pipeline and depth result
        mock_estimator = Mock()
        mock_pipeline.return_value = mock_estimator

        # Mock depth map result
        mock_depth_map = Mock()
        mock_depth_map.numpy.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_estimator.return_value = mock_depth_map

        # Create mock image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = blocks_11_20.estimate_depth_ml(mock_image, model_name="zoedepth")

        # Verify pipeline was called with correct model
        mock_pipeline.assert_called_once()
        assert "depth" in str(mock_pipeline.call_args)

    def test_supports_multiple_models(self):
        """Test that different models can be specified."""
        import blocks_11_20

        supported_models = ["zoedepth", "midas", "depth-anything"]

        for model in supported_models:
            # Just verify the model name is accepted
            # (actual call will be mocked)
            assert model in blocks_11_20.DepthModels.all()


class TestDepthComparison:
    """Test comparison of all three depth methods."""

    def test_generates_comparison_report(self, tmp_path):
        """Test that comparison generates JSON report with all methods."""
        import blocks_11_20

        client = Mock()
        image_id = SAMPLE_IMAGE_ID
        output_dir = tmp_path / "comparison"

        # Create a mock image
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock all three methods to return sample depths
        with patch("blocks_11_20.fetch_mapillary_depth") as mock_mapillary, \
             patch("blocks_11_20.estimate_depth_ml") as mock_ml:

            mock_mapillary.return_value = str(output_dir / "mapillary.npy")
            mock_ml.return_value = np.ones((100, 100)) * 5.0  # 5m depth

            result = blocks_11_20.compare_depth_methods(client, image_id, mock_image, str(output_dir))

            # Check report was generated
            report_path = output_dir / "depth_comparison_report.json"
            assert report_path.exists()

            # Check report structure
            with open(report_path) as f:
                report = json.load(f)

            assert "image_id" in report
            assert "methods" in report
            assert "mapillary" in report["methods"]
            assert "ml_depth" in report["methods"]


class TestDepthVisualization:
    """Test depth map visualization (Block 12)."""

    def test_visualize_depth_map_creates_plot(self, tmp_path):
        """Test that depth visualization creates matplotlib figure."""
        import blocks_11_20

        # Skip if matplotlib not available
        if not blocks_11_20.HAS_MATPLOTLIB:
            pytest.skip("matplotlib not installed")

        # Create sample depth map
        depth_map = np.random.rand(100, 100) * 10  # 0-10m range
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        output_path = tmp_path / "depth_visualization.png"

        blocks_11_20.visualize_depth_map(image, depth_map, str(output_path))

        # Verify file was created
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
