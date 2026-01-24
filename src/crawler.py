"""Street View crawler for GSV and Mapillary with metadata extraction.

Focus: Extract depth maps and camera parameters for geo-localization.
"""

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from urllib.parse import urlencode

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class StreetViewImage:
    """Data contract for Street View image with required metadata."""

    image_id: str
    camera_loc: Dict[str, float]  # {"lat": float, "lon": float}
    heading: float
    pitch: float
    fov: float
    depth_map: Optional[str]  # Base64 encoded or file path
    capture_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save_to_file(self, filepath: Path) -> None:
        """Save metadata to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: Path) -> "StreetViewImage":
        """Load metadata from JSON file."""
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(**data)


class GSVClient:
    """Google Street View API client.

    Focus: Extract metadata including depth availability.
    """

    BASE_URL = "https://maps.googleapis.com/maps/api/streetview"
    METADATA_URL = f"{BASE_URL}/metadata"

    def __init__(self, api_key: str):
        """Initialize GSV client with API key."""
        self.api_key = api_key
        self.base_url = self.BASE_URL

    def _build_metadata_url(
        self,
        lat: float,
        lon: float,
        radius: int = 50,
    ) -> str:
        """Build metadata API URL for a location."""
        params = {
            "location": f"{lat},{lon}",
            "radius": radius,
            "key": self.api_key,
            "return_error_code": "true",
        }
        return f"{self.METADATA_URL}?{urlencode(params)}"

    def _build_panorama_url(
        self,
        pano_id: str,
        size: Tuple[int, int] = (640, 640),
        fov: int = 90,
        heading: int = 0,
        pitch: int = 0,
    ) -> str:
        """Build panorama image URL."""
        params = {
            "pano": pano_id,
            "size": f"{size[0]}x{size[1]}",
            "fov": fov,
            "heading": heading,
            "pitch": pitch,
            "key": self.api_key,
        }
        return f"{self.BASE_URL}?{urlencode(params)}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_metadata(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Fetch panorama metadata for a location.

        Returns:
            Metadata dict or None if no panorama available.
        """
        url = self._build_metadata_url(lat, lon)

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            return None

        return data

    def _check_depth_support(self, metadata: Dict[str, Any]) -> bool:
        """Check if panorama supports depth data.

        Note: GSV metadata API doesn't directly indicate depth support.
        We need to infer from capture date (post-2017 panoramas likely have depth)
        or test via the separate depth endpoint.

        For now, we use a heuristic based on date.
        """
        date_str = metadata.get("date", "")
        if not date_str:
            return False

        # Panoramas from 2017 onwards likely have computed depth
        # Format: "YYYY-MM" or "YYYY-MM"
        try:
            year = int(date_str.split("-")[0])
            return year >= 2017
        except (ValueError, IndexError):
            return False

    def fetch_street_view(
        self,
        lat: float,
        lon: float,
        radius: int = 50,
    ) -> Optional[StreetViewImage]:
        """Fetch Street View image with full metadata.

        Args:
            lat: Latitude
            lon: Longitude
            radius: Search radius in meters

        Returns:
            StreetViewImage or None if no panorama available.
        """
        metadata = self.get_metadata(lat, lon)
        if not metadata:
            return None

        pano_id = metadata.get("pano_id")
        location = metadata.get("location", {})

        image = StreetViewImage(
            image_id=pano_id,
            camera_loc={"lat": location.get("lat", lat), "lon": location.get("lng", lon)},
            heading=metadata.get("heading", 0),
            pitch=metadata.get("pitch", 0),
            fov=90.0,  # Default FOV
            depth_map=None,  # Will be fetched separately
            capture_date=metadata.get("date"),
        )

        return image


class MapillaryClient:
    """Mapillary API client.

    Mapillary provides SfM-computed depth for all images.

    Key advantages over GSV:
    - Depth maps available via tiles endpoint
    - No API key required for public images
    - Original images available in high resolution
    """

    GRAPH_API_URL = "https://graph.mapillary.com"
    TILES_URL = "https://tiles.mapillary.com"
    IMAGE_URL = "https://images.mapillary.com"

    def __init__(self, access_token: str):
        """Initialize Mapillary client."""
        self.access_token = access_token
        self.base_url = self.GRAPH_API_URL

    def _build_search_url(
        self,
        bbox: str,
        limit: int = 10,
    ) -> str:
        """Build image search URL.

        Args:
            bbox: Bounding box "min_lon,min_lat,max_lon,max_lat"
            limit: Max results

        Note: Mapillary requires specifying fields to return.
        We need: id, geometry, compass_angle, is_pano, captured_at
        """
        # Mapillary v4 API requires fields parameter
        fields = "id,geometry,compass_angle,computed_compass_angle,is_pano,captured_at,quality_score"
        params = {
            "bbox": bbox,
            "limit": limit,
            "fields": fields,
            "access_token": self.access_token,
        }
        return f"{self.GRAPH_API_URL}/images?{urlencode(params)}"

    def _build_image_url(
        self,
        image_id: str,
        size: Tuple[int, int] = (1024, 1024),
    ) -> str:
        """Build URL to download original image.

        Args:
            image_id: Mapillary image ID
            size: Image dimensions (width, height)

        Returns:
            URL to fetch the image
        """
        # Mapillary image URL format:
        # https://images.mapillary.com/{image_id}/thumb-{width}x{height}.jpg
        width, height = size
        return f"{self.IMAGE_URL}/{image_id}/thumb-{width}x{height}.jpg"

    def _build_depth_url(
        self,
        image_id: str,
        zoom: int = 17,
    ) -> str:
        """Build URL to fetch depth map tiles.

        Mapillary provides depth via vector tiles (Planar Depth).
        Format: https://tiles.mapillary.com/...

        Note: This returns tile metadata. Actual depth data needs
        to be fetched from the tile service.

        Args:
            image_id: Mapillary image ID
            zoom: Tile zoom level

        Returns:
            URL to depth tile endpoint
        """
        # Mapillary depth tiles are served via their tiles API
        # Format: https://tiles.mapillary.com/...
        # We'll use the graph API to get the sequence info first
        return f"{self.TILES_URL}/maps/points/{image_id}?access_token={self.access_token}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def search_images(
        self,
        bbox: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for images within a bounding box.

        Args:
            bbox: "min_lon,min_lat,max_lon,max_lat"
            limit: Max results

        Returns:
            List of image metadata dicts.
        """
        url = self._build_search_url(bbox, limit)

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data.get("data", [])

    def _parse_image_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Mapillary image response to standard format.

        Mapillary returns:
        - id: Image ID
        - geometry.coordinates: [lon, lat]
        - compass_angle/computed_compass_angle: Heading
        - is_pano: Whether it's a panorama
        - captured_at: Timestamp
        - quality_score: Image quality

        For depth: Mapillary provides depth via their tiles endpoint.
        """
        images = []

        for item in response.get("data", []):
            coords = item.get("geometry", {}).get("coordinates", [0, 0])

            # Use computed_compass_angle if available, fallback to compass_angle
            heading = item.get("computed_compass_angle") or item.get("compass_angle", 0)

            images.append({
                "image_id": item.get("id"),
                "camera_loc": {"lon": coords[0], "lat": coords[1]},
                "heading": heading,
                "pitch": 0,  # Mapillary doesn't provide pitch in basic response
                "fov": 90.0 if item.get("is_pano") else 60.0,
                "depth_map": None,  # Fetchable via tiles endpoint
                "capture_date": item.get("captured_at"),
                "quality_score": item.get("quality_score"),
            })

        return images

    def fetch_street_views(
        self,
        bbox: str,
        limit: int = 10,
    ) -> List[StreetViewImage]:
        """Fetch multiple street views within a bounding box.

        Args:
            bbox: "min_lon,min_lat,max_lon,max_lat"
            limit: Max results

        Returns:
            List of StreetViewImage objects.
        """
        response_data = self.search_images(bbox, limit)
        parsed = self._parse_image_response({"data": response_data})

        images = []
        for item in parsed:
            # Swap lon/lat order for our standard format
            images.append(StreetViewImage(
                image_id=item["image_id"],
                camera_loc={"lat": item["camera_loc"]["lat"], "lon": item["camera_loc"]["lon"]},
                heading=item["heading"],
                pitch=item["pitch"],
                fov=item["fov"],
                depth_map=None,
                capture_date=item.get("capture_date"),
            ))

        return images

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def fetch_image(self, image_id: str, size: Tuple[int, int] = (1024, 1024)) -> bytes:
        """Download the original image.

        Args:
            image_id: Mapillary image ID
            size: Image dimensions

        Returns:
            Image data as bytes
        """
        url = self._build_image_url(image_id, size)

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        return response.content

    def check_depth_available(self, image_id: str) -> bool:
        """Check if depth data is available for an image.

        Mapillary has depth for most modern images (post-2020).

        Args:
            image_id: Mapillary image ID

        Returns:
            True if depth likely available
        """
        # Mapillary computes depth for all images via SfM
        # Most recent images have depth available
        return True

    def fetch_street_view_with_image(
        self,
        bbox: str,
        download_images: bool = True,
        limit: int = 10,
    ) -> List[StreetViewImage]:
        """Fetch street views with optional image download.

        Args:
            bbox: "min_lon,min_lat,max_lon,max_lat"
            download_images: Whether to download actual images
            limit: Max results

        Returns:
            List of StreetViewImage objects
        """
        response_data = self.search_images(bbox, limit)
        parsed = self._parse_image_response({"data": response_data})

        images = []
        for item in parsed:
            image_id = item["image_id"]
            depth_available = self.check_depth_available(image_id)

            sv_image = StreetViewImage(
                image_id=image_id,
                camera_loc={"lat": item["camera_loc"]["lat"], "lon": item["camera_loc"]["lon"]},
                heading=item["heading"],
                pitch=item["pitch"],
                fov=item["fov"],
                depth_map=self._build_depth_url(image_id) if depth_available else None,
                capture_date=item.get("capture_date"),
            )

            if download_images:
                # Save image to data/raw/
                output_path = Path("data/raw") / f"{image_id}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                image_data = self.fetch_image(image_id)
                with open(output_path, "wb") as f:
                    f.write(image_data)

                # Also save metadata
                metadata_path = Path("data/raw") / f"{image_id}_metadata.json"
                sv_image.save_to_file(metadata_path)

            images.append(sv_image)

        return images

    # ========== Tile-based Scraping (Better Coverage) ==========

    def _build_tile_url(self, z: int, x: int, y: int, layer: str = "image") -> str:
        """Build Mapillary tile URL.

        Args:
            z: Zoom level (14 for image layer)
            x: Tile X coordinate
            y: Tile Y coordinate
            layer: Layer name ("image" for zoom 14)

        Returns:
            Tile URL
        """
        return f"{self.TILES_URL}/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token={self.access_token}"

    def _lon_lat_to_tile(self, lon: float, lat: float, zoom: int) -> Tuple[int, int]:
        """Convert longitude/latitude to tile coordinates.

        Uses Web Mercator projection (EPSG:3857).

        Args:
            lon: Longitude
            lat: Latitude
            zoom: Zoom level

        Returns:
            (tile_x, tile_y) tuple
        """
        import math

        n = 2 ** zoom
        tile_x = int((lon + 180) / 360 * n)
        lat_rad = math.radians(lat)
        tile_y = int((1 - math.asinh(math.tan(lat_rad)) / math.pi) / 2 * n)

        return tile_x, tile_y

    def _tile_to_bbox(self, tile_x: int, tile_y: int, zoom: int) -> Tuple[float, float, float, float]:
        """Convert tile coordinates to bounding box.

        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level

        Returns:
            (min_lon, max_lon, min_lat, max_lat) tuple
        """
        import math

        n = 2 ** zoom
        min_lon = tile_x / n * 360 - 180
        max_lon = (tile_x + 1) / n * 360 - 180

        lat_rad1 = math.asin(math.tanh(math.pi * (1 - 2 * tile_y / n)))
        lat_rad2 = math.asin(math.tanh(math.pi * (1 - 2 * (tile_y + 1) / n)))
        min_lat = math.degrees(lat_rad2)
        max_lat = math.degrees(lat_rad1)

        return min_lon, max_lon, min_lat, max_lat

    def _bbox_to_tiles(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 14,
    ) -> List[Dict[str, int]]:
        """Get all tile coordinates covering a bounding box.

        Uses mercantile library for accurate tile calculation.

        Args:
            min_lon: Minimum longitude (west)
            min_lat: Minimum latitude (south)
            max_lon: Maximum longitude (east)
            max_lat: Maximum latitude (north)
            zoom: Zoom level

        Returns:
            List of tile coordinate dicts with keys: z, x, y
        """
        import mercantile

        tiles = []
        for tile in mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom):
            tiles.append({"z": tile.z, "x": tile.x, "y": tile.y})

        return tiles

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _fetch_tile(self, z: int, x: int, y: int, layer: str = "image") -> Dict[str, Any]:
        """Fetch a single tile from Mapillary and convert to GeoJSON.

        Mapillary returns Mapbox Vector Tiles (MVT) format - raw protobuf.
        Uses vt2geojson to convert to GeoJSON.

        Args:
            z: Zoom level
            x: Tile X coordinate
            y: Tile Y coordinate
            layer: Layer name ("image", "sequence", "overview")

        Returns:
            Tile data as GeoJSON dict
        """
        from vt2geojson.tools import vt_bytes_to_geojson

        url = self._build_tile_url(z, x, y, layer)

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Convert MVT bytes to GeoJSON
        # response.content is the raw protobuf bytes
        geojson = vt_bytes_to_geojson(
            response.content,
            x,
            y,
            z,
            layer=layer
        )

        return geojson

    def _parse_tile_response(self, tile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse image data from tile response.

        Args:
            tile_data: GeoJSON FeatureCollection from tile

        Returns:
            List of image metadata dicts
        """
        images = []

        for feature in tile_data.get("features", []):
            geometry = feature.get("geometry", {})
            coords = geometry.get("coordinates", [0, 0])
            props = feature.get("properties", {})

            images.append({
                "image_id": props.get("id"),
                "camera_loc": {"lon": coords[0], "lat": coords[1]},
                "heading": props.get("compass_angle", 0),
                "pitch": 0,  # Not provided in tiles
                "fov": 90.0 if props.get("is_pano") else 60.0,
                "depth_map": None,  # Fetchable separately
                "capture_date": props.get("captured_at"),
                "is_pano": props.get("is_pano", False),
                "sequence_id": props.get("sequence_id"),
                "creator_id": props.get("creator_id"),
            })

        return images

    def get_tiles_for_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 14,
    ) -> List[Dict[str, int]]:
        """Get all tile coordinates covering a bounding box.

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            zoom: Zoom level (14 for image layer)

        Returns:
            List of tile coordinate dicts
        """
        return self._bbox_to_tiles(min_lon, min_lat, max_lon, max_lat, zoom)

    def scrape_tiles(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        zoom: int = 14,
        save_tiles: bool = True,
    ) -> List[Dict[str, Any]]:
        """Scrape all tiles covering a bounding box.

        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            zoom: Zoom level
            save_tiles: Whether to save raw tile data

        Returns:
            List of all images found, with tile info
        """
        tiles = self.get_tiles_for_bbox(min_lon, min_lat, max_lon, max_lat, zoom)

        all_images = []

        for tile in tiles:
            try:
                tile_data = self._fetch_tile(tile["z"], tile["x"], tile["y"])
                images = self._parse_tile_response(tile_data)

                # Add tile info to each image
                for img in images:
                    img["tile_z"] = tile["z"]
                    img["tile_x"] = tile["x"]
                    img["tile_y"] = tile["y"]

                all_images.extend(images)

                if save_tiles:
                    # Save raw tile data
                    output_dir = Path("data/raw/tiles")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    tile_file = output_dir / f"tile_{tile['z']}_{tile['x']}_{tile['y']}.json"
                    with open(tile_file, "w") as f:
                        json.dump(tile_data, f, indent=2)

            except Exception as e:
                print(f"Warning: Failed to fetch tile {tile['z']}/{tile['x']}/{tile['y']}: {e}")
                continue

        return all_images

    def _generate_tile_visualization_html(
        self,
        tiles: List[Dict[str, Any]],
        bbox: Dict[str, float],
        output_path: str = "data/debug/tiles_viz.html",
    ) -> None:
        """Generate HTML visualization of tiles.

        Args:
            tiles: List of tile dicts with z, x, y, and optionally image_count
            bbox: Bounding box dict with min_lon, min_lat, max_lon, max_lat
            output_path: Where to save the HTML file
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mapillary Tiles - HCMC Coverage</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .tile-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }}
        .tile {{ border: 2px solid #ccc; padding: 10px; border-radius: 5px; text-align: center; }}
        .tile.has-images {{ background: #d4edda; border-color: #28a745; }}
        .tile.no-images {{ background: #f8d7da; border-color: #dc3545; }}
        .tile-coords {{ font-weight: bold; margin-bottom: 5px; }}
        .image-count {{ font-size: 14px; color: #666; }}
        .bbox {{ margin-top: 20px; padding: 10px; background: #e9ecef; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Mapillary Tiles - HCMC Coverage (Zoom 14)</h1>

    <div class="info">
        <p><strong>Total Tiles:</strong> {len(tiles)}</p>
        <p><strong>Bounding Box:</strong></p>
        <ul>
            <li>Min Lon: {bbox.get('min_lon', 'N/A')}</li>
            <li>Max Lon: {bbox.get('max_lon', 'N/A')}</li>
            <li>Min Lat: {bbox.get('min_lat', 'N/A')}</li>
            <li>Max Lat: {bbox.get('max_lat', 'N/A')}</li>
        </ul>
    </div>

    <div class="tile-grid">
"""

        for tile in tiles:
            image_count = tile.get("image_count", 0)
            css_class = "has-images" if image_count > 0 else "no-images"

            html += f"""
        <div class="tile {css_class}">
            <div class="tile-coords">Z{tile['z']}/{tile['x']}/{tile['y']}</div>
            <div class="image-count">{image_count} images</div>
        </div>
"""

        html += """
    </div>

    <div class="bbox">
        <p><strong>Mapillary Coverage Tiles API</strong></p>
        <p>Zoom 14 = Image layer with detailed metadata</p>
    </div>
</body>
</html>
"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(html)

    # ========== Entity Metadata Fetching (for Depth Investigation) ==========

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def fetch_image_entity(self, image_id: str) -> Dict[str, Any]:
        """Fetch full image metadata including depth/SfM fields.

        Fetches detailed image entity from Mapillary's Graph API.
        Fields requested (based on Mapillary API v4 documentation):
        - id, geometry, computed_geometry
        - camera_parameters (focal, k1, k2), camera_type
        - make, model (camera manufacturer and model)
        - altitude, computed_altitude
        - sequence, is_pano
        - thumb_original_url
        - creator (username, id)
        - sfm_cluster (point cloud data if available)

        Args:
            image_id: Mapillary image ID

        Returns:
            Full JSON response with image metadata

        Raises:
            requests.HTTPError: If API request fails
        """
        fields = [
            "id",
            "geometry",
            "computed_geometry",
            "camera_parameters",
            "camera_type",
            "make",
            "model",
            "altitude",
            "computed_altitude",
            "sequence",
            "is_pano",
            "thumb_original_url",
            "creator",
            "captured_at",
            "quality_score",
            "exif_orientation",
            "height",
            "width",
            "compass_angle",
            "computed_compass_angle",
            "computed_rotation",
            "atomic_scale",
            "merge_cc",
            "merge_version",
            "thumb_2048_url",
            "thumb_1024_url",
            "thumb_640_url",
            "thumb_320_url",
            "sfm_cluster",
        ]

        fields_str = ",".join(fields)
        params = {
            "fields": fields_str,
            "access_token": self.access_token,
        }

        url = f"{self.GRAPH_API_URL}/images/{image_id}"
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        return response.json()

    def group_tiles_by_sequence(self, tile_data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group images from tiles by sequence_id.

        Since sequence_id is already in tile data, we can group directly
        without additional API calls. This enables:
        - Device-specific depth analysis (same sequence = same device)
        - Temporal ordering for 3D reconstruction
        - Efficient processing by device/camera type

        Args:
            tile_data: List of image dicts from tile scraping, each containing
                      at least 'sequence_id' and 'image_id' keys

        Returns:
            Dict mapping sequence_id to list of images in that sequence:
                {
                    "sequence_id_1": [image1, image2, ...],
                    "sequence_id_2": [image3, image4, ...]
                }
        """
        sequences: Dict[str, List[Dict[str, Any]]] = {}

        for image in tile_data:
            sequence_id = image.get("sequence_id")
            if not sequence_id:
                continue

            if sequence_id not in sequences:
                sequences[sequence_id] = []

            sequences[sequence_id].append(image)

        return sequences

    def get_sequence_summary(
        self,
        grouped_sequences: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Generate summary statistics for each sequence.

        Args:
            grouped_sequences: Output from group_tiles_by_sequence()

        Returns:
            List of sequence summaries with keys:
                - sequence_id: str
                - image_count: int
                - creator_id: str (from first image)
                - first_capture_time: Optional timestamp
                - last_capture_time: Optional timestamp
        """
        summaries = []

        for seq_id, images in grouped_sequences.items():
            if not images:
                continue

            # Get creator_id from first image (should be same for all)
            creator_id = images[0].get("creator_id", "unknown")

            # Extract capture times if available
            capture_times = [img.get("capture_date") for img in images if img.get("capture_date")]

            summary = {
                "sequence_id": seq_id,
                "image_count": len(images),
                "creator_id": creator_id,
                "first_capture_time": min(capture_times) if capture_times else None,
                "last_capture_time": max(capture_times) if capture_times else None,
            }

            summaries.append(summary)

        # Sort by image count descending
        summaries.sort(key=lambda x: x["image_count"], reverse=True)

        return summaries
