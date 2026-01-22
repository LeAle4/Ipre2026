"""
Test script for generating threshold-based crops from geoglifs using polygon overlap.
Uses make_polygon_thresholds_crops and loads polygon coordinates from metadata JSON files.
Organizes output by area, window size, stride, threshold, and noise method.
"""

import cv2
import json
import numpy as np
from pathlib import Path
from format import make_polygon_thresholds_crops, fill_with_noise, TEST_DIR

# Configuration
WINDOW_SIZES = [122, 244, 488]
N_CROPS_PER_IMAGE = 10
NOISE_LEVELS = [0.0, 0.5, 1.0]  # 0=avg color, 0.5=blend, 1.0=full noise
NOISE_TYPES = ['gaussian', 'uniform']
THRESHOLDS = [0.1, 0.3, 0.5]  # 10%, 30%, 50% overlap thresholds

# Set to None to auto-compute stride based on image size and n_crops
# or provide an int or (x, y) tuple for explicit stride.
STRIDE = None

OUTPUT_BASE = Path(__file__).resolve().parent / "crops_test/thresholds_crops_output"


def extract_area_from_filename(filename):
	"""Extract area name (unita, lluta, chug) from filename."""
	name = filename.lower()
	if name.startswith('unita_'):
		return 'unita'
	if name.startswith('lluta_'):
		return 'lluta'
	if name.startswith('chug_'):
		return 'chug'
	return 'unknown'


def stride_label(stride):
	"""Generate a short label for stride directory naming."""
	if stride is None:
		return "auto"
	if isinstance(stride, (list, tuple, np.ndarray)):
		if len(stride) == 2:
			return f"{stride[0]}x{stride[1]}"
		return "invalid"
	return str(stride)


def load_polygon_from_metadata(json_path):
	"""
	Load polygon vertices from metadata JSON file.
	
	Expects JSON format with 'polygon_points', 'polygon' or 'geometry' field containing coordinates.
	"""
	try:
		with open(json_path, 'r') as f:
			metadata = json.load(f)
		
		# Try different possible keys for polygon data
		if 'polygon_points' in metadata:
			polygon_points = metadata['polygon_points']
			if polygon_points and isinstance(polygon_points, list) and 'exterior' in polygon_points[0]:
				coords = polygon_points[0]['exterior']
			else:
				coords = polygon_points
		elif 'polygon' in metadata:
			coords = metadata['polygon']
		elif 'geometry' in metadata:
			geom = metadata['geometry']
			if isinstance(geom, dict) and 'coordinates' in geom:
				coords = geom['coordinates']
				# Handle nested coordinate arrays (e.g., from GeoJSON)
				if coords and isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (list, tuple)):
					coords = coords[0]
			else:
				coords = geom
		elif 'coordinates' in metadata:
			coords = metadata['coordinates']
		else:
			print(f"    Warning: Could not find polygon data in {json_path.name}")
			return None
		
		# Convert to list of tuples if needed
		if isinstance(coords, list):
			polygon_vertices = [(float(x), float(y)) for x, y in coords]
		else:
			polygon_vertices = coords
		
		return polygon_vertices
	
	except Exception as e:
		print(f"    Error loading polygon from {json_path.name}: {e}")
		return None


def find_metadata_file(image_path):
	"""
	Find the corresponding metadata JSON file for an image in test_geos.
	
	Args:
		image_path: Path to the image file (.jpg or .tif)
	
	Returns:
		Path to metadata file or None if not found
	"""
	# Extract the area and geoglif ID from the filename
	# E.g., unita_geoglif_0000_ortho.jpg -> unita_geoglif_0000_metadata.json
	stem = image_path.stem.replace('_ortho', '')
	json_path = image_path.parent / f"{stem}_metadata.json"
	
	return json_path if json_path.exists() else None


def process_geoglif(image_path, metadata_path, area, output_dir):
	"""Process a single geoglif image with polygon threshold configurations."""
	print(f"\nProcessing: {image_path.name}")

	# Load image
	image = cv2.imread(str(image_path))
	if image is None:
		print(f"  Error: Could not load {image_path}")
		return

	# Load metadata
	try:
		with open(metadata_path, 'r') as f:
			metadata = json.load(f)
	except Exception as e:
		print(f"  Error: Could not load metadata: {e}")
		return

	# Load polygon
	polygon_vertices = load_polygon_from_metadata(metadata_path)
	if polygon_vertices is None:
		print(f"  Error: Could not load polygon from metadata")
		return

	# Get image metadata
	if 'bounds' not in metadata or 'image_shape' not in metadata:
		print(f"  Error: Missing bounds or image_shape in metadata")
		return

	bounds = metadata['bounds']
	img_shape = metadata['image_shape']
	
	# Convert geographic coordinates to pixel coordinates
	geo_minx, geo_miny = bounds['minx'], bounds['miny']
	geo_maxx, geo_maxy = bounds['maxx'], bounds['maxy']
	img_width, img_height = img_shape['width'], img_shape['height']
	
	# Create conversion functions
	def geo_to_pixel(lon, lat):
		"""Convert geographic coordinates to pixel coordinates."""
		# Normalize to [0, 1]
		x_norm = (lon - geo_minx) / (geo_maxx - geo_minx)
		y_norm = (geo_maxy - lat) / (geo_maxy - geo_miny)  # Invert Y axis
		
		# Convert to pixel coordinates
		x_pixel = x_norm * img_width
		y_pixel = y_norm * img_height
		
		return (x_pixel, y_pixel)
	
	# Convert polygon vertices from geo to pixel coordinates
	pixel_polygon = [geo_to_pixel(lon, lat) for lon, lat in polygon_vertices]

	h, w = image.shape[:2]
	print(f"  Image size: {w}x{h}")
	print(f"  Polygon vertices: {len(pixel_polygon)}")

	geoglif_id = image_path.stem.replace('_ortho', '')

	for window_size in WINDOW_SIZES:
		print(f"  Testing window size: {window_size}px")

		if window_size > min(h, w):
			print("    Skipped (image too small)")
			continue

		for threshold in THRESHOLDS:
			crops = make_polygon_thresholds_crops(
				image, pixel_polygon, window_size, N_CROPS_PER_IMAGE, 
				stride=STRIDE, threshold=threshold
			)
			print(f"    Threshold {threshold:.1%}: {len(crops)} crops (requested {N_CROPS_PER_IMAGE})")

			stride_dir = f"stride_{stride_label(STRIDE)}"
			threshold_str = f"threshold_{int(threshold*100)}"

			for noise_type in NOISE_TYPES:
				for noise_level in NOISE_LEVELS:
					method_name = f"{noise_type}_noise{int(noise_level*100)}"

					crop_output_dir = (output_dir / area / f"window_{window_size}" / 
									   stride_dir / threshold_str / method_name)
					crop_output_dir.mkdir(parents=True, exist_ok=True)

					for idx, (crop, mask) in enumerate(crops):
						filled_crop = fill_with_noise(crop, mask, noise_level, noise_type, seed=42 + idx)

						output_filename = f"{geoglif_id}_crop_{idx:02d}.jpg"
						output_path = crop_output_dir / output_filename
						cv2.imwrite(str(output_path), filled_crop)


def main():
	"""Main test execution."""
	print("=" * 80)
	print("POLYGON THRESHOLD CROPS GENERATION TEST")
	print("=" * 80)
	print("\nConfiguration:")
	print(f"  Window sizes: {WINDOW_SIZES}")
	print(f"  Crops per image: {N_CROPS_PER_IMAGE}")
	print(f"  Thresholds: {THRESHOLDS}")
	print(f"  Noise types: {NOISE_TYPES}")
	print(f"  Noise levels: {NOISE_LEVELS}")
	print(f"  Stride: {stride_label(STRIDE)}")
	print(f"  Input directory: {TEST_DIR}")
	print(f"  Output directory: {OUTPUT_BASE}")

	OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

	# Find all .jpg files (ortho images)
	jpg_images = sorted(TEST_DIR.glob("*_ortho.jpg"))

	if not jpg_images:
		print("\nError: No ortho images found in test_geos directory!")
		return

	print(f"\nFound {len(jpg_images)} images to process")

	# Check which images have corresponding metadata in test_geos
	images_with_metadata = []
	for img_path in jpg_images:
		metadata_path = find_metadata_file(img_path)
		if metadata_path:
			images_with_metadata.append((img_path, metadata_path))
		else:
			print(f"Warning: No metadata found for {img_path.name}")

	if not images_with_metadata:
		print("Error: No images with metadata found!")
		return

	print(f"Found {len(images_with_metadata)} images with metadata")

	images_by_area = {'unita': [], 'lluta': [], 'chug': []}
	for img_path, meta_path in images_with_metadata:
		area = extract_area_from_filename(img_path.name)
		if area != 'unknown':
			images_by_area[area].append((img_path, meta_path))

	print("\nImages by area:")
	for area, imgs in images_by_area.items():
		print(f"  {area.upper()}: {len(imgs)} images")

	total_images = len(images_with_metadata)
	for idx, (image_path, metadata_path) in enumerate(images_with_metadata, 1):
		area = extract_area_from_filename(image_path.name)
		print(f"\n[{idx}/{total_images}] Processing {area.upper()} image")
		process_geoglif(image_path, metadata_path, area, OUTPUT_BASE)

	print("\n" + "=" * 80)
	print("SUMMARY")
	print("=" * 80)

	total_crops = 0
	for area in ['unita', 'lluta', 'chug']:
		area_dir = OUTPUT_BASE / area
		if area_dir.exists():
			crop_count = len(list(area_dir.rglob("*.jpg")))
			print(f"{area.upper()}: {crop_count} crops generated")
			total_crops += crop_count

	print(f"\nTotal crops generated: {total_crops}")
	print(f"Output directory: {OUTPUT_BASE}")
	print("\nDirectory structure:")
	print("  thresholds_crops_output/")
	print("    ├── unita/")
	print("    ├── lluta/")
	print("    └── chug/")
	print("\nTest completed!")


if __name__ == "__main__":
	main()
