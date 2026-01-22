"""
Test script for generating fixed (deterministic) crops from geoglifs in test_geos.
Uses make_fixed_crops instead of make_random_crops and keeps the same noise tests.
Organizes output by area, window size, stride, and noise method.
"""

import cv2
import numpy as np
from pathlib import Path
from format import make_fixed_crops, fill_with_noise, TEST_DIR

# Configuration
WINDOW_SIZES = [122, 244, 488]
N_CROPS_PER_IMAGE = 10
NOISE_LEVELS = [0.0, 0.5, 1.0]  # 0=avg color, 0.5=blend, 1.0=full noise
NOISE_TYPES = ['gaussian', 'uniform']

# Set to None to auto-compute stride based on image size and n_crops
# or provide an int or (x, y) tuple for explicit stride.
STRIDE = None

OUTPUT_BASE = Path(__file__).resolve().parent / "crops_test/fixed_crops_output"


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


def process_geoglif(image_path, area, output_dir):
	"""Process a single geoglif image with multiple crop configurations."""
	print(f"\nProcessing: {image_path.name}")

	image = cv2.imread(str(image_path))
	if image is None:
		print(f"  Error: Could not load {image_path}")
		return

	h, w = image.shape[:2]
	print(f"  Image size: {w}x{h}")

	geoglif_id = image_path.stem.replace('_ortho', '')

	for window_size in WINDOW_SIZES:
		print(f"  Testing window size: {window_size}px")

		if window_size > min(h, w):
			print("    Skipped (image too small)")
			continue

		crops = make_fixed_crops(image, window_size, N_CROPS_PER_IMAGE, stride=STRIDE)
		print(f"    Generated {len(crops)} crops (requested {N_CROPS_PER_IMAGE})")

		stride_dir = f"stride_{stride_label(STRIDE)}"

		for noise_type in NOISE_TYPES:
			for noise_level in NOISE_LEVELS:
				method_name = f"{noise_type}_noise{int(noise_level*100)}"

				crop_output_dir = output_dir / area / f"window_{window_size}" / stride_dir / method_name
				crop_output_dir.mkdir(parents=True, exist_ok=True)

				for idx, (crop, mask) in enumerate(crops):
					filled_crop = fill_with_noise(crop, mask, noise_level, noise_type, seed=42 + idx)

					output_filename = f"{geoglif_id}_crop_{idx:02d}.jpg"
					output_path = crop_output_dir / output_filename
					cv2.imwrite(str(output_path), filled_crop)


def main():
	"""Main test execution."""
	print("=" * 80)
	print("FIXED CROPS GENERATION TEST")
	print("=" * 80)
	print("\nConfiguration:")
	print(f"  Window sizes: {WINDOW_SIZES}")
	print(f"  Crops per image: {N_CROPS_PER_IMAGE}")
	print(f"  Noise types: {NOISE_TYPES}")
	print(f"  Noise levels: {NOISE_LEVELS}")
	print(f"  Stride: {stride_label(STRIDE)}")
	print(f"  Input directory: {TEST_DIR}")
	print(f"  Output directory: {OUTPUT_BASE}")

	OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

	ortho_images = sorted(TEST_DIR.glob("*_ortho.jpg"))

	if not ortho_images:
		print("\nError: No ortho images found in test_geos directory!")
		return

	print(f"\nFound {len(ortho_images)} images to process")

	images_by_area = {'unita': [], 'lluta': [], 'chug': []}
	for img_path in ortho_images:
		area = extract_area_from_filename(img_path.name)
		if area != 'unknown':
			images_by_area[area].append(img_path)

	print("\nImages by area:")
	for area, imgs in images_by_area.items():
		print(f"  {area.upper()}: {len(imgs)} images")

	total_images = len(ortho_images)
	for idx, image_path in enumerate(ortho_images, 1):
		area = extract_area_from_filename(image_path.name)
		print(f"\n[{idx}/{total_images}] Processing {area.upper()} image")
		process_geoglif(image_path, area, OUTPUT_BASE)

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
	print("  fixed_crops_output/")
	print("    ├── unita/")
	print("    ├── lluta/")
	print("    └── chug/")
	print("\nTest completed!")


if __name__ == "__main__":
	main()
