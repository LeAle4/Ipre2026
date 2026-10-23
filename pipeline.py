import json
from pathlib import Path

from extract import get_polygon_imgs
from resize import resize_datapoint_image
from crop import generate_crops
from negatives import generate_negative_samples

from datamanager import DataWriter, get_sites


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "pipeline_config.json"
REQUIRED_CONFIG_KEYS = {
    "data_dir",
    "target_scale",
    "window_size",
    "stride",
    "threshold_crop_content",
    "negatives_ratio",
}


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    missing_keys = REQUIRED_CONFIG_KEYS - config.keys()
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing configuration keys in {CONFIG_PATH}: {missing}")

    config["data_dir"] = ROOT_DIR / config["data_dir"]
    return config

def main():
    config = load_config()
    data_dir = config["data_dir"]
    target_scale = config["target_scale"]
    window_size = config["window_size"]
    stride = config["stride"]
    threshold_crop_content = config["threshold_crop_content"]
    negatives_ratio = config["negatives_ratio"]

    sites = get_sites(data_dir)
    print(f"Found {len(sites)} sites in {data_dir}: {', '.join(sites.keys())}")
    writer = DataWriter(sites)
    for site in sites.values():
        writer.start_site(site)
        polygon_boundaries = []
        
        print(f"Processing site: {site.name}")
        for geo_datapoint in get_polygon_imgs(site, window_size, stride, target_scale):
            polygon_boundaries.append(geo_datapoint.polygon_bounds)
            writer.add_polygon_datapoint(geo_datapoint)
            
            print(f"Processing polygon datapoint: {geo_datapoint.id}")
            resize_datapoint_image(geo_datapoint, target_scale, window_size=window_size, stride=stride)
            
            print(f"Resized polygon datapoint: {geo_datapoint.id} to scale {target_scale}.")
            for n, crop in enumerate(generate_crops(geo_datapoint, window_size, stride, threshold=threshold_crop_content)):
                print(f"Processing crop N°{n} for {geo_datapoint.id}")
                writer.add_crop(geo_datapoint, crop, n)
        
        print(f"Generating negative samples for site: {site.name}")
        for n, negative_datapoint in enumerate(generate_negative_samples(site, num_positive_samples=writer.positive_count, positive_polygon_boundaries=tuple(polygon_boundaries), window_size=window_size, desired_scale=target_scale, negative_ratio=negatives_ratio)):
            
            print(f"Processing negative datapoint N°{n}/{writer.positive_count*negatives_ratio}: {negative_datapoint.id}")
            writer.add_crop(negative_datapoint)
        
        print(f"Finished processing site: {site.name}. Total positive samples: {writer.positive_count}, Total negative samples: {writer.negative_count}.")
        writer.close_site()
        
        print(f"Site {site.name} metadata saved.")
    
    print("Saving sites metadata...")
    writer.save_sites_metadata()  # Save the sites metadata to a CSV file
    print("All sites processed and metadata saved.")

if __name__ == "__main__":
    main()