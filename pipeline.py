from extract import get_polygon_imgs
from resize import resize_datapoint_image
from crop import generate_crops
from negatives import generate_negative_samples

from datamanager import DataWriter, get_sites, DataPoint
from parameters import DATA_DIR, DEFAULT_NEGATIVES_RATIO, DEFAULT_TARGET_SCALE, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_THRESHOLD_CROP_CONTENT, GEO_CLASS

def main():
    sites = get_sites(DATA_DIR)
    print(f"Found {len(sites)} sites in {DATA_DIR}: {', '.join(sites.keys())}")
    writer = DataWriter(sites)
    for site in sites.values():
        writer.start_site(site)
        polygon_boundaries = []
        
        print(f"Processing site: {site.name}")
        for geo_datapoint in get_polygon_imgs(site, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_TARGET_SCALE):
            polygon_boundaries.append(geo_datapoint.polygon_bounds)
            
            print(f"Processing polygon datapoint: {geo_datapoint.id}")
            resize_datapoint_image(geo_datapoint, DEFAULT_TARGET_SCALE, window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE)
            
            print(f"Resized polygon datapoint: {geo_datapoint.id} to scale {DEFAULT_TARGET_SCALE}.")
            for n, crop in enumerate(generate_crops(geo_datapoint, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, threshold=DEFAULT_THRESHOLD_CROP_CONTENT)):
                print(f"Processing crop N°{n} for {geo_datapoint.id}")
                writer.add_crop(geo_datapoint, crop, n)
        
        print(f"Generating negative samples for site: {site.name}")
        for n, negative_datapoint in enumerate(generate_negative_samples(site, num_positive_samples=writer.positive_count, positive_polygon_boundaries=tuple(polygon_boundaries), window_size=DEFAULT_WINDOW_SIZE, desired_scale=DEFAULT_TARGET_SCALE, negative_ratio=DEFAULT_NEGATIVES_RATIO)):
            
            print(f"Processing negative datapoint N°{n}/{writer.positive_count*DEFAULT_NEGATIVES_RATIO}: {negative_datapoint.id}")
            writer.add_crop(negative_datapoint)
        
        print(f"Finished processing site: {site.name}. Total positive samples: {writer.positive_count}, Total negative samples: {writer.negative_count}.")
        writer.close_site()
        
        print(f"Site {site.name} metadata saved.")
    
    print("Saving sites metadata...")
    writer.save_sites_metadata()  # Save the sites metadata to a CSV file
    print("All sites processed and metadata saved.")

if __name__ == "__main__":
    main()