#!/usr/bin/env python3
import os
import glob
from pathlib import Path
from usaf_projection_script import run_local_projection_pipeline

def generate_datasets():
    # Central output data directory (Dynamic for portability)
    script_dir = Path(__file__).resolve().parent
    output_base_dir = script_dir.parent / "data"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    output_base_dir_str = str(output_base_dir)
    
    # Locate all STLs
    stl_files = glob.glob("STL/*.stl")
    if not stl_files:
        print("No STL files found in the STL/ directory.")
        return
        
    print(f"Found {len(stl_files)} STL files: {stl_files}")
    
    # Define our dataset variations
    variations = [
        {
            "suffix": "standard",
            "add_noise": True,
            "i0": 50000.0,
            "gaussian_std": 10.0,
            "downsample": 2
        },
        {
            "suffix": "fast_scan_high_noise",
            "add_noise": True,
            "i0": 10000.0,
            "gaussian_std": 20.0,
            "downsample": 2
        },
        {
            "suffix": "perfect_no_noise",
            "add_noise": False,
            "i0": 50000.0, # ignored when no_noise=True
            "gaussian_std": 0.0,
            "downsample": 2
        },
        {
            "suffix": "downsample_4x",
            "add_noise": True,
            "i0": 50000.0,
            "gaussian_std": 10.0,
            "downsample": 4
        }
        # Add hundreds more variations here as needed!
    ]
    
    for stl_path in stl_files:
        stl_name = Path(stl_path).stem
        print(f"\n{'='*50}\nProcessing STL: {stl_name}\n{'='*50}")
        
        for var in variations:
            dataset_name = f"{stl_name}_{var['suffix']}"
            dataset_out_dir = os.path.join(output_base_dir_str, dataset_name)
            zip_out_name = f"{dataset_name}.zip"
            
            print(f"  -> Generating variation: {var['suffix']}")
            
            # Skip if already generated
            if os.path.exists(dataset_out_dir) and os.path.exists(os.path.join(dataset_out_dir, "settings.cto")):
                print(f"     Already exists at {dataset_out_dir}, skipping.")
                continue
                
            try:
                run_local_projection_pipeline(
                    stl_filepath=stl_path,
                    output_dir=dataset_out_dir,
                    supersample_pitch=0.025,
                    downsample_factor=var["downsample"],
                    add_noise=var["add_noise"],
                    i0=var["i0"],
                    gaussian_std=var["gaussian_std"]
                )
            except Exception as e:
                print(f"     Error generating {dataset_name}: {e}")

if __name__ == "__main__":
    generate_datasets()
