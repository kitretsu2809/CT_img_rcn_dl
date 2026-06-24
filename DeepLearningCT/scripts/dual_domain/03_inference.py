#!/usr/bin/env python3
import argparse
import sys
import glob
from pathlib import Path

import numpy as np
import tifffile
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ct_recon.geometry import parse_geometry
from ct_recon.train_dual_domain import DualDomainNet
from ct_recon.reconstruct_fdk_astra import convert_to_attenuation

def main():
    parser = argparse.ArgumentParser(description="Run inference using the trained Dual-Domain Network on a new scan.")
    parser.add_argument("--input-dir", required=True, help="Directory containing settings.cto and projection .tif files")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained model checkpoint (.pt file)")
    parser.add_argument("--output-file", default="reconstructed_dl_volume.tif", help="Path to save the reconstructed output TIFF")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of slices to process at once")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    settings_path = input_dir / "settings.cto"
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.cto not found in {input_dir}")
        
    print(f"Loading geometry from: {settings_path}")
    geometry = parse_geometry(settings_path)

    # Find projection files
    tif_files = sorted(glob.glob(str(input_dir / "projections" / "*.tif")) + glob.glob(str(input_dir / "*.tif")))
    if not tif_files:
        raise FileNotFoundError(f"No .tif projection files found in {input_dir}")
        
    print(f"Loading {len(tif_files)} projection images...")
    projections = np.stack([tifffile.imread(f) for f in tif_files], axis=0).astype(np.float32)
    attenuation = convert_to_attenuation(projections)

    # Initialize Model dynamically!
    print("Initializing Dual-Domain Network with dynamically parsed geometry...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualDomainNet(geometry=geometry).to(device)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Determine dimensions
    num_angles, num_rows, num_cols = attenuation.shape
    recon_rows, recon_cols = geometry.recon_rows, geometry.recon_cols
    output_volume = np.zeros((num_rows, recon_rows, recon_cols), dtype=np.float32)

    # The model expects sinograms scaled roughly [0, 1]. We should read sinogram_scale if available.
    # We fallback to standard scaling if metadata isn't present in checkpoint
    metadata = checkpoint.get("metadata", {})
    sinogram_scale = float(metadata.get("sinogram_scale", np.percentile(attenuation, 99.5)))
    image_min = float(metadata.get("image_min", 0.0))
    image_scale = float(metadata.get("image_max", 1.0)) - image_min
    if image_scale == 0: image_scale = 1.0

    print("Running Inference slice-by-slice...")
    with torch.no_grad():
        for i in range(0, num_rows, args.batch_size):
            end_idx = min(i + args.batch_size, num_rows)
            batch_sino = attenuation[:, i:end_idx, :].transpose(1, 0, 2) # [batch, angles, cols]
            
            # Normalize and convert to tensor
            batch_sino = np.clip(batch_sino / sinogram_scale, 0.0, None)
            sino_tensor = torch.from_numpy(batch_sino).unsqueeze(1).to(device) # [batch, 1, angles, cols]
            
            # Forward pass
            final_image, _ = model(sino_tensor)
            
            # Denormalize output
            output_slice = final_image.squeeze(1).cpu().numpy()
            output_slice = (output_slice * image_scale) + image_min
            output_volume[i:end_idx, :, :] = output_slice
            
            print(f"Processed slices {i} to {end_idx-1} / {num_rows}")

    print(f"Saving reconstructed volume to: {args.output_file}")
    tifffile.imwrite(args.output_file, output_volume)
    print("Inference Complete!")

if __name__ == "__main__":
    main()
