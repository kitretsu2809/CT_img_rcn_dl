import argparse
import json
import math
import random
from pathlib import Path
import numpy as np

from ct_recon.geometry import CTGeometry

# A skeletal representation of a State-of-the-Art Dual-Domain CT Reconstruction Network.
# In a full clinical deployment, the `DifferentiableBackprojection` layer would be powered 
# by a compiled CUDA extension (like astra.pytorch or torch-radon).

def _import_torch_or_exit():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ModuleNotFoundError:
        raise SystemExit("Missing PyTorch in this environment.")
    return torch, nn, Dataset, DataLoader

torch, nn, Dataset, DataLoader = _import_torch_or_exit()

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class SinogramNet(nn.Module):
    """
    Subnetwork operating in the Sensor (Sinogram) Domain.
    Cleans up noise and repairs missing angular projections before they cause streak artifacts.
    """
    def __init__(self, channels=32):
        super().__init__()
        self.inc = DoubleConv(1, channels)
        self.down = DoubleConv(channels, channels * 2)
        self.up = DoubleConv(channels * 2, channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down(nn.functional.max_pool2d(x1, 2))
        x_up = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.up(x_up + x1) # Skip connection
        return self.outc(x3) + x # Residual learning

class DifferentiableBackprojection(nn.Module):
    """
    Differentiable Radon transform (FBP).
    When running on the supercomputer with CUDA, this uses `torch_radon` to 
    mathematically project the 2D sinogram into a 2D image grid, allowing 
    gradients to flow backwards from the Image Domain to the Sinogram Domain.
    """
    def __init__(self, geometry: CTGeometry):
        super().__init__()
        self.image_size = geometry.recon_rows
        self.radon = None
        
        try:
            import torch_radon
            # Dynamically initialize using the specific scanner geometry from settings.cto
            angles_rad = geometry.angles_rad
            
            # Use RadonFanbeam for cone/fan beams if distances are provided, otherwise parallel
            if geometry.source_to_object_mm > 0 and geometry.source_to_detector_mm > 0:
                self.radon = torch_radon.RadonFanbeam(
                    geometry.recon_rows, 
                    angles_rad,
                    source_distance=geometry.source_to_object_mm,
                    det_distance=geometry.source_to_detector_mm,
                    det_spacing=geometry.detector_pixel_size_mm
                )
            else:
                self.radon = torch_radon.Radon(geometry.recon_rows, angles_rad)
                
            print("Successfully loaded dynamic torch-radon CUDA backend using settings.cto geometry.")
        except ImportError:
            print("WARNING: torch-radon not found. Using dummy placeholder backprojection.")

    def forward(self, sinogram):
        batch_size = sinogram.shape[0]
        
        if self.radon is not None:
            # The actual mathematically correct, differentiable conversion
            # Filters the sinogram and performs backprojection
            filtered_sinogram = self.radon.filter_sinogram(sinogram)
            reconstructed_image = self.radon.backprojection(filtered_sinogram)
            return reconstructed_image
        else:
            # Dummy placeholder for local testing without CUDA
            return torch.zeros((batch_size, 1, self.image_size, self.image_size), device=sinogram.device)

class ImageNet(nn.Module):
    """
    Subnetwork operating in the Image Domain.
    Refines the FBP-reconstructed image to recover high-frequency textures and structures.
    """
    def __init__(self, channels=64):
        super().__init__()
        self.inc = DoubleConv(1, channels)
        self.down = DoubleConv(channels, channels * 2)
        self.up = DoubleConv(channels * 2, channels)
        self.outc = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down(nn.functional.max_pool2d(x1, 2))
        x_up = nn.functional.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = self.up(x_up + x1)
        return self.outc(x3) + x # Residual learning

class DualDomainNet(nn.Module):
    """
    State-of-the-Art Dual-Domain CT Reconstruction Architecture.
    Processes data systematically across both the sensor and image domains.
    """
    def __init__(self, geometry: CTGeometry):
        super().__init__()
        self.sino_net = SinogramNet()
        self.fbp_layer = DifferentiableBackprojection(geometry)
        self.image_net = ImageNet()

    def forward(self, noisy_sinogram):
        # 1. Denoise in Sensor Domain
        clean_sinogram = self.sino_net(noisy_sinogram)
        
        # 2. Differentiable Domain Transform
        fbp_image = self.fbp_layer(clean_sinogram)
        
        # 3. Refine in Image Domain
        final_image = self.image_net(fbp_image)
        
        return final_image, clean_sinogram

def main():
    print("Dual-Domain Network Architecture Loaded.")
    print("This architecture represents the State-of-the-Art for CT Reconstruction.")
    print("To train end-to-end, ensure the data loader supplies both Sinogram and Target Image pairs.")
    print("Dual-Domain Network requires a CTGeometry object to initialize.")

if __name__ == "__main__":
    main()
