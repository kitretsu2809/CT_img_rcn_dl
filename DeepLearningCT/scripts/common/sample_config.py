"""Universal sample configuration."""
from pathlib import Path

# Sample configurations with their specific parameters
SAMPLE_CONFIG = {
    "sample_1": {
        "downsample_factor": 2,
        "detector_size": 1000,  # Original projection size
        "z_range": (180, 1060),  # zmin, zmax from settings
    },
    "sample_2": {
        "downsample_factor": 4,
        "detector_size": 2850,
        "z_range": (280, 440),  # Only active slices (std > 0.001)
    },
}


def get_sample_config(sample_name: str) -> dict:
    """Get configuration for a sample."""
    if sample_name in SAMPLE_CONFIG:
        return SAMPLE_CONFIG[sample_name]
    
    try:
        from ct_recon.paths import SAMPLE_DIRS
        from ct_recon.data_loader import load_cto_settings
        
        if sample_name in SAMPLE_DIRS:
            settings_path = SAMPLE_DIRS[sample_name] / "settings.cto"
            settings = load_cto_settings(settings_path)
            
            # Default downsample to 2 if not defined
            downsample_factor = 2 
            
            # We need to find detector size, from rows/cols
            recon_settings = settings.get("CT reconstruction settings", {})
            detector_size = recon_settings.get("columns", 1000)
            zmin = recon_settings.get("zmin", 0)
            zmax = recon_settings.get("zmax", detector_size)
            
            # Add to config cache
            SAMPLE_CONFIG[sample_name] = {
                "downsample_factor": downsample_factor,
                "detector_size": detector_size,
                "z_range": (zmin, zmax),
            }
            return SAMPLE_CONFIG[sample_name]
    except ImportError:
        pass
        
    raise ValueError(f"Unknown sample: {sample_name}. Available statically: {list(SAMPLE_CONFIG.keys())}")


def get_sample_paths(sample_name: str, repo_root: Path | None = None) -> dict:
    """Get all paths for a sample."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    
    parent_data = repo_root.parent / "data"
    if (parent_data / sample_name).exists():
        sample_dir = parent_data / sample_name
        output_base = repo_root.parent / "outputs" / f"{sample_name}_pipeline"
    else:
        sample_dir = repo_root / "data" / sample_name
        output_base = repo_root / "outputs" / f"{sample_name}_pipeline"
        
    config = get_sample_config(sample_name)
    
    return {
        "repo_root": repo_root,
        "sample_dir": sample_dir,
        "sample_name": sample_name,
        "downsample_factor": config["downsample_factor"],
        # Classical reconstruction outputs
        "fdk_volume": output_base / "classical" / "fdk_volume.tif",
        # Sinogram reconstruction pipeline (sparse input)
        "sinogram_dataset": output_base / "sinogram_recon" / "dataset.npz",
        "sinogram_checkpoint": output_base / "sinogram_recon" / "training" / "best_model.pt",
        "sinogram_inference": output_base / "sinogram_recon" / "inference",
        # Enhance pipeline (full input)
        "enhance_dataset": output_base / "enhance" / "dataset.npz",
        "enhance_checkpoint": output_base / "enhance" / "training" / "best_model.pt",
        "enhance_inference": output_base / "enhance" / "inference",
        # U-Net enhancement pipeline  
        "unet_training_pairs": output_base / "unet_enhance" / "training_pairs",
        "unet_checkpoint": output_base / "unet_enhance" / "training" / "best_model.pt",
        "unet_inference": output_base / "unet_enhance" / "inference",
    }
