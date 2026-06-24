#!/usr/bin/env python3
"""
Nikon XT H 225 ST Converter
Converts Nikon `.xtekct` or `.ctdata` (INI-style) metadata files into the `settings.cto` format
required by the Dual-Domain Deep Learning CT Reconstruction pipeline.
"""

import argparse
import configparser
import math
from pathlib import Path

def convert_nikon_to_cto(nikon_path: str | Path, output_dir: str | Path):
    nikon_path = Path(nikon_path)
    output_dir = Path(output_dir)
    
    if not nikon_path.exists():
        raise FileNotFoundError(f"Could not find Nikon file: {nikon_path}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    output_cto_path = output_dir / "settings.cto"
    
    # Nikon .xtekct files are typically INI format. We can use configparser.
    # We use strict=False because some Nikon files have duplicate keys.
    parser = configparser.ConfigParser(strict=False)
    
    try:
        parser.read(nikon_path)
    except Exception as e:
        print(f"Failed to parse {nikon_path} as standard INI. Attempting fallback parse... ({e})")
        # Fallback manual parse if the file is poorly formatted
        fallback_dict = {}
        with open(nikon_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if '=' in line:
                    key, val = line.split('=', 1)
                    fallback_dict[key.strip()] = val.strip()
        parser["XTekCT"] = fallback_dict

    # Extract Nikon specific keys (fallback to defaults if missing)
    try:
        section = parser["XTekCT"]
    except KeyError:
        # If no explicit XTekCT section, try grabbing from DEFAULT or building a flat dict
        section = {k: v for sect in parser.sections() for k, v in parser.items(sect)}

    def get_val(key, default=0.0):
        return float(section.get(key, default))

    # --- Mapping Nikon parameters to .cto ---
    # Nikon typical parameters:
    # SrcToObject, SrcToDetector, Projections, DetectorPixelsX, DetectorPixelsY, DetectorPixelSizeX
    
    sod = get_val("SrcToObject", 160.0)
    sdd = get_val("SrcToDetector", 200.0)
    projections = int(get_val("Projections", 360))
    
    cols = int(get_val("DetectorPixelsX", 1000))
    rows = int(get_val("DetectorPixelsY", 1000))
    pixel_size = get_val("DetectorPixelSizeX", 0.048) # assuming square pixels
    
    # kV and mA (Nikon usually provides uA)
    kv = get_val("X-ray kV", 35.0)
    ua = get_val("X-ray uA", 1000.0)
    ma = ua / 1000.0

    # Build the settings.cto structure
    class CaseSensitiveConfigParser(configparser.ConfigParser):
        def optionxform(self, optionstr: str) -> str:
            return optionstr
            
    cto_config = CaseSensitiveConfigParser()

    cto_config["Device settings"] = {
        "mA": f"{ma:.6f}",
        "kV": f"{kv:.6f}"
    }

    cto_config["Detector settings"] = {
        "binning": "0",
        "frames": "1",
        "exp time (ms)": "500.000000",
        "CORunbinned": f"{cols / 2.0:.6f}",
        "pixel size": f"{pixel_size:.6f}",
        "Xmin": "0",
        "Xmax": str(cols),
        "Ymin": "0",
        "Ymax": str(rows),
        "VC": f"{rows / 2.0:.6f}"
    }

    cto_config["CT scan settings"] = {
        "projections": str(projections),
        "angle range": "360.000000",
        "CWCCW": "FALSE",
        "SOD": f"{sod:.6f}",
        "SDD": f"{sdd:.6f}"
    }

    cto_config["CT reconstruction settings"] = {
        "SOD": f"{sod:.6f}",
        "SDD": f"{sdd:.6f}",
        "COR": f"{cols / 2.0:.6f}", # Center of Rotation
        "vertical center": f"{rows / 2.0:.6f}",
        "last angle": "360.000000",
        "bhc": "0.000000",
        "filter strength": "0.000000",
        "projections": str(projections),
        "rows": str(rows),
        "columns": str(cols),
        "pixel_size (mm)": f"{pixel_size:.6f}",
        "zmax": str(rows),
        "zmin": "0",
        "direction": "1",
        "tilt": "0.000000",
        "xminrec": "0",
        "xmaxrec": str(cols),
        "yminrec": "0",
        "ymaxrec": str(rows),
        "interpolate": "FALSE"
    }

    with open(output_cto_path, 'w') as f:
        # CTO files typically don't have spaces around the '=' for section headers but do for keys.
        # configparser writes spaces around '=' by default.
        cto_config.write(f)

    print(f"\nSuccessfully extracted Nikon geometry from: {nikon_path.name}")
    print(f"Generated CTO file at: {output_cto_path}")
    print("\nExtracted Details:")
    print(f" - SOD: {sod} mm")
    print(f" - SDD: {sdd} mm")
    print(f" - Projections: {projections}")
    print(f" - Detector: {cols}x{rows} (Pixel Size: {pixel_size} mm)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Nikon .xtekct/.ctdata to settings.cto")
    parser.add_argument("input_file", help="Path to the Nikon .xtekct or .ctdata file")
    parser.add_argument("--output-dir", default=".", help="Directory to save the settings.cto file")
    
    args = parser.parse_args()
    convert_nikon_to_cto(args.input_file, args.output_dir)
