#!/usr/bin/env python3
"""
Master pipeline script for the Dual-Domain CT Reconstruction workflow.

Usage:
    # Run the state-of-the-art Dual-Domain pipeline on all datasets
    python scripts/run_pipeline.py dual-domain --sample all --epochs 50
    
    # Classical FDK reconstruction only
    python scripts/run_pipeline.py classical --sample sample_1
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Get repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_DIR))

from scripts.common.sample_config import get_sample_paths, get_sample_config


def run_command(cmd: list[str], description: str, check: bool = True) -> int:
    """Run a command with error handling."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        return result.returncode
    return 0


def run_classical(sample_name: str, skip_if_exists: bool = True, no_downsample: bool = False, downsample_factor_override: int | None = None):
    """Run classical FDK reconstruction."""
    paths = get_sample_paths(sample_name)
    config = get_sample_config(sample_name)
    
    fdk_volume = paths["fdk_volume"]
    
    if skip_if_exists and fdk_volume.exists():
        print(f"Classical reconstruction already exists: {fdk_volume}")
        return 0
    
    fdk_volume.parent.mkdir(parents=True, exist_ok=True)
    
    if no_downsample:
        ds = 1
        print("Running FULL RESOLUTION reconstruction (no downsampling)")
    elif downsample_factor_override:
        ds = downsample_factor_override
    else:
        ds = config["downsample_factor"]
    
    if sample_name == "sample_1":
        output_name = f"fdk_full_ds{ds}" if ds > 1 else "fdk_full"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "classical_reconstruction" / "reconstruct_fdk.py"),
             "--downsample", str(ds),
             "--output-dir", str(REPO_ROOT / "outputs" / output_name)],
            "Running FDK reconstruction for sample_1"
        )
        if result != 0:
            return result
        source = REPO_ROOT / "outputs" / output_name / "fdk_volume.tif"
        if source.exists():
            shutil.copy(source, fdk_volume)
    else:
        output_name = f"sample_2_fdk_ds{ds}" if ds > 1 else "sample_2_fdk_full"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "classical_reconstruction" / "reconstruct_fdk_sample2.py"),
             "--downsample", str(ds)],
            "Running FDK reconstruction for sample_2"
        )
        if result != 0:
            return result
        source = REPO_ROOT / "outputs" / output_name / "fdk_volume.tif"
        if source.exists():
            shutil.copy(source, fdk_volume)
            
    print(f"Classical reconstruction complete: {fdk_volume}")
    return 0


def run_dual_domain_pipeline(sample_name: str, epochs: int = 50, sparse_step: int = 1):
    """Run SOTA Dual-Domain Enhancement pipeline."""
    data_dir = REPO_ROOT / "data"
    
    if sample_name == "all":
        print(f"\n{'='*60}")
        print("Dual-Domain Pipeline (MULTI-DATASET AGGREGATION)")
        print(f"{'='*60}")
        
        sample_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        print(f"Discovered {len(sample_dirs)} physical datasets.")
        
        for s_dir in sample_dirs:
            s_name = s_dir.name
            print(f"\n--- Processing Dataset: {s_name} ---")
            run_classical(s_name)
            
        print("\n[Step 2/3] Building unified Dual-Domain training dataset...")
        dataset_path = REPO_ROOT / "outputs" / "dual_domain_dataset_all.npz"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "dual_domain" / "01_build_dataset.py"),
             "--output-path", str(dataset_path),
             "--sparse-step", str(sparse_step)],
            f"Building unified Dual-Domain dataset (sparse-step={sparse_step})"
        )
        if result != 0:
            return result
        
        checkpoint_dir = REPO_ROOT / "outputs" / "dual_domain_all_datasets"
        
    else:
        paths = get_sample_paths(sample_name)
        
        print(f"\n{'='*60}")
        print(f"Dual-Domain Pipeline for {sample_name}")
        print(f"{'='*60}")
        
        result = run_classical(sample_name)
        if result != 0:
            return result
            
        print("\n[Step 2/3] Building dataset (Sensor & Image Domains)...")
        dataset_path = REPO_ROOT / "outputs" / f"dual_domain_dataset_{sample_name}.npz"
        result = run_command(
            [sys.executable, str(SCRIPTS_DIR / "dual_domain" / "01_build_dataset.py"),
             "--sample-dir", str(paths["sample_dir"]),
             "--output-path", str(dataset_path),
             "--sparse-step", str(sparse_step)],
            f"Building Dual-Domain dataset (sparse-step={sparse_step})"
        )
        if result != 0:
            return result
            
        checkpoint_dir = REPO_ROOT / "outputs" / f"dual_domain_{sample_name}_model"

    print("\n[Step 3/3] Training Dual-Domain Network...")
    result = run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "dual_domain" / "02_train_model.py"),
            "--dataset-path", str(dataset_path),
            "--output-dir", str(checkpoint_dir),
            "--epochs", str(epochs),
            "--batch-size", "2",
            "--learning-rate", "1e-3",
        ],
        "Training Dual-Domain Model",
    )
    if result != 0:
        return result

    print(f"\n{'='*60}")
    print("Dual-Domain pipeline complete!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"{'='*60}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="CT Reconstruction Master Pipeline")
    parser.add_argument("mode", choices=["dual-domain", "classical"], help="Pipeline mode to run")
    parser.add_argument("--sample", default="sample_1", help="Sample directory name (or 'all' for multi-dataset training)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--sparse-step", type=int, default=1, help="Downsample factor for projections. 1 = All data (Artifact Reduction), >1 = Sparse View.")
    
    args = parser.parse_args()
    
    if args.mode == "classical":
        if args.sample == "all":
            data_dir = REPO_ROOT / "data"
            for s_dir in data_dir.iterdir():
                if s_dir.is_dir():
                    run_classical(s_dir.name)
        else:
            run_classical(args.sample)
    elif args.mode == "dual-domain":
        run_dual_domain_pipeline(args.sample, args.epochs, args.sparse_step)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
