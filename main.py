#!/usr/bin/env python3
"""
Main execution script for radar pulse deinterleaving.

This script provides a complete pipeline from raw IQ data to deinterleaved pulse trains.

Usage:
    python main.py --iq data/raw_iq/your_dataset.iq --output results/
"""

import argparse
import os
import sys
import time
from pathlib import Path

def run_preprocessing(iq_file, fs, output_file, format="hdf5", normalize=True, limit=None):
    """Run PDW extraction preprocessing."""
    print("\n" + "="*50)
    print("STEP 1: PDW EXTRACTION")
    print("="*50)
    
    cmd = f"python scripts/preprocess.py --iq {iq_file} --fs {fs} --out {output_file} --format {format}"
    if normalize:
        cmd += " --normalize"  # Ensure normalize flag is passed only if supported
    if limit:
        cmd += f" --limit {limit}"  # Add limit argument to the command
    
    print(f"Command: {cmd}")
    ret = os.system(cmd)
    if ret != 0 or not Path(output_file).exists():
        print(f"ERROR: Preprocessing failed or output file {output_file} not created!")
        sys.exit(1)
    return ret

def run_clustering(pdw_file, output_dir, visualize=True, adaptive_params=False, max_clusters=25, reduce_dim=False):
    """Run adaptive clustering."""
    print("\n" + "="*50)
    print("STEP 2: ADAPTIVE CLUSTERING")
    print("="*50)
    
    cmd = f"python scripts/clustering.py --pdw {pdw_file} --output {output_dir}"
    if visualize:
        cmd += " --visualize"
    if adaptive_params:
        cmd += " --adaptive_params"
    if max_clusters:
        cmd += f" --max_clusters {max_clusters}"
    if reduce_dim:
        cmd += " --reduce_dim"
    
    print(f"Command: {cmd}")
    ret = os.system(cmd)
    clustering_output = Path(output_dir) / 'cluster_labels.npy'
    if ret != 0 or not clustering_output.exists():
        print(f"ERROR: Clustering failed or output file {clustering_output} not created!")
        sys.exit(1)
    return ret

def run_evaluation(labels_file, pdw_file, output_dir, plot=True, advanced=False):
    print("\n" + "="*50)
    print("STEP 3: EVALUATION")
    print("="*50)
    cmd = f"python scripts/evaluate.py --labels {labels_file} --pdw {pdw_file} --output {output_dir}"
    if plot:
        cmd += " --plot"
    if advanced:
        cmd += " --advanced"
    print(f"Command: {cmd}")
    ret = os.system(cmd)
    evaluation_report = Path(output_dir) / 'evaluation_report.txt'
    if ret != 0 or not evaluation_report.exists():
        print(f"ERROR: Evaluation failed or report file {evaluation_report} not created!")
        sys.exit(1)
    return ret

def main():
    parser = argparse.ArgumentParser(description='Complete radar pulse deinterleaving pipeline')
    parser.add_argument('--iq', required=True, help='Path to IQ data file')
    parser.add_argument('--fs', type=float, default=20e6, help='Sampling frequency (Hz)')
    parser.add_argument('--output', default='results/', help='Output directory')
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing step')
    parser.add_argument('--skip_evaluation', action='store_true', help='Skip evaluation step')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of data points')
    parser.add_argument('--adaptive_params', action='store_true', help='Automatically adjust clustering parameters')
    parser.add_argument('--max_clusters', type=int, default=25, help='Maximum number of clusters')
    parser.add_argument('--reduce_dim', action='store_true', help='Apply dimensionality reduction before clustering')
    parser.add_argument('--advanced', action='store_true', help='Enable advanced radar mode analyses')  # <-- Add this line
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdw_file = Path('data/pdw') / 'dataset.hdf5'  # Changed to use Path consistently and updated format
    (output_dir / 'clustering').mkdir(exist_ok=True)
    (output_dir / 'evaluation').mkdir(exist_ok=True)
    
    # Create data/pdw directory if it doesn't exist
    Path('data/pdw').mkdir(parents=True, exist_ok=True)
    
    # File paths
    pdw_file = Path('data/pdw/dataset.hdf5')  # Changed to use data/pdw and updated format
    labels_file = output_dir / 'clustering' / 'cluster_labels.npy'  # Changed from .hdf5 to .npy
    
    start_time = time.time()
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        print(f"Processing IQ file: {args.iq}")
        if args.limit:
            print(f"Limiting to the first {args.limit} raw signals.")
        print(f"Sampling frequency: {args.fs} Hz")
        ret = run_preprocessing(args.iq, args.fs, str(pdw_file), format="hdf5", normalize=True, limit=args.limit)
        if ret != 0 or not pdw_file.exists():
            print("ERROR: Preprocessing failed or PDW file not created.")
            sys.exit(1)
    else:
        print("Skipping preprocessing step")
    
    # Step 2: Clustering
    print(f"\nClustering PDW file: {pdw_file}")
    ret = run_clustering(str(pdw_file), str(output_dir / 'clustering'), 
                         visualize=True, adaptive_params=args.adaptive_params, 
                         max_clusters=args.max_clusters, reduce_dim=args.reduce_dim)
    if ret != 0 or not labels_file.exists():
        print("ERROR: Clustering failed or cluster label file not created.")
        sys.exit(1)
    
    # Step 3: Evaluation (remove ckpt parameter which is no longer needed)
    if not args.skip_evaluation:
        print(f"\nEvaluating results: {labels_file}")
        ret = run_evaluation(str(labels_file), str(pdw_file), str(output_dir / 'evaluation'), plot=True, advanced=args.advanced)
    else:
        print("Skipping evaluation step")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("PIPELINE COMPLETED!")
    print("="*50)
    print(f"Total processing time: {total_time:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print("\nOutput files:")
    print(f"  - PDW data: {pdw_file}")
    print(f"  - Cluster labels: {labels_file}")
    print(f"  - Visualizations: {output_dir}/clustering/")
    print(f"  - Evaluation: {output_dir}/evaluation/")
    
    print("\nNext steps:")
    print("1. Review clustering visualization plots")
    print("2. Check evaluation metrics in evaluation_report.txt")
    print("3. Analyze individual emitter characteristics")
    print("4. Verify the existence and integrity of output files:")
    print("   - PDW data file: Ensure it is not empty and correctly formatted.")
    print("   - Cluster labels file: Confirm it exists and contains valid data.")
    print("5. If any step failed, check logs and rerun the corresponding step with adjusted parameters.")
    print("6. Ensure sufficient disk space and permissions for output directories.")
    
    print("\nPipeline status: All steps completed successfully.")

if __name__ == "__main__":
    main()