import numpy as np
import h5py
import argparse
import os
import pandas as pd
from pathlib import Path

def main():
    """Extract training data from radar dataset using segment IDs with known emitter counts."""
    parser = argparse.ArgumentParser(description="Extract training data from radar segments with known emitter counts")
    parser.add_argument("--iq", required=True, help="Path to IQ data file")
    parser.add_argument("--labels", required=True, help="Path to radar_labels.csv with emitter counts")
    parser.add_argument("--output", default="data/pdw/", help="Output directory for training data")
    parser.add_argument("--segment", type=int, required=True, help="Segment ID to use for training")
    args = parser.parse_args()

    # Read radar labels to get segment info
    labels_df = pd.read_csv(args.labels)
    segment_info = labels_df[labels_df['SegmentID'] == args.segment]
    
    if len(segment_info) == 0:
        print(f"Error: Segment ID {args.segment} not found in labels file")
        return
    
    num_emitters = segment_info.iloc[0]['NumActiveEmitters']
    print(f"Using segment {args.segment} with {num_emitters} known emitters for training data")
    
    # Process this segment to extract PDW data
    print(f"Extracting PDWs from IQ file: {args.iq}")
    os.system(f"python scripts/preprocess.py --iq {args.iq} --out {args.output}/train_segment.hdf5 --segment {args.segment}")
    
    # Run clustering with high confidence settings to get clean labels
    print("Running clustering to get initial labels...")
    os.makedirs(f"{args.output}/temp", exist_ok=True)
    cluster_cmd = f"python scripts/clustering.py --pdw {args.output}/train_segment.hdf5 --output {args.output}/temp"
    cluster_cmd += f" --min_cluster_size 10 --cluster_selection_epsilon 0.05 --max_clusters {num_emitters}"
    os.system(cluster_cmd)
    
    # Load the clustered data and labels
    with h5py.File(f"{args.output}/train_segment.hdf5", 'r') as f:
        pdw_data = f['pdw_data'][:]
    
    labels = np.load(f"{args.output}/temp/cluster_labels.npy")
    
    # Filter out noise points
    mask = labels != -1
    clean_pdw = pdw_data[mask]
    clean_labels = labels[mask]
    
    # Save as training data
    os.makedirs(args.output, exist_ok=True)
    np.save(f"{args.output}/train_pdws.npy", clean_pdw)
    np.save(f"{args.output}/train_labels.npy", clean_labels)
    
    print(f"Saved {len(clean_pdw)} pulses with {len(np.unique(clean_labels))} emitter labels")
    print(f"Training data saved to: {args.output}/train_pdws.npy and {args.output}/train_labels.npy")

if __name__ == "__main__":
    main()
