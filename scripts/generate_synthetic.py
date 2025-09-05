#!/usr/bin/env python3
"""
Generate synthetic PDW data for testing and development.
This script creates synthetic radar pulses with different emitter characteristics.

Usage:
    python scripts/generate_synthetic.py --output data/pdw/synthetic.hdf5 --emitters 5 --pulses 50
"""

import numpy as np
import argparse
import os
import h5py
from pathlib import Path
import matplotlib.pyplot as plt

def generate_synthetic_pulses(n_emitters=5, n_pulses_per_emitter=50, jitter=0.01, staggered=False):
    """
    Generate synthetic PDW data with configurable parameters.
    
    Args:
        n_emitters: Number of emitters to simulate
        n_pulses_per_emitter: Number of pulses per emitter
        jitter: Amount of jitter in parameters (0-1)
        staggered: Whether to generate staggered PRI patterns
    
    Returns:
        pdw_array: Array of shape (n_pulses, 5) with [TOA, CF, PW, AOA, Amp]
        labels: Array of shape (n_pulses,) with emitter labels
    """
    total_pulses = n_emitters * n_pulses_per_emitter
    
    # Initialize arrays
    pdw_data = []
    labels = []
    
    print(f"Generating {total_pulses} synthetic pulses for {n_emitters} emitters...")
    
    for emitter_id in range(n_emitters):
        # Each emitter has different carrier frequency and pulse width
        cf_base = 1000 * (emitter_id + 1)  # MHz
        pw_base = 0.1 * (emitter_id + 1)   # microseconds
        
        # Set up PRI patterns - regular or staggered
        if staggered and emitter_id % 2 == 1:
            # Create a staggered PRI pattern with 2-3 levels
            n_levels = np.random.randint(2, 4)
            pri_pattern = np.random.uniform(0.0005, 0.002, size=n_levels)
            print(f"Emitter {emitter_id}: Staggered PRI with {n_levels} levels: {pri_pattern}")
        else:
            # Regular PRI
            pri_base = 0.001 * (emitter_id + 1)
            pri_pattern = [pri_base]
            print(f"Emitter {emitter_id}: Regular PRI: {pri_base}")
        
        # Generate pulses for this emitter
        current_time = np.random.uniform(0, 0.1)  # Random start time
        for i in range(n_pulses_per_emitter):
            # Select PRI for this pulse
            pri_idx = i % len(pri_pattern)
            pri = pri_pattern[pri_idx]
            
            # Add jitter to TOA
            toa = current_time + np.random.normal(0, pri * jitter)
            current_time += pri
            
            # Add variation to CF and PW
            cf_actual = cf_base + np.random.normal(0, cf_base * jitter)
            pw_actual = pw_base + np.random.normal(0, pw_base * jitter)
            
            # Random AOA and amplitude
            aoa = np.random.uniform(0, 360)
            amp = np.random.uniform(0.5, 1.0)
            
            # Create PDW: [TOA, CF, PW, AOA, Amp]
            pdw_data.append([toa, cf_actual, pw_actual, aoa, amp])
            labels.append(emitter_id)
    
    # Sort by TOA to simulate mixed pulse streams
    combined = list(zip(pdw_data, labels))
    combined.sort(key=lambda x: x[0][0])  # Sort by TOA
    
    pdw_data = [item[0] for item in combined]
    labels = [item[1] for item in combined]
    
    return np.array(pdw_data, dtype=np.float32), np.array(labels, dtype=np.int32)

def visualize_synthetic_data(pdws, labels, output_path):
    """Create visualizations of the synthetic data"""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Pulse train visualization
    plt.figure(figsize=(12, 6))
    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    # Plot each emitter with different colors
    for i, emitter_id in enumerate(unique_labels):
        mask = labels == emitter_id
        emitter_pulses = pdws[mask]
        
        # Sort by TOA
        sorted_idx = np.argsort(emitter_pulses[:, 0])
        toa = emitter_pulses[sorted_idx, 0]
        
        # Plot pulse train as vertical lines
        for t in toa:
            plt.axvline(x=t, color=cmap(i), alpha=0.7, linewidth=1)
            
    plt.title(f"Synthetic Pulse Train - {len(unique_labels)} Emitters", fontsize=14)
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"{output_path}_pulse_train.png", dpi=300)
    plt.close()
    
    # 2. Feature scatter plot
    plt.figure(figsize=(10, 8))
    for i, emitter_id in enumerate(unique_labels):
        mask = labels == emitter_id
        plt.scatter(pdws[mask, 1], pdws[mask, 2], 
                   label=f"Emitter {emitter_id}", 
                   color=cmap(i),
                   alpha=0.7)
    
    plt.title("Synthetic PDW Feature Distribution", fontsize=14)
    plt.xlabel("Carrier Frequency (MHz)")
    plt.ylabel("Pulse Width (µs)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}_features.png", dpi=300)
    plt.close()
    
    # 3. PRI histograms
    plt.figure(figsize=(12, 8))
    for i, emitter_id in enumerate(unique_labels):
        plt.subplot(2, (len(unique_labels)+1)//2, i+1)
        
        mask = labels == emitter_id
        emitter_pulses = pdws[mask]
        
        # Sort by TOA and calculate PRI
        sorted_idx = np.argsort(emitter_pulses[:, 0])
        toa = emitter_pulses[sorted_idx, 0]
        pri = np.diff(toa) * 1000  # convert to ms
        
        plt.hist(pri, bins=20, alpha=0.7, color=cmap(i))
        plt.title(f"Emitter {emitter_id} PRI")
        plt.xlabel("PRI (ms)")
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_pri_histograms.png", dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {output_path}_*.png")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PDW data")
    parser.add_argument("--output", default="data/pdw/synthetic", help="Output file prefix")
    parser.add_argument("--emitters", type=int, default=5, help="Number of emitters")
    parser.add_argument("--pulses", type=int, default=50, help="Pulses per emitter")
    parser.add_argument("--jitter", type=float, default=0.01, help="Parameter jitter (0-1)")
    parser.add_argument("--staggered", action="store_true", help="Include staggered PRI patterns")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--format", default="hdf5", choices=["hdf5", "npy"], help="Output format")
    args = parser.parse_args()
    
    # Generate synthetic data
    pdw_array, label_array = generate_synthetic_pulses(
        n_emitters=args.emitters,
        n_pulses_per_emitter=args.pulses,
        jitter=args.jitter,
        staggered=args.staggered
    )
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save data
    if args.format == "hdf5":
        with h5py.File(f"{args.output}.hdf5", 'w') as f:
            f.create_dataset('pdw_data', data=pdw_array)
            f.create_dataset('labels', data=label_array)
        print(f"Saved {len(pdw_array)} synthetic pulses to {args.output}.hdf5")
    else:
        np.save(f"{args.output}_pdws.npy", pdw_array)
        np.save(f"{args.output}_labels.npy", label_array)
        print(f"Saved {len(pdw_array)} synthetic pulses to {args.output}_pdws.npy")
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_synthetic_data(pdw_array, label_array, args.output)
    
    print(f"Generated {len(pdw_array)} synthetic pulses for {args.emitters} emitters")
    print("Statistics:")
    for i in range(args.emitters):
        mask = label_array == i
        print(f"  Emitter {i}: {np.sum(mask)} pulses")

if __name__ == "__main__":
    main()
