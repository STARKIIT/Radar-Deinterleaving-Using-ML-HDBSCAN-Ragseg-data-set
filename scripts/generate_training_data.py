import numpy as np
import os
from pathlib import Path
import h5py

def generate_synthetic_emitter(n_pulses, cf_mean, cf_std, pw_mean, pw_std, pri_mean, pri_std):
    """Generate synthetic pulses for a single emitter."""
    # Generate carrier frequencies with some variance
    cf = np.random.normal(cf_mean, cf_std, n_pulses)
    
    # Generate pulse widths with some variance
    pw = np.random.normal(pw_mean, pw_std, n_pulses)
    pw = np.clip(pw, pw_mean * 0.5, pw_mean * 1.5)  # Ensure reasonable values
    
    # Generate time of arrivals based on PRI
    pri = np.random.normal(pri_mean, pri_std, n_pulses - 1)
    pri = np.clip(pri, pri_mean * 0.8, pri_mean * 1.2)  # Limit PRI jitter
    toa = np.zeros(n_pulses)
    toa[1:] = np.cumsum(pri)
    
    # Generate amplitudes (somewhat correlated with distance/TOA)
    amp = 1.0 / (1 + 0.01 * toa) + np.random.normal(0, 0.1, n_pulses)
    amp = np.clip(amp, 0.3, 1.0)
    
    # Generate AOA (fixed for a single emitter with small variation)
    aoa = np.random.normal(np.random.uniform(0, 360), 2.0, n_pulses)
    
    # Combine into PDW format [TOA, CF, PW, AOA, Amp]
    pdw_data = np.column_stack([toa, cf, pw, aoa, amp])
    
    return pdw_data

def main():
    """Generate synthetic PDW dataset with ground truth labels."""
    # Create output directory
    Path('data/pdw').mkdir(parents=True, exist_ok=True)
    
    # Define emitter characteristics (using realistic radar parameters)
    emitters = [
        # Format: [num_pulses, CF(MHz), CF_std, PW(μs), PW_std, PRI(ms), PRI_std]
        [100, 1200, 0.5, 1.0, 0.05, 1.0, 0.02],  # Surveillance radar
        [80, 2400, 1.0, 0.5, 0.02, 0.5, 0.01],   # Fire control radar
        [120, 5600, 2.0, 0.2, 0.01, 0.2, 0.005], # Weather radar
        [60, 9200, 5.0, 0.1, 0.01, 0.1, 0.002],  # Airborne radar
        [90, 3500, 1.5, 0.3, 0.02, 0.3, 0.01],   # Marine radar
    ]
    
    all_pdws = []
    all_labels = []
    
    # Generate data for each emitter
    for i, (n_pulses, cf, cf_std, pw, pw_std, pri, pri_std) in enumerate(emitters):
        print(f"Generating emitter {i+1} with {n_pulses} pulses")
        pdw_data = generate_synthetic_emitter(
            n_pulses, cf, cf_std, pw, pw_std, pri, pri_std
        )
        all_pdws.append(pdw_data)
        all_labels.append(np.full(n_pulses, i))
    
    # Combine all emitters
    pdw_data = np.vstack(all_pdws)
    labels = np.concatenate(all_labels)
    
    # Sort by TOA to interleave the pulses
    sort_idx = np.argsort(pdw_data[:, 0])
    pdw_data = pdw_data[sort_idx]
    labels = labels[sort_idx]
    
    print(f"Generated dataset with {len(pdw_data)} pulses from {len(emitters)} emitters")
    
    # Save the dataset and ground truth labels
    np.save('data/pdw/synthetic_pdws.npy', pdw_data)
    np.save('data/pdw/ground_truth_labels.npy', labels)
    
    # Also save as HDF5 for direct use in clustering pipeline
    with h5py.File('data/pdw/synthetic_dataset.hdf5', 'w') as f:
        f.create_dataset('pdw_data', data=pdw_data)
    
    # Also create training data for the transformer
    np.save('data/pdw/train_pdws.npy', pdw_data)
    np.save('data/pdw/train_labels.npy', labels)
    
    print("Saved files:")
    print("- data/pdw/synthetic_pdws.npy (PDW data)")
    print("- data/pdw/ground_truth_labels.npy (Truth labels)")
    print("- data/pdw/synthetic_dataset.hdf5 (HDF5 format)")
    print("- data/pdw/train_pdws.npy (For transformer training)")
    print("- data/pdw/train_labels.npy (For transformer training)")

if __name__ == "__main__":
    main()
