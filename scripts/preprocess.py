"""
Convert a raw complex-64 IQ binary file to a PDW numpy array.
"""
import numpy as np
import argparse
import h5py
import os
import sys
import time
from tqdm import tqdm  # For progress bars

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import pdw_io as pio

def decode_iq(path, fs, normalize=False, limit=None, show_progress=True):
    """
    Decode IQ data and extract PDW features.
    
    Args:
        path: Path to IQ data file
        fs: Sampling frequency in Hz
        normalize: Whether to normalize signal magnitude
        limit: Maximum number of signals to process
        show_progress: Whether to display progress bars
        
    Returns:
        Array of PDWs with shape (n_pulses, 5)
    """
    try:
        with h5py.File(path, "r") as f:
            # Robust key detection
            keys = list(f.keys())
            if "signals" in keys:
                signals = f["signals"][:limit] if limit else f["signals"][:]
            else:
                print(f"ERROR: No 'signals' key found in {path}. Available keys: {keys}")
                return np.zeros((0, 5), dtype=np.float32)
            print(f"Loaded signals shape: {signals.shape}")
            if signals.size == 0:
                print("WARNING: No signals found in file.")
                return np.zeros((0, 5), dtype=np.float32)
    except Exception as e:
        print(f"ERROR: Failed to load IQ file {path}: {e}")
        return np.zeros((0, 5), dtype=np.float32)

    start_time = time.time()
    pulses = []
    
    # Create progress bar for signal processing
    signal_iterator = tqdm(enumerate(signals), total=len(signals), 
                          desc="Processing signals", 
                          disable=not show_progress)
    
    for i, iq in signal_iterator:
        # Skip updating description for every signal to improve performance
        if i % 10 == 0:
            signal_iterator.set_description(f"Processing signal {i+1}/{len(signals)}")
        
        mag = np.abs(iq)
        if np.max(mag) == 0:
            continue  # Skip empty signals

        mag_for_thr = mag / np.max(mag) if normalize else mag
        
        noise_floor = np.median(mag_for_thr)
        peak_height = np.max(mag_for_thr) - noise_floor
        thr = noise_floor + 0.5 * peak_height
        
        # Find rising and falling edges more robustly
        rising = np.where((mag_for_thr[1:] >= thr) & (mag_for_thr[:-1] < thr))[0]
        falling = np.where((mag_for_thr[1:] < thr) & (mag_for_thr[:-1] >= thr))[0]
        
        # Handle case where no rising edges are found
        if len(rising) == 0:
            continue
        
        # Improved edge handling for pulses
        for idx_start in rising:
            end_candidates = falling[falling > idx_start]
            if len(end_candidates) == 0:
                # Handle case where rising edge has no corresponding falling edge
                # Use end of signal as fallback
                idx_end = len(mag_for_thr) - 1
                # If the remaining signal is too long, we might be detecting noise
                # so we'll skip this potential pulse
                if idx_end - idx_start > 2000:
                    continue
            else:
                idx_end = end_candidates[0]
            
            pulse_samples = idx_end - idx_start
            if not (4 <= pulse_samples <= 2000):
                continue
                
            window = iq[idx_start:idx_end]
            
            # Calculate Time of Arrival
            toa = idx_start / fs
            
            # Calculate Pulse Width
            pw = pulse_samples / fs
            
            # Calculate Carrier Frequency
            try:
                if len(window) >= 8:
                    # Use FFT for longer pulses
                    spectrum = np.fft.fft(window * np.blackman(len(window)))
                    freqs = np.fft.fftfreq(len(window), 1/fs)
                    cf = np.abs(freqs[np.argmax(np.abs(spectrum[:len(spectrum)//2]))])
                else:
                    # Use phase difference for shorter pulses
                    phase_diff = np.diff(np.unwrap(np.angle(window)))
                    cf = np.abs(np.mean(phase_diff) * fs / (2 * np.pi))
            except Exception as e:
                continue  # Skip this pulse on error
            
            # Calculate amplitude
            amp = np.max(np.abs(window))
            
            # Angle of Arrival (placeholder - would normally come from DF antenna)
            aoa = 0.0
            
            pulses.append([toa, cf, pw, aoa, amp])

    if not pulses:
        print("WARNING: No valid pulses were detected.")
        return np.zeros((0, 5), dtype=np.float32)
    
    pulse_array = np.asarray(sorted(pulses), np.float32)
    
    # Print summary statistics
    elapsed = time.time() - start_time
    print(f"\nProcessed {len(signals)} signals in {elapsed:.1f}s ({len(signals)/elapsed:.1f} signals/sec)")
    print(f"Detected {len(pulse_array)} pulses ({len(pulse_array)/len(signals):.1f} pulses/signal)")
    
    # Print pulse parameter ranges
    if len(pulse_array) > 0:
        print("\nPulse parameter ranges:")
        print(f"  TOA:        {pulse_array[:,0].min():.6f} - {pulse_array[:,0].max():.6f} s")
        print(f"  CF:         {pulse_array[:,1].min():.2f} - {pulse_array[:,1].max():.2f} MHz")
        print(f"  PW:         {pulse_array[:,2].min()*1e6:.2f} - {pulse_array[:,2].max()*1e6:.2f} μs")
        print(f"  Amplitude:  {pulse_array[:,4].min():.2f} - {pulse_array[:,4].max():.2f}")
    
    return pulse_array

def main():
    ap = argparse.ArgumentParser(description="Extract PDWs from IQ data")
    ap.add_argument("--iq", required=True, help="Path to IQ data file")
    ap.add_argument("--fs", type=float, default=20e6, help="Sampling frequency in Hz")
    ap.add_argument("--out", default="pdw/dataset.csv", help="Output file path")
    ap.add_argument("--format", default="csv", choices=["npy", "hdf5", "csv"], help="Output file format")
    ap.add_argument("--normalize", action="store_true", help="Normalize signal magnitude")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of signals to process")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ns = ap.parse_args()

    # Check input file exists
    if not os.path.exists(ns.iq):
        print(f"ERROR: Input file {ns.iq} does not exist")
        sys.exit(1)

    print(f"Processing {ns.iq} with sampling rate {ns.fs/1e6:.1f} MHz")
    
    # Extract PDWs with progress reporting
    pdw = decode_iq(ns.iq, ns.fs, normalize=ns.normalize, 
                   limit=ns.limit, show_progress=not ns.no_progress)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(ns.out), exist_ok=True)
    out_path = ns.out

    # Save in the requested format
    try:
        if ns.format == "npy":
            np.save(out_path, pdw)
        elif ns.format == "csv":
            np.savetxt(out_path, pdw, delimiter=",")
        else:
            with h5py.File(out_path, 'w') as f:
                f.create_dataset('pdw_data', data=pdw)
        print(f"PDW data saved to {out_path}")
    except Exception as e:
        print(f"ERROR saving PDW data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
