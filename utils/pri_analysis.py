import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import time

def analyze_pri_stability(pulses, labels, output_dir=None):
    """
    Analyze PRI (Pulse Repetition Interval) stability for each detected emitter.
    
    Args:
        pulses: Array of shape (n_pulses, 5) with [TOA, CF, PW, AOA, Amp]
        labels: Cluster labels for each pulse
        output_dir: Directory to save visualization plots
        
    Returns:
        Dictionary with PRI statistics for each emitter
    """
    emitter_stats = {}
    unique_emitters = np.unique(labels[labels != -1])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for emitter in unique_emitters:
        # Get pulses for this emitter
        emitter_pulses = pulses[labels == emitter]
        
        # Sort by Time of Arrival
        sorted_indices = np.argsort(emitter_pulses[:, 0])
        sorted_pulses = emitter_pulses[sorted_indices]
        
        # Calculate PRIs
        if len(sorted_pulses) > 1:
            pri_values = np.diff(sorted_pulses[:, 0])
            
            # Basic statistics
            pri_mean = np.mean(pri_values)
            pri_std = np.std(pri_values)
            pri_cv = pri_std / pri_mean if pri_mean > 0 else np.inf
            
            # Detect staggered PRIs
            is_staggered = False
            num_staggers = 1
            stagger_values = [pri_mean]
            
            if len(pri_values) > 5:
                # Use kernel density estimation to identify PRI modes
                # Only if there's significant variation - improves detection of jittered PRIs
                if np.std(pri_values) / np.mean(pri_values) > 0.05:
                    try:
                        # Use KDE with adaptive bandwidth for better peak detection in jittered data
                        bw = "scott"  # Adaptive bandwidth selector
                        kde = stats.gaussian_kde(pri_values, bw_method=bw)
                        x = np.linspace(np.min(pri_values) * 0.9, np.max(pri_values) * 1.1, 1000)
                        y = kde(x)
                        
                        # Find peaks with improved parameters for jittered PRIs
                        from scipy.signal import find_peaks
                        # Use more sensitive peak detection for jittered PRIs
                        peaks, properties = find_peaks(y, height=np.max(y) * 0.2, 
                                                 distance=len(x)//50,
                                                 prominence=np.max(y) * 0.1)
                        
                        if len(peaks) > 1:
                            is_staggered = True
                            num_staggers = len(peaks)
                            stagger_values = x[peaks]
                            
                            # Improve stagger pattern analysis using peak prominence
                            prominence_ratio = properties["prominences"] / np.max(properties["prominences"])
                            significant_peaks = peaks[prominence_ratio > 0.3]
                            
                            # If plotting is enabled
                            if output_dir:
                                plt.figure(figsize=(10, 6))
                                plt.hist(pri_values, bins=30, alpha=0.5, density=True, label="PRI Distribution")
                                plt.plot(x, y, 'r-', label="Density Estimate")
                                plt.plot(x[peaks], y[peaks], 'go', label="Detected Staggers")
                                plt.title(f"Emitter {emitter} PRI Analysis: {num_staggers} staggers detected")
                                plt.xlabel("PRI (s)")
                                plt.ylabel("Density")
                                plt.legend()
                                plt.grid(True, alpha=0.3)
                                plt.savefig(os.path.join(output_dir, f"emitter_{emitter}_pri.png"), dpi=300)
                                plt.close()
                    except Exception as e:
                        print(f"Error in KDE analysis for emitter {emitter}: {e}")
            
            # Store statistics
            emitter_stats[emitter] = {
                "mean_pri": pri_mean,
                "std_pri": pri_std,
                "cv_pri": pri_cv,
                "is_staggered": is_staggered,
                "num_staggers": num_staggers,
                "stagger_values": stagger_values,
                "valid_emitter": True,
                "num_pulses": len(sorted_pulses),
                # Enhanced analysis for agile radars
                "frequency_agility": detect_frequency_agility(sorted_pulses),
                "pulse_density": len(sorted_pulses) / (sorted_pulses[-1, 0] - sorted_pulses[0, 0]) if len(sorted_pulses) > 1 else 0
            }
        else:
            emitter_stats[emitter] = {
                "valid_emitter": False,
                "reason": "insufficient_pulses",
                "num_pulses": len(sorted_pulses)
            }
    
    return emitter_stats

def detect_frequency_agility(pulses):
    """
    Detect if the emitter shows frequency agility patterns.
    
    Args:
        pulses: Array of sorted pulses for a single emitter
        
    Returns:
        Dictionary with frequency agility metrics
    """
    if len(pulses) < 5:
        return {"agile": False, "reason": "insufficient_data"}
    
    # Extract CF values
    cf_values = pulses[:, 1]
    
    # Calculate statistics
    cf_mean = np.mean(cf_values)
    cf_std = np.std(cf_values)
    cf_range = np.max(cf_values) - np.min(cf_values)
    cf_cv = cf_std / cf_mean if cf_mean > 0 else 0
    
    # Detect frequency hopping
    is_hopping = False
    hop_pattern = None
    
    # Check for significant frequency variations indicating agility
    if cf_cv > 0.05 and cf_range / cf_mean > 0.1:
        # Look for patterns in frequency changes
        try:
            from scipy.fft import fft
            
            # Normalize and compute FFT of frequency sequence
            cf_norm = (cf_values - np.mean(cf_values)) / np.std(cf_values)
            fft_result = fft(cf_norm)
            fft_mag = np.abs(fft_result[:len(cf_norm)//2])
            
            # Check for strong peaks in FFT (indicating periodic hopping)
            max_peak = np.max(fft_mag[1:])  # Skip DC component
            if max_peak > 3 * np.mean(fft_mag[1:]):
                is_hopping = True
                hop_pattern = "regular"
            else:
                is_hopping = True
                hop_pattern = "irregular"
        except Exception:
            # If FFT analysis fails, use simpler heuristics
            cf_diffs = np.abs(np.diff(cf_values))
            threshold = 0.1 * cf_mean
            if np.mean(cf_diffs > threshold) > 0.5:
                is_hopping = True
                hop_pattern = "irregular"
    
    return {
        "agile": is_hopping,
        "pattern": hop_pattern,
        "cf_cv": cf_cv,
        "cf_range_ratio": cf_range / cf_mean if cf_mean > 0 else 0
    }

def validate_emitters(pulses, labels, max_cv=0.3, output_dir=None, timeout=300):
    """
    Validate emitter clusters based on PRI consistency.
    
    Args:
        pulses: Array of shape (n_pulses, 5) with [TOA, CF, PW, AOA, Amp]
        labels: Cluster labels for each pulse
        max_cv: Maximum coefficient of variation for valid emitters
        output_dir: Directory to save validation plots
        timeout: Maximum execution time in seconds
        
    Returns:
        Refined labels with inconsistent emitters marked as noise,
        Dictionary with PRI statistics for each emitter
    """
    # Start timer to prevent hanging on large datasets
    start_time = time.time()
    
    # For large datasets, use a more efficient approach
    if len(pulses) > 10000:
        print(f"Large dataset detected ({len(pulses)} pulses). Using optimized validation.")
        # Sample strategy for large datasets
        sample_size = min(10000, int(len(pulses) * 0.2))
        sample_indices = np.random.choice(len(pulses), sample_size, replace=False)
        sample_indices.sort()  # Keep temporal ordering
        
        # Analyze the sample
        sample_pulses = pulses[sample_indices]
        sample_labels = labels[sample_indices]
        emitter_stats = analyze_pri_stability(sample_pulses, sample_labels, output_dir)
    else:
        # For smaller datasets, analyze all pulses
        emitter_stats = analyze_pri_stability(pulses, labels, output_dir)
    
    # Create a copy of labels to modify
    refined_labels = labels.copy()
    
    invalid_emitters = []
    
    # Check each emitter with timeout protection
    for emitter, stats in emitter_stats.items():
        # Check if we're exceeding the timeout
        if time.time() - start_time > timeout:
            print(f"WARNING: Validation timeout ({timeout}s) reached. Stopping validation.")
            break
            
        if not stats.get("valid_emitter", False):
            invalid_emitters.append(emitter)
            refined_labels[labels == emitter] = -1
            continue
        
        # Skip staggered PRIs - they're naturally more variable
        if stats.get("is_staggered", False):
            continue
        
        # Check PRI consistency with improved handling for jittered PRIs
        # Adjust max_cv based on pulse count (more pulses = more likely to have jitter)
        pulse_count = stats.get("num_pulses", 0)
        adjusted_max_cv = max_cv
        if pulse_count > 100:
            # For many pulses, allow more variation
            adjusted_max_cv = max_cv * 1.5
        
        if stats.get("cv_pri", np.inf) > adjusted_max_cv:
            # Further check if it might be a jittered PRI before rejecting
            if not stats.get("is_staggered", False) and stats.get("cv_pri", np.inf) > adjusted_max_cv * 2:
                invalid_emitters.append(emitter)
                refined_labels[labels == emitter] = -1
    
    if invalid_emitters:
        print(f"Removed {len(invalid_emitters)} inconsistent emitters: {invalid_emitters}")
    
    return refined_labels, emitter_stats

def enhanced_staggered_pri_analysis(pulses, labels, output_dir=None):
    """
    Enhanced staggered PRI detection using autocorrelation and pattern mining.
    Args:
        pulses: Array of shape (n_pulses, 5)
        labels: Cluster labels
        output_dir: Directory for plots (optional)
    Returns:
        Dict of staggered PRI metrics per emitter
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import correlate, find_peaks

    results = {}
    unique_emitters = np.unique(labels[labels != -1])
    for emitter in unique_emitters:
        emitter_mask = labels == emitter
        emitter_pdws = pulses[emitter_mask]
        toas = np.sort(emitter_pdws[:, 0])
        pris = np.diff(toas)
        if len(pris) < 5 or np.std(pris) == 0:
            results[emitter] = {"staggered": False, "pattern_length": None, "autocorr_peaks": []}
            continue
        try:
            norm_pris = (pris - np.mean(pris)) / (np.std(pris) + 1e-8)
            autocorr = correlate(norm_pris, norm_pris, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            peaks, _ = find_peaks(autocorr, height=0.3 * np.max(autocorr))
            pattern_length = int(peaks[1]) if len(peaks) > 1 else None
            results[emitter] = {
                "staggered": bool(pattern_length),
                "pattern_length": pattern_length,
                "autocorr_peaks": peaks.tolist(),
            }
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    plt.figure(figsize=(8, 4))
                    plt.plot(autocorr, label="Autocorrelation")
                    if len(peaks) > 1:
                        plt.scatter(peaks, autocorr[peaks], color='red', label="Pattern Peaks")
                    plt.title(f"Emitter {emitter} PRI Autocorrelation")
                    plt.xlabel("Lag")
                    plt.ylabel("Autocorr")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"emitter_{emitter}_pri_autocorr.png"), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Staggered PRI plot error for emitter {emitter}: {e}")
        except Exception as e:
            results[emitter] = {"staggered": False, "pattern_length": None, "autocorr_peaks": []}
            print(f"Staggered PRI analysis error for emitter {emitter}: {e}")
    return results
