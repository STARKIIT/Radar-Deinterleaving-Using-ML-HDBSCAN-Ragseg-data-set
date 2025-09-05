import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from scipy.signal import find_peaks

def analyze_pulse_timing(pulses, labels, output_dir=None, max_emitters=10):
    """
    Analyze temporal characteristics of pulses for each emitter.
    
    Args:
        pulses: Array of shape (n_pulses, 5) with [TOA, CF, PW, AOA, Amp]
        labels: Cluster labels for each pulse
        output_dir: Directory to save visualization plots
        max_emitters: Maximum number of emitters to analyze in detail
        
    Returns:
        Dictionary with temporal statistics for each emitter
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get unique emitters (excluding noise)
    unique_emitters = np.unique(labels[labels != -1])
    if len(unique_emitters) == 0:
        print("No valid emitters found for temporal analysis")
        return {}
    
    # Dictionary to store results
    timing_stats = {}
    
    # Limit number of emitters for detailed analysis if there are too many
    if len(unique_emitters) > max_emitters:
        # Choose the emitters with the most pulses
        emitter_sizes = [(emitter, np.sum(labels == emitter)) for emitter in unique_emitters]
        emitter_sizes.sort(key=lambda x: x[1], reverse=True)
        selected_emitters = [e for e, _ in emitter_sizes[:max_emitters]]
        print(f"Limiting detailed temporal analysis to top {max_emitters} largest emitters out of {len(unique_emitters)} total.")
    else:
        selected_emitters = unique_emitters
    
    # Create a plot with subplots for the selected emitters
    if output_dir:
        n_emitters = len(selected_emitters)
        n_cols = min(3, n_emitters)
        n_rows = (n_emitters + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_emitters == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Generate summary statistics for all emitters
    emitter_summary = {}
    for emitter in unique_emitters:
        emitter_pulses = pulses[labels == emitter]
        emitter_summary[emitter] = {
            "num_pulses": len(emitter_pulses),
            "cf_mean": np.mean(emitter_pulses[:, 1]) if len(emitter_pulses) > 0 else 0,
            "cf_std": np.std(emitter_pulses[:, 1]) if len(emitter_pulses) > 0 else 0,
            "pw_mean": np.mean(emitter_pulses[:, 2]) if len(emitter_pulses) > 0 else 0,
        }
    
    # Analyze selected emitters in detail
    for i, emitter in enumerate(selected_emitters):
        # Get pulses for this emitter
        emitter_pulses = pulses[labels == emitter]
        
        # Sort by Time of Arrival
        sorted_indices = np.argsort(emitter_pulses[:, 0])
        sorted_pulses = emitter_pulses[sorted_indices]
        toa = sorted_pulses[:, 0]
        
        # Calculate inter-pulse intervals (PRIs)
        if len(toa) > 1:
            pri = np.diff(toa)
            
            # Calculate basic statistics
            stats_dict = {
                "num_pulses": len(toa),
                "total_duration": toa[-1] - toa[0] if len(toa) > 1 else 0,
                "mean_pri": np.mean(pri) if len(pri) > 0 else 0,
                "median_pri": np.median(pri) if len(pri) > 0 else 0,
                "min_pri": np.min(pri) if len(pri) > 0 else 0,
                "max_pri": np.max(pri) if len(pri) > 0 else 0,
                "std_pri": np.std(pri) if len(pri) > 0 else 0,
            }
            
            # Check for missing pulses (gaps in the pulse train)
            if len(pri) > 2:
                mean_pri = stats_dict["mean_pri"]
                # Look for gaps that are significantly larger than the mean PRI
                large_gaps = pri[pri > 2 * mean_pri]
                stats_dict["missing_pulses_estimate"] = np.sum((large_gaps / mean_pri) - 1)
                stats_dict["large_gaps"] = len(large_gaps)
                
                # Add missing pulses indices for further analysis
                gap_indices = np.where(pri > 2 * mean_pri)[0]
                stats_dict["gap_indices"] = gap_indices.tolist()
                
                # Calculate time ranges with missing pulses
                if len(gap_indices) > 0:
                    gap_starts = toa[gap_indices]
                    gap_ends = toa[gap_indices + 1]
                    stats_dict["gap_ranges"] = [(start, end) for start, end in zip(gap_starts, gap_ends)]
            else:
                stats_dict["missing_pulses_estimate"] = 0
                stats_dict["large_gaps"] = 0
                stats_dict["gap_indices"] = []
                stats_dict["gap_ranges"] = []
            
            # Check for PRI jitter
            if len(pri) > 2:
                stats_dict["pri_jitter"] = stats_dict["std_pri"] / stats_dict["mean_pri"] if stats_dict["mean_pri"] > 0 else 0
                
                # Detailed jitter analysis
                sorted_pri = np.sort(pri)
                q1 = np.percentile(pri, 25)
                q3 = np.percentile(pri, 75)
                iqr = q3 - q1
                stats_dict["pri_iqr"] = iqr
                stats_dict["pri_q1"] = q1
                stats_dict["pri_q3"] = q3
                stats_dict["pri_range"] = sorted_pri[-1] - sorted_pri[0]
            else:
                stats_dict["pri_jitter"] = 0
            
            # Improved stagger PRI detection
            stats_dict["staggered_pri"] = False
            stats_dict["stagger_pattern"] = 1
            stats_dict["stagger_values"] = [stats_dict["mean_pri"]]
            
            if len(pri) > 5:
                # Use histogram analysis for stagger detection
                hist, bin_edges = np.histogram(pri, bins='auto')
                
                # Use kernel density estimation for better peak detection
                try:
                    kde = stats.gaussian_kde(pri)
                    x = np.linspace(np.min(pri) * 0.9, np.max(pri) * 1.1, 1000)
                    y = kde(x)
                    
                    # Find peaks with appropriate parameters for stagger detection
                    peaks, _ = find_peaks(y, height=np.max(y) * 0.2, distance=len(x)//20)
                    
                    if len(peaks) > 1:
                        # Additional validation of stagger pattern
                        peak_values = x[peaks]
                        peak_heights = y[peaks]
                        
                        # Require peaks to be sufficiently distinct
                        min_peak = np.min(peak_values)
                        peak_ratios = peak_values / min_peak
                        distinct_peaks = peak_ratios > 1.2  # At least 20% difference
                        
                        if np.sum(distinct_peaks) > 0:
                            stats_dict["staggered_pri"] = True
                            stats_dict["stagger_pattern"] = len(peaks)
                            stats_dict["stagger_values"] = peak_values.tolist()
                            
                            # Enhanced stagger analysis
                            if len(pri) > 20:
                                # Try to detect the stagger sequence
                                sequence_length = min(10, len(peaks))
                                stats_dict["possible_sequence_length"] = detect_stagger_sequence(pri, sequence_length)
                except Exception as e:
                    print(f"Error in stagger detection for emitter {emitter}: {e}")
            
            # Plot if output directory is specified
            if output_dir and i < len(axes):
                ax = axes[i]
                
                # Enhanced visualization with more information
                
                # Plot pulse train as vertical lines
                for t in toa:
                    ax.axvline(x=t, color='blue', alpha=0.3, linewidth=1)
                
                # Highlight large gaps in red
                for gap_idx in stats_dict.get("gap_indices", []):
                    start = toa[gap_idx]
                    end = toa[gap_idx + 1]
                    ax.axvspan(start, end, color='red', alpha=0.2)
                
                # Annotate with PRI information
                if stats_dict.get("staggered_pri", False):
                    title = f"Emitter {emitter}: {len(toa)} pulses\nStaggered PRI: {stats_dict['stagger_pattern']} levels"
                    
                    # Add inset with PRI histogram for staggered emitters
                    if len(pri) > 10:
                        axins = ax.inset_axes([0.6, 0.6, 0.35, 0.35])
                        axins.hist(pri, bins='auto', alpha=0.7)
                        axins.set_title('PRI Distribution')
                        axins.tick_params(labelsize=8)
                else:
                    title = f"Emitter {emitter}: {len(toa)} pulses\nPRI: {stats_dict['mean_pri']:.6f}s, Jitter: {stats_dict['pri_jitter']:.1%}"
                
                ax.set_title(title)
                ax.set_xlabel("Time (s)")
                ax.set_yticks([])
                
                # Improved marking of gaps
                for j, (start, end) in enumerate(stats_dict.get("gap_ranges", [])):
                    ax.annotate(f"Gap", 
                                xy=((start + end)/2, 0.5), 
                                xycoords=('data', 'axes fraction'),
                                ha='center', va='center', 
                                color='red', fontsize=9)
        else:
            stats_dict = {
                "num_pulses": len(toa),
                "insufficient_data": True
            }
            
            # Clear the plot if not enough data
            if output_dir and i < len(axes):
                axes[i].text(0.5, 0.5, f"Emitter {emitter}: Insufficient pulses for analysis",
                            horizontalalignment='center', verticalalignment='center')
                axes[i].set_yticks([])
                axes[i].set_xticks([])
        
        # Store stats for this emitter
        timing_stats[emitter] = stats_dict
    
    # Hide any unused axes
    if output_dir and 'axes' in locals():
        for i in range(len(selected_emitters), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pulse_timing_analysis.png"), dpi=300)
        plt.close()
        
        # Create a separate plot showing all emitters on the same timeline
        if len(unique_emitters) > 0:
            plt.figure(figsize=(12, 6))
            
            # Get overall time range
            all_times = []
            for emitter in unique_emitters:
                emitter_toa = pulses[labels == emitter, 0]
                if len(emitter_toa) > 0:
                    all_times.extend(emitter_toa)
            
            if all_times:
                t_min = min(all_times)
                t_max = max(all_times)
                
                # Plot each emitter with different colors
                cmap = plt.cm.get_cmap('viridis', len(unique_emitters))
                
                for i, emitter in enumerate(unique_emitters):
                    emitter_toa = pulses[labels == emitter, 0]
                    plt.scatter(emitter_toa, np.ones_like(emitter_toa) * i, 
                                marker='|', s=20, color=cmap(i), 
                                label=f"Emitter {emitter} ({len(emitter_toa)} pulses)")
                
                plt.yticks(range(len(unique_emitters)), [f"Emitter {e}" for e in unique_emitters])
                plt.xlabel("Time (s)")
                plt.title("Pulse Timeline for All Detected Emitters")
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "all_emitters_timeline.png"), dpi=300)
                plt.close()
    
    # Add global statistics
    timing_stats["global"] = {
        "total_emitters": len(unique_emitters),
        "total_pulses": np.sum(labels != -1),
        "noise_pulses": np.sum(labels == -1),
        "emitter_summary": emitter_summary
    }
    
    return timing_stats

def detect_stagger_sequence(pri_values, max_length=10):
    """
    Attempt to detect the sequence length of a staggered PRI pattern.
    
    Args:
        pri_values: Array of PRI values
        max_length: Maximum sequence length to check
        
    Returns:
        Most likely sequence length or None if no pattern found
    """
    if len(pri_values) < max_length * 2:
        return None
    
    # Calculate autocorrelation to find repeating patterns
    from scipy import signal
    
    # Normalize PRI values
    normalized_pri = (pri_values - np.mean(pri_values)) / np.std(pri_values)
    
    # Calculate autocorrelation
    autocorr = signal.correlate(normalized_pri, normalized_pri, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
    
    # Find peaks in autocorrelation
    peaks, _ = find_peaks(autocorr, height=0.3*max(autocorr))
    
    if len(peaks) < 2:
        return None
    
    # The first peak is at lag 0 (self-correlation)
    # The second peak indicates the pattern length
    candidate_lengths = peaks[1:max_length+1] if len(peaks) > 1 else []
    
    if not candidate_lengths.size:
        return None
    
    # Return the lag with the highest autocorrelation as the pattern length
    best_length = candidate_lengths[np.argmax(autocorr[candidate_lengths])]
    
    return int(best_length)
