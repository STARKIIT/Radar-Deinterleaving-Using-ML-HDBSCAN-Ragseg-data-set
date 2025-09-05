import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_frequency_hopping(pdw_data, labels, output_dir=None):
    """
    Analyze CF sequences for frequency hopping patterns per emitter.
    Args:
        pdw_data: Array of shape (n_pulses, 5) [TOA, CF, PW, AOA, Amp]
        labels: Cluster labels for each pulse
        output_dir: Directory to save visualizations (optional)
    Returns:
        Dict of hopping metrics per emitter
    """
    results = {}
    unique_emitters = np.unique(labels[labels != -1])
    for emitter in unique_emitters:
        emitter_mask = labels == emitter
        emitter_pdws = pdw_data[emitter_mask]
        if emitter_pdws.shape[0] < 2:
            results[emitter] = {"hop_events": 0, "hop_rate": 0, "fft_peak": 0, "cf_range": 0, "cf_mean": 0}
            continue
        cf_seq = emitter_pdws[:, 1]
        toa_seq = emitter_pdws[:, 0]
        cf_diff = np.abs(np.diff(cf_seq))
        mean_cf = np.mean(cf_seq) if np.std(cf_seq) > 0 else 0
        hop_events = np.sum(cf_diff > 0.05 * mean_cf) if mean_cf > 0 else 0
        hop_rate = hop_events / len(cf_diff) if len(cf_diff) > 0 else 0
        fft_peak = 0
        if len(cf_seq) > 10 and np.std(cf_seq) > 0:
            try:
                from scipy.fft import fft
                cf_norm = (cf_seq - np.mean(cf_seq)) / (np.std(cf_seq) + 1e-8)
                fft_mag = np.abs(fft(cf_norm))
                fft_peak = float(np.max(fft_mag[1:])) if len(fft_mag) > 1 else 0
            except Exception as e:
                fft_peak = 0
        results[emitter] = {
            "hop_events": int(hop_events),
            "hop_rate": hop_rate,
            "fft_peak": float(fft_peak),
            "cf_range": float(np.ptp(cf_seq)),
            "cf_mean": float(mean_cf),
        }
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                plt.figure(figsize=(10, 4))
                plt.plot(toa_seq, cf_seq, marker='o', linestyle='-', alpha=0.7)
                plt.title(f"Emitter {emitter} Carrier Frequency vs TOA")
                plt.xlabel("TOA (s)")
                plt.ylabel("CF (MHz)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"emitter_{emitter}_cf_hopping.png"), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Frequency hopping plot error for emitter {emitter}: {e}")
    return results
