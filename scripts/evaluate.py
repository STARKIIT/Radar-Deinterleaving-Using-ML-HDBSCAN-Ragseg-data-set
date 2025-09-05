import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import h5py
import argparse
import os
import pandas as pd
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
from model.embedding_transformer import PDWEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import filter_pdw_data # Import the shared function

def analyze_emitters(pdw_data, labels):
    """Generates a detailed report for each detected emitter."""
    report = []
    pdw_df = pd.DataFrame(pdw_data, columns=['TOA', 'CF', 'PW', 'AOA', 'Amplitude'])
    pdw_df['label'] = labels
    
    unique_labels = sorted([l for l in pdw_df['label'].unique() if l != -1])
    
    for label in unique_labels:
        emitter_df = pdw_df[pdw_df['label'] == label]
        report.append(f"\n--- Emitter {label} ---")
        report.append(f"  Number of Pulses: {len(emitter_df)}")
        
        # Parameter Analysis
        for param in ['CF', 'PW', 'Amplitude']:
            stats = emitter_df[param].describe()
            report.append(f"  {param}: Mean={stats['mean']:.3e}, Std={stats['std']:.3e}, "
                          f"Min={stats['min']:.3e}, Max={stats['max']:.3e}")
            
        # PRI Analysis
        if len(emitter_df) > 1:
            toas = np.sort(emitter_df['TOA'].values)
            pris = np.diff(toas)
            pri_stats = pd.Series(pris).describe()
            report.append(f"  PRI: Mean={pri_stats['mean']:.3e}, Std={pri_stats['std']:.3e} (Jitter)")
    
    return "\n".join(report)

def create_evaluation_plots(pdw_data, labels, output_dir):
    """Creates insightful plots for the evaluation report."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    pdw_df = pd.DataFrame(pdw_data, columns=['TOA', 'CF', 'PW', 'AOA', 'Amplitude'])
    pdw_df['label'] = labels
    pdw_df['label_str'] = pdw_df['label'].astype(str)

    # Plot 1: UMAP scatter plot with cluster colors and shapes
    try:
        import umap
        features = pdw_df[['CF', 'PW', 'Amplitude']].values
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(features)
        plt.figure(figsize=(10, 8))

        # Prepare color and marker maps
        cluster_labels = sorted([l for l in pdw_df['label'].unique() if l != -1])
        palette = sns.color_palette("tab20", len(cluster_labels))
        color_map = {str(label): palette[i % len(palette)] for i, label in enumerate(cluster_labels)}
        color_map[str(-1)] = (0.5, 0.5, 0.5)  # grey for noise

        marker_map = {str(label): 'o' for label in cluster_labels}
        marker_map[str(-1)] = 'X'  # noise as 'X'

        # Plot each cluster with its color and marker
        for label in pdw_df['label_str'].unique():
            mask = pdw_df['label_str'] == label
            plt.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[color_map[label]],
                marker=marker_map[label],
                s=40 if label != str(-1) else 60,
                alpha=0.8,
                label=f'Cluster {label}' if label != str(-1) else 'Noise'
            )
        plt.title("UMAP Projection of Pulses by Cluster", fontsize=16)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300)
        plt.close()
        print(f"UMAP cluster plot saved to {output_dir}/umap_clusters.png")
    except ImportError:
        print("UMAP not installed. Skipping UMAP plot.")

    # Plot 2: Cluster size bar plot
    plt.figure(figsize=(8, 5))
    cluster_sizes = pdw_df['label'].value_counts().sort_index()
    bar_colors = [color_map.get(l, (0.5, 0.5, 0.5)) for l in cluster_sizes.index]
    cluster_sizes.plot(kind='bar', color=bar_colors)
    plt.title("Cluster Sizes", fontsize=14)
    plt.xlabel("Cluster Label")
    plt.ylabel("Number of Pulses")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=300)
    plt.close()
    print(f"Cluster size bar plot saved to {output_dir}/cluster_sizes.png")

    # Plot 3: Feature distributions per cluster (boxplots)
    for feature in ['CF', 'PW', 'Amplitude']:
        plt.figure(figsize=(12, 6))
        # Convert label column to string for seaborn compatibility
        pdw_df['label_str'] = pdw_df['label'].astype(str)
        sns.boxplot(x='label_str', y=feature, data=pdw_df, hue='label_str', palette=color_map, legend=False)
        plt.title(f"{feature} Distribution by Cluster", fontsize=14)
        plt.xlabel("Cluster Label")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{feature}_by_cluster_boxplot.png'), dpi=300)
        plt.close()
        print(f"{feature} boxplot saved to {output_dir}/{feature}_by_cluster_boxplot.png")

    # Plot 4: Feature pairplot (with improved palette)
    sns.pairplot(
        pdw_df[pdw_df['label'] != -1], 
        hue='label_str',  # Use string labels for hue
        vars=['CF', 'PW', 'Amplitude'],
        palette=color_map,
        plot_kws={'alpha': 0.6, 's': 20}
    )
    plt.suptitle('Emitter Feature Distributions (Clusters Only)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emitter_feature_pairplot.png'), dpi=300)
    plt.close()
    print(f"Emitter feature pairplot saved to {output_dir}/emitter_feature_pairplot.png")

    # Plot 5: PRI Histograms for each emitter (with color)
    unique_labels = cluster_labels
    if not unique_labels: return

    n_emitters = len(unique_labels)
    n_cols = min(3, n_emitters)
    n_rows = (n_emitters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.ravel()
    for i, label in enumerate(unique_labels):
        emitter_df = pdw_df[pdw_df['label'] == label]
        if len(emitter_df) > 2:
            toas = np.sort(emitter_df['TOA'].values)
            pris = np.diff(toas) * 1e6
            if len(pris) > 1 and np.ptp(pris) > 0:
                axes[i].hist(pris, bins=min(20, max(1, len(np.unique(pris)))), alpha=0.7, color=color_map[str(label)])
                axes[i].set_title(f'Emitter {label} PRI Distribution')
                axes[i].set_xlabel('PRI (µs)')
                axes[i].set_ylabel('Count')
            else:
                axes[i].set_visible(False)
        else:
            axes[i].set_visible(False)
    for i in range(n_emitters, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'emitter_pri_histograms.png'), dpi=300)
    plt.close()
    print(f"PRI histograms saved to {output_dir}/emitter_pri_histograms.png")

    # 3D UMAP scatter plot
    try:
        import umap
        from mpl_toolkits.mplot3d import Axes3D
        features = pdw_df[['CF', 'PW', 'Amplitude']].values
        reducer3d = umap.UMAP(n_components=3, random_state=42)
        embedding_3d = reducer3d.fit_transform(features)

        cluster_labels = sorted([l for l in pdw_df['label'].unique() if l != -1])
        palette = sns.color_palette("tab20", len(cluster_labels))
        color_map = {str(label): palette[i % len(palette)] for i, label in enumerate(cluster_labels)}
        color_map[str(-1)] = (0.5, 0.5, 0.5)
        marker_map = {str(label): 'o' for label in cluster_labels}
        marker_map[str(-1)] = 'X'

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for label in pdw_df['label_str'].unique():
            mask = pdw_df['label_str'] == label
            ax.scatter(
                embedding_3d[mask, 0],
                embedding_3d[mask, 1],
                embedding_3d[mask, 2],
                c=[color_map[label]],
                marker=marker_map[label],
                s=40 if label != str(-1) else 60,
                alpha=0.8,
                label=f'Cluster {label}' if label != str(-1) else 'Noise'
            )
        ax.set_title("3D UMAP Projection of Pulses by Cluster", fontsize=16)
        ax.set_xlabel("UMAP Dim 1")
        ax.set_ylabel("UMAP Dim 2")
        ax.set_zlabel("UMAP Dim 3")
        ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_clusters_3d.png'), dpi=300)
        plt.close()
        print(f"3D UMAP cluster plot saved to {output_dir}/umap_clusters_3d.png")
    except ImportError:
        print("UMAP not installed. Skipping 3D UMAP plot.")
    except Exception as e:
        print(f"3D UMAP plot error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering results.")
    parser.add_argument("--pdw", required=True, help="Path to PDW data file (.hdf5)")
    parser.add_argument("--labels", required=True, help="Path to cluster labels file (.npy)")
    parser.add_argument("--output", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--ckpt", required=False, help="Path to trained transformer checkpoint, if used for clustering")
    parser.add_argument("--plot", action="store_true", help="Enable generation of plots.")
    parser.add_argument("--ground_truth", required=False, help="Path to ground truth labels for purity evaluation")
    parser.add_argument("--advanced", action="store_true", help="Enable advanced radar mode analyses")
    args = parser.parse_args()

    # Instead of filtering again, load the filtered PDW data
    filtered_pdw_path = os.path.join(os.path.dirname(args.labels), "filtered_pdw_data.npy")
    if os.path.exists(filtered_pdw_path):
        features = np.load(filtered_pdw_path)
        print(f"Loaded filtered PDW data from {filtered_pdw_path}")
    else:
        # Fallback: filter raw PDW data (legacy)
        with h5py.File(args.pdw, 'r') as f:
            all_features = f['pdw_data'][:]
        features = filter_pdw_data(all_features, filter_cf_zero=True, filter_outliers=True)
        print("WARNING: Filtered PDW data file not found. Filtering raw PDW data (may cause length mismatch).")
    
    labels = np.load(args.labels)
    
    if features.shape[0] != labels.shape[0]:
        print(f"ERROR: Filtered PDW data ({features.shape[0]}) and labels ({labels.shape[0]}) have different lengths. This indicates a filtering mismatch. Please ensure clustering and evaluation use the same filtered data.")
        return

    report = ["# Radar Deinterleaving Evaluation Report"]
    report.append(f"Evaluation for: {args.labels}")

    # Cluster summary
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster summary:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} points")
    if np.all(labels == -1):
        print("WARNING: All points labeled as noise. Check clustering parameters or input data.")

    # --- Feature Selection for Scoring ---
    if args.ckpt and os.path.exists(args.ckpt):
        # This part remains for future use with a trained model
        print("Evaluating on learned embeddings from transformer.")
        net = PDWEncoder()
        net.load_state_dict(torch.load(args.ckpt, map_location=torch.device('cpu')))
        net.eval()
        with torch.no_grad():
            # Bug fix: Use the filtered 'features' for the transformer
            pdw_tensor = torch.from_numpy(features).float().unsqueeze(0)
            features_for_scoring = net(pdw_tensor).squeeze(0).numpy()
    else:
        print("Evaluating on intrinsic features (CF, PW, Amp).")
        features_for_scoring = features[:, [1, 2, 4]]

    # --- Report Generation ---
    
    # Overall Metrics
    report.append("\n### Overall Performance")
    num_clusters = len(np.unique(labels[labels != -1]))
    noise_points = np.sum(labels == -1)
    noise_ratio = noise_points / len(labels) if len(labels) > 0 else 0
    report.append(f"- Detected Emitters: {num_clusters}")
    report.append(f"- Noise Points: {noise_points} ({noise_ratio:.1%})")

    if num_clusters > 1:
        mask = labels != -1
        # Use RobustScaler to match clustering step
        scaler = RobustScaler()
        # Ensure we only scale the clustered points
        features_for_scoring_masked = features_for_scoring[mask]
        if features_for_scoring_masked.shape[0] > 0:
            features_scaled = scaler.fit_transform(features_for_scoring_masked)
            score = silhouette_score(features_scaled, labels[mask])
            report.append(f"- Silhouette Score: {score:.4f} (higher is better)")
        else:
            report.append("- Silhouette Score: N/A (no points in clusters)")
    else:
        report.append("- Silhouette Score: N/A (less than 2 clusters found)")

    # Purity and Completeness Metrics
    if args.ground_truth:
        print("Attempting to evaluate with ground truth...")
        gt_labels_full = np.load(args.ground_truth)
        
        # This is a critical assumption: ground truth must correspond to the *original* unfiltered data
        if len(gt_labels_full) == len(all_features):
            # Create the same filter mask that was applied to the data
            cf_zero_mask = all_features[:, 1] != 0
            temp_features = all_features[cf_zero_mask]
            
            outlier_mask = np.ones(len(temp_features), dtype=bool)
            if len(temp_features) > 0:
                q_low = np.percentile(temp_features[:, [1, 2, 4]], 1, axis=0)
                q_high = np.percentile(temp_features[:, [1, 2, 4]], 99, axis=0)
                outlier_mask = np.all((temp_features[:, [1, 2, 4]] >= q_low) & (temp_features[:, [1, 2, 4]] <= q_high), axis=1)

            final_mask = cf_zero_mask.copy()
            final_mask[final_mask] = outlier_mask
            
            gt_labels = gt_labels_full[final_mask]

            if len(gt_labels) == len(labels):
                purity = homogeneity_score(gt_labels, labels)
                completeness = completeness_score(gt_labels, labels)
                report.append(f"- Cluster Purity: {purity:.4f}")
                report.append(f"- Cluster Completeness: {completeness:.4f}")
            else:
                print(f"WARNING: Ground truth labels ({len(gt_labels)}) could not be aligned with filtered data labels ({len(labels)}). Skipping purity metrics.")
        else:
            print(f"WARNING: Ground truth file has {len(gt_labels_full)} labels, but original PDW file has {len(all_features)} pulses. Skipping purity metrics.")

    # Per-Emitter Analysis
    report.append("\n### Emitter-Specific Analysis")
    emitter_report = analyze_emitters(features, labels)
    report.append(emitter_report)

    # Save Report
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("\n".join(report))
    
    print(f"\nDetailed evaluation report saved to: {report_path}")

    # Create Plots
    if args.plot:
        create_evaluation_plots(features, labels, args.output)

    # After loading and filtering data, add:
    # Perform temporal analysis
    try:
        from utils.temporal_analysis import analyze_pulse_timing
        print("Performing temporal analysis...")
        timing_stats = analyze_pulse_timing(
            features,
            labels,
            output_dir=os.path.join(args.output, "temporal_analysis")
        )
        report.append("\n### Temporal Analysis")
        report.append("Pulse train analysis for each emitter:")
        
        for emitter, stats in timing_stats.items():
            if stats.get("insufficient_data", False) or "num_pulses" not in stats:
                continue
                
            report.append(f"\n**Emitter {emitter}**:")
            report.append(f"- Pulses: {stats['num_pulses']}")
            report.append(f"- Duration: {stats['total_duration']:.6f}s")
            report.append(f"- PRI: {stats['mean_pri']:.6f}s ± {stats['std_pri']:.6f}s")
            report.append(f"- PRI Jitter: {stats['pri_jitter']:.1%}")
            
            if stats.get("staggered_pri", False):
                report.append(f"- Staggered PRI detected with {stats['stagger_pattern']} levels")
                report.append(f"- Stagger values: {', '.join([f'{x:.6f}s' for x in stats['stagger_values']])}")
            
            if stats.get("large_gaps", 0) > 0:
                report.append(f"- Detected {stats['large_gaps']} large gaps in the pulse train")
                report.append(f"- Estimated missing pulses: {int(stats['missing_pulses_estimate'])}")
    except Exception as e:
        print(f"Temporal analysis error: {e}")
        report.append("\n### Temporal Analysis")
        report.append("Error performing temporal analysis.")
    
    # Advanced analyses (optional)
    if args.advanced:
        print("Running advanced radar mode analyses...")
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Frequency hopping analysis
        try:
            from utils.frequency_hopping import analyze_frequency_hopping
            fh_results = analyze_frequency_hopping(features, labels, output_dir=os.path.join(args.output, "frequency_hopping"))
            with open(os.path.join(args.output, "frequency_hopping", "fh_report.json"), "w") as f:
                import json
                json.dump(make_json_safe(fh_results), f, indent=2)
            print("Frequency hopping analysis complete. See fh_report.json for details.")
        except Exception as e:
            print(f"Frequency hopping analysis error: {e}")

        # Enhanced staggered PRI analysis
        try:
            from utils.pri_analysis import enhanced_staggered_pri_analysis
            staggered_results = enhanced_staggered_pri_analysis(features, labels, output_dir=os.path.join(args.output, "pri_analysis"))
            with open(os.path.join(args.output, "pri_analysis", "staggered_report.json"), "w") as f:
                import json
                json.dump(make_json_safe(staggered_results), f, indent=2)
            print("Enhanced staggered PRI analysis complete. See staggered_report.json for details.")
        except Exception as e:
            print(f"Staggered PRI analysis error: {e}")

    # Additional check for clustering output
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) == 1 and unique[0] == 0:
        print("WARNING: All points assigned to a single cluster (0). Check clustering parameters and input data.")
    if np.all(labels == -1):
        print("WARNING: All points labeled as noise. Check clustering parameters or input data.")

if __name__ == "__main__":
    main()
    print("Evaluation complete. Results saved to output directory.")