import hdbscan
import numpy as np
import argparse
import h5py
import os
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_processing import filter_pdw_data

# Import error handling
try:
    from utils.pri_analysis import validate_emitters
    PRI_VALIDATION_AVAILABLE = True
except ImportError:
    print("WARNING: PRI validation module not available. Skipping PRI validation.")
    PRI_VALIDATION_AVAILABLE = False

def main():
    ap = argparse.ArgumentParser(description="HDBSCAN Clustering for Radar Deinterleaving")
    ap.add_argument("--pdw", required=True, help="Path to PDW data file")
    ap.add_argument("--output", required=True, help="Output directory for clustering results")
    ap.add_argument("--visualize", action="store_true", help="Enable visualization")
    ap.add_argument("--min_cluster_size", type=int, default=50, help="HDBSCAN min_cluster_size")  # Increase default
    ap.add_argument("--min_samples", type=int, default=None, help="HDBSCAN min_samples. If None, defaults to min_cluster_size.")
    ap.add_argument("--cluster_selection_epsilon", type=float, default=0.05, help="HDBSCAN cluster_selection_epsilon")  # Lower default
    ap.add_argument("--adaptive_params", action="store_true", help="Automatically adjust clustering parameters based on dataset size")
    ap.add_argument("--outlier_percentile", type=int, default=10, help="Percentile for outlier removal (default 10)")  # Keep more data
    ap.add_argument("--features", nargs='+', default=["CF", "PW", "Amp"], help="Features to use for clustering: CF, PW, Amp, TOA, AOA")  # Add Amplitude
    ap.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--max_clusters", type=int, default=25, help="Maximum number of clusters to keep")
    ap.add_argument("--reduce_dim", action="store_true", help="Enable dimensionality reduction before clustering")
    ap.add_argument("--allow_single_cluster", action="store_true", help="Allow HDBSCAN to assign all points to a single cluster")
    ns = ap.parse_args()

    np.random.seed(ns.random_seed)

    # 1. Load Data
    with h5py.File(ns.pdw, 'r') as f:
        all_features = f['pdw_data'][:]
    print(f"Loaded {len(all_features)} pulses.")

    # 2. Filter Data using the shared utility
    # Print filtering stats
    print(f"Loaded {len(all_features)} pulses before filtering.")
    all_features = filter_pdw_data(
        all_features,
        filter_cf_zero=True,
        filter_outliers=True,
        outlier_percentile=ns.outlier_percentile
    )
    print(f"{all_features.shape[0]} pulses after filtering.")

    # After filtering
    filtered_pdw_path = os.path.join(ns.output, "filtered_pdw_data.npy")
    np.save(filtered_pdw_path, all_features)
    print(f"Filtered PDW data saved to {filtered_pdw_path}")

    if all_features.shape[0] < ns.min_cluster_size:
        print(f"ERROR: Not enough data ({all_features.shape[0]}) to form a cluster of size {ns.min_cluster_size}.")
        os.makedirs(ns.output, exist_ok=True)
        np.save(os.path.join(ns.output, "cluster_labels.npy"), np.array([]))
        return

    # 3. Feature Selection
    feature_map = {"CF": 1, "PW": 2, "Amp": 4, "TOA": 0, "AOA": 3}
    feature_indices = [feature_map[f] for f in ns.features if f in feature_map]
    if not feature_indices:
        print("ERROR: No valid features selected for clustering.")
        return
    print(f"Using features for clustering: {ns.features}")

    # 4. Use raw features directly
    features_for_clustering = all_features[:, feature_indices]
    
    # 5. Apply dimensionality reduction if requested
    if "reduce_dim" in ns and ns.reduce_dim:
        print("Applying dimensionality reduction before clustering...")
        try:
            from sklearn.decomposition import PCA
            import umap
            
            # First, scale the features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features_for_clustering)
            
            # Choose dimensionality reduction method based on data size
            if len(features_scaled) < 500:
                # For smaller datasets, PCA is more stable, faster, and uses less memory
                n_components = min(3, features_for_clustering.shape[1])
                reducer = PCA(n_components=n_components)
                print(f"Using PCA with {n_components} components for {len(features_scaled)} points")
                
            else:
                # For larger datasets, UMAP gives better results but is more resource-intensive
                n_components = min(3, features_for_clustering.shape[1])
                
                # Optimize parameters for M1 Mac
                # - For medium datasets (500-5000), use moderate n_neighbors and high min_dist
                # - For large datasets (>5000), use larger n_neighbors and higher min_dist
                if len(features_scaled) > 5000:
                    n_neighbors = min(50, max(30, len(features_scaled) // 500))
                    min_dist = 0.5
                    low_memory = True
                    print("Using UMAP with memory optimization for large dataset")
                else:
                    n_neighbors = min(30, len(features_scaled) // 20)
                    min_dist = 0.3
                    low_memory = False
                
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='euclidean',
                    low_memory=low_memory,
                    random_state=ns.random_seed
                )
                print(f"Using UMAP with {n_components} components, {n_neighbors} neighbors for {len(features_scaled)} points")
            
            features_reduced = reducer.fit_transform(features_scaled)
            print(f"Reduced dimensions from {features_for_clustering.shape[1]} to {features_reduced.shape[1]}")
            
            # Set the reduced features for clustering
            features_for_clustering = features_reduced
            
            # No need to scale again, UMAP output is already well-distributed
            features_scaled = features_reduced
            
        except ImportError:
            print("WARNING: Could not import dimensionality reduction libraries. Using raw features.")
            # Scale the raw features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features_for_clustering)
        except Exception as e:
            print(f"ERROR in dimensionality reduction: {e}. Using raw features.")
            # Scale the raw features
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features_for_clustering)
    else:
        # Standard scaling for raw features
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_for_clustering)
        print("Features scaled using RobustScaler.")
    
    # Adaptive parameter selection now properly scales with dimensionality
    if ns.adaptive_params:
        data_size = all_features.shape[0]
        feature_dim = len(feature_indices)
        
        # Density-based parameters following research by Campello et al. (2013)
        # - Min points should scale with dimensionality: min_pts ≈ 2^d (where d is dimensionality)
        # - For high dimensions, this gets capped to avoid excessive requirements
        dim_factor = min(10, 2**feature_dim)  # Cap at 10 for high dimensions
        
        # 2. Establish core distance estimation parameter based on nearest neighbors
        # - Research by McInnes et al. suggests min_samples around 0.5% of data size with a 
        #   minimum based on the dimensionality of the feature space
        min_samples_by_size = max(feature_dim + 1, int(data_size * 0.005))
        
        # 3. Estimate point density requirements for cluster formation
        # - Clusters should have enough points to form statistically significant distributions
        # - Base on sample size requirements for mean estimation with confidence interval
        min_cluster_by_size = max(dim_factor, int(data_size * 0.01))
        
        # 4. Apply practical constraints for very small or very large datasets
        if data_size < 100:
            # For very small datasets, we need looser constraints
            ns.min_cluster_size = max(3, min(int(data_size * 0.1), min_cluster_by_size))
            ns.min_samples = max(2, min(ns.min_cluster_size - 1, min_samples_by_size))
        elif data_size > 5000:
            # For very large datasets, use log scaling to avoid excessive parameters
            log_scale = np.log10(data_size) / np.log10(5000)
            ns.min_cluster_size = min(100, max(20, int(min_cluster_by_size * log_scale)))
            ns.min_samples = max(feature_dim + 1, int(ns.min_cluster_size * 0.7))
        else:
            # For normal-sized datasets
            ns.min_cluster_size = min_cluster_by_size
            ns.min_samples = min_samples_by_size
        
        # 5. Epsilon parameter depends on the feature density and dimensionality
        # - With higher dimensions, we need larger epsilon to counteract curse of dimensionality
        # - Scale with the feature space density
        base_epsilon = 0.05 + (0.02 * feature_dim)
        density_factor = np.power(data_size, -1/feature_dim)  # Density decreases as dim increases
        ns.cluster_selection_epsilon = base_epsilon / max(0.01, density_factor)  # Limit to reasonable range
        ns.cluster_selection_epsilon = min(0.3, max(0.05, ns.cluster_selection_epsilon))  # Constrain to practical range
        
        print(f"Adaptive parameters using theoretical approach:")
        print(f"  Feature dimensionality: {feature_dim}")
        print(f"  min_cluster_size = {ns.min_cluster_size}")
        print(f"  min_samples = {ns.min_samples}")
        print(f"  cluster_selection_epsilon = {ns.cluster_selection_epsilon:.4f}")

    # 6. Perform HDBSCAN Clustering
    min_samples = ns.min_samples if ns.min_samples is not None else ns.min_cluster_size
    
    print(f"Running HDBSCAN with: min_cluster_size={ns.min_cluster_size}, min_samples={min_samples}, epsilon={ns.cluster_selection_epsilon}")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=ns.min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        cluster_selection_epsilon=ns.cluster_selection_epsilon,
        allow_single_cluster=ns.allow_single_cluster
    )
    
    labels = clusterer.fit_predict(features_scaled)
    
    # Retry with more aggressive settings if we have too much noise
    noise_ratio = np.sum(labels == -1) / len(labels)
    if noise_ratio > 0.5:
        print(f"High noise ratio ({noise_ratio:.1%}). Trying more aggressive parameters...")
        retry_min_cluster_size = max(3, int(ns.min_cluster_size * 0.7))
        retry_epsilon = ns.cluster_selection_epsilon + 0.1
        
        retry_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=retry_min_cluster_size,
            min_samples=max(3, int(retry_min_cluster_size * 0.7)),
            metric='euclidean',
            cluster_selection_method='eom',
            cluster_selection_epsilon=retry_epsilon,
            allow_single_cluster=True
        )
        
        retry_labels = retry_clusterer.fit_predict(features_scaled)
        retry_noise = np.sum(retry_labels == -1) / len(retry_labels)
        
        if retry_noise < noise_ratio:
            print(f"Better results with retry: noise {noise_ratio:.1%} → {retry_noise:.1%}")
            labels = retry_labels
            noise_ratio = retry_noise

    # Cluster summary
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster summary:")
    for u, c in zip(unique, counts):
        print(f"  Cluster {u}: {c} points")
    print(f"Noise ratio: {noise_ratio:.2%}")

    # Warning for problematic clustering
    if len(unique) == 1 and unique[0] == 0:
        print("WARNING: All points assigned to a single cluster (0). Check clustering parameters and input data.")
    if np.all(labels == -1):
        print("WARNING: All points labeled as noise. Try adjusting clustering parameters.")

    # Post-processing: merge excessive clusters if needed
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels[unique_labels != -1])
    
    if num_clusters > ns.max_clusters:
        print(f"WARNING: {num_clusters} clusters detected, which exceeds the maximum of {ns.max_clusters}.")
        print("Performing cluster merging to reduce to a practical number...")
        
        # Calculate cluster centroids
        cluster_features = {}
        for label in unique_labels:
            if label == -1: continue
            cluster_features[label] = np.mean(features_for_clustering[labels == label], axis=0)
        
        # Calculate pairwise distances between clusters
        from scipy.spatial.distance import pdist, squareform
        centroid_vectors = np.array([cluster_features[label] for label in sorted(cluster_features.keys())])
        distance_matrix = squareform(pdist(centroid_vectors))
        
        # Hierarchical merging of closest clusters
        sorted_cluster_labels = sorted(cluster_features.keys())
        while len(cluster_features) > ns.max_clusters:
            # Find the two closest clusters
            i, j = np.unravel_index(np.argmin(distance_matrix + np.eye(len(distance_matrix)) * 9999), distance_matrix.shape)
            if i > j:
                i, j = j, i  # Ensure i < j
            
            cluster1, cluster2 = sorted_cluster_labels[i], sorted_cluster_labels[j]
            print(f"  Merging clusters {cluster1} and {cluster2} (distance: {distance_matrix[i, j]:.3f})")
            
            # Merge cluster2 into cluster1
            labels[labels == cluster2] = cluster1
            
            # Update centroid for the merged cluster
            cluster_features[cluster1] = np.mean(features_for_clustering[labels == cluster1], axis=0)
            
            # Remove cluster2
            del cluster_features[cluster2]
            sorted_cluster_labels.remove(cluster2)
            
            # Recalculate distance matrix
            centroid_vectors = np.array([cluster_features[label] for label in sorted_cluster_labels])
            distance_matrix = squareform(pdist(centroid_vectors))
        
        # Renumber clusters to be consecutive
        unique_labels = np.unique(labels)
        mapping = {label: i for i, label in enumerate(unique_labels) if label != -1}
        mapping[-1] = -1  # Keep noise as -1
        labels = np.array([mapping[label] for label in labels])
        
        print(f"After merging: {len(np.unique(labels)) - 1} clusters")

    # 7. Save Results
    os.makedirs(ns.output, exist_ok=True)
    np.save(os.path.join(ns.output, "cluster_labels.npy"), labels)
    print(f"Labels saved to {os.path.join(ns.output, 'cluster_labels.npy')}")

    # 8. Visualization
    if ns.visualize:
        try:
            import umap
            plt.figure(figsize=(10, 8))
            n_neighbors = min(15, len(features_scaled) - 1)
            if n_neighbors < 2:
                print("Not enough data points for UMAP visualization.")
                return

            reducer = umap.UMAP(n_components=2, random_state=ns.random_seed, n_neighbors=n_neighbors)
            embedding_2d = reducer.fit_transform(features_scaled)
            
            unique_labels = np.unique(labels)
            num_clusters = len(unique_labels[unique_labels != -1])
            try:
                colors = plt.cm.get_cmap('viridis', max(1, num_clusters))
                color_map = {label: colors(i) for i, label in enumerate(unique_labels[unique_labels != -1])}
                color_map[-1] = (0.5, 0.5, 0.5, 0.3)
                point_colors = [color_map[label] for label in labels]
            except Exception as e:
                print(f"Visualization warning: {e}")
                point_colors = ['blue' if label != -1 else 'gray' for label in labels]
            
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=point_colors, s=15, alpha=0.8)
            plt.title(f"HDBSCAN Visualization of {num_clusters} Detected Emitters", fontsize=16)
            plt.xlabel("UMAP Dimension 1")
            plt.ylabel("UMAP Dim 2")
            plt.grid(True)

            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Noise',
                                      markerfacecolor=color_map.get(-1, 'gray'), markersize=10)]
            for label in sorted([l for l in unique_labels if l != -1]):
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              label=f'Emitter {int(label)}', 
                                              markerfacecolor=color_map[label], markersize=10))

            plt.legend(handles=legend_elements, title="Clusters")
            plt.tight_layout()
            viz_path = os.path.join(ns.output, "clustering_visualization.png")
            plt.savefig(viz_path, dpi=300)
            print(f"Visualization saved to: {viz_path}")
            plt.close()
        except Exception as e:
            print(f"Visualization error: {e}. Please ensure all dependencies are installed.")

    # Post-process clusters with PRI validation if enough pulses
    if len(all_features) > 50 and PRI_VALIDATION_AVAILABLE:
        try:
            print("Validating emitters based on PRI consistency...")
            
            # For very large datasets, sample a subset of pulses for faster validation
            if len(all_features) > 10000:
                print(f"Large dataset detected ({len(all_features)} pulses). Sampling for PRI validation.")
                sample_size = 10000
                sample_indices = np.random.choice(len(all_features), sample_size, replace=False)
                
                # Sort indices to maintain temporal ordering
                sample_indices = np.sort(sample_indices)
                
                # Use sample for initial validation
                refined_labels_sample, emitter_stats = validate_emitters(
                    all_features[sample_indices], 
                    labels[sample_indices],
                    max_cv=0.5,
                    output_dir=os.path.join(ns.output, "pri_analysis")
                )
                
                # Convert results from sample back to full dataset
                invalid_emitters = [e for e, stats in emitter_stats.items() 
                                   if not stats.get("valid_emitter", True) or 
                                   (stats.get("cv_pri", 0) > 0.5 and not stats.get("is_staggered", False))]
                
                refined_labels = labels.copy()
                for emitter in invalid_emitters:
                    refined_labels[labels == emitter] = -1
                
                print(f"PRI validation on {sample_size} sample pulses identified {len(invalid_emitters)} invalid emitters")
                
            else:
                # Regular PRI validation for normal-sized datasets
                refined_labels, emitter_stats = validate_emitters(
                    all_features, 
                    labels, 
                    max_cv=0.5,  # Allowing up to 50% variation in PRI
                    output_dir=os.path.join(ns.output, "pri_analysis")
                )
            
            # Count how many emitters were refined
            orig_emitters = len(np.unique(labels[labels != -1]))
            refined_emitters = len(np.unique(refined_labels[refined_labels != -1]))
            
            if refined_emitters < orig_emitters:
                print(f"PRI validation refined emitter count from {orig_emitters} to {refined_emitters}")
                labels = refined_labels
                
                # Save PRI statistics
                import json
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

                with open(os.path.join(ns.output, "emitter_stats.json"), 'w') as f:
                    serializable_stats = {str(int(emitter)): make_json_safe(stats) for emitter, stats in emitter_stats.items()}
                    json.dump(serializable_stats, f, indent=2)
        except Exception as e:
            print(f"WARNING: PRI validation failed: {e}")
            print(f"Traceback: {sys.exc_info()[2]}")

if __name__ == "__main__":
    main()
