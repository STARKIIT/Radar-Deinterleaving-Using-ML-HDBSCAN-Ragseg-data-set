import itertools
import subprocess
import pandas as pd
import os

# This script implements a grid search for clustering parameters.
# It runs clustering and evaluation for each combination of:
#   - min_cluster_size
#   - cluster_selection_epsilon
#   - outlier_percentile
#   - features (CF, PW, Amp)
# After each run, it parses the evaluation report to extract:
#   - Detected emitters
#   - Noise points
#   - Silhouette score
# Results are saved to an Excel file for easy comparison.

# What does it do?
# - Automates the process of tuning clustering parameters.
# - Records the results (metrics and parameters) for each run.
# - Helps you find the best parameter set for your data.
# - Enables reproducible and systematic optimization.

# How does it work?
# 1. Iterates over all combinations of parameters.
# 2. Runs clustering and evaluation scripts for each combination.
# 3. Parses the evaluation report for key metrics.
# 4. Saves all results to 'parameter_search_results.xlsx'.

# Usage:
#   python parameter_search.py
# (Make sure your pipeline scripts and data paths are correct.)

# Define parameter grid
param_grid = {
    "min_cluster_size": [20, 50, 100],
    "cluster_selection_epsilon": [0.05, 0.1, 0.15],
    "outlier_percentile": [5, 10, 20],
    "features": [["CF", "PW"], ["CF", "PW", "Amp"]],
}

# Paths
pdw_file = "data/pdw/dataset.hdf5"
output_dir = "results/clustering"
eval_dir = "results/evaluation"
eval_report = os.path.join(eval_dir, "evaluation_report.txt")
labels_file = os.path.join(output_dir, "cluster_labels.npy")

results = []

# Iterate over all parameter combinations
for min_cluster_size, epsilon, outlier_percentile, features in itertools.product(
    param_grid["min_cluster_size"],
    param_grid["cluster_selection_epsilon"],
    param_grid["outlier_percentile"],
    param_grid["features"]
):
    # Build clustering command
    features_str = " ".join(features)
    clustering_cmd = (
        f"python scripts/clustering.py --pdw {pdw_file} --output {output_dir} "
        f"--min_cluster_size {min_cluster_size} --cluster_selection_epsilon {epsilon} "
        f"--outlier_percentile {outlier_percentile} --features {features_str} --visualize"
    )
    print(f"Running: {clustering_cmd}")
    subprocess.run(clustering_cmd, shell=True, check=True)

    # Run evaluation
    eval_cmd = (
        f"python scripts/evaluate.py --labels {labels_file} --pdw {pdw_file} --output {eval_dir} --plot"
    )
    print(f"Running: {eval_cmd}")
    subprocess.run(eval_cmd, shell=True, check=True)

    # Parse evaluation report for metrics
    with open(eval_report, "r") as f:
        lines = f.readlines()
    detected_emitters = None
    noise_points = None
    silhouette_score = None
    for line in lines:
        if "Detected Emitters:" in line:
            detected_emitters = int(line.split(":")[1].strip())
        if "Noise Points:" in line:
            noise_points = line.split(":")[1].strip().split(" ")[0]
        if "Silhouette Score:" in line:
            try:
                silhouette_score = float(line.split(":")[1].split("(")[0].strip())
            except:
                silhouette_score = None

    results.append({
        "min_cluster_size": min_cluster_size,
        "cluster_selection_epsilon": epsilon,
        "outlier_percentile": outlier_percentile,
        "features": features_str,
        "detected_emitters": detected_emitters,
        "noise_points": noise_points,
        "silhouette_score": silhouette_score,
    })

# Save results to Excel
df = pd.DataFrame(results)
df.to_excel("parameter_search_results.xlsx", index=False)
print("Parameter search complete. Results saved to parameter_search_results.xlsx")
