# Radar Deinterleaving Pipeline — Technical Documentation

## Overview

Radar deinterleaving is the process of separating pulses from multiple emitters in a mixed signal environment. This project implements a modular pipeline for pulse extraction, feature engineering, clustering, and evaluation.

---

## Pipeline Steps

### 1. Preprocessing (`scripts/preprocess.py`)
- **Input:** Raw IQ data (complex64, HDF5 format).
- **Process:** Detects pulses using thresholding/spectrogram, extracts features:
    - TOA (Time of Arrival)
    - CF (Carrier Frequency)
    - PW (Pulse Width)
    - AOA (Angle of Arrival, placeholder)
    - Amplitude
- **Output:** PDW array (Pulse Descriptor Word), saved as HDF5/CSV/NPY.

### 2. Feature Filtering (`utils/data_processing.py`)
- **Removes:** Pulses with CF=0 (noise/artifacts).
- **Outlier Removal:** Uses percentile-based filtering on CF, PW, Amplitude.

### 3. Clustering (`scripts/clustering.py`)
- **Algorithm:** HDBSCAN (Hierarchical Density-Based Spatial Clustering).
- **Features:** Selectable (default: CF, PW, Amplitude).
- **Scaling:** RobustScaler for outlier resistance.
- **Parameters:** `min_cluster_size`, `min_samples`, `cluster_selection_epsilon`.
- **Output:** Cluster labels (NPY), summary, visualization (UMAP).

### 4. Evaluation (`scripts/evaluate.py`)
- **Metrics:** Silhouette score, purity, completeness, cluster sizes.
- **Emitter Analysis:** Per-emitter statistics (mean, std, min, max for features and PRI).
- **Visualizations:**
    - UMAP 2D/3D scatter plots (color and shape for clusters/noise)
    - Cluster size bar plot
    - Feature boxplots per cluster
    - Pairplot of features
    - PRI histograms per emitter

### 5. Model Training (`scripts/train.py`)
- **Model:** Transformer encoder for pulse embeddings.
- **Losses:** Triplet, contrastive, temporal consistency.
- **Config:** Training parameters in `config/train.yaml`.

---

## Configuration Files

- **`config/config.yaml`:** Main pipeline, model, clustering, and evaluation settings.
- **`config/train.yaml`:** Training-specific parameters.

---

## Output Files

- **PDW Data:** `data/pdw/dataset.hdf5`
- **Cluster Labels:** `results/clustering/cluster_labels.npy`
- **Evaluation Report:** `results/evaluation/evaluation_report.txt`
- **Visualizations:** PNG files in `results/clustering/` and `results/evaluation/`

---

## How It Works

1. **Extract pulses** from raw IQ data and compute features.
2. **Filter and scale features** for robust clustering.
3. **Cluster pulses** into emitters using HDBSCAN.
4. **Evaluate clustering** with metrics and visualizations.
5. **(Optional) Train transformer model** for advanced feature embeddings.

---

## Extending the Pipeline

- **Feature Engineering:** Add SNR, bandwidth, or other features.
- **Model:** Use transformer embeddings for clustering.
- **Clustering:** Try other algorithms (GMM, hierarchical).
- **Evaluation:** Add more metrics or export formats.

---

## Troubleshooting

- **Data size mismatch:** Ensure filtering is consistent in clustering and evaluation.
- **High noise ratio:** Tune clustering parameters or outlier filtering.
- **Visualization errors:** Check palette and marker mapping for clusters/noise.

---

## Contact

For questions or contributions, please contact the project maintainer.

Likhit -mail at 22ee01013@iitbbs.ac.in
Thanay - 22ee01007@iitbbs.ac.in
