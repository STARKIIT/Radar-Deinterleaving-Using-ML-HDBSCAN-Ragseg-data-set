# High-Level Flow and Control of Radar Deinterleaving Project (Updated)

## 1. Data Engineering & Dataset Structure

### **Raw Data**
- **Location:** `data/raw_iq/radseg_iq.hdf5`
- **Format:** HDF5 file containing IQ (In-phase and Quadrature) radar signals.
- **Shape:** Typically `(num_signals, num_samples_per_signal)`; each signal is a time series of complex values.

### **Key Files for Data Engineering**
- **`scripts/preprocess.py`**: Extracts pulses (PDWs) from raw IQ signals.
  - **How:** For each signal, detects pulses using thresholding/spectrogram, computes features (TOA, CF, PW, AOA, Amplitude).
  - **Output:** PDW array (one row per pulse), saved as HDF5/CSV/NPY in `data/pdw/dataset.hdf5`.
- **`utils/pdw_io.py`**: Utility for loading/saving PDW data in various formats.
- **`utils/data_processing.py`**: Applies filtering to PDW data (removes CF=0, outliers).

---

## 2. Feature Engineering & Filtering

- **`utils/data_processing.py`**: 
  - **Removes**: Pulses with CF=0 (likely noise).
  - **Outlier Removal**: Uses percentile-based filtering on CF, PW, Amplitude.
  - **Importance:** Ensures only valid, representative pulses are used for clustering.

---

## 3. Clustering

### **Main File**
- **`scripts/clustering.py`**: Clusters PDWs into emitters using HDBSCAN.
  - **How:** 
    - Loads filtered PDW data.
    - Selects features (CF, PW, Amp, etc.).
    - Scales features using RobustScaler.
    - Optionally applies dimensionality reduction (UMAP/PCA).
    - Runs HDBSCAN with adaptive/manual parameters.
    - Optionally merges clusters if too many.
    - Validates clusters using PRI consistency.
  - **Output:** Cluster labels (`results/clustering/cluster_labels.npy`), visualizations (`clustering_visualization.png`), PRI stats (`emitter_stats.json`).

---

## 4. Post-Clustering Advanced Analysis

### **A. PRI Validation (Pulse Repetition Interval)**
- **File:** `scripts/clustering.py` (after HDBSCAN clustering)
- **Process:** For each detected emitter (cluster), analyze the sequence of pulse arrival times (TOA) to:
    - Compute PRI statistics (mean, std, jitter)
    - Detect staggered PRI patterns
    - Identify and remove inconsistent emitters (clusters with high PRI variance or non-radar-like patterns)
- **Output:** Refined cluster labels, PRI statistics (`emitter_stats.json`), and PRI analysis visualizations.

### **B. Temporal Analysis**
- **File:** `scripts/evaluate.py` (after clustering and PRI validation)
- **Process:** For each emitter:
    - Analyze pulse timing for gaps, missing pulses, and train duration
    - Detect temporal patterns (e.g., burst/staggered emission, large gaps)
    - Estimate missing pulses and train integrity
- **Output:** Temporal analysis report (`temporal_analysis/`), per-emitter timing statistics.

### **C. Advanced Radar Mode Analysis**
- **File:** `scripts/evaluate.py` (enabled with `--advanced`)
- **Process:** 
    - **Frequency Hopping Detection:** Analyze frequency changes within clusters to detect hopping patterns.
    - **Enhanced Staggered PRI Detection:** Identify complex PRI patterns beyond simple periodicity.
- **Output:** Frequency hopping report (`frequency_hopping/fh_report.json`), enhanced PRI report (`pri_analysis/staggered_report.json`).

---

## 5. Evaluation & Analysis

### **Main File**
- **`scripts/evaluate.py`**: Evaluates clustering results.
  - **How:** 
    - Loads filtered PDW data and cluster labels.
    - Computes metrics: silhouette score, purity, completeness, noise ratio.
    - Generates per-emitter statistics (mean, std, min, max for features and PRI).
    - Creates visualizations: UMAP plots, cluster size bar plots, feature boxplots, pairplots, PRI histograms.
    - Performs temporal analysis and advanced radar mode analysis (if enabled).
  - **Output:** Evaluation report (`evaluation_report.txt`), plots, advanced analysis results (`frequency_hopping/fh_report.json`, `pri_analysis/staggered_report.json`, `temporal_analysis/`).

---

## 6. Model Training (Optional)

- **`scripts/train.py`**: Trains a transformer encoder for pulse embeddings.
  - **How:** 
    - Loads PDW sequences and labels.
    - Creates overlapping sequences for training.
    - Trains transformer model with triplet/contrastive/temporal losses.
    - Saves best model checkpoint.
  - **Output:** Model weights (`models/transformer_encoder.pth`).

---

## 7. Utilities

- **`utils/inspect_h5.py`**: Inspects HDF5 files, prints structure, can convert datasets to CSV.
- **`parameter_search.py`**: Automates parameter tuning, logs results to Excel.

---

## 8. Configuration & Scripts

- **`config/config.yaml`**: Pipeline, model, clustering, and evaluation settings.
- **`config/train.yaml`**: Training-specific parameters.
- **`run_pipeline.sh` / `run.sh`**: Shell scripts to run the pipeline with default or custom parameters.
- **`main.py`**: Orchestrates the full pipeline (preprocessing, clustering, evaluation).

---

## 9. Control Flow (Pipeline Execution)

### **Flow Chart (Textual, Updated)**

1. **Start**
2. **Preprocessing** (`scripts/preprocess.py`)
    - Input: IQ data
    - Output: PDW data
3. **Feature Filtering** (`utils/data_processing.py`)
    - Input: PDW data
    - Output: Filtered PDW data
4. **Clustering** (`scripts/clustering.py`)
    - Input: Filtered PDW data
    - Output: Cluster labels, visualizations
5. **PRI Validation** (`scripts/clustering.py`)
    - Input: Cluster labels, PDW data
    - Output: Refined cluster labels, PRI stats
6. **Evaluation** (`scripts/evaluate.py`)
    - Input: Refined cluster labels, filtered PDW data
    - Output: Evaluation report, plots
7. **Temporal Analysis** (`scripts/evaluate.py`)
    - Input: Refined cluster labels, filtered PDW data
    - Output: Temporal analysis report
8. **Advanced Radar Mode Analysis** (`scripts/evaluate.py`, optional)
    - Input: Refined cluster labels, filtered PDW data
    - Output: Frequency hopping and enhanced PRI reports
9. **(Optional) Model Training** (`scripts/train.py`)
    - Input: PDW sequences, labels
    - Output: Model checkpoint
10. **End**

---

## 10. Results Generated at Each Step

### **After Clustering**
- **Cluster Labels:** Initial emitter assignments.
- **Visualizations:** Cluster separation plots.

### **After PRI Validation**
- **Refined Cluster Labels:** Emitters with consistent PRI retained.
- **PRI Stats:** Per-emitter PRI statistics and validation.

### **After Temporal Analysis**
- **Temporal Analysis Report:** Pulse train integrity, gaps, burst patterns.

### **After Advanced Analysis**
- **Frequency Hopping Report:** Detection of hopping emitters.
- **Enhanced PRI Report:** Complex PRI pattern identification.

### **After Evaluation**
- **Evaluation Report:** Metrics, per-emitter analysis, visualizations.

---

## 11. Why Each Result Is Important

- **PDW Data**: Enables pulse-level analysis and clustering.
- **Cluster Labels**: Core output for emitter identification.
- **Evaluation Metrics**: Quantify clustering quality, guide tuning.
- **Per-Emitter Analysis**: Understand emitter characteristics.
- **Visualizations**: Make results interpretable and actionable.
- **Advanced Analysis**: Detect complex radar behaviors.
- **Parameter Logs**: Enable reproducible, optimized pipeline runs.

---

## 12. Summary

This updated workflow clarifies that after HDBSCAN clustering, the pipeline performs advanced analyses (PRI validation, temporal analysis, and radar mode analysis) to refine emitter identification and provide deeper insight into radar signal behavior.
