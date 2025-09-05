# How Adaptive Parameters Work in Your Radar Deinterleaving Pipeline

## Purpose

Adaptive parameters automatically set clustering settings (like `min_cluster_size`, `min_samples`, and `cluster_selection_epsilon`) based on the size and dimensionality of your data, so you don't have to manually tune them for each dataset.

---

## How It Works

### 1. **Data Size and Feature Dimensionality**

- The pipeline calculates:
  - `data_size`: Number of pulses after filtering.
  - `feature_dim`: Number of features used for clustering (e.g., CF, PW, Amp).

### 2. **Parameter Calculation**

- **min_cluster_size**:  
  - Scales with data size and feature dimension.
  - Formula:  
    - For small datasets: proportional to data size and feature dimension.
    - For large datasets: logarithmic scaling to avoid excessive cluster size.
    - Example:  
      ```python
      min_cluster_by_size = max(dim_factor, int(data_size * 0.01))
      ```
- **min_samples**:  
  - Ensures enough points for core cluster density.
  - Formula:  
    - Based on feature dimension and a fraction of data size.
    - Example:  
      ```python
      min_samples_by_size = max(feature_dim + 1, int(data_size * 0.005))
      ```
- **cluster_selection_epsilon**:  
  - Controls how tightly clusters are formed.
  - Formula:  
    - Scales with feature dimension and data density.
    - Example:  
      ```python
      base_epsilon = 0.05 + (0.02 * feature_dim)
      density_factor = np.power(data_size, -1/feature_dim)
      cluster_selection_epsilon = base_epsilon / max(0.01, density_factor)
      cluster_selection_epsilon = min(0.3, max(0.05, cluster_selection_epsilon))
      ```

### 3. **Practical Constraints**

- For very small datasets: Looser constraints to allow clustering.
- For very large datasets: Stricter constraints to avoid too many clusters.

### 4. **Automatic Adjustment**

- The pipeline chooses these values **dynamically** for each run, based on the actual data loaded.
- You can override them manually if needed.

---

## Why Is This Useful?

- **No manual tuning required** for each dataset.
- **Scales automatically** with your data.
- **Reduces risk** of poor clustering due to bad parameter choices.
- **Makes pipeline robust** for both small and large datasets.

---

## Where Is It Implemented?

- In `scripts/clustering.py`, inside the block:
  ```python
  if ns.adaptive_params:
      # ...adaptive parameter logic...
  ```

---

## Summary

Adaptive parameters make your clustering step smarter and more robust by automatically setting key values based on your data, improving usability and reliability for radar deinterleaving.
