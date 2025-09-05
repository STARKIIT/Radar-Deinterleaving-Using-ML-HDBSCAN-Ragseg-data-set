# Radar Pulse Deinterleaving System

A state-of-the-art signal processing pipeline for radar pulse deinterleaving using density-based clustering with adaptive parameters.

## Overview

This project implements a complete pipeline for radar pulse deinterleaving:

1. **Preprocessing**: Extract Pulse Descriptor Words (PDWs) from raw IQ data
2. **Clustering**: Group pulses by emitter using adaptive HDBSCAN clustering
3. **Validation**: Verify emitter consistency using PRI analysis
4. **Evaluation**: Generate reports and visualizations of identified emitters

## Installation

1. Create a conda environment (recommended):

    ```bash
    conda create -n radar_env python=3.10
    conda activate radar_env
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Quick Start

```bash
./run_pipeline.sh
```

This will run the full pipeline with default settings. Results will be saved to the `results/` directory.

### Custom Parameters

```bash
./run.sh --iq data/raw_iq/your_dataset.hdf5 --output custom_results --limit 200 --reduce-dim --adaptive
```

### Manual Execution

You can also run each step of the pipeline individually:

```bash
# 1. Extract PDWs from IQ data
python scripts/preprocess.py --iq data/raw_iq/radseg_iq.hdf5 --out data/pdw/dataset.hdf5 --format hdf5 --limit 100

# 2. Cluster PDWs into emitters
python scripts/clustering.py --pdw data/pdw/dataset.hdf5 --output results/clustering --visualize --adaptive_params --reduce_dim

# 3. Evaluate results
python scripts/evaluate.py --labels results/clustering/cluster_labels.npy --pdw data/pdw/dataset.hdf5 --output results/evaluation --plot
```

### run the cmd and debug and tune:
 # 1. preprocessing script : 
python scripts/preprocess.py --iq data/raw_iq/radseg_iq.hdf5 --out data/pdw/dataset.hdf5 --format hdf5 --limit 1000 --normalize
 # 2. clustering script :
python scripts/clustering.py --pdw data/pdw/dataset.hdf5 --output results/clustering \
  --min_cluster_size 50 --cluster_selection_epsilon 0.05 --outlier_percentile 10 --features CF PW Amp --visualize
  # 3. evalutae script :
  python scripts/evaluate.py --labels results/clustering/cluster_labels.npy --pdw data/pdw/dataset.hdf5 --output results/evaluation --plot
# one line piple cmd: 
./run.sh --iq data/raw_iq/radseg_iq.hdf5 --output results --limit 1000 --fs 20000000 --max-clusters 25 --reduce-dim --adaptive --advanced

# To run the entire pipeline :
chmod +x run_pipeline.sh #to build the file after changes
./run_pipeline.sh #to run the recently build file


## Advanced Features

- **Dimensionality Reduction**: Apply PCA/UMAP before clustering with `--reduce_dim`
- **Adaptive Parameters**: Automatically tune HDBSCAN parameters with `--adaptive_params`
- **PRI Validation**: Verify emitter consistency based on Pulse Repetition Intervals
- **Temporal Analysis**: Analyze temporal patterns and staggered PRIs
- **Advanced Radar Mode Analysis**: Optional, enabled with `--advanced` flag:
    - Frequency hopping detection
    - Enhanced staggered PRI detection

## Error Handling

- The pipeline checks for missing files, empty data, and failed steps.
- Advanced analyses handle short or constant sequences gracefully.
- All errors are reported with clear messages in the console and logs.

## Hardware Requirements

- **Minimum**: 8GB RAM, dual-core CPU
- **Recommended**: 16GB RAM, quad-core CPU
- **For large datasets**: 32GB RAM, SSD storage

Processing speed depends on dataset size:
- Small datasets (<1000 pulses): ~30 seconds
- Medium datasets (1000-10000 pulses): 1-5 minutes
- Large datasets (>10000 pulses): 5-30 minutes

## Directory Structure

- `scripts/`: Python scripts for each pipeline stage
- `utils/`: Utility modules for data processing and analysis
- `model/`: Deep learning models for radar signal embedding (optional)
- `data/`: Directory for storing input/output data
- `results/`: Default output directory for clustering results

## License

MIT
