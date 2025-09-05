#!/bin/bash
set -e

# Set your input IQ file and output directory
IQ_FILE="data/raw_iq/radseg_iq.hdf5"
OUTPUT_DIR="results"
LIMIT=6000
FS=20000000

# Check if required directories exist
mkdir -p data/raw_iq
mkdir -p "$OUTPUT_DIR"

# Check if input file exists
if [ ! -f "$IQ_FILE" ]; then
    echo "ERROR: Input file $IQ_FILE not found."
    echo "Please place your IQ data file in data/raw_iq/ directory."
    exit 1
fi

# Display execution parameters
echo "==== Radar Deinterleaving Pipeline ===="
echo "Input file: $IQ_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Sample limit: $LIMIT"
echo "Sampling frequency: $FS Hz"
echo "====================================="

# Run the full pipeline with enhanced features
# Remove timeout for macOS compatibility
export PYTHONPATH=$(pwd)
python -m main \
  --iq "$IQ_FILE" \
  --output "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --fs "$FS" \
  --adaptive_params \
  --max_clusters 25 \
  --reduce_dim \
  --advanced

# Check if pipeline completed successfully
if [ $? -ne 0 ]; then
    echo "ERROR: Pipeline execution failed."
    exit 1
fi

# Print where to find results
echo ""
echo "Pipeline completed successfully!"
echo "Results are available at:"
echo "- Clustering visualization: $OUTPUT_DIR/clustering/clustering_visualization.png"
echo "- Evaluation report: $OUTPUT_DIR/evaluation/evaluation_report.txt"
echo "- PRI analysis: $OUTPUT_DIR/clustering/pri_analysis/"
echo "- Temporal analysis: $OUTPUT_DIR/evaluation/temporal_analysis/"
echo ""
echo "Pipeline completed successfully!"
echo "Results are available at:"
echo "- Clustering visualization: $OUTPUT_DIR/clustering/clustering_visualization.png"
echo "- Evaluation report: $OUTPUT_DIR/evaluation/evaluation_report.txt"
echo "- PRI analysis: $OUTPUT_DIR/clustering/pri_analysis/"
echo "- Temporal analysis: $OUTPUT_DIR/evaluation/temporal_analysis/"
