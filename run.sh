#!/bin/bash
# Simple script to run the radar deinterleaving pipeline

set -e  # Exit on error

# Default values
IQ_FILE="data/raw_iq/radseg_iq.hdf5"
OUTPUT_DIR="results"
LIMIT=50
FS=20000000
REDUCE_DIM=false
ADAPTIVE=false
MAX_CLUSTERS=25
ADVANCED=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --iq)
      IQ_FILE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --fs)
      FS="$2"
      shift 2
      ;;
    --reduce-dim)
      REDUCE_DIM=true
      shift
      ;;
    --adaptive)
      ADAPTIVE=true
      shift
      ;;
    --max-clusters)
      MAX_CLUSTERS="$2"
      shift 2
      ;;
    --advanced)
      ADVANCED=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p data/raw_iq

# Set PYTHONPATH
export PYTHONPATH=.

# Build command with optional arguments
CMD="python main.py --iq \"$IQ_FILE\" --output \"$OUTPUT_DIR\" --limit \"$LIMIT\" --fs \"$FS\""

if [ "$ADAPTIVE" = true ]; then
  CMD="$CMD --adaptive_params"
fi

if [ "$REDUCE_DIM" = true ]; then
  CMD="$CMD --reduce_dim"
fi

CMD="$CMD --max_clusters $MAX_CLUSTERS"

if [ "$ADVANCED" = true ]; then
  CMD="$CMD --advanced"
fi

# Run the pipeline
echo "Running radar deinterleaving pipeline..."
echo "Command: $CMD"
eval "$CMD"

echo "Pipeline completed successfully!"
echo "Results available in: $OUTPUT_DIR"
echo "Clustering visualization: $OUTPUT_DIR/clustering/clustering_visualization.png"
echo "Evaluation report: $OUTPUT_DIR/evaluation/evaluation_report.txt"
echo "PRI analysis: $OUTPUT_DIR/evaluation/frequency_hopping/fh_report.json"
echo "Staggered PRI analysis: $OUTPUT_DIR/evaluation/pri_analysis/staggered_report.json"
echo "Temporal analysis: $OUTPUT_DIR/evaluation/temporal_analysis/"
