#!/bin/bash
set -e

# Activate Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate photoens

# Add the parent directory of src to PYTHONPATH
export PYTHONPATH="/app/rlenh:$PYTHONPATH"

# Run the main command
exec "$@"