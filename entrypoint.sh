#!/bin/bash
set -e

# Activate Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate photoens

# Run the main command
exec "$@"