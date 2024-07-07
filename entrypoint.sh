#!/bin/bash

# Activate the Conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate your_env_name

# Execute the command provided as arguments to the script
exec "$@"
