#!/usr/bin/env zsh

# Load modules required for NVIDIA GPUs
module purge
module load DEVELOP intel
module load cuda/11.0 # just used for e.g., cuML

# Load module for Singularity container
module load CONTAINERS rapids/nvcr-22.02-cuda11.0-runtime

singularity exec -B $(pwd)/exercises/:/project/ \
    -e --no-home --home /project --nv ${R_CONTAINER} \
    bash -c "conda init && source ~/.bashrc && conda activate rapids && export MPLCONFIGDIR=~/.config/matplotlib && cd ~; jupyter lab --ip='0.0.0.0'"
