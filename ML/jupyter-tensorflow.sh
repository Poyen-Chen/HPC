#!/usr/bin/env zsh

# Load modules required for NVIDIA GPUs
module purge
module load DEVELOP intel
module load cuda/11.0
module load cudnn/8.0.5

# Load module for Singularity container
module load CONTAINERS tensorflow/nvcr-22.01-tf2-py3

singularity exec -B $(pwd)/exercises/:/project/ \
    -e --no-home --home /project --nv ${R_CONTAINER} \
    bash -c "export MPLCONFIGDIR=~/.config/matplotlib && cd ~; jupyter-lab --NotebookApp.custom_display_url=''"
