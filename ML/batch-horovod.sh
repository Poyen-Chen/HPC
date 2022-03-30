#!/usr/bin/env zsh

# Note:
#   - The reservation "ppces_gpu_22" is not available after the workshop anymore
#   - For Horovod we request a GPU machine exclusively to be able to distribute work on both GPUs

srun --time=01:00:00 --partition=c18g --ntasks=1 --cpus-per-task=48 --gres=gpu:volta:2 --reservation=ppces_gpu_22 zsh ./jupyter-tensorflow.sh
