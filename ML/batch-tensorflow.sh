#!/usr/bin/env zsh

# Note:
#   - --cpus-per-task depends on how many cores a system has and what fraction of GPUs is used on the system. Here we use half of the node.
#   - The reservation "ppces_gpu_22" is not available after the workshop anymore

srun --time=01:00:00 --partition=c18g --ntasks=1 --cpus-per-task=24 --gres=gpu:volta:1 --reservation=ppces_gpu_22 zsh ./jupyter-tensorflow.sh
