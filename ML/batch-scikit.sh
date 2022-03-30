#!/usr/bin/env zsh

# Note:
#   - --cpus-per-task depends on how many cores a system has and what fraction of GPUs is used on the system. Here we use half of the node.
#   - The reservation "ppces_gpu_22" is not available after the workshop anymore
#   - For scikit-learn you usually don't need GPUs. Everything is running on the CPU. However, there are extensions like h2o4gpu or cuML that port some functions to a GPU.

srun --time=01:00:00 --partition=c18g --ntasks=1 --cpus-per-task=24 --gres=gpu:volta:1 --reservation=ppces_gpu_22 zsh ./jupyter-scikit.sh
