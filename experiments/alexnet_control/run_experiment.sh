#!/usr/bin/env bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 4
#SBATCH --mem=8g
#SBATCH --job-name "Example GPU Job"
#SBATCH --partition short
#SBATCH --time 0-2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="A30"
#SBATCH --output logs/%j.out

# source and setup the runtime
source ../hpc_scripts/experiment_setup.sh
setup-env 

# run main entrypoint
uv run ./src/main.py
