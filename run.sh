#!/usr/bin/env bash

set -e

# set filenames for configs and entrypoints
DEFAULT_CONFIG_FILENAME="default.env"
USER_CONFIG_FILENAME="config.env" # values in this file override the defaults

PROJECT_ROOT="$(pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"

SRC_DIR="$PROJECT_ROOT/src"

DEFAULT_CONFIG="$CONFIG_DIR/$DEFAULT_CONFIG_FILENAME"
USER_CONFIG="$PROJECT_ROOT/$USER_CONFIG_FILENAME"

export PROJECT_ROOT
export CONFIG_DIR
export SCRIPTS_DIR
export SRC_DIR

function setup-env() {
    module load uv 
    
    # load config
    set -a
    source "$DEFAULT_CONFIG"
    set +a

    set -a
    source "$USER_CONFIG"
    set +a
    
    load modules
    module load cuda
    uv sync
}

function run_all_experiments() {
    if ! [ -d $EXPERIMENTS_DIR ]; then 
        echo "No directory '$EXPERIMENTS_DIR' found."
    fi
    
    # loop over experiment directories and run their slurm scripts
    shopt -s nullglob
    for dir in $EXPERIMENTS_DIR; do 
        sbatch "${dir%/}/run_experiment.sh"
    done
}

run_all_experiments
