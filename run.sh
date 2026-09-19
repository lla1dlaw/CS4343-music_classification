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
USER_CONFIG="$CONFIG_DIR/$USER_CONFIG_FILENAME"

export PROJECT_ROOT
export CONFIG_DIR
export SCRIPTS_DIR
export SRC_DIR

function usage() {
    echo "Usage: $0 [-h] [-a remote-address]"
    echo "  -h                  Display this message"
    echo "  -a remote-address   Specify the remote address to run these experiments on"
}

function setup-env() {
    module load uv 
    
    # load configs
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

function run_on_hpc() {
    if ! [ -d "$EXPERIMENTS_DIR" ]; then 
        echo "No directory '$EXPERIMENTS_DIR' found."
        exit 1
    fi

    setup-env
    
    # loop over experiment directories and run their slurm scripts
    shopt -s nullglob
    for dir in $EXPERIMENTS_DIR; do 
        sbatch "${dir%/}/run_experiment.sh"
    done
}

function run_local() {
    if ! [ -d "$EXPERIMENTS_DIR" ]; then 
        echo "No directory '$EXPERIMENTS_DIR' found."
        exit 1
    fi

    setup-env
}

function main() {
    while getopts ":hla:" opt; do 
        case "${opt}" in 
            h)
                usage
                ;;
            a)
                HPC_HOSTNAME="${OPTARG}"
                ;;

            l)  
                RUN_LOCAL=true
                ;;
            \?)
                echo "Error: Invalid option -${OPTARG}" >&2
                usage
                ;;
            :)
                echo "Error: Option -${OPTARG} requires an argument." >&2
                usage
                ;;
        esac
    done

    if RUN_LOCAL; then 
        run_local 
    else 
        run_on_hpc 
    fi
}

