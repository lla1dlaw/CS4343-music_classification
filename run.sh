#!/usr/bin/env bash

set -e

# set filenames for configs and entrypoints
DEFAULT_CONFIG_FILENAME="defaults.env"
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

source "$EXPERIMENTS_DIR/hpc_scripts/experiment_setup.sh"

function run_on_hpc() {
    if ! [ -d "$EXPERIMENTS_DIR" ]; then 
        echo "No directory '$EXPERIMENTS_DIR' found."
        exit 1
    fi

    # loop over experiment directories and run their slurm scripts
    shopt -s nullglob
    local dirs=()
    if [ ${#TARGET_EXPERIMENTS[@]} -gt 0 ]; then
        for exp in "${TARGET_EXPERIMENTS[@]}"; do
            dirs+=("$EXPERIMENTS_DIR/$exp")
        done
    else
        dirs=("$EXPERIMENTS_DIR"/*)
    fi

    for dir in "${dirs[@]}"; do 
        if [ -d "$dir" ] && [ -f "$dir/run_experiment.sh" ]; then
            echo "Submitting experiment: $(basename "$dir")"
            (cd "$dir" && mkdir -p logs && sbatch run_experiment.sh)
        else
            if [ ${#TARGET_EXPERIMENTS[@]} -gt 0 ]; then
                echo "Warning: Experiment directory '$dir' or run_experiment.sh not found."
            fi
        fi
    done
}

function run_local() {
    if ! [ -d "$EXPERIMENTS_DIR" ]; then 
        echo "No directory '$EXPERIMENTS_DIR' found."
        exit 1
    fi

    # setup env logic is now sourced directly
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

    shift $((OPTIND-1))
    TARGET_EXPERIMENTS=("$@")

    if [ "$RUN_LOCAL" = true ]; then 
        run_local 
    else 
        run_on_hpc 
    fi
}

main "$@"

