#!/usr/bin/env bash

set -e

PROJECT_ROOT="$(pwd)"
CONFIG_DIR="$PROJECT_ROOT/config"
SCRIPTS_DIR="$PROJECT_ROOT/scripts"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments"

SRC_DIR="$PROJECT_ROOT/src"

export PROJECT_ROOT
export CONFIG_DIR
export SCRIPTS_DIR
export SRC_DIR

function usage() {
    echo "Usage: $0 [-h] [-l] [-a remote-address] [-s] [experiment1 ...]"
    echo "  -h                  Display this message"
    echo "  -l                  Run locally instead of on HPC"
    echo "  -a remote-address   Specify the remote address to run these experiments on (triggers remote execution via SSH)"
    echo "  -s                  Sync local changes to remote using rsync before remote execution (requires -a)"
}

function run_remote() {
    local remote_host="${HPC_HOSTNAME:-turing.wpi.edu}"
    local remote_user="${HPC_USERNAME:-$USER}"
    local remote_dir="${REMOTE_PROJECT_DIR}"
    local remote_key="${HPC_SSH_KEY}"

    if [ -z "$remote_dir" ]; then
        echo "Error: REMOTE_PROJECT_DIR is not set in config/defaults.env or config/config.env"
        exit 1
    fi

    if [ -z "$remote_key" ]; then
        echo "Error: HPC_SSH_KEY is not set in config/defaults.env or config/config.env"
        exit 1
    fi

    if [ "$SYNC_REMOTE" = true ]; then
        echo "Syncing local changes to $remote_user@$remote_host:$remote_dir..."
        rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'data' ./ "$remote_user@$remote_host:$remote_dir/"
    fi

    echo "Running remotely on $remote_user@$remote_host in $remote_dir"
    local run_cmd="cd $remote_dir && ./run.sh ${TARGET_EXPERIMENTS[*]}"
    ssh -i "$remote_key" "$remote_user@$remote_host" "$run_cmd"
}

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
}

function main() {
    # Load configs immediately to make sure variables like REMOTE_PROJECT_DIR are available
    if [ -f "$CONFIG_DIR/defaults.env" ]; then
        set -a; source "$CONFIG_DIR/defaults.env"; set +a
    fi
    if [ -f "$CONFIG_DIR/config.env" ]; then
        set -a; source "$CONFIG_DIR/config.env"; set +a
    fi

    RUN_REMOTE=false
    SYNC_REMOTE=false
    while getopts ":hla:s" opt; do 
        case "${opt}" in 
            h)
                usage
                exit 0
                ;;
            a)
                HPC_HOSTNAME="${OPTARG}"
                RUN_REMOTE=true
                ;;
            l)  
                RUN_LOCAL=true
                ;;
            s)
                SYNC_REMOTE=true
                ;;
            \?)
                echo "Error: Invalid option -${OPTARG}" >&2
                usage
                exit 1
                ;;
            :)
                echo "Error: Option -${OPTARG} requires an argument." >&2
                usage
                exit 1
                ;;
        esac
    done

    shift $((OPTIND-1))
    TARGET_EXPERIMENTS=("$@")

    if [ "$RUN_REMOTE" = true ]; then
        run_remote
    elif [ "$RUN_LOCAL" = true ]; then 
        run_local 
    else 
        run_on_hpc 
    fi
}

main "$@"

