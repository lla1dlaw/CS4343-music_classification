#!/usr/bin/env bash

set -e

# Setup paths to config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"

# Load configs
if [ -f "$CONFIG_DIR/defaults.env" ]; then
    set -a
    source "$CONFIG_DIR/defaults.env"
    set +a
fi

if [ -f "$CONFIG_DIR/config.env" ]; then
    set -a
    source "$CONFIG_DIR/config.env"
    set +a
fi

# Default values
REMOTE_HOST="${HPC_HOSTNAME:-turing.wpi.edu}" 
REMOTE_USER="${HPC_USERNAME:-$USER}"
PROJECT_DIR="${REMOTE_PROJECT_DIR:-~/Projects/deep_learning/music_classification}"

function usage() {
    echo "Usage: $0 [options] [experiment1 experiment2 ...]"
    echo "Options:"
    echo "  -h              Show this help message"
    echo "  -u USER         Remote SSH user (default: $USER)"
    echo "  -H HOST         Remote SSH host (default: $REMOTE_HOST)"
    echo "  -d DIR          Remote project directory (default: $PROJECT_DIR)"
    echo "  -s              Sync local changes to remote using rsync before running"
}

SYNC=false

while getopts ":hu:H:d:s" opt; do
    case "${opt}" in
        h) usage; exit 0 ;;
        u) REMOTE_USER="${OPTARG}" ;;
        H) REMOTE_HOST="${OPTARG}" ;;
        d) PROJECT_DIR="${OPTARG}" ;;
        s) SYNC=true ;;
        \?) echo "Invalid option: -${OPTARG}"; usage; exit 1 ;;
    esac
done

shift $((OPTIND-1))
TARGET_EXPERIMENTS=("$@")

if [ "$SYNC" = true ]; then
    echo "Syncing local changes to $REMOTE_USER@$REMOTE_HOST:$PROJECT_DIR..."
    rsync -avz --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude 'data' ./ "$REMOTE_USER@$REMOTE_HOST:$PROJECT_DIR/"
fi

# Build the command to run remotely
# Note: we quote the target experiments to pass them properly to run.sh
RUN_CMD="cd $PROJECT_DIR && ./run.sh ${TARGET_EXPERIMENTS[*]}"

echo "Running on remote: $RUN_CMD"
ssh "$REMOTE_USER@$REMOTE_HOST" "$RUN_CMD"
