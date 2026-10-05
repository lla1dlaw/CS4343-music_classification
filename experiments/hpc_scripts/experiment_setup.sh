#!/usr/bin/env bash

function setup-env() {
    # If variables aren't set by run.sh, set them relative to this script
    if [ -z "$PROJECT_ROOT" ]; then
        local SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
        export PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
        export CONFIG_DIR="$PROJECT_ROOT/config"
    fi

    module load uv
    module load cuda

    # load configs
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
    
    # Sync UV dependencies in the project root
    (cd "$PROJECT_ROOT" && uv sync)
}
