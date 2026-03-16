#!/usr/bin/env bash
# Set up τ²-bench. Run once.
set -euo pipefail

if [ ! -d "tau2-bench" ]; then
    echo "Cloning τ²-bench..."
    git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
fi

echo "Installing τ²-bench..."
pip install -e tau2-bench/

echo "Done. Run: bash eval/eval.sh"
