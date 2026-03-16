#!/usr/bin/env bash
# Evaluate agent.py on τ²-bench test set (airline + retail + telecom = 100 tasks).
# Prints accuracy summary at the end.
set -euo pipefail

cd "$(dirname "$0")/.."
python eval/run_eval.py
