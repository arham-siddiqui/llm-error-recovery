#!/usr/bin/env bash
# Run the v2 study one model at a time (concurrent models make Ollama evict/reload weights on a 24 GB machine).
# Usage: scripts/run_v2_all.sh [output_dir] [extra run_v2.py args...]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-results/v2}; shift || true
MODELS=(llama3.2-3b-instruct qwen3-vl-4b-instruct llama3.1-8b-instruct)
mkdir -p "$OUT/logs"
for m in "${MODELS[@]}"; do
  .venv/bin/python scripts/run_v2.py --config config_v2.yaml --output-dir "$OUT" --models "$m" "$@" \
    > "$OUT/logs/$m.log" 2>&1
done
echo "ALL DONE"
