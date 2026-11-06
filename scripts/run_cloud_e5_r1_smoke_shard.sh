#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
CONFIG="${1:?config path required}"
BUDGET="${2:?budget required}"
REP_START="${3:?rep start required}"
REP_END="${4:?rep end required}"
OUT="${5:?output dir required}"
mkdir -p "$OUT"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT/pydeps:$ROOT/src:$ROOT/scripts"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
exec "$PY" "$ROOT/scripts/run_gas_semisynthetic.py" \
  --config "$CONFIG" \
  --data "$ROOT/data/gas/processed/gas_processed.npz" \
  --budget "$BUDGET" \
  --rep-range "$REP_START" "$REP_END" \
  --output-path "$OUT" \
  --resume --profile
