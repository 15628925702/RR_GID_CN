#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
BUDGET="${1:?budget required}"
REP_START="${2:?rep start required}"
REP_END="${3:?rep end required}"
SHARD="${4:?shard id required}"
OUT="$ROOT/results/e5_r1/shards/shard_${SHARD}"
mkdir -p "$OUT"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT/pydeps:$ROOT/src:$ROOT/scripts"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
exec "$PY" "$ROOT/scripts/run_gas_semisynthetic.py" \
  --config "$ROOT/configs/paper/gas_semisynthetic_empirical_cloud.yaml" \
  --data "$ROOT/data/gas/processed/gas_processed.npz" \
  --budget "$BUDGET" \
  --rep-range "$REP_START" "$REP_END" \
  --output-path "$OUT" \
  --resume --profile
