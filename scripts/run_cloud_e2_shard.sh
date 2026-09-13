#!/usr/bin/env bash
set -euo pipefail

ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
CONFIG="$ROOT/configs/paper/nonlinearity_order14_cloud.yaml"
ALPHA="${1:?alpha required}"
SHARD="${2:?shard id required}"
OUT="$ROOT/results/e2/shards/shard_${SHARD}"

mkdir -p "$OUT"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT/pydeps:$ROOT/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$PY" "$ROOT/scripts/run_nonlinearity.py" \
  --config "$CONFIG" \
  --alpha "$ALPHA" \
  --output-path "$OUT" \
  --artifact-storage-dir "$ROOT/experiments/paper" \
  --seed-manifest "$ROOT/results/e2/seed_manifest.json" \
  --resume --profile
