#!/usr/bin/env bash
set -euo pipefail

# One isolated fixed-QMC shard for veMLP.  Every shard has its own output
# directory and seed manifest so several single-GPU jobs can run concurrently
# without sharing mutable state.
ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
CONFIG="$ROOT/configs/paper/synthetic_main_validated_fixed_order14_j2_200.yaml"
PREPARED="$ROOT/experiments/paper/oracle_artifact.pkl"
CERTIFICATE="$ROOT/certificates/fixed_cached_equivalence_order14.json"
BUDGET="${1:?budget required}"
REP_START="${2:?rep start required}"
REP_END="${3:?rep end required}"
SHARD="${4:?shard id required}"
OUT="$ROOT/results/shards/shard_${SHARD}"

mkdir -p "$OUT"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT/pydeps:$ROOT/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "$PY" "$ROOT/scripts/run_synthetic_main.py" \
  --config "$CONFIG" \
  --prepared "$PREPARED" \
  --budget "$BUDGET" \
  --rep-range "$REP_START" "$REP_END" \
  --output-path "$OUT" \
  --seed-manifest "$OUT/seed_manifest.json" \
  --certificate "$CERTIFICATE" \
  --resume --profile
