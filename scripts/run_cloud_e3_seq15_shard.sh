#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
SEQ_START="${1:?sequence start required}"
SEQ_END="${2:?sequence end required}"
SHARD="${3:?shard id required}"
OUT="$ROOT/results/e3_seq15/shards/shard_${SHARD}"
mkdir -p "$OUT"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
export PYTHONPATH="$ROOT/pydeps:$ROOT/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
exec "$PY" "$ROOT/scripts/run_reuse.py" \
  --config "$ROOT/configs/paper/reuse_order14_t50_seq15_cloud.yaml" \
  --prepared "$ROOT/experiments/paper/oracle_artifact.pkl" \
  --checkpoint "$ROOT/checkpoints/vaeac_reference.pt" \
  --sequence-range "$SEQ_START" "$SEQ_END" \
  --output-path "$OUT" \
  --resume --profile
