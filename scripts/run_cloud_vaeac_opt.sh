#!/usr/bin/env bash
set -euo pipefail
ROOT="${CLOUD_ROOT:-/kairos_vepfs_volc/autodrive/manlichen/intern/ivan/rr_gid_cloud_20260912}"
PY="${CLOUD_PYTHON:-/root/miniconda3/bin/python}"
OUTDIR="$ROOT/results/opt/vaeac_gas_opt_v1"
mkdir -p "$OUTDIR"
export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= ALL_PROXY=
export PYTHONPATH="$ROOT/pydeps:$ROOT/src:$ROOT/scripts"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
exec "$PY" "$ROOT/scripts/train_gas_vaeac_opt.py" \
  --data "$ROOT/data/gas/processed/gas_processed.npz" \
  --output "$OUTDIR/vaeac_gas_opt_v1.pt" \
  --log "$OUTDIR/vaeac_gas_opt_v1.json" \
  --epochs 150 \
  --hidden 512 \
  --latent 64 \
  --batch 512 \
  --device cuda
