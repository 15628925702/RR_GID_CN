#!/usr/bin/env bash
# RISK-1 order-14 10-replication driver.
#
# Runs the missing adaptive-gold and fixed_qmc fast replications for the
# frozen order-14 RISK-1 configuration, one GPU process at a time.
# Idempotent: a replication whose output JSON already exists is skipped.
#
# Usage: bash scripts/run_risk1_order14_10rep.sh gold|fast
set -u

MODE="${1:-gold}"
PY="G:/Anaconda/envs/dermagent-xh/python.exe"
ROOT="G:/0-newResearch/4.RR_GID_CN"
GATE="H:/RR_GID_CN_data/active/risk_gates"
RUN="${GATE}/risk1_order14_10rep"
CFG="configs/paper/oracle_calibration_risk1_order13_10rep.yaml"
PREPARED="experiments/paper/oracle_artifact.pkl"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${ROOT}/src"

cd "${ROOT}" || exit 1

for REP in $(seq "${2:-0}" "${3:-9}"); do
  case "${MODE}" in
    gold)
      OUT="${RUN}/gold_order18_rep${REP}.json"
      [ -s "${OUT}" ] && { echo "[skip] gold rep${REP} exists"; continue; }
      echo "[run ] gold rep${REP} $(date -Is)"
      "${PY}" scripts/run_gold_calibration_probe.py \
        --config "${CFG}" --prepared "${PREPARED}" \
        --replication "${REP}" --top-panels 20 --budget 8000 \
        --score-order 14 --info-order 12 \
        --max-order 18 --start-order 8 \
        --atol 2e-5 --rtol 2e-4 --scrambles 4 --chunk-rows 64 \
        --scoring-steps 2 --information-inner exact_adaptive \
        --out "${OUT}" \
        --basis-root "${RUN}/basis" \
        > "${RUN}/gold_rep${REP}.log" 2>&1
      RC=$?
      ;;
    fast)
      OUT="${RUN}/fast_order14_rep${REP}.json"
      [ -s "${OUT}" ] && { echo "[skip] fast rep${REP} exists"; continue; }
      echo "[run ] fast rep${REP} $(date -Is)"
      "${PY}" scripts/run_fast_calibration_probe.py \
        --config "${CFG}" --prepared "${PREPARED}" \
        --replication "${REP}" --top-panels 20 --budget 8000 \
        --score-order 14 --score-backend fixed_qmc --score-scrambles 1 \
        --info-order 12 --reuse-information-basis \
        --out "${OUT}" \
        --basis-root "${RUN}/basis" \
        > "${RUN}/fast_rep${REP}.log" 2>&1
      RC=$?
      ;;
    *) echo "unknown mode ${MODE}"; exit 2 ;;
  esac
  if [ "${RC}" -ne 0 ]; then
    echo "[FAIL] ${MODE} rep${REP} rc=${RC} $(date -Is)"
    tail -n 20 "${RUN}/${MODE}_rep${REP}.log" 2>/dev/null
  else
    echo "[done] ${MODE} rep${REP} rc=0 $(date -Is)"
  fi
done
echo "ALL DONE ${MODE} $(date -Is)"
