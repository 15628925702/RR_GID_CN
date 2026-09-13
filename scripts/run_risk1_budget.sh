#!/usr/bin/env bash
# RISK-1 paired cached-QMC vs adaptive-gold calibration at one budget.
#
# Idempotent: a replication whose output JSON already exists is skipped, so the
# script can be relaunched after an interruption without losing completed work.
#
# Usage: bash scripts/run_risk1_budget.sh <gold|fast> <budget> <run-dir> [first] [last]
set -u

MODE="${1:?mode}"
BUDGET="${2:?budget}"
RUN="${3:?run-dir}"
FIRST="${4:-0}"
LAST="${5:-9}"

PY="${RISK1_PY:-G:/Anaconda/envs/dermagent-xh/python.exe}"
ROOT="G:/0-newResearch/4.RR_GID_CN"
CFG="configs/paper/oracle_calibration_risk1_order13_10rep.yaml"
PREPARED="experiments/paper/oracle_artifact.pkl"
SCORE_ORDER="${RISK1_SCORE_ORDER:-14}"
INFO_ORDER="${RISK1_INFO_ORDER:-12}"
MAX_ORDER="${RISK1_MAX_ORDER:-18}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${ROOT}/src"

cd "${ROOT}" || exit 1
mkdir -p "${RUN}"

for REP in $(seq "${FIRST}" "${LAST}"); do
  case "${MODE}" in
    gold) OUT="${RUN}/gold_order${MAX_ORDER}_rep${REP}.json" ;;
    fast) OUT="${RUN}/fast_order${SCORE_ORDER}_rep${REP}.json" ;;
    *) echo "unknown mode ${MODE}"; exit 2 ;;
  esac
  if [ -s "${OUT}" ]; then
    echo "[skip] ${MODE} B=${BUDGET} rep${REP}"
    continue
  fi
  echo "[run ] ${MODE} B=${BUDGET} rep${REP} $(date -Is)"
  ATTEMPT=0
  RC=1
  while [ "${ATTEMPT}" -lt 2 ] && [ "${RC}" -ne 0 ]; do
    ATTEMPT=$((ATTEMPT + 1))
    if [ "${MODE}" = "gold" ]; then
      "${PY}" scripts/run_gold_calibration_probe.py \
        --config "${CFG}" --prepared "${PREPARED}" \
        --replication "${REP}" --top-panels 20 --budget "${BUDGET}" \
        --score-order "${SCORE_ORDER}" --info-order "${INFO_ORDER}" \
        --max-order "${MAX_ORDER}" --start-order 8 \
        --atol 2e-5 --rtol 2e-4 --scrambles 4 --chunk-rows 64 \
        --scoring-steps 2 --information-inner exact_adaptive \
        --out "${OUT}" --basis-root "${RUN}/basis" \
        > "${RUN}/gold_rep${REP}.log" 2>&1
    else
      "${PY}" scripts/run_fast_calibration_probe.py \
        --config "${CFG}" --prepared "${PREPARED}" \
        --replication "${REP}" --top-panels 20 --budget "${BUDGET}" \
        --score-order "${SCORE_ORDER}" --score-backend fixed_qmc --score-scrambles 1 \
        --info-order "${INFO_ORDER}" --reuse-information-basis \
        --out "${OUT}" --basis-root "${RUN}/basis" \
        > "${RUN}/fast_rep${REP}.log" 2>&1
    fi
    RC=$?
    if [ "${RC}" -ne 0 ]; then
      echo "[warn] ${MODE} B=${BUDGET} rep${REP} attempt ${ATTEMPT} rc=${RC}"
      tail -n 5 "${RUN}/${MODE}_rep${REP}.log" 2>/dev/null
      sleep 20
    fi
  done
  if [ "${RC}" -ne 0 ]; then
    echo "[FAIL] ${MODE} B=${BUDGET} rep${REP} after ${ATTEMPT} attempts"
  else
    echo "[done] ${MODE} B=${BUDGET} rep${REP}"
  fi
done
echo "BUDGET DONE ${MODE} B=${BUDGET} $(date -Is)"
