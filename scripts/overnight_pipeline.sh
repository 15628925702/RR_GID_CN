#!/usr/bin/env bash
# Unattended overnight pipeline for RR-GID_CN.
#
# Runs every remaining gate and diagnostic in dependency order, one GPU process
# at a time, and is safe to relaunch: each phase skips work already present on
# disk.  Every phase appends to the master log and to a machine-readable status
# file so progress can be read without inspecting the terminal.
#
# Usage: bash scripts/overnight_pipeline.sh
set -u

PY="G:/Anaconda/envs/dermagent-xh/python.exe"
ROOT="G:/0-newResearch/4.RR_GID_CN"
GATE="H:/RR_GID_CN_data/active/risk_gates"
MASTER="${GATE}/OVERNIGHT_PIPELINE_20260910.log"
STATUS="${GATE}/OVERNIGHT_STATUS_20260910.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${ROOT}/src"

cd "${ROOT}" || exit 1
mkdir -p "${GATE}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

phase() {
  PHASE_NAME="$1"
  log "=================================================================="
  log "PHASE START: ${PHASE_NAME}"
  PHASE_START=$(date +%s)
}

phase_end() {
  local rc=$1
  local secs=$(( $(date +%s) - PHASE_START ))
  log "PHASE END: ${PHASE_NAME} rc=${rc} elapsed=${secs}s"
  printf '%s\t%s\t%s\t%s\n' "${PHASE_NAME}" "${rc}" "${secs}" "$(date -Is)" >> "${GATE}/overnight_phase_results.tsv"
}

# Skip a phase entirely when its completion marker already exists.
done_marker() { [ -s "$1" ]; }

RUN8000="${GATE}/risk1_order14_10rep"
RUN2000="${GATE}/risk1_order14_10rep_b2000"
RUN32000="${GATE}/risk1_order14_10rep_b32000"
Q="${GATE}/risk1_query_200.json"
I="${GATE}/risk1_information_20x5.json"

log "pipeline boot: pid=$$"

########################################################################
phase "1. RISK-1 B=8000 adaptive gold (10 paired replications)"
if [ "$(ls "${RUN8000}"/gold_order18_rep*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  log "already complete, skipping"
  phase_end 0
else
  bash scripts/run_risk1_budget.sh gold 8000 "${RUN8000}" 0 9 >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "2. RISK-1 B=8000 fixed_qmc fast (10 paired replications)"
if [ "$(ls "${RUN8000}"/fast_order14_rep*.json 2>/dev/null | wc -l)" -ge 10 ]; then
  log "already complete, skipping"
  phase_end 0
else
  bash scripts/run_risk1_budget.sh fast 8000 "${RUN8000}" 0 9 >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "3. RISK-1 B=8000 assemble certificate"
if done_marker "${RUN8000}/report.json"; then
  log "already complete, skipping"
  phase_end 0
else
  "${PY}" scripts/summarize_risk1_order14_10rep.py \
    --run-dir "${RUN8000}" --query "${Q}" --information "${I}" \
    --replications 10 --budget 8000 --score-order 14 \
    --information-order 12 --adaptive-max-order 18 \
    >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "4. RISK-2 generator information fidelity"
if done_marker "${GATE}/risk2_generator_information_v1/report.json"; then
  log "already complete, skipping"
  phase_end 0
else
  "${PY}" scripts/validate_generator_information.py \
    --config configs/validation/generator_information_v1.yaml \
    --scope all --device cuda >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "5. RISK-3 Gas secondary-metric gate"
if done_marker "${GATE}/risk3_gas_secondary_v1/report.json"; then
  log "already complete, skipping"
  phase_end 0
else
  "${PY}" scripts/run_gas_secondary_gate.py \
    --config configs/validation/gas_secondary_v1.yaml >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "6. E1 score-order diagnostic (B=8000, score order 12 vs 8)"
if done_marker "${GATE}/e1_order12_score_diag/summary.json"; then
  log "already complete, skipping"
  phase_end 0
else
  "${PY}" scripts/run_synthetic_main.py \
    --config configs/validation/e1_order12_score_diag_v1.yaml --resume \
    >> "${MASTER}" 2>&1
  RC=$?
  if [ "${RC}" -eq 0 ]; then
    "${PY}" - "$GATE" <<'PYEOF' >> "${MASTER}" 2>&1
import json, sys, statistics, collections
from pathlib import Path
gate = Path(sys.argv[1])
rows = [json.loads(l) for l in
        (gate / "e1_order12_score_diag/rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = collections.defaultdict(list)
for r in rows:
    by[r["method"]].append(r["risk_ratio_raw"])
summary = {m: {"n": len(v), "mean_risk_ratio": statistics.mean(v)} for m, v in sorted(by.items())}
# reference: the frozen order-8/6 E1 stage at the same budget
ref_path = Path("H:/RR_GID_CN_data/active/paper_runs/synthetic_main_b8000_v1/rows.jsonl")
ref = [json.loads(l) for l in ref_path.read_text(encoding="utf-8").splitlines() if l.strip()]
ref_by = collections.defaultdict(list)
for r in ref:
    ref_by[r["method"]].append(r["risk_ratio_raw"])
summary["_reference_order8_score6_info_b8000"] = {
    m: {"n": len(v), "mean_risk_ratio": statistics.mean(v)} for m, v in sorted(ref_by.items())}
(gate / "e1_order12_score_diag/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PYEOF
  fi
  phase_end "${RC}"
fi

########################################################################
phase "7. RISK-1 B=2000 (second planned budget)"
if [ "$(ls "${RUN2000}"/fast_order14_rep*.json 2>/dev/null | wc -l)" -ge 10 ] && [ -s "${RUN2000}/report.json" ]; then
  log "already complete, skipping"
  phase_end 0
else
  bash scripts/run_risk1_budget.sh gold 2000 "${RUN2000}" 0 9 >> "${MASTER}" 2>&1
  bash scripts/run_risk1_budget.sh fast 2000 "${RUN2000}" 0 9 >> "${MASTER}" 2>&1
  "${PY}" scripts/summarize_risk1_order14_10rep.py \
    --run-dir "${RUN2000}" --query "${Q}" --information "${I}" \
    --replications 10 --budget 2000 --score-order 14 \
    --information-order 12 --adaptive-max-order 18 \
    >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
phase "8. RISK-1 B=32000 (third planned budget)"
if [ "$(ls "${RUN32000}"/fast_order14_rep*.json 2>/dev/null | wc -l)" -ge 10 ] && [ -s "${RUN32000}/report.json" ]; then
  log "already complete, skipping"
  phase_end 0
else
  bash scripts/run_risk1_budget.sh gold 32000 "${RUN32000}" 0 9 >> "${MASTER}" 2>&1
  bash scripts/run_risk1_budget.sh fast 32000 "${RUN32000}" 0 9 >> "${MASTER}" 2>&1
  "${PY}" scripts/summarize_risk1_order14_10rep.py \
    --run-dir "${RUN32000}" --query "${Q}" --information "${I}" \
    --replications 10 --budget 32000 --score-order 14 \
    --information-order 12 --adaptive-max-order 18 \
    >> "${MASTER}" 2>&1
  phase_end $?
fi

########################################################################
log "PIPELINE COMPLETE $(date -Is)"
"${PY}" - "${GATE}" "${STATUS}" <<'PYEOF' >> "${MASTER}" 2>&1
import json, sys
from pathlib import Path
gate, status_path = Path(sys.argv[1]), Path(sys.argv[2])
tsv = gate / "overnight_phase_results.tsv"
phases = []
if tsv.exists():
    for line in tsv.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            phases.append({"phase": parts[0], "rc": int(parts[1]),
                           "seconds": int(parts[2]), "finished_at": parts[3]})
def read(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
payload = {
    "schema_version": "overnight-status-v1",
    "phases": phases,
    "risk1_b8000": read(gate / "risk1_order14_10rep/report.json"),
    "risk1_b8000_passed": (read(gate / "risk1_order14_10rep/report.json") or {}).get("passed"),
    "risk2": read(gate / "risk2_generator_information_v1/report.json"),
    "risk2_passed": (read(gate / "risk2_generator_information_v1/report.json") or {}).get("passed"),
    "risk3": read(gate / "risk3_gas_secondary_v1/report.json"),
    "risk3_passed": (read(gate / "risk3_gas_secondary_v1/report.json") or {}).get("passed"),
    "e1_order12_diag": read(gate / "e1_order12_score_diag/summary.json"),
    "risk1_b2000_passed": (read(gate / "risk1_order14_10rep_b2000/report.json") or {}).get("passed"),
    "risk1_b32000_passed": (read(gate / "risk1_order14_10rep_b32000/report.json") or {}).get("passed"),
}
status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in payload.items() if k.endswith("_passed") or k == "phases"}, indent=2))
PYEOF
