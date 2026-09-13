#!/usr/bin/env bash
# Watchdog for the unattended overnight pipeline.
#
# The orchestrator is one long-lived shell process; if the app that spawned it
# closes, the whole night is lost.  This loop relaunches it whenever neither
# the orchestrator nor any GPU worker is running, and exits once the pipeline
# reports completion.
#
# Relaunching is safe: every phase is idempotent and skips completed work, so a
# restarted orchestrator resumes instead of repeating.
set -u

ROOT="G:/0-newResearch/4.RR_GID_CN"
GATE="H:/RR_GID_CN_data/active/risk_gates"
MASTER="${GATE}/OVERNIGHT_PIPELINE_20260910.log"
WDLOG="${GATE}/OVERNIGHT_WATCHDOG_20260910.log"
STATUS="${GATE}/OVERNIGHT_STATUS_20260910.json"

log() { echo "[$(date -Is)] $*" >> "${WDLOG}"; }

# Print comma-joined PIDs matching a Win32_Process CommandLine filter.
pids() {
  powershell.exe -NoProfile -Command \
    "(Get-CimInstance Win32_Process -Filter \"Name='${1}'\") | Where-Object { \$_.CommandLine -match '${2}' } | Select-Object -ExpandProperty ProcessId" \
    2>/dev/null | tr -d '\r' | paste -sd, - | tr -d ' '
}

orchestrator_pids() { pids 'bash.exe' 'overnight_pipeline'; }
worker_pids() {
  pids 'python.exe' 'calibration_probe|validate_generator_information|gas_secondary|run_synthetic_main|summarize_risk1'
}

log "watchdog boot pid=$$"
QUIET=0
while true; do
  sleep 180
  if [ -s "${STATUS}" ] && grep -q "PIPELINE COMPLETE" "${MASTER}" 2>/dev/null; then
    log "pipeline complete; watchdog exiting"
    exit 0
  fi
  O=$(orchestrator_pids)
  W=$(worker_pids)
  if [ -n "${O}" ] || [ -n "${W}" ]; then
    QUIET=0
    log "alive: orchestrator=${O:-none} worker=${W:-none}"
    continue
  fi
  QUIET=$((QUIET + 1))
  log "quiet check ${QUIET}/2 (nothing running)"
  if [ "${QUIET}" -lt 2 ]; then
    continue
  fi
  QUIET=0
  log "nothing running for ~6 min; relaunching pipeline"
  cd "${ROOT}" || continue
  nohup bash scripts/overnight_pipeline.sh >> "${MASTER}" 2>&1 &
  log "relaunched orchestrator pid=$!"
done
