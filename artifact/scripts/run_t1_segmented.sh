#!/usr/bin/env bash
# Run the 72-cell ClickBench T1 matrix as six independently recoverable
# 12-cell checkpoints. Start this script inside screen/tmux.

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${COLBENCH_WORK:?export COLBENCH_WORK to the qualified block-device path}"
: "${COLBENCH_DATA:?export COLBENCH_DATA to the source-data path}"
: "${COLBENCH_RESULTS:?export COLBENCH_RESULTS to the repository results directory}"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || {
  echo "STOP: missing $PYTHON; create the venv and run the tests first" >&2
  exit 2
}
[[ -f "$COLBENCH_DATA/clickbench_32p.parquet" ]] || {
  echo "STOP: missing $COLBENCH_DATA/clickbench_32p.parquet" >&2
  exit 2
}
[[ -f "$COLBENCH_RESULTS/gate_report.json" ]] || {
  echo "STOP: missing authorized repeated-gate report" >&2
  exit 2
}

mkdir -p "$COLBENCH_RESULTS"

CHUNKS=(1024 8192 32768 131072 524288 1048576)
PRUNE_COMPLETED_LAYOUTS="${COLBENCH_PRUNE_COMPLETED_LAYOUTS:-0}"
REQUIRE_RAW_SAMPLES="${COLBENCH_REQUIRE_RAW_SAMPLES:-0}"

raid_name() {
  "$PYTHON" -c 'import json, pathlib, sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); x=r.get("raid") or {}; print(pathlib.Path(x.get("device", "")).name if x.get("detected") else "")' \
    "$COLBENCH_RESULTS/gate_report.json"
}

raid_action() {
  local name
  name="$(raid_name)"
  if [[ -z "$name" ]]; then
    printf '%s\n' "none"
  elif [[ -r "/sys/block/$name/md/sync_action" ]]; then
    tr -d '\n' < "/sys/block/$name/md/sync_action"
    printf '\n'
  else
    printf '%s\n' "unreadable"
  fi
}

raid_field() {
  local field="$1"
  local name
  name="$(raid_name)"
  if [[ -z "$name" ]]; then
    printf '%s\n' "none"
  elif [[ -r "/sys/block/$name/md/$field" ]]; then
    tr -d '\n' < "/sys/block/$name/md/$field"
    printf '\n'
  else
    printf '%s\n' "unreadable"
  fi
}

raid_members() {
  local name
  name="$(raid_name)"
  if [[ -z "$name" ]]; then
    printf '%s\n' "none"
    return
  fi
  awk -v name="$name" '
    $1 == name && $2 == ":" { found = 1; next }
    found && match($0, /\[[U_]+\]/) {
      print substr($0, RSTART, RLENGTH)
      exit
    }
    found && NF == 0 { exit }
  ' /proc/mdstat
}

raid_health_signature() {
  printf '%s|%s|%s|%s\n' \
    "$(raid_action)" "$(raid_field mismatch_cnt)" \
    "$(raid_field degraded)" "$(raid_members)"
}

require_raid_healthy() {
  local context="$1"
  local name action mismatch degraded members
  name="$(raid_name)"
  [[ -n "$name" ]] || return 0
  action="$(raid_action)"
  mismatch="$(raid_field mismatch_cnt)"
  degraded="$(raid_field degraded)"
  members="$(raid_members)"
  if [[ "$action" != "idle" || "$mismatch" != "0" || "$degraded" != "0" || \
        ! "$members" =~ ^\[U+\]$ ]]; then
    echo "STOP: unhealthy RAID at $context: action=$action mismatch=$mismatch degraded=$degraded members=$members" >&2
    return 2
  fi
}

gate_report_sha256() {
  sha256sum "$COLBENCH_RESULTS/gate_report.json" | awk '{print $1}'
}

lsblk_mount_columns() {
  if lsblk --help 2>/dev/null | grep -qw MOUNTPOINTS; then
    printf '%s\n' "NAME,TYPE,SIZE,ROTA,MODEL,SERIAL,MOUNTPOINTS"
  else
    printf '%s\n' "NAME,TYPE,SIZE,ROTA,MODEL,SERIAL,MOUNTPOINT"
  fi
}

capture_boundary() {
  local destination="$1"
  local phase="$2"
  local exit_code="${3:-not-run}"
  {
    echo "timestamp_utc=$(date -u --iso-8601=seconds)"
    echo "phase=$phase"
    echo "exit_code=$exit_code"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unavailable)"
    echo "raid_action=$(raid_action)"
    echo "raid_mismatch_cnt=$(raid_field mismatch_cnt)"
    echo "raid_degraded=$(raid_field degraded)"
    echo "raid_members=$(raid_members)"
    echo "gate_report_sha256=$(gate_report_sha256)"
    echo "COLBENCH_WORK=$COLBENCH_WORK"
    echo "COLBENCH_DATA=$COLBENCH_DATA"
    echo "COLBENCH_RESULTS=$COLBENCH_RESULTS"
    findmnt -T "$COLBENCH_WORK" -o TARGET,SOURCE,FSTYPE,OPTIONS
    df -hT "$COLBENCH_WORK"
    lsblk -o "$(lsblk_mount_columns)"
    cat /proc/mdstat
    uptime
  } > "$destination"
}

validate_checkpoint() {
  local output="$1"
  local chunk="$2"
  local start_boundary="$3"
  local end_boundary="$4"
  local args=("$output" "$chunk")
  [[ "$REQUIRE_RAW_SAMPLES" == "1" ]] && args+=(--require-raw-samples)
  args+=(
    --start-boundary "$start_boundary"
    --end-boundary "$end_boundary"
    --gate-report "$COLBENCH_RESULTS/gate_report.json"
    --work "$COLBENCH_WORK"
    --data "$COLBENCH_DATA"
    --results "$COLBENCH_RESULTS"
  )
  "$PYTHON" scripts/validate_t1_checkpoint.py "${args[@]}"
}

quarantine_partial() {
  local output="$1"
  local reason="$2"
  if [[ -e "$output" ]]; then
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$output" "${output%.jsonl}.partial-${reason}-${stamp}.jsonl"
  fi
}

prune_layouts() {
  local chunk="$1"
  [[ "$PRUNE_COMPLETED_LAYOUTS" == "1" ]] || return 0
  [[ -n "$COLBENCH_WORK" && "$COLBENCH_WORK" != "/" ]] || {
    echo "STOP: refusing unsafe layout cleanup path: $COLBENCH_WORK" >&2
    exit 2
  }
  local parquet="$COLBENCH_WORK/clickbench_32p__parquet__c${chunk}.parquet"
  local lance="$COLBENCH_WORK/clickbench_32p__lance__c${chunk}.lance"
  rm -f -- "$parquet"
  rm -rf -- "$lance"
  echo "pruned regenerable layouts for completed chunk=$chunk"
}

for chunk in "${CHUNKS[@]}"; do
  output="$COLBENCH_RESULTS/fig4_clickbench_32p_c${chunk}.jsonl"
  start_boundary="$COLBENCH_RESULTS/fig4_clickbench_32p_c${chunk}.boundary_start.txt"
  end_boundary="$COLBENCH_RESULTS/fig4_clickbench_32p_c${chunk}.boundary_end.txt"

  require_raid_healthy "checkpoint $chunk preflight"

  if [[ -e "$output" && -e "$start_boundary" && -e "$end_boundary" ]] && \
     grep -qx 'exit_code=0' "$end_boundary" && \
     grep -Eq '^raid_action=(idle|none)$' "$start_boundary" && \
     grep -Eq '^raid_action=(idle|none)$' "$end_boundary" && \
     validate_checkpoint "$output" "$chunk" "$start_boundary" "$end_boundary"; then
    echo "checkpoint chunk=$chunk already valid; skipping"
    prune_layouts "$chunk"
    continue
  fi

  quarantine_partial "$output" "preexisting"
  start_health="$(raid_health_signature)"

  capture_boundary "$start_boundary" "start" "not-run"
  echo "=== checkpoint chunk=$chunk start $(date -u --iso-8601=seconds) ==="

  set +e
  "$PYTHON" scripts/run_experiment.py fig4 \
    --dataset clickbench_32p --formats parquet,lance --chunks "$chunk" \
    --repeats 7 --threads 1 --out "$output"
  rc=$?
  set -e

  capture_boundary "$end_boundary" "end" "$rc"
  if [[ "$rc" -ne 0 ]]; then
    quarantine_partial "$output" "exit${rc}"
    echo "STOP: chunk=$chunk failed with exit=$rc; earlier checkpoints remain valid" >&2
    exit "$rc"
  fi

  if ! require_raid_healthy "checkpoint $chunk end"; then
    quarantine_partial "$output" "raid-health"
    echo "STOP: RAID health failed during chunk=$chunk" >&2
    exit 2
  fi
  if [[ "$(raid_health_signature)" != "$start_health" ]]; then
    quarantine_partial "$output" "boundary"
    echo "STOP: storage/RAID state changed during chunk=$chunk" >&2
    exit 2
  fi
  if ! validate_checkpoint "$output" "$chunk" "$start_boundary" "$end_boundary"; then
    quarantine_partial "$output" "invalid"
    echo "STOP: chunk=$chunk did not produce a complete 12-record checkpoint" >&2
    exit 2
  fi

  echo "=== checkpoint chunk=$chunk complete $(date -u --iso-8601=seconds) ==="
  prune_layouts "$chunk"
done

echo "All six T1 checkpoints are complete and individually boundary-qualified."
echo "Report with:"
echo "$PYTHON scripts/report.py $COLBENCH_RESULTS/fig4_clickbench_32p_c{1024,8192,32768,131072,524288,1048576}.jsonl --cache-mode cold --threads 1 --dataset clickbench_32p"
