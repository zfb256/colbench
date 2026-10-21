#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${COLBENCH_WORK:?set COLBENCH_WORK}"
: "${COLBENCH_DATA:?set COLBENCH_DATA}"
: "${COLBENCH_RESULTS:?set COLBENCH_RESULTS}"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
GATE="$COLBENCH_RESULTS/gate_report.json"
DATASETS=(tpch_sf1_lineitem tpcds_sf1_store_sales clickbench_32p)
CHUNKS=(1024 8192 32768 131072 524288 1048576)

[[ -x "$PYTHON" ]] || { echo "STOP: missing $PYTHON" >&2; exit 2; }
[[ -f "$GATE" ]] || { echo "STOP: missing authorized gate $GATE" >&2; exit 2; }
for dataset in "${DATASETS[@]}"; do
  [[ -f "$COLBENCH_DATA/$dataset.parquet" ]] || {
    echo "STOP: missing $COLBENCH_DATA/$dataset.parquet" >&2
    exit 2
  }
done
mkdir -p "$COLBENCH_RESULTS"

raid_name() {
  "$PYTHON" -c 'import json,pathlib,sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text()); x=r.get("raid") or {}; print(pathlib.Path(x.get("device","")).name if x.get("detected") else "")' "$GATE"
}
RAID_NAME="$(raid_name)"

raid_action() {
  if [[ -z "$RAID_NAME" ]]; then printf '%s\n' none
  elif [[ -r "/sys/block/$RAID_NAME/md/sync_action" ]]; then
    tr -d '\n' < "/sys/block/$RAID_NAME/md/sync_action"; printf '\n'
  else printf '%s\n' unreadable
  fi
}

raid_field() {
  local field="$1"
  if [[ -z "$RAID_NAME" ]]; then printf '%s\n' none
  elif [[ -r "/sys/block/$RAID_NAME/md/$field" ]]; then
    tr -d '\n' < "/sys/block/$RAID_NAME/md/$field"; printf '\n'
  else printf '%s\n' unreadable
  fi
}

raid_members() {
  if [[ -z "$RAID_NAME" ]]; then printf '%s\n' none; return; fi
  awk -v name="$RAID_NAME" '
    $1 == name && $2 == ":" { found=1; next }
    found && match($0, /\[[U_]+\]/) {
      print substr($0, RSTART, RLENGTH); exit
    }
    found && NF == 0 { exit }
  ' /proc/mdstat
}

raid_signature() {
  printf '%s|%s|%s|%s\n'     "$(raid_action)" "$(raid_field mismatch_cnt)"     "$(raid_field degraded)" "$(raid_members)"
}

require_raid_healthy() {
  local context="$1" action mismatch degraded members
  [[ -n "$RAID_NAME" ]] || return 0
  action="$(raid_action)"
  mismatch="$(raid_field mismatch_cnt)"
  degraded="$(raid_field degraded)"
  members="$(raid_members)"
  if [[ "$action" != idle || "$mismatch" != 0 || "$degraded" != 0 ||
        ! "$members" =~ ^\[U+\]$ ]]; then
    echo "STOP: unhealthy RAID at $context: action=$action mismatch=$mismatch degraded=$degraded members=$members" >&2
    return 2
  fi
}

gate_sha256() {
  sha256sum "$GATE" | awk '{print $1}'
}

capture_boundary() {
  local destination="$1" phase="$2" exit_code="${3:-not-run}"
  {
    echo "timestamp_utc=$(date -u --iso-8601=seconds)"
    echo "phase=$phase"
    echo "exit_code=$exit_code"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "boot_id=$(cat /proc/sys/kernel/random/boot_id)"
    echo "raid_action=$(raid_action)"
    echo "raid_mismatch_cnt=$(raid_field mismatch_cnt)"
    echo "raid_degraded=$(raid_field degraded)"
    echo "raid_members=$(raid_members)"
    echo "gate_report_sha256=$(gate_sha256)"
    echo "COLBENCH_WORK=$COLBENCH_WORK"
    echo "COLBENCH_DATA=$COLBENCH_DATA"
    echo "COLBENCH_RESULTS=$COLBENCH_RESULTS"
    findmnt -T "$COLBENCH_WORK" -o TARGET,SOURCE,FSTYPE,OPTIONS
    df -hT "$COLBENCH_WORK"
    lsblk -o NAME,TYPE,SIZE,ROTA,MODEL,SERIAL,MOUNTPOINTS
    cat /proc/mdstat
    uptime
  } > "$destination"
}

validate_checkpoint() {
  local output="$1" dataset="$2" chunk="$3" start="$4" end="$5"
  "$PYTHON" scripts/validate_p2_checkpoint.py     "$output" "$dataset" "$chunk"     --source "$COLBENCH_DATA/$dataset.parquet"     --start-boundary "$start" --end-boundary "$end"     --gate-report "$GATE" --work "$COLBENCH_WORK"     --data "$COLBENCH_DATA" --results "$COLBENCH_RESULTS"
}

quarantine_partial() {
  local output="$1" reason="$2"
  if [[ -e "$output" ]]; then
    local stamp
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    mv -- "$output" "${output%.jsonl}.partial-$reason-$stamp.jsonl"
  fi
}

WATCHER_LOG="$COLBENCH_RESULTS/p2_raid_watcher.log"
raid_watcher() {
  while true; do
    printf '%s action=%s mismatch=%s degraded=%s members=%s\n'       "$(date -u --iso-8601=seconds)" "$(raid_action)"       "$(raid_field mismatch_cnt)" "$(raid_field degraded)"       "$(raid_members)" >> "$WATCHER_LOG"
    sleep 30
  done
}
raid_watcher &
WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null || true' EXIT

require_raid_healthy "P2 start"
native_readahead="$(cat /sys/block/$RAID_NAME/queue/read_ahead_kb)"
[[ "$native_readahead" == 2048 ]] || {
  echo "STOP: P2 requires native read_ahead_kb=2048, got $native_readahead" >&2
  exit 2
}

for dataset in "${DATASETS[@]}"; do
  rate_file="$COLBENCH_RESULTS/p2_${dataset}_orc_bytes_per_row.txt"
  if [[ ! -s "$rate_file" ]]; then
    rate_tmp="$rate_file.tmp.$$"
    "$PYTHON" -c 'import sys; from pathlib import Path; from colbench.runner import Dataset; from colbench.formats import measure_bytes_per_row; ds=Dataset(sys.argv[1], source=Path(sys.argv[2])); print(format(measure_bytes_per_row(ds.reader(), compression="zstd"), ".12g"))' "$dataset" "$COLBENCH_DATA/$dataset.parquet" > "$rate_tmp"
    mv -- "$rate_tmp" "$rate_file"
  fi
  orc_rate="$(tr -d '\n' < "$rate_file")"
  "$PYTHON" -c 'import math,sys; x=float(sys.argv[1]); assert math.isfinite(x) and x > 0' "$orc_rate"
  echo "fixed ORC rate dataset=$dataset bytes_per_row=$orc_rate"

  for chunk in "${CHUNKS[@]}"; do
    stem="p2_${dataset}_c${chunk}"
    output="$COLBENCH_RESULTS/$stem.jsonl"
    start="$COLBENCH_RESULTS/$stem.boundary_start.txt"
    end="$COLBENCH_RESULTS/$stem.boundary_end.txt"

    require_raid_healthy "$dataset chunk=$chunk preflight"
    if [[ -e "$output" && -e "$start" && -e "$end" ]] &&
       grep -qx 'exit_code=0' "$end" &&
       validate_checkpoint "$output" "$dataset" "$chunk" "$start" "$end"; then
      echo "checkpoint $dataset chunk=$chunk already valid; skipping"
      continue
    fi

    quarantine_partial "$output" preexisting
    start_health="$(raid_signature)"
    capture_boundary "$start" start not-run
    echo "=== P2 checkpoint dataset=$dataset chunk=$chunk start $(date -u --iso-8601=seconds) ==="

    set +e
    "$PYTHON" scripts/run_experiment.py p2 --dataset "$dataset" --formats parquet,lance,orc --chunks "$chunk" --repeats 7 --threads 1 --orc-rate "$orc_rate" --out "$output"
    rc=$?
    set -e

    capture_boundary "$end" end "$rc"
    if [[ "$rc" -ne 0 ]]; then
      quarantine_partial "$output" "exit$rc"
      echo "STOP: $dataset chunk=$chunk failed with exit=$rc" >&2
      exit "$rc"
    fi
    require_raid_healthy "$dataset chunk=$chunk end"
    if [[ "$(raid_signature)" != "$start_health" ]]; then
      quarantine_partial "$output" boundary
      echo "STOP: RAID state changed during $dataset chunk=$chunk" >&2
      exit 2
    fi
    if ! validate_checkpoint "$output" "$dataset" "$chunk" "$start" "$end"; then
      quarantine_partial "$output" invalid
      echo "STOP: invalid checkpoint $dataset chunk=$chunk" >&2
      exit 2
    fi
    echo "=== P2 checkpoint dataset=$dataset chunk=$chunk complete $(date -u --iso-8601=seconds) ==="
  done
done

echo "All 18 P2 dataset/chunk checkpoints are complete and boundary-qualified."

