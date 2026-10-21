#!/usr/bin/env bash

set -u

RESULTS="${COLBENCH_RESULTS:-/root/autodl-tmp/colbench/artifact/results/autodl_nvme_raid5_20260820}"
INTERVAL_SECONDS="${P2_WATCH_INTERVAL_SECONDS:-900}"
LOG="$RESULTS/p2_light_watchdog.log"
ALERT="$RESULTS/p2_light_watchdog.alert"

mkdir -p "$RESULTS"

while true; do
  timestamp="$(date -u --iso-8601=seconds)"
  completed=0
  for boundary in "$RESULTS"/p2_*.boundary_end.txt; do
    [[ -f "$boundary" ]] || continue
    if grep -qx 'exit_code=0' "$boundary"; then
      completed=$((completed + 1))
    fi
  done

  if pgrep -f '[r]un_p2_segmented.sh' >/dev/null; then
    process=running
  else
    process=stopped
  fi

  raid_action="$(cat /sys/block/md0/md/sync_action 2>/dev/null || echo unreadable)"
  raid_degraded="$(cat /sys/block/md0/md/degraded 2>/dev/null || echo unreadable)"
  raid_mismatch="$(cat /sys/block/md0/md/mismatch_cnt 2>/dev/null || echo unreadable)"
  available_kb="$(df -Pk /root/autodl-tmp | awk 'NR==2 {print $4}')"

  printf '%s completed=%s/18 process=%s raid_action=%s degraded=%s mismatch=%s available_kb=%s\n' \
    "$timestamp" "$completed" "$process" "$raid_action" \
    "$raid_degraded" "$raid_mismatch" "$available_kb" >> "$LOG"

  if [[ "$completed" -ge 18 ]]; then
    printf '%s COMPLETE completed=18/18\n' "$timestamp" >> "$LOG"
    rm -f "$ALERT"
    exit 0
  fi

  if [[ "$process" != running || "$raid_action" != idle ||
        "$raid_degraded" != 0 || "$raid_mismatch" != 0 ||
        ! "$available_kb" =~ ^[0-9]+$ || "$available_kb" -lt 20971520 ]]; then
    printf '%s ALERT completed=%s/18 process=%s raid_action=%s degraded=%s mismatch=%s available_kb=%s\n' \
      "$timestamp" "$completed" "$process" "$raid_action" \
      "$raid_degraded" "$raid_mismatch" "$available_kb" > "$ALERT"
  else
    rm -f "$ALERT"
  fi

  sleep "$INTERVAL_SECONDS"
done
