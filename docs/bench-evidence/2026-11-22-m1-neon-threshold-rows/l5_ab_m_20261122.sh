#!/bin/bash
# M1 fleet leg, step L5: table_ab_bench, current kNeon table against the proposed rows, old/new/old/new.
# Same ambient gate as L4 before each pass.
# before a quiet window came (mediaanalysisd, 02:30-12:00). Same gate, new deadline.
# M1 fleet leg, step L4 (PLAN.md "Fleet legs for v3.0.0"), at the head stated in run.txt:
#   1 ctest -j1 -L timing   2 simd_verification
#   3 distributions_bench and elementary_bench, main against dev, alternately, three passes
#   4 three strategy_profile --large runs
# Waits for the accuracy oracle jobs to end first. Gate as on 2026-10-13: ambient CPU =
# 100 - top's idle %, two consecutive 10 s windows below 15%; 20% after 3 h of waiting; no
# gated phase starts after DEADLINE (noon next day). Runs under caffeinate -is.
set -u
W=/Users/wolfman/Development/libstats
O=/Users/wolfman/Development/libstats-m-2026-11-21/l5
B=$W/build-release
LOG=$O/run.txt
MAXAMB=15; MAXAMB_RELAXED=20; RELAX_AFTER=$((3*3600)); WIN=10
DEADLINE=$(date -j -v+1d -f '%H:%M:%S' '12:00:00' +%s)
caffeinate -is -w $$ &

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
ambient() { /usr/bin/top -l 2 -s $WIN -n 0 2>/dev/null | grep 'CPU usage' | tail -1 \
              | sed -E 's/.* ([0-9.]+)% idle.*/\1/' | awk '{printf "%.2f", 100-$1}'; }
top3() { /bin/ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{n=$2; sub(/.*\//,"",n); printf "%s%%%s ", $1, n}'; }
log() { echo "$(date '+%F %T') load=$(load1) $*" >> $LOG; }
gate() {
  local q=0 a m=$MAXAMB t0; t0=$(date +%s)
  log "$1: waiting for ambient < ${m}% over two consecutive ${WIN}s windows"
  while :; do
    [ "$(date +%s)" -gt "$DEADLINE" ] && { log "$1: DEADLINE passed, SKIPPED"; return 1; }
    if [ "$m" = "$MAXAMB" ] && [ $(( $(date +%s) - t0 )) -gt $RELAX_AFTER ]; then
      m=$MAXAMB_RELAXED; q=0; log "$1: waited $((RELAX_AFTER/3600)) h, gate RELAXED to ${m}%"
    fi
    a=$(ambient)
    if awk -v a="$a" -v m="$m" 'BEGIN{exit !(a<m)}'; then
      q=$((q+1)); [ $q -ge 2 ] && { log "$1: gate passed (<${m}%) ambient=${a}% top: $(top3)"; return 0; }
    else
      q=0; echo "$(date '+%F %T') [$1] ambient=${a}% top: $(top3)" >> $O/gate_samples.txt; sleep 50
    fi
  done
}
noise() { while kill -0 "$1" 2>/dev/null; do echo "$(date '+%T') load=$(load1) top: $(top3)" >> "$2"; sleep 60; done; }
run() {
  local label=$1 nf=$2; shift 2
  log "$label: start"
  "$@" > "$OUT" 2>&1 & local pid=$!
  noise $pid "$nf" & local np=$!
  wait $pid; local rc=$?; wait $np 2>/dev/null
  log "$label: end exit=$rc"
  return $rc
}

log "table_ab start (old = cfbb992 table, new = l5/dispatch_thresholds.h.proposed)"
for p in 1 2; do
  for t in old new; do
    gate "ab $t pass$p" || continue
    log "ab $t pass$p: start"
    $O/table_ab_$t > $O/tab_${t}_$p.csv 2> /dev/null & pid=$!
    noise $pid $O/noise_ab_${t}_$p.txt & np=$!
    wait $pid; rc=$?; wait $np 2>/dev/null
    log "ab $t pass$p: end exit=$rc rows=$(wc -l < $O/tab_${t}_$p.csv | tr -d ' ')"
  done
done
log "DONE"
