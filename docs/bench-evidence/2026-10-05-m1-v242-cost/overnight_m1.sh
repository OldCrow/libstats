#!/bin/bash
# M1 overnight quiet runs for the v2.5.0 runbook (dev/v2.4.2 PLAN.md): R8 NEON
# captures (three strategy_profile --large runs), then R3 (cost bench pair and
# the timing ctest). Each phase waits for a quiet machine.
#
# Quiet on the M1 is ambient CPU (100 - top's idle %, all 8 cores), not the
# 1-min load: load sits at 4-10 here with the efficiency cores busy. Gate:
# two consecutive 10 s windows below MAXAMB (15%); a gate that has waited
# RELAX_AFTER (3 h) relaxes to MAXAMB_RELAXED (20%) and logs that it did.
# No phase starts after DEADLINE; it is logged SKIPPED.
#
# Outputs (outside the repo; bundled and committed after review):
#   run.txt            phase log with load, ambient and top consumers
#   gate_samples.txt   every failed gate sample
#   r8/                strategy_profile_run{1,2,3}.csv, run{i}.out, noise_run{i}.txt,
#                      system_inspector.txt, tier.txt, start_utc.txt
#   r3/                v241.csv, v242.csv, compare.txt, timing.txt, noise_*.txt
set -u
W=/Users/wolfman/Development/libstats-v2.4.2
O=/Users/wolfman/Development/libstats-r3m
B=$W/build-release
PY=/Users/wolfman/Development/libstats-log-cleanroom/.venv/bin/python
LOG=$O/run.txt
MAXAMB=15; MAXAMB_RELAXED=20; RELAX_AFTER=$((3*3600)); WIN=10
DEADLINE=$(date -j -v+1d -f '%H:%M:%S' '12:00:00' +%s)
mkdir -p $O/r8 $O/r3

load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
ambient() { /usr/bin/top -l 2 -s $WIN -n 0 2>/dev/null | grep 'CPU usage' | tail -1 \
              | sed -E 's/.* ([0-9.]+)% idle.*/\1/' | awk '{printf "%.2f", 100-$1}'; }
top3() { /bin/ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{n=$2; sub(/.*\//,"",n); printf "%s%%%s ", $1, n}'; }
log() { echo "$(date '+%F %T') load=$(load1) $*" >> $LOG; }

gate() {  # $1 label; 0 when quiet, 1 when past DEADLINE
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

noise() {  # $1 pid, $2 file: load and top consumers once a minute while pid runs
  while kill -0 "$1" 2>/dev/null; do
    echo "$(date '+%T') load=$(load1) top: $(top3)" >> "$2"; sleep 60
  done
}

run() {  # $1 label, $2 noise file, rest: command with stdout to $OUT
  local label=$1 nf=$2; shift 2
  log "$label: start"
  "$@" > "$OUT" 2>&1 & local pid=$!
  noise $pid "$nf" & local np=$!
  wait $pid; local rc=$?; wait $np 2>/dev/null
  log "$label: end exit=$rc"
  return $rc
}

cd "$W" || exit
SHA=$(git rev-parse --short HEAD)
log "batch start sha=$SHA build=$B gate=${MAXAMB}% (relaxed ${MAXAMB_RELAXED}% after 3 h) deadline=$(date -r $DEADLINE '+%F %T')"
$B/tools/system_inspector --quick > $O/r8/system_inspector.txt 2>&1
{ echo "Defined kernel symbols (nm -U) in build-release/libstats.a:"
  echo "  vector_*_neon: $(nm -U $B/libstats.a 2>/dev/null | grep -oE 'vector_[a-z0-9_]*_neon' | sort -u | wc -l | tr -d ' ')"
  echo "system_inspector: $(grep -m1 '^System:' $O/r8/system_inspector.txt)"; } > $O/r8/tier.txt

# R8: three NEON captures
for i in 1 2 3; do
  gate "R8 run$i" || continue
  [ $i = 1 ] && date -u '+%Y-%m-%dT%H-%M-%SZ' > $O/r8/start_utc.txt
  OUT=$O/r8/run$i.out run "R8 run$i" $O/r8/noise_run$i.txt \
    $B/tools/strategy_profile --large -o $O/r8/strategy_profile_run$i.csv
done

# R3: cost bench pair back to back, then the timing suite
if gate "R3 bench"; then
  OUT=$O/r3/v241.csv run "R3 bench_v241" $O/r3/noise_bench.txt $O/bench_v241
  OUT=$O/r3/v242.csv run "R3 bench_v242" $O/r3/noise_bench.txt $O/bench_v242
  $PY tools/bench/v242_compare.py $O/r3/v241.csv $O/r3/v242.csv > $O/r3/compare.txt 2>&1
  log "R3 compare exit=$?"
fi
if gate "R3 timing"; then
  OUT=$O/r3/timing.txt run "R3 timing ctest" $O/r3/noise_timing.txt \
    ctest --test-dir $B -j1 -L timing --output-on-failure
fi
log "DONE"
