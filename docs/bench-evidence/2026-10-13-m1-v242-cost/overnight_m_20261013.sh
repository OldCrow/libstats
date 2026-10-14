#!/bin/bash
# M1 overnight quiet runs at d1adc95 (dev/v2.4.2 PLAN.md, "At d1adc95, per machine", M steps 3-6):
#   gates : test_parallel_batch_gates alone (Release), ratios must be < 0.75
#   R8    : three strategy_profile --large runs (NEON)
#   R3    : cost bench pair (v2.4.1 vs v2.4.2) back to back, compare, then ctest -j1 -L timing
#   R9    : ThreadSanitizer (build-tsan, RelWithDebInfo): stress test at scale 1, the six
#           targeted tests, then the whole correctness ctest -j2. Ungated: not timing-sensitive.
# Gate (2026-10-05 settings, kept [user, 2026-10-13]): ambient CPU = 100 - top's idle %, two
# consecutive 10 s windows below 15%; relaxes to 20% after 3 h of waiting; no gated phase starts
# after DEADLINE (noon). The whole script runs under caffeinate -is (AC sleep is 1 min here).
set -u
W=/Users/wolfman/Development/libstats-v2.4.2
O=/Users/wolfman/Development/libstats-m-20261013
B=$W/build-release
T=$W/build-tsan
PY=/Users/wolfman/Development/libstats-log-cleanroom/.venv/bin/python
LOG=$O/run.txt
MAXAMB=15; MAXAMB_RELAXED=20; RELAX_AFTER=$((3*3600)); WIN=10
DEADLINE=$(date -j -v+1d -f '%H:%M:%S' '12:00:00' +%s)
mkdir -p $O/r8 $O/r3 $O/r9 $O/gates
caffeinate -is -w $$ &

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

run() {  # $1 label, $2 noise file, rest: command with stdout+stderr to $OUT
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
log "batch start sha=$SHA dirty=$(git status --porcelain | wc -l | tr -d ' ') build=$B tsan=$T gate=${MAXAMB}% (relaxed ${MAXAMB_RELAXED}% after 3 h) deadline=$(date -r $DEADLINE '+%F %T') l2cachesize=$(sysctl -n hw.l2cachesize)"
$B/tools/system_inspector --quick > $O/r8/system_inspector.txt 2>&1
{ echo "Defined kernel symbols (nm -U) in build-release/libstats.a:"
  echo "  vector_*_neon: $(nm -U $B/libstats.a 2>/dev/null | grep -oE 'vector_[a-z0-9_]*_neon' | sort -u | wc -l | tr -d ' ')"
  echo "system_inspector: $(grep -m1 '^System:' $O/r8/system_inspector.txt)"; } > $O/r8/tier.txt
log "tier: $(tr '\n' ' ' < $O/r8/tier.txt)"

# Step 3: parallel batch gates, alone and quiet
if gate "gates"; then
  OUT=$O/gates/test_parallel_batch_gates.txt run "gates test_parallel_batch_gates" $O/gates/noise.txt \
    $B/tests/Release/test_parallel_batch_gates
  log "gates ratios: $(grep -iE 'ratio|PARALLEL/VECTORIZED' $O/gates/test_parallel_batch_gates.txt | head -8 | tr -s ' ' | tr '\n' ';')"
fi

# Step 4 (R8): three NEON captures
for i in 1 2 3; do
  gate "R8 run$i" || continue
  [ $i = 1 ] && date -u '+%Y-%m-%dT%H-%M-%SZ' > $O/r8/start_utc.txt
  OUT=$O/r8/run$i.out run "R8 run$i" $O/r8/noise_run$i.txt \
    $B/tools/strategy_profile --large -o $O/r8/strategy_profile_run$i.csv
  log "R8 run$i rows=$(wc -l < $O/r8/strategy_profile_run$i.csv | tr -d ' ')"
done

# Step 5 (R3): cost bench pair back to back, then the timing suite
if gate "R3 bench"; then
  OUT=$O/r3/v241.csv run "R3 bench_v241" $O/r3/noise_bench.txt $O/r3/bench_v2.4.1
  OUT=$O/r3/v242.csv run "R3 bench_v242" $O/r3/noise_bench.txt $O/r3/bench_v2.4.2
  $PY tools/bench/v242_compare.py $O/r3/v241.csv $O/r3/v242.csv > $O/r3/compare.txt 2>&1
  log "R3 compare exit=$?"
fi
if gate "R3 timing"; then
  OUT=$O/r3/timing.txt run "R3 timing ctest" $O/r3/noise_timing.txt \
    ctest --test-dir $B -j1 -L timing --output-on-failure
  log "R3 timing: $(grep -E 'tests passed|tests failed' $O/r3/timing.txt | head -1)"
fi

# Step 6 (R9): ThreadSanitizer, ungated
export TSAN_OPTIONS="log_path=$O/r9/tsan_report suppressions=$W/tests/tsan.supp halt_on_error=0 second_deadlock_stack=1"
log "R9 TSan start TSAN_OPTIONS=$TSAN_OPTIONS"
OUT=$O/r9/stress_scale1.txt run "R9 stress scale=1" $O/r9/noise_stress.txt \
  env LIBSTATS_STRESS_SCALE=1 $T/tests/RelWithDebInfo/test_concurrency_stress
for t in test_concurrency_gates test_snapshot_consistency test_thread_pool test_work_stealing_pool \
         test_parallel_execution_integration test_parallel_exception_propagation; do
  OUT=$O/r9/$t.txt run "R9 $t" $O/r9/noise_$t.txt $T/tests/RelWithDebInfo/$t
done
OUT=$O/r9/ctest_correctness_tsan.txt run "R9 ctest correctness -j2 (scale 0.1)" $O/r9/noise_ctest.txt \
  env LIBSTATS_STRESS_SCALE=0.1 ctest --test-dir $T -LE "timing|benchmark" -j2 --output-on-failure
log "R9 ctest: $(grep -E 'tests passed|tests failed' $O/r9/ctest_correctness_tsan.txt | head -1) tsan_report_files=$(ls $O/r9/tsan_report.* 2>/dev/null | wc -l | tr -d ' ')"
log "DONE"
