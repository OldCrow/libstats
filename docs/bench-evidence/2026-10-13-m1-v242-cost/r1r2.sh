#!/bin/bash
# M at d1adc95: R1 (correctness ctest) and R2 (sweep, diff vs last M sweep, oracle). Run OUTSIDE the sandbox.
set -u
W=/Users/wolfman/Development/libstats-v2.4.2
O=/Users/wolfman/Development/libstats-m-20261013
B=$W/build-release
PY=/Users/wolfman/Development/libstats-log-cleanroom/.venv/bin/python
BASE=$O/r2/sweep_base_5b20fc2_dirty_26885.csv
log() { echo "$(date '+%F %T') load=$(sysctl -n vm.loadavg | awk '{print $2}') $*"; }
cd "$W" || exit 1
log "start sha=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l | tr -d ' ') sandbox_check: $(sysctl -n hw.l2cachesize)"
$B/tools/system_inspector --quick > $O/r1/system_inspector.txt 2>&1
log "inspector: $(grep -m1 '^System:' $O/r1/system_inspector.txt) L2=$(grep -E '^L2 ' $O/r1/system_inspector.txt | awk '{print $2}')KB"
log "R1 ctest start"
ctest --test-dir $B -LE "timing|benchmark" -j8 --output-on-failure > $O/r1/ctest_correctness.txt 2>&1
log "R1 ctest exit=$? : $(grep -E 'tests passed|tests failed' $O/r1/ctest_correctness.txt | head -2 | tr '\n' ' ')"
log "R2 sweep start"
$B/tools/accuracy_sweep $O/r2/sweep_d1adc95.csv > $O/r2/sweep.log 2>&1
log "R2 sweep exit=$? : $(head -1 $O/r2/sweep.log) banner: $(sed -n 2p $O/r2/sweep_d1adc95.csv)"
grep -v '^#' $BASE > $O/r2/base_rows.txt; grep -v '^#' $O/r2/sweep_d1adc95.csv > $O/r2/new_rows.txt
diff $O/r2/base_rows.txt $O/r2/new_rows.txt > $O/r2/sweep_diff.txt
log "R2 diff: base=$(wc -l < $O/r2/base_rows.txt) new=$(wc -l < $O/r2/new_rows.txt) changed_lines=$(grep -c '^[<>]' $O/r2/sweep_diff.txt)"
grep '^>' $O/r2/sweep_diff.txt | cut -d, -f1,2 | sort | uniq -c | sort -rn > $O/r2/sweep_diff_by_dist_method.txt
log "R2 oracle start"
$PY tools/accuracy_vs_mpmath.py $O/r2/sweep_d1adc95.csv > $O/r2/oracle.log 2>&1
log "R2 oracle exit=$? : $(grep -iE 'contract violation|self-check' $O/r2/oracle.log | tail -3 | tr '\n' ' ')"
git diff --stat > $O/r2/doc_diff_stat.txt; git diff > $O/r2/doc_block.diff
log "doc diff: $(tail -1 $O/r2/doc_diff_stat.txt)"
log "DONE"
