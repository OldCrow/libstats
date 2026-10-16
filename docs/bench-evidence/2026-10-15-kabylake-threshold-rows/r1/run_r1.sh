#!/bin/zsh
# R1 on K at the final K table header (uncommitted over 31d3ec3): Release 0 warnings, correctness,
# then the quiet parallel-batch gate (build/, Dev) and the timing label.
D=${0:A:h}; REPO=/Users/wolfman/Development/libstats; LOG=$D/r1.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date -u '+%FT%TZ') load=$(load1) $*" >> $LOG; }
quiet() { while (( $(load1) >= 2.0 )); do sleep 30; done; }
cd $REPO
stamp "start head=$(git rev-parse --short HEAD) diff=$(git diff --stat | tail -1)"
cmake --build build-release --parallel 4 > $D/build.txt 2>&1
stamp "build-release exit=$? warnings=$(grep -c 'warning:' $D/build.txt)"
ctest --test-dir build-release -LE "timing|benchmark" -j4 --timeout 1800 > $D/ctest.txt 2>&1
stamp "ctest exit=$? $(grep -E 'tests passed' $D/ctest.txt)"
cmake --build build --parallel 4 --target test_parallel_batch_gates > $D/build_gates.txt 2>&1
stamp "gates build exit=$?"
quiet; stamp "gates start"
./build/tests/test_parallel_batch_gates > $D/gates.txt 2>&1
stamp "gates exit=$? $(grep -oE 'PARALLEL/VECTORIZED [0-9.]+' $D/gates.txt | awk '{print $2}' | sort -n | sed -n '1p;$p' | tr '\n' ' ')"
quiet; stamp "timing start"
ctest --test-dir build-release -j1 -L timing > $D/timing.txt 2>&1
stamp "timing exit=$? $(grep -E 'tests passed' $D/timing.txt)"
stamp "DONE"
