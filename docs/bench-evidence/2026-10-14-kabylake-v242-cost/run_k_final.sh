#!/bin/zsh
# K at d1adc95: R1 (Release correctness), R9 (TSan concurrency suite), then R3 (quiet costs
# against v2.4.1 and the timing tests). Runbook: PLAN.md R1, R9 step 4, R3.
D=${0:A:h}
REPO=/Users/wolfman/Development/libstats
OLD=/Users/wolfman/Development/libstats-v2.4.1
LOG=$D/k_final.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date -u '+%FT%TZ') load=$(load1) $*" >> $LOG; }
quiet() { while (( $(load1) >= 2.0 )); do sleep 30; done; }
cd $REPO
stamp "start sha=$(git rev-parse --short HEAD) code=$(git log -1 --format=%h -- src include CMakeLists.txt cmake tests) dirty=$(git status --porcelain | wc -l | tr -d ' ')"

# R1: fresh build-release (configured by the orchestrator), correctness
cmake --build build-release --parallel 4 > $D/r1_build.txt 2>&1
stamp "R1 build exit=$? warnings=$(grep -c 'warning:' $D/r1_build.txt)"
./build-release/tools/system_inspector --quick > $D/r1_system_inspector.txt 2>&1
stamp "R1 $(grep '^System:' $D/r1_system_inspector.txt)"
ctest --test-dir build-release -LE "timing|benchmark" -j4 --timeout 1800 > $D/r1_ctest.txt 2>&1
stamp "R1 ctest exit=$? $(grep -E 'tests passed' $D/r1_ctest.txt)"

# R9: fresh TSan build, the concurrency binaries, then the whole correctness suite
rm -rf build-tsan
cmake -S . -B build-tsan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBSTATS_BUILD_TOOLS=OFF \
    -DCMAKE_CXX_FLAGS=-fsanitize=thread -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread \
    -DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=thread > $D/r9_configure.txt 2>&1
stamp "R9 configure exit=$?"
cmake --build build-tsan --parallel 4 > $D/r9_build.txt 2>&1
stamp "R9 build exit=$? warnings=$(grep -c 'warning:' $D/r9_build.txt)"
export LIBSTATS_STRESS_SCALE=0.1
export TSAN_OPTIONS="halt_on_error=0 second_deadlock_stack=1"
for t in test_concurrency_stress test_concurrency_gates test_snapshot_consistency test_thread_pool \
         test_work_stealing_pool test_parallel_execution_integration \
         test_parallel_exception_propagation; do
    ./build-tsan/tests/$t > $D/r9_$t.txt 2>&1
    rc=$?
    stamp "R9 $t exit=$rc tsan_reports=$(grep -c 'WARNING: ThreadSanitizer' $D/r9_$t.txt) $(grep -E '^\[  (PASSED|FAILED)  \]' $D/r9_$t.txt | head -2 | tr '\n' ' ')"
done
ctest --test-dir build-tsan -LE "timing|benchmark" -j2 --timeout 3600 --output-on-failure > $D/r9_ctest.txt 2>&1
stamp "R9 ctest exit=$? $(grep -E 'tests passed' $D/r9_ctest.txt) tsan_reports=$(grep -c 'WARNING: ThreadSanitizer' $D/r9_ctest.txt)"
unset LIBSTATS_STRESS_SCALE TSAN_OPTIONS

# R3: v2.4.1 comparison tree and both bench binaries
[[ -d $OLD ]] || git worktree add --detach $OLD v2.4.1 >> $LOG 2>&1
(cd $OLD && cmake --preset release -G Ninja -DLIBSTATS_BUILD_TESTS=OFF -DLIBSTATS_BUILD_TOOLS=OFF \
   && cmake --build build-release --target libstats_static --parallel 4) > $D/r3_prep_v241.txt 2>&1
stamp "R3 v2.4.1 build exit=$?"
for T in $OLD $REPO; do
  V=$([[ $T == $OLD ]] && echo v241 || echo v242)
  clang++ -std=c++20 -O2 -DNDEBUG -I$T/include -I$T/build-release/generated \
    tools/bench/v242_cost_bench.cpp $T/build-release/libstats.a -lpthread -o $D/bench_$V >> $D/r3_prep_bench.txt 2>&1
  stamp "R3 bench_$V build exit=$?"
done
stamp "R3 bench waiting for load < 2.0"
quiet
stamp "R3 start bench_v241"; $D/bench_v241 > $D/v241.csv 2> $D/bench_v241.err
stamp "R3 end bench_v241 exit=$?; start bench_v242"; $D/bench_v242 > $D/v242.csv 2> $D/bench_v242.err
stamp "R3 end bench_v242 exit=$?"
python3 tools/bench/v242_compare.py $D/v241.csv $D/v242.csv > $D/compare.txt 2>&1
stamp "R3 compare exit=$?; timing waiting for load < 2.0"
quiet
stamp "R3 start timing ctest"
ctest --test-dir build-release -j1 -L timing > $D/timing.txt 2>&1
stamp "R3 end timing ctest exit=$? $(grep -E 'tests passed' $D/timing.txt)"
stamp "DONE sha=$(git rev-parse --short HEAD)"
