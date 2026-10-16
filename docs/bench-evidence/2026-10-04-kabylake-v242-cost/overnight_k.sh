#!/bin/zsh
# Freeze-head overnight on K: prep builds, then R3 (quiet), then R8 (AVX2, AVX, SSE2 x3).
D=${0:A:h}   # ran from the session scratchpad; outputs copied here and to data/profiles/dispatcher
REPO=/Users/wolfman/Development/libstats
OLD=/Users/wolfman/Development/libstats-v2.4.1
LOG=$D/overnight.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date '+%F %T') load=$(load1) $*" >> $LOG; }

stamp "waiting for R1/R2 to finish"
while pgrep -f r1r2.sh > /dev/null; do sleep 30; done
stamp "start prep sha=$(git -C $REPO rev-parse --short HEAD)"

# v2.4.1 comparison tree and both bench binaries
[[ -d $OLD ]] || git -C $REPO worktree add --detach $OLD v2.4.1 >> $LOG 2>&1
(cd $OLD && cmake --preset release -G Ninja -DLIBSTATS_BUILD_TESTS=OFF -DLIBSTATS_BUILD_TOOLS=OFF \
   && cmake --build build-release --target libstats_static) > $D/prep_v241.txt 2>&1
stamp "v2.4.1 build exit=$?"
cd $REPO
cmake --build build-release --target libstats_static > $D/prep_head.txt 2>&1
for T in $OLD $REPO; do
  V=$([[ $T == $OLD ]] && echo v241 || echo v242)
  clang++ -std=c++20 -O2 -DNDEBUG -I$T/include -I$T/build-release/generated \
    tools/bench/v242_cost_bench.cpp $T/build-release/libstats.a -lpthread -o $D/bench_$V >> $D/prep_bench.txt 2>&1
  stamp "bench_$V build exit=$?"
done
for B in build-cap-avx build-cap-sse2; do
  cmake --build $B > $D/prep_$B.txt 2>&1
  stamp "$B build exit=$? warnings=$(grep -c 'warning:' $D/prep_$B.txt)"
done

# R3: quiet cost bench and timing tests
stamp "R3 waiting for load < 2.0"
while (( $(load1) >= 2.0 )); do sleep 30; done
stamp "start bench_v241"; $D/bench_v241 > $D/v241.csv 2> $D/bench_v241.err
stamp "end bench_v241 exit=$?; start bench_v242"; $D/bench_v242 > $D/v242.csv 2> $D/bench_v242.err
stamp "end bench_v242 exit=$?"
python3 tools/bench/v242_compare.py $D/v241.csv $D/v242.csv > $D/compare.txt 2>&1
stamp "compare exit=$?; start timing ctest"
ctest --test-dir build-release -j1 -L timing > $D/timing.txt 2>&1
stamp "end timing ctest exit=$?"

# R8: three quiet captures per tier
$D/run_profiles.sh AVX2 build-release 1 2 3;  stamp "AVX2 batch done"
$D/run_profiles.sh AVX  build-cap-avx 1 2 3;  stamp "AVX batch done"
$D/run_profiles.sh SSE2 build-cap-sse2 1 2 3; stamp "SSE2 batch done"
stamp "DONE"
