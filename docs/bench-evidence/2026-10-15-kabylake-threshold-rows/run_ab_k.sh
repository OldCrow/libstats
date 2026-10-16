#!/bin/zsh
# K table A/B, round 1 (PLAN.md R8, "Table A/B on K and Z"): the header before the rows (10a7ff7)
# against the current head, per tier: native AVX2 (kAvx2), capped AVX and SSE2 (kAvx). Release
# libraries with tests and tools off; table_ab_bench from the head tree's source against each;
# old, new, old, new back to back, each run started once the 1-min load is below 2.0.
D=${0:A:h}
NEW=/Users/wolfman/Development/libstats
OLD=/Users/wolfman/Development/libstats-oldtable
LOG=$D/ab.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date -u '+%FT%TZ') load=$(load1) $*" >> $LOG; }
quiet() { while (( $(load1) >= 2.0 )); do sleep 30; done; }

stamp "start new=$(git -C $NEW rev-parse --short HEAD) dirty=$(git -C $NEW status --porcelain | wc -l | tr -d ' ')"
[[ -d $OLD ]] || git -C $NEW worktree add --detach $OLD 10a7ff7 >> $LOG 2>&1
stamp "old=$(git -C $OLD rev-parse --short HEAD) header_diff_files=$(git -C $NEW diff --name-only 10a7ff7 HEAD -- include src cmake CMakeLists.txt | tr '\n' ' ')"

for spec in AVX2: AVX:AVX SSE2:SSE2; do
    TIER=${spec%%:*}; CAP=${spec#*:}
    B=build-ab-${(L)TIER}
    for T in $OLD $NEW; do
        V=$([[ $T == $OLD ]] && echo old || echo new)
        (cd $T && cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DLIBSTATS_BUILD_TESTS=OFF \
             -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_EXAMPLES=OFF ${CAP:+-DLIBSTATS_MAX_SIMD_TIER=$CAP} \
         && cmake --build $B --target libstats_static --parallel 4) > $D/prep_${TIER}_$V.txt 2>&1
        stamp "$TIER $V lib exit=$? warnings=$(grep -c 'warning:' $D/prep_${TIER}_$V.txt) $(grep -o 'capped at [A-Z0-9]*' $D/prep_${TIER}_$V.txt | head -1)"
        mkdir -p $D/$TIER
        clang++ -std=c++20 -O2 -DNDEBUG -I$T/include -I$T/$B/generated \
            $NEW/tools/bench/table_ab_bench.cpp $T/$B/libstats.a -lpthread -o $D/$TIER/table_ab_$V \
            >> $D/prep_bench.txt 2>&1
        stamp "$TIER $V bench exit=$?"
    done
done

for TIER in AVX2 AVX SSE2; do
    for i in 1 2; do
        for V in old new; do
            stamp "$TIER $V$i waiting for load < 2.0"
            quiet
            stamp "$TIER $V$i start"
            $D/$TIER/table_ab_$V > $D/$TIER/tab_${V}_$i.csv 2> $D/$TIER/tab_${V}_$i.err
            stamp "$TIER $V$i end exit=$? rows=$(wc -l < $D/$TIER/tab_${V}_$i.csv)"
        done
    done
done
stamp "DONE new=$(git -C $NEW rev-parse --short HEAD)"
