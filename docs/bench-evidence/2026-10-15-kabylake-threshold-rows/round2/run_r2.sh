#!/bin/zsh
# Round 2: K's 13 decided rows (round-2 header in the main tree) against the current head
# (round 1's "new" bench binaries, built from 31d3ec3). Same tiers, ABAB, each run quiet.
D=${0:A:h}; R1=${D:h}
NEW=/Users/wolfman/Development/libstats
LOG=$D/ab.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date -u '+%FT%TZ') load=$(load1) $*" >> $LOG; }
quiet() { while (( $(load1) >= 2.0 )); do sleep 30; done; }
stamp "start head=$(git -C $NEW rev-parse --short HEAD) header_changed=$(git -C $NEW diff --stat -- include/libstats/core/dispatch_thresholds.h | tail -1)"
for spec in AVX2: AVX:AVX SSE2:SSE2; do
    TIER=${spec%%:*}; B=build-ab-${(L)TIER}
    mkdir -p $D/$TIER
    /bin/cp -f $R1/$TIER/table_ab_new $D/$TIER/table_ab_old
    (cd $NEW && cmake --build $B --target libstats_static --parallel 4) > $D/prep_$TIER.txt 2>&1
    stamp "$TIER lib exit=$? warnings=$(grep -c 'warning:' $D/prep_$TIER.txt)"
    clang++ -std=c++20 -O2 -DNDEBUG -I$NEW/include -I$NEW/$B/generated \
        $NEW/tools/bench/table_ab_bench.cpp $NEW/$B/libstats.a -lpthread -o $D/$TIER/table_ab_new >> $D/prep_bench.txt 2>&1
    stamp "$TIER bench exit=$?"
done
for TIER in AVX2 AVX SSE2; do
    for i in 1 2; do
        for V in old new; do
            quiet
            stamp "$TIER $V$i start"
            $D/$TIER/table_ab_$V > $D/$TIER/tab_${V}_$i.csv 2> $D/$TIER/tab_${V}_$i.err
            stamp "$TIER $V$i end exit=$? rows=$(wc -l < $D/$TIER/tab_${V}_$i.csv)"
        done
    done
done
stamp "DONE"
