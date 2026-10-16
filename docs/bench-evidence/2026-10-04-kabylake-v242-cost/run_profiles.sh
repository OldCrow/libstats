#!/bin/zsh
# Quiet strategy_profile --large captures. Usage: run_profiles.sh <tier> <build-dir> <run#>...
# Waits for 1-min load < 2.0 before each run (the v2.4.0 bundle's entry rule) and logs
# load, sha and the active SIMD level, so each CSV is attributable.
TIER=$1; BUILD=$2; shift 2
REPO=/Users/wolfman/Development/libstats
OUT=${0:A:h}/$TIER
mkdir -p $OUT
LOG=$OUT/run.log
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
stamp() { echo "$(date '+%F %T') load=$(load1) $*" >> $LOG; }

stamp "batch start tier=$TIER build=$BUILD runs=$* sha=$(git -C $REPO rev-parse --short HEAD)"
$REPO/$BUILD/tools/system_inspector --quick > $OUT/system_inspector.txt 2>&1
stamp "system_inspector: $(grep '^System:' $OUT/system_inspector.txt)"
for i in "$@"; do
    stamp "run$i waiting for load < 2.0"
    while (( $(load1) >= 2.0 )); do sleep 30; done
    stamp "run$i start"
    $REPO/$BUILD/tools/strategy_profile --large -o $OUT/strategy_profile_run$i.csv > $OUT/stdout_run$i.txt 2>&1
    stamp "run$i end exit=$? rows=$(wc -l < $OUT/strategy_profile_run$i.csv)"
done
stamp "batch DONE"
