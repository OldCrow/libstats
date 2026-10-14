#!/bin/bash
# Assemble M's R8 NEON bundle at d1adc95 from the overnight outputs, K's 2026-10-13 layout.
set -eu
W=/Users/wolfman/Development/libstats-v2.4.2
O=/Users/wolfman/Development/libstats-m-20261013
P=$W/data/profiles/dispatcher
PREV=2026-10-05T04-36-25Z_darwin-arm64_dev-v2.4.2_sha-2c5230a
UTC=$(cat $O/r8/start_utc.txt)
ID=${UTC}_darwin-arm64_dev-v2.4.2_sha-d1adc95
D=$P/$ID
mkdir -p $D/logs
cp $O/r8/strategy_profile_run{1,2,3}.csv $D/
cp $O/r8/system_inspector.txt $O/r8/tier.txt $D/logs/
cp $O/r8/noise_run{1,2,3}.txt $D/logs/ 2>/dev/null || true
cp $O/run.txt $D/logs/run.txt; [ -f $O/gate_samples.txt ] && cp $O/gate_samples.txt $D/logs/
cp $W/scripts/analyze_crossovers.py $D/
(cd $D && python3 analyze_crossovers.py --bundle . --table kNeon --header $W/include/libstats/core/dispatch_thresholds.h > sustained_crossovers.txt)
sed -e "s#$PREV#$ID#" \
    -e 's#OLDD = {"NEON": f"{PROF}/2026-09-04T04-22-28Z_darwin-arm64_dev-v2.4.0_sha-5f27ee1"}#OLDD = {"NEON": f"{PROF}/'"$PREV"'"}#' \
    -e 's/Assessment of the 2026-10-05 M1 NEON captures at 2c5230a (dev\/v2.4.2 freeze head plus tests\/docs)/Assessment of the '"${UTC%%T*}"' M1 NEON captures at d1adc95 (dev\/v2.4.2 freeze head, fix-9 round)/' \
    -e 's/and to the 2026-09-04 v2.4.0 NEON runs/and to M'"'"'s 2026-10-05 2c5230a NEON runs/' \
    -e 's/since the 2026-09-04 capture/since the 2026-10-05 capture/' \
    -e 's/since 2026-09-04 (v2.4.0 -> v2.4.2)/since 2026-10-05 (2c5230a -> d1adc95)/' \
    -e 's/Sep runs/Oct05 runs/; s/MOVED since Sep/MOVED since Oct05/; s/same as Sep/same as Oct05/; s/Sep {fmt/Oct05 {fmt/' \
    $P/$PREV/cross_tier_assessment.py > $D/cross_tier_assessment.py
(cd $D && python3 cross_tier_assessment.py > cross_tier_assessment.txt)
cat > $D/metadata.json <<JSON
{
  "captured_at_utc": "$UTC",
  "run_id": "$ID",
  "git_branch": "dev/v2.4.2",
  "git_sha": "$(cd $W && git rev-parse --short HEAD)",
  "code_sha": "d1adc95",
  "project_root": "$W",
  "build_dir": "$W/build-release",
  "build_type": "Release",
  "cmake_options": "",
  "cxx_compiler": "/usr/bin/c++ (Apple clang 21.0.0, clang-2100.3.34.2)",
  "cmake": "4.4.4",
  "os": "darwin (macOS 27.0.1)",
  "arch": "arm64",
  "simd_tier_active": "NEON",
  "simd_tier_capped": "",
  "cpu_brand": "$(sysctl -n machdep.cpu.brand_string)",
  "physical_cores": "$(sysctl -n hw.physicalcpu)",
  "logical_cores": "$(sysctl -n hw.logicalcpu)"
}
JSON
echo "bundle at $D"; ls $D
