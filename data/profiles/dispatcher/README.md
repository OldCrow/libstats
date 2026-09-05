# Dispatcher Profiling Data

This directory contains the profiling bundles behind the `constexpr`
dispatch-threshold tables in `include/libstats/core/dispatch_thresholds.h`.
Each subdirectory is a timestamped bundle from a single architecture run,
named `<timestamp>_<platform>_<branch>_sha-<sha>`. Bundles are committed so
raw calibration data accumulates across machines via normal git workflow.

There are two bundle classes:

## Harness bundles (June 2026, `capture_dispatcher_profile.sh`)

- `metadata.json` — machine, OS, SIMD level, compiler, git state
- `manifest.txt` — file listing for the bundle
- `strategy_profile_results.csv` — canonical raw timing data (distribution × operation × batch size × strategy)
- `crossovers.csv` — derived SCALAR→VECTORIZED, VECTORIZED→PARALLEL, PARALLEL→WORK_STEALING crossover points (first-crossing heuristic — superseded, see below)
- `best_strategies.csv` — per-(distribution, operation, batch size) best strategy and speedup vs scalar
- `summary.json` — coverage, strategy win counts, crossover summary
- `logs/` — console output from `system_inspector` and `strategy_profile`

## Direct-capture bundles (v2.4.0, 2026-09-04T*)

Captured via `strategy_profile --large -o <csv>` directly (three quiet
runs), analyzed with the SUSTAINED-crossover rule
(`scripts/PROFILING_METHOD.md`, 2026-09-04 amendment; #146 tracks folding
the rule into `threshold_validator`):

- `metadata.json`, `manifest.txt` — as above (the manifest records capture caveats)
- `strategy_profile_run{1,2,3}.csv` — raw per-run timing data
- `analyze_crossovers.py` — the sustained-crossover extraction used
- `sustained_crossovers.txt` — its per-run output, the direct input to the table update
- `logs/*.txt` — suite/tier/configure logs (`.txt`, not `.log` — `.gitignore` excludes `*.log`)
- the Kaby Lake bundle also carries that leg's `accuracy_sweep` CSV

## Current table provenance (post-#143 fork repair, sustained crossovers)

| Table | Machine | Bundle |
|---|---|---|
| kNeon | Mac Mini M1 (native) | `2026-09-04T04-22-28Z_darwin-arm64_…` |
| kAvx2 | Kaby Lake i7-7820HQ (native) | `2026-09-04T02-36-22Z_darwin-x86_64_…` |
| kAvx512 | Asus TUF A16 Zen 4 (native) | PR #143 record (no 2026-09 bundle checked in); von Mises CDF cell provisional — #144 |
| kAvx | Kaby Lake, `LIBSTATS_MAX_SIMD_TIER=AVX` capped build — first measured kAvx (the 2012 AVX MBP is retired; its June bundles are historical) | `2026-09-04T23-51-14Z_darwin-x86_64_…` |
| kSse2 | delegates to kAvx by design | — |

June bundles remain as the historical record of the pre-repair calibration;
do not derive new thresholds from them (their parallel timings measured
secretly-serial paths for the sliced batch families — see #143).

## Capturing a new profile

```bash
# Build first (Release), then either use the harness:
scripts/capture_dispatcher_profile.sh
# ...or capture directly, three quiet runs (the v2.4.0 approach):
./build-release/tools/strategy_profile --large -o run1.csv
# Assemble manifest.txt + metadata.json per an existing 2026-09-04T* bundle,
# commit and push the new bundle.
```
