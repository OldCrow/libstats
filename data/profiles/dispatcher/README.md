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
(`scripts/PROFILING_METHOD.md`, 2026-09-04 amendment). The canonical tool
is `scripts/analyze_crossovers.py` (#146); the `analyze_crossovers.py`
copies inside these two bundles are historical record only:

- `metadata.json`, `manifest.txt` — as above (the manifest records capture caveats)
- `strategy_profile_run{1,2,3}.csv` — raw per-run timing data
- `analyze_crossovers.py` — the extraction used at the time (historical copy; use `scripts/analyze_crossovers.py`)
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

## v2.4.2 captures, not yet applied (2026-10-04/05)

Kaby Lake bundles at the freeze head (code `5367999`, captured at
`6f86996`), one per tier: `2026-10-05T01-50-52Z_…` (AVX2),
`2026-10-05T03-09-27Z_…` (AVX-capped), `2026-10-05T04-31-33Z_…`
(SSE2-capped). They supersede the `d9384f8` bundles of 2026-10-04
(`2026-10-04T02-07-22Z_…`, `T03-51-16Z_…`, `T05-16-48Z_…`; the first
measured SSE2 profile), which predate the speed work: that moved
VonMises CDF to NEVER on every tier and Weibull and Pareto CDF later.
`cross_tier_assessment.txt` in the new AVX2 bundle has the comparison.
The M1 and Zen 4 bundles of the same week are listed in PLAN.md. The
tables above stay as they are until the R8 decision (PLAN.md).

June bundles remain as the historical record of the pre-repair calibration;
do not derive new thresholds from them (their parallel timings measured
secretly-serial paths for the sliced batch families — see #143).

## Analyzing a bundle

```bash
scripts/analyze_crossovers.py --bundle <bundle-dir> --table kAvx2   # or kNeon|kAvx|kAvx512|kNone
scripts/analyze_crossovers.py run1.csv run2.csv run3.csv            # explicit files, no table
scripts/analyze_crossovers.py --self-test
```

Per (distribution, PDF/LogPDF/CDF) it prints the sustained V→P crossover of
each run (smallest grid size where PARALLEL beats VECTORIZED there and at
every larger size; NEVER if none) and a candidate: the median across runs,
or the max when max/min > 2, NEVER counting as infinity. A crossover at the
64-element grid floor is flagged `floor-run`/`floor` (timing-resolution
artifact). With `--table`, each row is flagged `ok` or `DIFF` against the
encoded value (within one grid step), and `override?` when the table line
carries a comment. The candidate is a starting point, not the encoding:
many encoded rows are deliberate overrides (sweep data, bimodal runs,
contaminated runs) or predate the v2.4.0 legs. Future bundles reference this
tool and its commit instead of embedding a copy.

## Capturing a new profile

```bash
# Build first (Release), then either use the harness:
scripts/capture_dispatcher_profile.sh
# ...or capture directly, three quiet runs (the v2.4.0 approach):
./build-release/tools/strategy_profile --large -o run1.csv
# Assemble manifest.txt + metadata.json per an existing 2026-09-04T* bundle,
# commit and push the new bundle.
```
