# Dispatcher Threshold Profiling Method

This document is the authoritative reference for how dispatch thresholds in
`include/core/dispatch_thresholds.h` are measured and derived.  All agents
that capture profiling runs or encode table values **must** follow this method
exactly to ensure consistency across architectures and sessions.

---

## Overview

The dispatch table encodes, for each (SIMD tier, distribution, operation)
triple, the minimum batch size at which a parallel execution strategy
(PARALLEL or WORK_STEALING) sustainably outperforms VECTORIZED.  Below that
threshold the library uses VECTORIZED; at or above it, it dispatches to the
parallel strategy chosen at runtime by `selectMultiThreadedStrategy()`.

---

## Prerequisites

- **Release build** only.  Debug and Dev builds include instrumentation that
  distorts timing.  All profiling runs must use `CMAKE_BUILD_TYPE=Release`.
- **Quiet machine.**  Close background applications.  On macOS, disable
  Time Machine, Spotlight indexing, and any sync processes during runs.
- **No cross-contamination.**  Run on the target architecture only.  Do not
  profile a Kaby Lake binary on an M1 or vice versa.
- **Three runs minimum.**  The aggregation rules require at least three runs
  to apply the outlier-discard check.  Use `--runs 5` when bimodal instability
  is suspected on a particular distribution/operation.

---

## Step 1 — Capture profiling bundles

### macOS / Linux (Bash)

```bash
# Standard capture (3 runs, default):
bash scripts/capture_dispatcher_profile.sh

# Extended batch sizes (use when V→P = 500000 is reported by any run):
bash scripts/capture_dispatcher_profile.sh --large

# Custom run count:
bash scripts/capture_dispatcher_profile.sh --runs 5
```

### Windows (PowerShell)

The Bash script has a PS1 translation guide embedded in its header comments.
Generate `scripts/capture_dispatcher_profile.ps1` by following those comments.
Key differences:
- `.exe` suffix on tool binaries
- `python` instead of `python3`
- `(Get-Date).ToUniversalTime().ToString(...)` for timestamps
- `Copy-Item -Recurse` instead of `cp -R`

The script automatically commits bundles to `data/profiles/dispatcher/`.
**Commit the bundles before deriving the table** so the raw data is always
available for re-analysis.

---

## Step 2 — Per-run threshold computation

For each (distribution, operation) pair in `crossovers.csv`:

```
V2P  = vectorized_to_parallel column
BEST = best_strategy_at_max_size column

if BEST == "VECTORIZED":
    threshold = NEVER          # Crossover was transient — parallel did not sustain
elif BEST == "SCALAR":
    threshold = NEVER          # Measurement anomaly / scheduler artifact; discard
elif V2P is blank (empty string):
    threshold = 64             # Parallel wins at max_size but crossover below floor
else:
    threshold = max(64, int(V2P))   # Clamp any value below 64 to 64
```

> **Why BEST matters:** A V→P crossover at a small batch size can reflect a
> brief parallel win that is reversed at larger sizes (e.g., GCD's warm-pool
> effect).  Only encoding the crossover when BEST is a parallel strategy
> ensures the threshold represents a *sustainable* advantage.

> **Why blank V→P with a parallel BEST is 64:** The profiler's measurement
> resolution floors out below 64 elements.  If parallel wins at max_size but
> no explicit crossover was detected, the real crossover is at or below the
> measurement floor; encoding 64 is the correct conservative value.

> **SCALAR BEST:** SCALAR winning at max_size indicates a scheduler anomaly
> (e.g., GCD put all 500k elements on the calling thread).  Treat as NEVER.

---

## Step 3 — Three-run aggregation

Given per-run thresholds `{t1, t2, t3}` where NEVER is represented as `None`:

```python
finite   = [t for t in (t1, t2, t3) if t is not None]
n_never  = (t1, t2, t3).count(None)

if len(finite) == 0 or n_never >= 2:
    result = NEVER

elif len(finite) == 3:
    lo, hi = min(finite), max(finite)
    if hi / lo <= 10:
        result = hi           # All three within one OOM → take max (conservative)
    else:
        a, b, c = sorted(finite)
        if b / a <= 10:
            result = b        # Lower two agree → discard high outlier
        elif c / b <= 10:
            result = c        # Upper two agree → discard low outlier
        else:
            result = NEVER    # All three mutually incoherent

elif len(finite) == 2:
    lo, hi = min(finite), max(finite)
    result = hi if hi / lo <= 10 else NEVER

else:  # 1 finite, 2 NEVER
    result = NEVER
```

---

## Step 4 — Bimodal flag and manual override

**Bimodal flag:** When two finite values differ by more than 2 orders of
magnitude (>100×), flag the entry for human review.  The algorithm still
produces a result, but the encoded value must include a comment in
`dispatch_thresholds.h` explaining the decision.

**Common bimodal cause:** GCD (macOS) and Windows Thread Pool exhibit
warm-pool / cold-pool behaviour.  If the thread pool is warm from a prior
batch, parallel wins at small batch sizes (crossover at 64).  On a cold pool,
the startup overhead dominates until 500k or beyond (crossover at 500k).
These two conditions produce the bimodal distribution.  In such cases, the
conservative approach is to use NEVER (or the larger of the coherent pair)
because encoding 64 would dispatch to parallel when the pool is cold and
parallel is actually slower.

**Manual override rule:** After the algorithm produces a result, a human
reviewer may override it when:
1. The bimodal flag is set and the instability is clearly GCD scheduling, not
   a real performance crossover.
2. The change vs. the previous table value looks physically implausible for the
   SIMD tier (e.g., an entry moving from 64 to NEVER when the underlying code
   path is unchanged and identical distributions on the same SIMD tier moved to
   64).

Every manual override **must** be documented in the table comment with the raw
values and the reason.

---

## Step 5 — Ceiling advisory and --large

If the summarizer prints a ceiling advisory:

```
⚠  V→P crossover at measurement ceiling — re-run with --large to resolve:
   Gaussian CDF: V→P = 500000
```

Re-run with `--large` and apply the same three-run method to the extended
data.  The `--large` flag adds batch sizes 750000, 1000000, 1500000, and
2000000.

The `--large` re-run is **advisory** (not mandatory) because:
- It adds ~50% to per-run wall-clock time.
- A V→P at 500000 with BEST = parallel is still a valid conservative threshold
  (it encodes "do not dispatch to parallel below 500k"), even if the true
  crossover is slightly above 500000.
- Use `--large` when precision matters (e.g., the entry affects a hot path in
  production use) or when the current encoding seems implausibly conservative.

---

## Step 6 — Encoding in dispatch_thresholds.h

Update the `kXxx` table for the relevant SIMD tier:

```cpp
// --- ARCH (CPU description, width, cores, OS/scheduler) ---
// data/profiles/dispatcher/TIMESTAMP_arch_branch_sha-SHA/  (run 1)
// data/profiles/dispatcher/TIMESTAMP_arch_branch_sha-SHA/  (run 2)
// data/profiles/dispatcher/TIMESTAMP_arch_branch_sha-SHA/  (run 3)
//
// Three sequential Release-mode bundles on BRANCH (SHA).
// Method: see scripts/PROFILING_METHOD.md.
//
// Key changes vs prior table:
//   - DistributionName Op: OLD → NEW  (reason)
//   ...
// Manual overrides:
//   - DistributionName Op: rule gave X from {a,b,c}; held at Y because REASON.
constexpr ArchTable kArch = {
    /* uniform     */ {PDF, LogPDF, CDF},
    ...
};
```

---

## Architecture status

| SIMD tier | Machine | Status |
|---|---|---|
| AVX2+FMA (kAvx2) | Kaby Lake i7-7820HQ | ✅ current (3-run, canonical method) |
| NEON (kNeon)     | Mac Mini M1         | ✅ current (3-run, corrected) |
| AVX-512 (kAvx512)| Asus TUF A16 Zen 4  | ⚠ needs re-validation — bundles missing strategy_profile_results.csv |
| AVX (kAvx)       | Ivy Bridge (retired) | ⚠ hardware gone; values inferred from kAvx2 trends |
| SSE2 (kSse2)     | no dedicated hardware | delegates to kAvx by design |

---

## Known issues with historical bundles

**kAvx512 (Windows, June 2026):** The three Windows bundles committed under
`data/profiles/dispatcher/2026-06-22T02-*` contain only
`strategy_profile_output.txt` and `manifest.txt`.  The
`strategy_profile_results.csv` required by the summarizer was not committed.
Re-run on the Windows machine and commit the full bundles before the next
kAvx512 update.

**kNeon (M1, June 2026):** The three M1 bundles contain all required files.
The original kNeon encoding used the correct 100k crossover for Discrete
PDF/LogPDF (by reading best_strategies.csv directly) but applied the V→P
crossover from `crossovers.csv` for other entries — which at the time used the
buggy PARALLEL-only definition (not min(PARALLEL,WS)).  The summarizer has been
corrected; re-derive from the existing raw data before next update.
