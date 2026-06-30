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

## Measurement artifacts and warm-state bias

Two distinct artifact classes affect profiler measurements.  Understanding
them is necessary to interpret crossover data correctly and to recognise when
a threshold was derived from noise rather than a real performance difference.

### Timer jitter and the sub-64 measurement floor

The profiler measures wall-clock elapsed time per strategy per batch size.
On all target platforms the per-sample jitter (scheduler preemption, timer
interrupt alignment, memory-bus contention) is approximately 0.1–0.2 µs.
For batch sizes below ~64 elements, this jitter exceeds the actual compute
time for any strategy:

- A VECTORIZED loop over 8 doubles finishes in well under 0.1 µs.
- PARALLEL or WORK_STEALING also finish quickly at that size, but their
  GCD / Thread Pool dispatch latency (typically 1–5 µs warm, 50–200 µs
  cold) contributes an overhead that is also smaller than one timer tick
  when the pool is warm.

When the per-sample jitter exceeds the signal, comparisons are dominated by
noise.  A false "PARALLEL wins" outcome at batch=8 is then detected as a
crossover, which the Step 2 rule clamps to the minimum encodable threshold
(64).  The resulting table entry of 64 has no physical meaning — it reflects
timer granularity, not an actual performance advantage at that batch size.

**The grid floor requirement**: the profiler's measurement grid must start at
no lower than 64 elements.  The `capture_dispatcher_profile.sh` script
enforces this.  Do not add sub-64 batch sizes to `strategy_profile.cpp`'s
measurement grid.

**Effect on historical data**: the kNeon table encoded from sha-2904d63
bundles (2026-06-22) used a grid starting at 8 elements.  Approximately 12
entries were encoded as 64 from clamped sub-floor crossovers.  The fb8e8b6
recalibration (2026-06-24) with a 64-element grid floor revealed the true
crossovers, which range from 128 to 75 000 for the affected distributions.

### Warm-state bias and how it compounds the floor problem

The timer floor becomes worse when the machine is in a warm state — thread
pool workers active, CPU caches loaded from a preceding build or prior
profiling run.  The two warm-state components interact multiplicatively:

**Warm thread pool (GCD / Windows Thread Pool)**

After a build, GCD workers remain active for tens of seconds.  When the
profiler launches immediately after a build, the first run (R1) executes with
a warm pool.  GCD dispatch latency in this state drops to ~1–2 µs from a
cold-start baseline of 50–200 µs.  This makes parallel appear competitive at
much smaller batch sizes than it is in typical production use (where the pool
is cold on first call).  The effect is most visible at sub-64 sizes where it
pushes the already-noisy measurement toward a false crossover.

**Warm CPU cache**

The build also loads executable code and stack data into L1/L2 caches.  At
sub-64 batch sizes, the entire input array fits inside the warm L1 cache
(64 elements × 8 bytes = 512 bytes; L1 is 32–192 KB on all target CPUs).
VECTORIZED benefits disproportionately: its tight loop runs with zero cache
misses.  PARALLEL also benefits, but the fixed dispatch overhead — even when
pool-warm — still dominates at these sizes.  Paradoxically, a warm cache
worsens the floor problem rather than improving measurement quality, because
both strategies complete faster, narrowing the already-marginal signal below
the timer jitter threshold.

**The compound effect**

A session that starts immediately after a build experiences all of the
following simultaneously:

1. Timer jitter exceeds compute time at sub-64 sizes (floor condition).
2. Warm pool eliminates dispatch overhead → parallel looks faster than at
   cold-start (warm-pool condition).
3. Warm cache reduces compute time for both strategies → signal-to-noise
   ratio falls further (warm-cache condition).

In this combined state, false crossovers at batch=8–32 are highly probable.
The three-run sequential protocol spreads runs across time: by R2 or R3 the
GCD pool may have shed idle workers (cold-pool outlier in the opposite
direction) and cache state has changed.  This produces the run-to-run spread
that the Step 3 outlier-discard rule is designed to handle.

Warm-state bias at ≥64 is a separate, less severe phenomenon (covered in
Step 4 as the bimodal flag): pool-warm runs show earlier crossovers (e.g.,
64 or 256) while pool-cold runs show later crossovers (e.g., 500 000).  The
three-run rule discards the outlier; the resulting threshold is conservative
but valid.  This is expected and does not require a grid floor change.

**Practical implication**: always wait at least 30 seconds after a build
before starting a profiling session.  This allows GCD workers to idle down,
bringing the pool closer to the cold-start state that production callers
experience.  The capture script's built-in `--sleep 30` default handles this
automatically — it sleeps 30 seconds before each run.  Increase with
`--sleep 60` if bimodal instability persists; use `--sleep 0` only for
smoke tests where measurement accuracy is not required.

---

## Step 1 — Capture profiling bundles

### macOS / Linux (Bash)

```bash
# Standard capture (3 runs, 30 s sleep before each; default):
bash scripts/capture_dispatcher_profile.sh

# Extended batch sizes (use when V→P = 500000 is reported by any run):
bash scripts/capture_dispatcher_profile.sh --large

# Custom run count:
bash scripts/capture_dispatcher_profile.sh --runs 5

# Longer cool-down (use when bimodal instability persists):
bash scripts/capture_dispatcher_profile.sh --sleep 60

# Disable sleep — quick smoke test only, not for threshold calibration:
bash scripts/capture_dispatcher_profile.sh --sleep 0
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
| AVX2+FMA (kAvx2) | Kaby Lake i7-7820HQ | ✅ current (3-run standard + 3-run --large, canonical method, fb8e8b6) |
| NEON (kNeon)     | Mac Mini M1         | ✅ current (3-run, fb8e8b6; 64-floor grid; warm/cold pool recalibrated) |
| AVX-512 (kAvx512)| Asus TUF A16 Zen 4  | ✅ current (3-run --large, canonical method, 1b564ec; covers Geometric/Laplace/Cauchy) |
| AVX (kAvx)       | Ivy Bridge (retired) | ⚠ hardware gone; values inferred from kAvx2 trends |
| SSE2 (kSse2)     | no dedicated hardware | delegates to kAvx by design |

---

## Windows capture

Use `scripts/capture_dispatcher_profile.ps1` (requires `pwsh`, i.e. PowerShell 7):

```powershell
pwsh -ExecutionPolicy Bypass -File scripts\capture_dispatcher_profile.ps1
pwsh -ExecutionPolicy Bypass -File scripts\capture_dispatcher_profile.ps1 -Large
```

Do **not** invoke with the legacy `powershell` command — it runs Windows
PowerShell 5.x which lacks the `utf8NoBOM` encoding identifier.

---

## Known issues with historical bundles

*No outstanding issues.  All active bundles contain the full set of required
files.*  Resolved issues are recorded below for reference.

**kAvx512 (Windows, 2026-06-22):** Three bundles under
`2026-06-22T02-*_sha-9b2c1a3` contained only `strategy_profile_output.txt`
and `manifest.txt` — no `strategy_profile_results.csv`.  These bundles have
been removed and replaced by six complete bundles captured 2026-06-23.

**kNeon (M1, 2026-06-22, sha-2904d63):** These three bundles had two
compounding defects and have been removed.  First, they used a profiler grid
starting at 8 elements, producing false crossovers below the timer-jitter
floor that were clamped to 64 (see "Measurement artifacts" above).  Second,
the original kNeon encoding derived from them used a buggy PARALLEL-only V→P
definition; the corrected definition was applied to the raw data
(commit e927ebf) but could not undo the sub-64 grid contamination.  Replaced
by the fb8e8b6 bundles (2026-06-24) which use a 64-element grid floor and
reveal true crossovers of 128–75 000 for the previously mis-encoded entries.
