# libstats — Plan / Status

State only: what is decided, open, or next. Release contents are in
`CHANGELOG.md`, per-version validation in `docs/VALIDATION_HISTORY.md`,
conventions and commands in `AGENTS.md`.

History is in git, not here. This file was pruned on 2026-11-21: the
adoption branch's rebuild and CI narrative, the session logs of 2026-11-16
to 2026-11-21, the v2.5.0 release record and its runbook are in
`git show e1a11ec:PLAN.md`. Earlier history (shipped releases, closed
milestones, the 2026-08-21 review, the corvus spike) is in
`git show f508b12:PLAN.md`. Each fixed issue carries a status comment
naming its commits.

## v3.0.0 corvus adoption — where it stands [DERIVED, 2026-11-25]

Branch `dev/v3.0.0-corvus-adoption`, built on the v2.5.0 release. corvus
v1.2.1 is the special-function engine (found or fetched, pinned;
`cmake/FindOrFetchCorvus.cmake`). `main` is the v2.5.x line; **v2.5.1 was
released 2026-11-25** (tag on `c744eb3`) and is merged in here.

- **The legs are run on all four machines.** L1–L4 and L7 are done on I,
  K, M and Z (status lines below); every ISA block is regenerated with 0
  contract violations; L3 is bit-identical on NEON, AVX2 and AVX-512 and
  the `PROTOTYPE` tags are out. The last change under `src/` is
  `7879e9a` (#107, one trig table for every tier; comments only on x86),
  CI green on `66f8377`.
- **What is not settled is the threading, not the numerics**: the pool
  design ("Pool wake-up and batch spacing"; #323, #333) and, behind it,
  the threshold tables (kAvx512 and kAvx2 held, kNeon and kAvx
  provisional). **The design is decided [user, 2026-11-25]** ("Pool
  design — decided"): wake-up on submit, thresholds for sleeping
  workers, work stealing out of AUTO, #323 in the same change, nothing
  back-ported. **The prototype is built [Z, 2026-11-26]:**
  `proto/v3.0.0-pool-wake` at `974704a`, pushed, not merged. **Next
  step: each machine owes one sitting on it** (ThreadSanitizer and the
  guards on Linux and macOS, then the measurements; K first), and after
  the merge L1, with L2 again on I, K and M ("Three accuracy blocks"
  below).
- **corvus v1.2.1 is the pin [user, 2026-11-27; K, `ba1463f`]** (released 2026-11-27, tag on corvus `77bd92b`; one value
  through the span API; no result moves). Changed:
  `cmake/FindOrFetchCorvus.cmake` (the fetched tag; the `find_package`
  floor stays 1.2, since v1.2.0 returns the same bits), the two CI steps
  that install corvus to a prefix, and `tools/bench/README.md`.
  **Proof on AVX2 (K, AppleClang 15, at `7c4b809`)**: the same tree built
  at v1.2.0 and at v1.2.1, 0 warnings each; `accuracy_sweep`'s 26,882
  rows are bit-identical between the two; correctness 135/135 and
  `simd_verification` 140 pass at v1.2.1. I, M and Z owe the same sweep
  comparison, or L1 at the new pin, with their next sitting. Z, quiet
  [DERIVED, 2026-11-27,
  `docs/bench-evidence/2026-11-27-zen4-corvus-v121-quantiles/`]: this
  branch built against it reads the scalar Gaussian quantile at 76.3 ns
  and the LogNormal at 85.5, against 105.9 and 116.6 with v1.2.0 and 72
  and 76 in v2.5.1: 6% and 12% over v2.5.1, inside the 1.25× guard of
  #156. von Mises and Student's t gain 5% and 3%; nothing else moves.
  K's cost at the new pin is not measured (corvus reads 0.84× to 1.01×
  of v1.2.0 per one-value call on this machine). What corvus's v1.3.0
  asks of this repo is under "Cross-Repo Dependencies".
- **#156 is decided [user, 2026-11-25]: the cost is accepted** ("#156 —
  decided" below). No decision is owed before the pool design.
- **Then the release documents** ("After the legs", step 4):
  `MIGRATION_GUIDE.md` is still the v1 → v2 guide, `README.md` still
  says "Zero Dependencies", the per-family quantile accuracy notes #104
  asked for are not written, and #156's decision asks for the cost of
  the scalar special-function routes to be stated per route and tier.
- **Project version is 2.5.1** in `CMakeLists.txt` since the merge of the
  v2.5.1 release; the bump to 3.0.0 is a release step.
- **The merge of v2.5.1 (2026-11-25, Z).** It brings the version, the
  2.5.1 CHANGELOG section (placed under this branch's unreleased 3.0.0
  section), the v2.5.1 evidence and the documentation review's edits.
  Kept from this branch where the two sides conflicted: this file, the
  third-party notices, and all four generated accuracy blocks exactly as
  swept here. The accuracy document gains the "v2.5.1" section with a note
  that its blocks are the tag's, not this branch's. Z rebuilt the merged
  tree with clang-cl 22.1.3 and cl.exe 19.52: no compiler warnings of
  libstats' own, correctness 135/135 each.
- **Three accuracy blocks here predate the Cauchy fix; I, K and M owe
  L2 again [user, 2026-11-25: no separate sitting, do it with the next
  one each machine has].** #326 came in with `e4026b3` (2026-11-23). The
  AVX (`98810cf`), AVX2 (`09c7ea0`) and NEON (`cfbb992`) sweeps are from
  2026-11-21, so their seven `cauchy` rows are the pre-fix values;
  AVX-512 (`164cbce`) is after it and owes nothing. The merge did not
  copy `main`'s rows in: a block states what its banner commit measured.
  - **What each of I, K and M does**: L2 as in "The steps", from a head
    at or after the v2.5.1 merge: `accuracy_sweep`, the oracle, its own
    `isa=` block regenerated and committed. No L2b, L3 or timing is asked
    for by this.
  - **When**: in the sitting that follows the pool fix, where L1 is
    already owed on every machine. A threading fix does not by itself
    ask for L2 (the leg rule asks only when results change), so this is
    not covered unless it is written here. If the pool fix slips past
    the release documents, do it before step 4 of "After the legs".
  - **Expect**: 26,882 rows, 0 contract violations, and only the banner
    and the seven `cauchy` rows moving, to the figures in the accuracy
    document's "v2.5.1" section for that ISA (the Cauchy code is
    identical on `main` and here). Any other row moving is a
    stop-and-report.

## Fleet legs for v3.0.0 [OPEN; user, 2026-11-21]

Each machine runs one leg. The legs are independent and may run in
parallel: every machine reads only its own tier's threshold rows, and
nothing a leg needs is still open on another machine.

**Start from** the commit that carries this section, or later. Confirm
with `git log -1` and state the head in the leg's status line. CI is
green on `e890dec`, the last commit that changed code, tests or CI.

**Order if they are not run at once** [DERIVED]: M1, then Zen 4, then
Kaby Lake. The M1 differs most from what is validated (architecture,
AppleClang and libc++, GCD backend, hardware FMA), so a defect there is
the one most likely to change code the other legs would re-run. Zen 4 is
the longest leg (two compilers, the throttling opt-out). Kaby Lake covers
the least new ground but is the only native AVX2 tier. To clear the PR
gate soonest, run Kaby Lake first.

**What gates on the legs**: the dev-branch PR (one other leg complete);
removing the `PROTOTYPE` comment tags (all three report bit-identity:
**done 2026-11-24**, M, K and Z bit-identical, three tags removed);
#156's gates (all four machines' numbers); the release docs; the pool
design decision (all four machines' L7).

**Changed 2026-11-22, read before your next step** [user]:
- **L7 is new and every machine owes it**, including a machine whose
  L1–L6 are done. It is a 15-minute quiet sitting; see "The steps" and
  "Pool wake-up and batch spacing" below for why.
- **The profiler and three bench programs time differently from
  `8c55bfc`** (#320: each cell runs its own work for 0.15 s, then is
  timed for at least 0.25 s; 28 sizes, 32 with `--large`). A
  `strategy_profile --large` run is at least 0.4 s per cell, and the
  largest cells still take eight calls each whatever they cost, so
  expect one to two hours (an estimate: no machine has run one end to
  end yet; state your run's duration in the bundle manifest), and L4's three runs are an
  afternoon. The CSV has three more columns after the first five
  (`MinTime_us`, `MeanTime_us`, `Blocks`); the analysis scripts read by
  name and need no change. A bundle captured before `8c55bfc` is not
  comparable cell for cell with one captured after.
- **A leg already past L4 keeps what it captured.** Say in your status
  line which commit's profiler produced your bundle. Rows derived from a
  pre-`8c55bfc` bundle are provisional, as kAvx is; whether they are
  re-captured is decided after L7 [OPEN; user]. Do not re-capture on
  your own.
- A leg that has not started L4 pulls first and captures with the new
  profiler.

### Leg status (edit only your own line)

- **I — Ivy Bridge, AVX, Linux, GCC 13.3**: L1 done at `e890dec`
  (135/135). L2 done again 2026-11-21 at `98810cf`: 26,882 rows, 200
  self-checks, 0 contract violations, AVX block regenerated; row gate
  against `allow_AVX.csv` 0 failing. L2b done: `data/accuracy/allow_AVX.csv`; off-grid sweep
  found #316, #317, #318. L3 done at the time (`90d1bef` against
  `b766e45`, sweep row for row). **L4 done 2026-11-21** at `cfbb992`
  (code `e890dec`), quiet, on AC, screen locked: bundle
  `data/profiles/dispatcher/2026-11-21T21-09-06Z_linux-x86_64_…_sha-cfbb992`
  (three `--large` runs, timing tests 14/14, `simd_verification` all
  pass, the dev-against-`main` benches under `bench/`; notes in its
  manifest). K's `78e47eb` came after it and touches only Poisson CDF
  WORK_STEALING cells, so kAvx's derivation is unaffected. **L5 done 2026-11-21**: kAvx re-derived, 27 values across 19 distributions, all earlier; the fork
  zone ends at 4096 on this machine. A loaded `table_ab_bench` A/B held
  26 of the rule's 30 values, moved Exponential CDF to 8192 and kept
  Rayleigh CDF and InverseGamma PDF/LogPDF
  (`docs/bench-evidence/2026-11-21-ivybridge-threshold-rows/`). #291's
  column is in `docs/SIMD_BENCHMARK_RESULTS.md`. **The kAvx table at
  `98810cf` is provisional** [user, 2026-11-21]: a sustained forced
  PARALLEL-against-VECTORIZED run (0.15 s warm-up, 0.25 s timed per cell)
  has PARALLEL cleanly ahead from 25000 but erratic between 4096 and
  10000, so the rows moved into that zone are not confirmed (worst:
  Weibull LogPDF 1.4-1.9x slower at 8192-10000, Beta LogPDF 1.9x at
  10000, Pareto LogPDF 1.2x at 8192). It stays on dev and is re-captured
  once `strategy_profile` warms up each cell (#320). Causes found for the
  odd readings: the A/B's "InverseGamma PDF under AUTO 1.7x slower", the
  bench's lognormal PDF 1.44x, von Mises scalar 1.16x and elementary
  n=1000 0.72x are all the CPU clock recovering after a parallel section
  or a process start (#320, AGENTS.md Testing), none to be quoted on
  #156; InverseGamma PDF/LogPDF do over 90% of their work serially on the
  caller (#321). Also found: `parallelForSlices` on the four-thread pool
  costs ceil(slices/4) slice-times, so 5, 6, 9 and 10 slices are slow
  sizes on every row (#323); Cauchy LogPDF loses under sustained PARALLEL
  at 50k-500k (noted on #321). **L6 done 2026-11-21**: bench stated on #156.
  **2026-11-22**: #320 fixed at `8c55bfc` (profiler and benches time
  each cell in its own steady state). The kAvx bundle is from the
  earlier profiler, so the table stays provisional. The slow sizes and
  erratic readings above were mostly the package's low-clock state, not
  slice arithmetic ("Pool wake-up and batch spacing"). **L7 done 2026-11-22** at `7089876`, quiet
  (`docs/bench-evidence/2026-11-22-ivybridge-batch-gap/`). Crossover at
  gap 0 / 2 ms / 20 ms, plain run: Poisson 25,000 / 50,000 / 50,000,
  Gaussian 25,000 / 200,000 / 200,000. `--init` run: Poisson 4096 and
  Gaussian 25,000 at every gap. So on I the crossover moves with the gap
  when only the thread pool exists, and does not once the work-stealing
  pool is polling; forced PARALLEL at the same gap is up to 3x faster
  after `--init` (Poisson at 10,000: 47–57 against 18–22 ns/element),
  and VECTORIZED about 13% slower.
  **`pool_wake_bench` done 2026-11-23** at `761bb43`, quiet, under
  `schedutil` and again under `performance` (measurements 2, 5, 6, 7, 8
  and the governor test;
  `docs/bench-evidence/2026-11-22-ivybridge-pool-wake/`, README covers
  both). **The slow run on I is the governor's policy**: plain, under
  `schedutil`, pool workers run the integer workload at 40.2 ns per
  element against 13.8 (2.92x, the 3.5-to-1.2 GHz ratio) at every gap
  from 500 µs and, for calls under about 300 µs, back to back as well;
  under `performance` they run at 13.8 everywhere. What is left under
  `performance` is K's start delay: last worker 55–65 µs after the
  submit against 17–20, call 28–49 µs longer. So I = K's delay plus the
  governor's factor, which is why neither model fitted I's L7 data.
  Sleeping caller: under `schedutil` serial work is 3.1x slower too and
  the last worker starts 220 µs late; under `performance` serial is
  unchanged and the call costs 106–140 µs more. With the polling pool
  alive neither gap nor governor matters (last worker 21–30 µs), at 8%
  on serial work and 15% on pool work, and 2.5 cores idle (24.8 CPU-s in
  10 s). Eight plain threads match or beat work stealing at every size
  and are 1.6x faster than four at 204,800 (K's finding 3 holds). Work
  stealing's busy-gap delay sets in between 5 and 10 µs. The bench's
  four-thread local pool was 10–15 µs faster per call than the library's
  thread pool; **traced 2026-11-23** to `hardware_concurrency()` being
  called on every parallel batch (5.4 µs per call on glibc 2.39, about
  10 µs in place), **fixed in #324** and merged here at `f6e748e`. The
  THREAD_POOL figures in these CSVs predate the fix and carry that 10 µs
  (22% of a 4096-element call); K and Z have no figure for the
  lookup; M's is 0.38 µs per call (its line below). `main` also gained #325
  (Exponential `setParameters`, #268) and #326 (Cauchy quantile at
  subnormal p, #316), merged here at `e4026b3`.
  **Quiet sitting 2026-11-24** at `92ec8e1` (135/135), governor
  `schedutil`, 18 items each started at load 0.25–0.29
  (`docs/bench-evidence/2026-11-24-ivybridge-v300-bench-scaling-pool/`).
  **#156 A, re-run with the sustained timer**, `main` → dev, median of
  three: batch CDF gamma 3.36×, beta 6.96×, poisson 5.77×, binomial
  5.28×, student_t 0.67×; scalar CDF 4.0–9.8× on all five; quantile
  2.0–11.4×. Faster on dev: binomial and poisson PDF/LogPDF (0.5×,
  0.8×), batch CDF of gaussian, lognormal and von Mises (0.54–0.64×),
  `erf` (0.35×). **#156 B**: corvus selects SSE4 (2 lanes); a one-element
  call costs a whole vector for `erf` and `exp` and 1.3–1.9 elements at
  the long-span rate for the six costly kernels (K: 2.2–3.5), so a
  scalar entry point would leave gamma scalar CDF, beta scalar CDF and
  gamma quantile 3.7×, 6.3× and 8.7× above `main`. Per element at long
  spans SSE4 costs 2.5–2.8× K's AVX2 for the incomplete functions. Two
  bench artifacts explained in the README: the n = 4 row is the argument
  draw (same on K and Z), and `erf` at 256 and 4096 is 4K aliasing
  between the bench's two buffers (1.5×, this machine only). **Both stated
  on #156, 2026-11-24.** **Pool re-run after #324**: the thread
  pool now matches the local pool of four (34 / 67 / 232 / 842 µs
  against 45 / 77 / 246 / 854 before); plain and back to back, short
  calls are at the full clock (36 µs at 4096 against 111), and from a
  50 µs gap the governor behaviour is as before. Still owed on I: the
  kAvx re-capture, held on the pool design.
  **`turbostat` sampling skipped on I** [user, 2026-11-24]; reasons and
  the conditions for revisiting are under "Pool wake-up and batch
  spacing", measurement 3.
- **M — M1, NEON, macOS 27.2.1, AppleClang 21.0.0**: L1 done 2026-11-21
  at `cfbb992`, again 2026-11-22 at `c6ae2b9` and 2026-11-24 at
  `0f57256` (with #324, #325, #326): Release build
  warning-free, NEON, 135/135, backend GCD
  (`test_parallel_backend_selection` 3 pass, the OpenMP case skips;
  `WorkStealingCrossWaiting` skips here). **L2 done** at `cfbb992`:
  NEON block regenerated, 0 contract violations. **L2b done**:
  `data/accuracy/allow_NEON.csv` (2,971 rows). Row gate against v2.5.0
  on its grid: 874 values better, 126 worse, all small-shape densities
  (#314) except six groups that read the same in `allow_AVX.csv`
  (Student-t cdf and quantile at ν 39.9, Fisher-F survival, LogNormal
  quantile, Cauchy logpdf, Geometric pdf). Off-grid, seed 213: 2
  contract violations, #316 and #317; #318 reproduces; 142 rows skipped
  (120 NegativeBinomial survival, 21 Fisher-F quantile oracle gaps, 1
  Student-t quantile timeout). **L3 done**: `b766e45` and `90d1bef`
  sweeps bit-identical on NEON. **L4 done 2026-11-22** at `cfbb992`,
  screen locked, each item gated on ambient CPU below 15% (20% after
  3 h): bundle
  `data/profiles/dispatcher/2026-11-22T17-20-47Z_darwin-arm64_…_sha-cfbb992`
  (three `--large` runs of 26 min), timing tests 14/14,
  `simd_verification` 0 failed
  (`docs/bench-evidence/2026-11-21-m1-neon-v300-l4/`). **Its profiler
  and benches are `cfbb992`'s, before `8c55bfc`**, and the runs were not
  quiet throughout: `mediaanalysisd` held 1–2 cores in every profile
  run (34 rows spread more than 2×). Dev against `main`, old timer:
  densities level; batch CDF Gamma 2.5×, Poisson 5.0×, Binomial 6.5×,
  Beta 11.7×; scalar CDF up to 14.4× (Beta); quantile up to 12.6×
  (Beta). Superseded by the quiet run of 2026-11-24 below, which is the
  one stated on #156. **L5 done 2026-11-22, provisional**
  (`docs/bench-evidence/2026-11-22-m1-neon-threshold-rows/`): the
  derivation proposed 11 values; `table_ab_bench` (old timer) supported
  one, kNeon Poisson LogPDF 2048 → 10k, applied; seven spread-maximum
  candidates lost the A/B. Found: Bernoulli dispatches on Binomial's
  row, so AUTO Bernoulli CDF runs PARALLEL from 2048 and is 2.8–3.8×
  behind VECTORIZED at 4096 (a dispatch change, not a row; not acted
  on). L7 done 2026-11-23 at `7a1ec20`, two passes, indexing daemons on
  one to three cores in all but one run
  (`docs/bench-evidence/2026-11-22-m1-batch-gap/`): **the crossover does
  not move with the gap**, unlike Ivy Bridge — Poisson 10000 plain and
  4096 with `--init`, Gaussian 100000 in both, at 0, 2 ms and 20 ms
  alike. Forced PARALLEL after `--init` is faster at the same gap only
  where the work per call is small (Poisson 4096 by up to 43%, 10000 by
  9–15%, Gaussian 100000 by 15–26% at 2 ms and 20 ms); at Poisson
  100000 and 200000 it is about 11% slower, and VECTORIZED is 7% slower
  at every size. Without the pool the kNeon Poisson PDF threshold of
  4096 is wrong (PARALLEL 1.18–1.65× behind; 10000 is the crossover).
  `fit_gap_model.py`: where the plain run fits, a start delay of
  5–24 µs (K: 61–89), residuals up to 20% under the load. Not traced;
  **`pool_wake_bench` and idle burn done 2026-11-24** at `03cb724`,
  quiet, with `mediaanalysisd` and `corespotlightd` suspended around
  each run [user] (`docs/bench-evidence/2026-11-24-m1-pool-wake/`;
  measurements 2, 5, 6, 7; 8 not run). Busy caller, thread pool only: no
  cost up to 2 ms, 23–40 µs at 20 ms, work unchanged (K: 60–75 µs from
  500 µs). **Sleeping caller: the clock drops as on K**, serial 3.3×
  slower (15.0 to 49.4 ns), pool calls 2.1–4.0× longer; with the
  polling pool alive it loses nothing. The polling costs 7% on serial
  work, 5–10% on pool calls at 204,800 and 5.6–6.6 CPU-s per 10 s idle
  (I: 24.8). Work stealing matches the thread pool back to back (eight
  workers each) and is slower after any busy gap from 5 µs. So the
  polling pool is not pure cost on M1: it holds the clock for a caller
  that blocks, and an M1 guard that drops it would cost that caller
  3.3×. These runs predate #324, and need no re-run for it: the lookup
  costs 0.38 µs per call here (I: 5.4), and the thread pool reads the
  same after the fix within 2%. **L6 done 2026-11-24** at `92ec8e1`
  against `main` at `c049f23`, quiet (ambient CPU 2.4–3.9% at each
  start, screen locked, fourteen background processes suspended
  [user]), sustained timer, three passes
  (`docs/bench-evidence/2026-11-24-m1-v300-cost/`). #156 A: batch CDF
  Gamma 2.5×, Poisson 5.1×, Binomial 6.6×, Beta 12.4×; scalar CDF 3.5×
  to 14.4× (Beta); quantile 2.3× to 12.5× (Beta); densities and the
  elementary kernels level, discrete pmfs and the Gaussian and
  LogNormal quantiles faster. The old-timer run read the same. None of
  #156's NEON targets is met (Gamma batch CDF 46.9 ns per element,
  Beta 147.3, a scalar CDF call 435–1496 ns). #156 B, corvus at NEON:
  a one-element call costs a whole 2-lane vector for `erf`, `exp`,
  `lgamma` and `beta_p` and half of one for `gamma_p` and
  `gamma_p_inv`; 1.3–2.0 elements at the long-span rate for the costly
  kernels (K 2.2–3.5), so a scalar entry point recovers 1.3–1.6× at
  most and Gamma's scalar CDF, Beta's and Gamma's quantile would stay
  2.2×, 8.7× and 6.7× above `main`. Stated on #156. Pool
  measurement 8, same sitting: eight threads are 1.6× faster than four
  at 204,800 and level at 10,240 and below; work stealing is slower
  than the thread pool at 204,800 and 16–57 µs slower after a 2 ms
  gap. **#107 done 2026-11-24, not committed**: `simd_neon.cpp`
  includes `trig_cleanroom_data.inc` through five aliases, the NEON
  copy and its generator are removed; the NEON object's code and
  constants are byte-identical before and after, the generator
  reproduces the table, 135/135, `simd_verification` 140 pass. The
  three x86 files change in a comment only and were not built here.
  Nothing further owed on M.
  **Session state, 2026-11-24.** v2.5.1's legs on M are reported on
  `main` (#335, merged). Cleaned up: the
  work directory `libstats-m-2026-11-21` and its four worktrees
  (`v2.5.0`, `b766e45`, `90d1bef`, the L5 A/B tree) are removed; every
  result in them is under `docs/bench-evidence/`, the A/B tree's table
  as `2026-11-22-m1-neon-threshold-rows/dispatch_thresholds.h.proposed.txt`.
  The `libstats-main` worktree of `main` went with #335's merge, so
  this checkout is all that remains on this machine. Open: L5's rows
  stay provisional until the pool design (old timer; one value applied); the Bernoulli-on-Binomial's-row
  dispatch finding is not acted on. #107's comment change in the three
  x86 files is covered by CI, dispatched on `66f8377` and green on
  every leg (Linux gcc-14 and clang-17, Windows clang-cl and MSVC,
  macOS); the push itself starts no run on this branch. Method for a quiet
  run here: screen locked, Backblaze paused by the user, the fourteen
  processes listed in `2026-11-24-m1-v300-cost/quiet_run.sh.txt`
  suspended for the run [user approves each sitting], gate at two 10 s
  windows under 8% ambient CPU. Next on M: nothing until the #156
  decision or the pool design asks for a run.
- **Z — Zen 4, AVX-512, Windows 11, clang-cl 22.1.3 and cl.exe 19.52**:
  **L1 done 2026-11-23** at `164cbce` (code `e4026b3`): 135/135 under
  both, backend `std::execution`; `test_concurrency_stress` passes
  natively under `cl.exe` (195 s). Not warning-free as found: clang-cl
  gave two `getenv` deprecation warnings in
  `tools/bench/sustained_timing.h` (`strategy_profile`, since
  `8c55bfc`), fixed with the leg (`bench_timing::envValue`, 0 warnings
  after); cl.exe's eight are Highway's own (`D9025`). **L2 done**: 26,882 rows, 200 self-checks, 0 contract violations
  under both; AVX-512 block regenerated from the clang-cl sweep. **L2b
  done**: `data/accuracy/allow_AVX-512.csv` (3,025 rows). Row gate
  against v2.5.0 on its grid: 26,301 rows, 707 values better, 74 worse,
  all in the classes K and M reported (#314's small-shape densities;
  Student-t at ν 39.9 and at large ν, Fisher-F survival, LogNormal
  quantile, Geometric pdf), none re-checked at higher precision here.
  Off-grid, seed 213: 11,278 scored, 60 skipped, 1 contract violation
  (#317); #316 is gone. 14 quantile rows at 115–474 ε are real, each
  reference rechecked at 130 digits: small shape with a deep tail
  (Student-t, Fisher-F, InverseGamma, Weibull), 117–620 ulps from the
  correctly rounded value; `main` returns the same bits on 10 of the
  14 and is worse on the other four, so none is the adoption's;
  #104's subject, stated there. **L3 done**:
  `b766e45` and `90d1bef` sweeps bit-identical on AVX-512. **L4 done
  2026-11-24**, quiet throughout (CPU 2–3% at every gate, clock probe
  127 ms, every process opted out of throttling): timing 14/14,
  `simd_verification` 140 pass; bench against `main` (`7069adc`), three
  passes, sustained timer: discrete pmfs 0.48–0.63×, `erf` 0.47×, batch
  CDF Poisson 2.0×, Binomial 3.0×, Beta 5.7×, Gamma 0.85×; scalar CDF
  3.8–13.7×, quantile 2.8–15.6× on the gamma and beta families;
  **Gaussian and LogNormal quantile 1.47× and 1.53× slower, where K
  read 0.82×**: the probit is now `corvus::erfcinv` on a one-element
  span, 106 ns here against 178 on K, and `main`'s scalar probit is
  3× faster here than on K (corvus #110's cost, on `erfcinv`). Three `strategy_profile --large`
  runs with `164cbce`'s profiler (sustained-v1), 81 min each, 3 of 81
  rows spread > 2×: bundle `2026-11-24T07-53-40Z_windows-x86_64_…`.
  **L5 derived and A/B'd; rows held** [user, 2026-11-24: hold every
  kAvx512 value, Erlang LogPDF included; reconsider if another
  machine's capture moves its table materially, since L1 is re-run
  then anyway]: the script
  proposes 34 kAvx512 values, 30 earlier (most 25k → 8192 or 10240);
  quiet ABAB: AUTO rows > 1.25× behind 35 → 15, 22 faster, 3 slower.
  PARALLEL wins at 8192, loses at 10000 and wins again from 25000 on
  most rows, so the 15 rows read as 10240 leave AUTO 1.3–1.6× behind at
  8192; and Cauchy LogPDF regresses 1.29× at 8192 through Student-t's
  row. Only one row has a 2× win below its value (Erlang LogPDF,
  75k, PARALLEL 0.46× at 50000); the gains are all in the 1.3–1.6×
  band. L1 on the proposed header (clang-cl): 135/135.
  `docs/bench-evidence/2026-11-24-zen4-threshold-rows/`. **L6 done
  2026-11-24**: bench stated on #156. **L7 done 2026-11-24**, quiet, same sitting:
  Gaussian's crossover is 50000 at gap 0 and 100000 at 2 ms and 20 ms;
  Poisson's is 10000 (the first size that forks) at every gap; plain and
  `--init` read the same. Forced PARALLEL costs 1.9× more per element at
  2 ms and 2.6× more at 20 ms (Gaussian, 50000); `--init` does not
  remove it; VECTORIZED also slows 1.1–1.2× at 20 ms.
  **`pool_wake_bench`, sleeping caller, idle burn and clocks done
  2026-11-24** at `3662ae7`, quiet
  (`docs/bench-evidence/2026-11-24-zen4-pool-wake/`; measurements 1, 2,
  3, 5, 6, 7, 8). Busy caller, thread pool: a late start only, +4–26 µs
  a call at 2 ms and +24–48 µs at 20 ms, work unchanged (K: 60–75 µs
  from 500 µs); `--init` does not remove it. Workers run 1.5× slower
  per element than the caller at every gap, gap 0 included: four
  logical processors boost to 130–144% of nominal and the other eight
  sit at 99%; nothing is ever parked. Sleeping caller: serial 1.5×
  slower (10.1 to 15.2 ns), the pool call 87–93 µs longer, and the
  polling pool does not help, unlike M. Idle burn 0.19 CPU-s per 10 s
  (I: 24.8). A 12-thread pool is 1.6–1.9× faster than the library's 6
  on the integer workload. **Driven directly,
  the work-stealing pool starts 13–15 ms after the submit for a busy
  caller** (its idle `sleep_for(100 µs)` runs to the 15.6 ms Windows
  timer tick); the batch API does not show it on Z, where forced
  WORK_STEALING reads as PARALLEL; cause confirmed and filed as #333.
  **For #156, 2026-11-24**, quiet
  (`docs/bench-evidence/2026-11-24-zen4-v300-msvc-and-scaling/`): the
  all-MSVC build (corvus at AVX2) against `main`: batch CDF Beta 13.0×
  and Binomial 6.1× (clang-cl 5.7× and 3.0×), scalar CDF up to 18.8×,
  quantile up to 19.4×. `corvus_scaling_bench` with `erfinv` and
  `erfcinv` added: a one-element call costs half a vector to a whole one
  (`erfinv` 130 ns against 15.5 per element at long spans), so a
  dispatch-only scalar entry recovers little; at the per-element rate
  Beta's CDF and Gamma's quantile would still be 2–3× above `main`.
  Stated on #156.
  `fit_gap_model.py` on L7: Poisson a
  factor (1.13 at 2 ms), Gaussian neither. Evidence:
  `docs/bench-evidence/2026-11-24-zen4-v300-l4/`,
  `docs/bench-evidence/2026-11-24-zen4-batch-gap/`. **L1 again
  2026-11-25** at `321ad02` (with #107), fresh builds in a standing
  worktree: 135/135 under clang-cl (0 warnings) and cl.exe (Highway's
  eight). Nothing owed on Z; L5 stays held.
- **K — Kaby Lake, AVX2, macOS, AppleClang 15**: L1 done 2026-11-21 at
  the commit that carries this line: Release build warning-free, AVX2
  with FMA, 135/135, backend GCD. L1 found one defect at `cfbb992`:
  Poisson's WORK_STEALING batch CDF called `pool.waitForAll()` (#180's
  fix undone by the port, `30479d9`), which
  `ConcurrencyStress_Poisson.WorkStealingCrossWaiting` fails 10 of 10
  here and skips on every other machine and CI leg. Fixed on this branch;
  **no result changes, so no leg re-runs L2**. The same commit pair
  removes an AppleClang warning in `tools/accuracy_sweep.cpp`. CI green
  on `09c7ea0` (run 44959041879).
  **L2 done** at `09c7ea0`: 26,882 rows, 200 self-checks, 0 contract
  violations; AVX2 block regenerated. **L2b done**:
  `data/accuracy/allow_AVX2.csv` (3,037 rows). Row gate against v2.5.0
  on its grid: 26,301 rows, 912 values better, 102 worse; 83 are
  small-shape densities (#314) and the other 19 are rounding changes
  amplified by ill-conditioned formulas, each reference checked at 80
  digits (detail on #314, including the Student-t log normaliser at
  large ν, 20.7 ulp at ν = 1e18 against 11.4 on v2.5.0). Off-grid, seed
  213: 11,196 scored, 141 skipped, 2 contract violations, both known
  (#316, #317). Fisher-F pdf at 1,295 ε, shapes (21490, 9667),
  **examined 2026-11-24**: `main` returns the same bits; the log density
  is good to 1–15 ε relative and the pdf is its exponential 20 standard
  deviations from the mode (logpdf −416), so the pdf's relative error is
  the log density's absolute error (2.9e-13). Conditioning, not a
  regression or a wrong branch. **L3 done**: `b766e45` and `90d1bef` sweeps bit-identical on
  AVX2. Working files are in `../libstats-kleg/` (local). **L4 done
  2026-11-22** at `255e1fe` (code `ec8cdc9`), quiet (every item started
  below a 1-minute load of 2.0), on AC, screen locked: bundle
  `data/profiles/dispatcher/2026-11-22T00-28-35Z_darwin-x86_64_…_sha-255e1fe`
  (three `--large` runs of 40 min, timing tests 14/14,
  `simd_verification` all pass). **Its profiler is `255e1fe`'s, before
  `8c55bfc`**, so rows derived from it are provisional. Against kAvx2:
  66 ok, 9 DIFF, all on commented rows; the derivation proposes three
  values (Bernoulli CDF 4096→8192, Beta LogPDF 8192→25k, Logistic LogPDF
  25k→8192), not applied, no A/B run. Dev against `main`, same sitting,
  old timer (`docs/bench-evidence/2026-11-22-kabylake-v300-cost/`):
  densities level or faster; batch CDF Gamma 1.7×, Poisson 3.5×, Binomial
  4.8×, Beta 8.2×; scalar CDF up to 17.4× (Beta), quantile up to 16.5×
  (Beta). Re-run 2026-11-23 with the sustained timer (`sustained/` there): the
  ratios hold within 6% (`erf` 0.73× is real here). **L6 done 2026-11-23**: stated
  on #156. L5 held on the pool design. **L7 done 2026-11-23** at `c6ae2b9`,
  quiet (`docs/bench-evidence/2026-11-23-kabylake-batch-gap/`).
  Crossover at gap 0 / 2 ms / 20 ms, plain run: Poisson 4096 / 10,000 /
  25,000, Gaussian 50,000 / 50,000 / 100,000. `--init` run: Poisson 4096
  and Gaussian 50,000 at every gap. So the crossover moves with the gap
  when only the thread pool exists. Forced PARALLEL after `--init` is
  faster only with a gap (up to 2× at 2–20 ms) and 1.3–1.5× slower back
  to back, the opposite of I; VECTORIZED is 9% slower. **AUTO (work
  stealing) against forced PARALLEL, `--init` run**: faster back to back
  (0.52–0.82), slower with any gap below 25,000 for Poisson (1.1–1.9×)
  and at 50,000–100,000 for Gaussian (1.2–1.7×), faster at every gap
  from 25,000 (Poisson). Unlike I: Gaussian VECTORIZED at 4096 doubles
  (2.3 to 4.6 ns/element) at gaps of 2 ms and up in both pool states;
  cause not measured. **2026-11-24 at `164cbce`** (`main` merged, with
  #324), rebuilt warning-free, 135/135: measurements 4 to 8 of "Pool
  wake-up and batch spacing" done on K, quiet (results under each item;
  `docs/bench-evidence/2026-11-24-kabylake-{pool-threads,batch-gap-rows,gap-controls}/`);
  #47's AppleClang validation and the Fisher-F row done (below and
  above). **Owed on K: L5, held on the pool design.** v2.5.1's legs on
  K are reported on `main` (#331).

### What every leg produces

| Step | Output | Committed as |
|---|---|---|
| L1 native correctness | warning-free Release build; `system_inspector --quick`; correctness suite count; the backend `test_parallel_backend_selection` reports | status line |
| L2 accuracy sweep and block | this ISA's block in `docs/ACCURACY_CHARACTERIZATION.md`, 0 contract violations | `docs(accuracy): regenerate <ISA> block at <sha>` |
| L2b row gate and off-grid sweep | `data/accuracy/allow_<ISA>.csv`; a list of off-grid findings | the allowlist; findings filed as issues |
| L3 prototype bit-identity | sweep at `90d1bef` identical to sweep at `b766e45` | status line |
| L4 quiet session | timing tests, `simd_verification`, `tools/bench` dev against `main`, three `strategy_profile --large` runs | bundle under `data/profiles/dispatcher/`; bench evidence under `docs/bench-evidence/` |
| L5 threshold rows | this tier's rows in `dispatch_thresholds.h`, settled by `table_ab_bench` | header, script and bundle together |
| L6 report | bench numbers stated on #156; status line updated | — |
| L7 batch-gap check | three `batch_gap_bench` runs (plain, `--init`, `--init --auto`); the crossovers by gap in the status line | `docs/bench-evidence/<date>-<machine>-batch-gap/` |

### Rules on every machine

- Commit, push, comment, label or file only when the user asks. Commits
  are signed; never disable signing.
- **Commit only your own**: your tier's rows in `dispatch_thresholds.h`,
  your `isa=` block, your `allow_<ISA>.csv`, your bundle and evidence
  directories (L7's included), your status line here. `git pull --rebase` before every
  commit; this file is the one every leg edits.
- **Library code changes only by the user's decision.** A new defect or
  an unexpected oracle row: stop and report with evidence. If the fix
  also applies to `main`, it goes to `main` first on a small branch with
  a PR, then `main` is merged here; adoption-only fixes go straight to
  this branch. A fix that changes results means the other legs re-run L2
  from the new head, so say so in your status line.
- **A new guard must be shown to fail on the unfixed library**, on the
  platform it targets (AGENTS.md, Testing).
- CI: this branch has no push trigger. Dispatch with
  `gh workflow run ci.yml --ref dev/v3.0.0-corvus-adoption` after a
  change to `src/`, tests, tools or the workflow; do not poll.
- Performance numbers from Release builds only. **Only a quiet-machine
  run is a record**; anything measured on a machine in use is reported
  in conversation and written nowhere.
- Quiet runs alone on the machine: no builds, sweeps or other sessions.
  K: wait for the 1-minute load to fall below 2.0. I: the script waits
  for 0.30.
- Z: a locked screen puts running processes under Windows power
  throttling (EcoQoS), 1.5× slower, and it stays with the process after
  unlock. Opt each measured process out (`SetProcessInformation`,
  ProcessPowerThrottling, EXECUTION_SPEED in ControlMask, StateMask 0);
  children do not inherit it, so launch every binary yourself, not
  through ctest. Helper:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/nothrottle.ps1`.
  The CPU also steps down some 8–15 s into sustained load: set
  `LIBSTATS_BENCH_WARMUP_SECONDS=25` for the benches and precede each
  `strategy_profile` run with a discarded short pass.
- **Every timing bundle records the clock policy it ran under** [user,
  2026-11-25]. Linux: kernel version (`uname -r`) and, from
  `/sys/devices/system/cpu/cpu0/cpufreq/`, `scaling_driver`,
  `scaling_governor` and `energy_performance_preference` where it
  exists, plus `/sys/devices/system/cpu/intel_pstate/status` on Intel.
  macOS: `pmset -g` (power source and low-power mode). Windows: the
  active power plan (`powercfg /getactivescheme`). The driver, not the
  distribution, decides which governors exist: `schedutil` on I is the
  passive `intel_pstate` of a CPU without hardware P-states, and newer
  Intel and AMD CPUs pick their own frequency, so I's Linux readings are
  not assumed to hold for them.
- Macs: run every library binary outside the Claude Code sandbox. It
  blocks the `hw.*cachesize` sysctls, so cache sizes read 0 and the
  cache-derived tuning changes; it also blocks the SSH agent.

### The steps

**L1 — native correctness.** Fresh Release build (`cmake --preset release
-G Ninja`, or the Windows presets in AGENTS.md); expect no warnings. Run
`system_inspector --quick` and record the tier. Run
`ctest -LE "timing|benchmark"`; the count is 135 on Linux and may differ
by platform. Record what `test_parallel_backend_selection` selects: GCD
on the Macs, `std::execution` on Windows, and on Linux `std::execution`
with TBB or pthreads without. On the Macs this is the first native run of
the #179 change.

**L2 — accuracy sweep and block.** The tree must be clean and at a pushed
commit; the sweep stamps the commit it measured. `accuracy_sweep
sweep.csv` (write it outside the repo), then
`python3 tools/accuracy_vs_mpmath.py sweep.csv` with mpmath installed (ask
the user before installing anything). A full sweep rewrites only its own
`isa=` block; check with `git diff`. Expect 0 contract violations and no
row worse than the same row in that ISA's v2.5.0 block beyond noise.
Keep the sweep CSV and the per-row scores (`--rows-out`) for L2b.

**L2b — row gate and off-grid sweep.** Commands and how to read the
output: `docs/ACCURACY_CHARACTERIZATION.md`, "Row gate" and "Off-grid
sweep".
1. Row gate against v2.5.0, on the same inputs: build `v2.5.0` beside
   the checkout, sweep it, then sweep this branch with `--grid-from` that
   CSV, and score with `--baseline` the v2.5.0 rows. Two builds sweep
   neighbouring doubles otherwise, and the comparison is not row for row.
   Expected here: about 1,100 values better; the worse ones are densities
   below shape 20 (#314). Anything else worse is a stop-and-report.
2. Write this ISA's allowlist with `--allow-out
   data/accuracy/allow_<ISA>.csv` from the L2 sweep.
3. Off-grid: `accuracy_sweep offgrid.csv --offgrid` (default seed 213),
   then the oracle with `--no-doc --row-timeout 20 --budget 4`. Known on
   AVX: #316 (Cauchy quantile), #317 (Fisher-F at subnormal x), #318
   (TruncatedNormal narrow-window density). Report anything else.
   Known oracle gap: 120 NegativeBinomial survival rows exceed the time
   limit and are not scored.

**L3 — prototype bit-identity.** The batch-CDF span routing and the
integer `lgamma` table (`23f6fb3`, merge `90d1bef`) must not change a
bit. Build `b766e45` and `90d1bef` Release, run each one's
`accuracy_sweep`, and `diff` the two CSVs past the banner lines: expect
no difference. Record the batch timings the bench in L4 gives for the
discrete PMFs and the incomplete gamma and beta CDFs. The `PROTOTYPE`
tags came out on 2026-11-24, once M, Z and K had all reported
bit-identity (comments only: `src/math_utils.cpp`,
`src/negative_binomial.cpp`, `include/libstats/core/math_utils.h`).

**L4 — quiet session**, last on each machine, one sitting:
1. `ctest -j1 -L timing` (Z: launch each binary directly).
2. `tools/simd_verification`.
3. `tools/bench/distributions_bench.cpp` and `elementary_bench.cpp`,
   each built against a `main` build and against this branch with the
   library's own flags, run alternately, three passes. Build lines:
   `tools/bench/README.md`. This is the machine's number for #156.
4. Three `strategy_profile --large -o strategy_profile_run<i>.csv` runs.
5. Bundle as `data/profiles/dispatcher/README.md` describes
   (`metadata.json`, `manifest.txt` with the conditions and the spread,
   the three CSVs, `sustained_crossovers.txt`, logs).

**L5 — threshold rows.** Every table is stale: the adoption changed
kernel costs on every tier, and #311 changed them again on CPUs without
FMA. Derive this tier's rows from the L4 bundle with
`scripts/derive_threshold_rows.py` (recipe in the dispatcher README; its
`TABLES` entry for your tier must point at the new bundle), then settle
them with `tools/bench/table_ab_bench.cpp`, old table against new, run
old, new, old, new. Rules [user, 2026-10-14, still in force]: a candidate
at the machine's fork threshold is not a measured crossover; skip a row
whose three runs spread more than 2× unless its maximum still clears the
table by more than one grid step; per-row override comments stay unless
the comment is stale. Then L1 again. Tier ownership: kAvx512 is Z's
(clang-cl), kNeon is M's, kAvx2 is K's, **kAvx is I's** [user,
2026-11-19: native values from the only silicon whose native tier is
AVX; on I a candidate of 6144 stays 6144 where PARALLEL is behind at
4096]. kSse2 delegates to kAvx.

**L6 — report.** State the bench result on #156 and update your status
line. Do not decide #156's gates; that follows all four machines.

**L7 — batch-gap check** [user, 2026-11-22]. Independent of L1–L6: run
it in the L4 sitting if that is still ahead, otherwise in a short quiet
sitting of its own. It changes no code and decides nothing; it measures
whether this machine's V→P crossover moves when batches stop arriving
back to back, which is what the pool design waits on.
1. Build `tools/bench/batch_gap_bench.cpp` against this branch with the
   library's own flags, as the L4 benches are built
   (`tools/bench/README.md`). One binary; no `main` build needed.
   It has been compiled only with GCC 13 so far (against this branch and
   against `main`), and CI does not build `tools/bench`: if AppleClang,
   clang-cl or cl.exe rejects it, report the error; a portability fix to
   the bench itself is yours to make and commit.
2. On a quiet machine, under the same rules as L4 (Z: throttling
   opt-out, launch the binary yourself, one discarded pass first; Macs:
   outside the sandbox), run three times, each its own process:
   `batch_gap_bench > gap_plain.csv`,
   `batch_gap_bench --init > gap_init.csv`,
   `batch_gap_bench --init --auto > gap_init_auto.csv`.
   Three to five minutes each. Keep the default busy gap; `--sleep` is
   not asked for (Windows cannot sleep for 50 µs).
3. Commit the three CSVs with a README (machine, commit, compiler,
   conditions, load) under
   `docs/bench-evidence/<date>-<machine>-batch-gap/`.
4. In your status line state, for Poisson and for Gaussian, the
   crossover the `#` lines give at gap 0, 2 ms and 20 ms, in the plain
   run and in the `--init` run. That answers the two questions:
   does the crossover move with the gap, and is forced PARALLEL faster
   at the same gap once `--init` has started the work-stealing pool.
   **K also**: at each gap, is the AUTO column (work stealing there)
   faster than forced PARALLEL in the `--init` run.
5. If the numbers look unlike anything described below, say so and stop:
   the bench shows that a gap matters, not why, and a machine may have a
   different cause (M: woken workers on efficiency cores; Z: core
   parking). A per-machine trace is the follow-up, on the user's word.

### Machine-specific items

- **M**: #107 (one trig table; the change is on the NEON side) was done
  here on 2026-11-24, the NEON object byte-identical before and after;
  the three x86 files change in a comment only. #47 needs nothing further from M: K's AppleClang
  validation closed it (2026-11-24), and M's NEON block and its 135/135
  with the entropy guard cover this machine; the "≤ 1.7 ulp" once asked
  here does not hold at every point on either tree (K's item below).
  #156's original numbers are from this
  machine: `gamma_p` 272 and `beta_p` 823 ns per element against targets
  of 40 and 30 that corvus's accuracy contract puts out of reach.
- **Z**: clang-cl is the performance build and supplies kAvx512; cl.exe
  is the correctness build. CI's cl.exe leg excludes
  `test_concurrency_stress` (starvation at cl.exe-built corvus speed);
  say whether it passes natively. A capped-tier leg
  (`LIBSTATS_MAX_SIMD_TIER`) is optional.
- **K**: the only hyper-threaded macOS machine, so the only one where
  AUTO reaches WORK_STEALING. #47's AppleClang validation applies here
  too. **Done on K 2026-11-24** at `164cbce`, 996 points against mpmath:
  pdf and logpdf are bit-identical to `main`'s (v2.5.0's `8672a3a` had
  already taken the density off the Tier 2 normaliser); what the branch
  fixes on AppleClang is `getEntropy()`, 1.5e-11 to 2.1e-5 off on `main`
  for κ from 0.01 to 44 and within 1.2e-14 here. `main`'s values miss
  `DH2_M1_VonMisesEntropy`'s 1e-13 at four of its rows, so the guard
  fails on the unfixed state. "≤ 1.7 ulp" does not hold at every point
  in ulps of the result (pdf 177 at κ = 100, logpdf 9.5 at its zero
  crossing), on both trees alike: conditioning. Stated on #47.
- **I**: after the capture, assemble the bundle, derive kAvx, write the
  table and the documentation column #291 asks for, regenerate the AVX
  block, and state the bench on #156. Unverified here: that mixing
  libstats' VEX code with corvus's non-VEX SSE4 kernels costs nothing on
  this CPU (check `other_assists.avx_to_sse` with `perf stat` if the
  bench reads oddly).

### #156 — decided [user, 2026-11-25]

All four machines measured both items below, and the decision is on the
issue (comment of 2026-11-25):

1. **The cost is accepted for v3.0.0.** Against `main`, across the four
   machines: batch CDF 0.67–12.4×, scalar CDF 3.5–14.4×, quantile
   2.0–15.6× on the gamma, beta, poisson, student_t and binomial
   families; up to 19× on the all-MSVC build. The batch API is the fast
   path.
2. **No local fast path.** Keeping v2.5.x cores for scalar calls would
   bring back the defects the adoption fixes and make scalar and batch
   disagree.
3. **The gate is a regression guard, not a target.** The NEON targets
   named below are withdrawn. Each tier's figures on #156 are the
   v3.0.0 baseline; a later change may not worsen a cell by more than
   1.25×.
4. **The release documents state the cost** per route and tier, and
   advise the batch API for bulk work and clang-cl on Windows.
5. **Follow-ups in v3.1.0:** #338 (a batch quantile for the
   corvus-backed inverses) and #339 (corvus scalar code paths, which
   waits on corvus #110).

#156 is labelled `fixed-on-branch` and closes with the release. What
follows is the record of what was asked of each machine.


The gate on #156 was set on M as absolute NEON targets (`gamma_p` ≤ 40
and `beta_p` ≤ 30 ns per element at batch, a single-lane call ≤ 250 ns).
corvus's comment of 2026-10-30 says they cannot be met under its accuracy
contract, so the gate is re-baselined here, and the decision is whether
the measured cost is accepted or some routes keep a local fast path. Two
measurements feed it. Both are quiet runs under the leg rules above.

**A — the bench against `main`.** `tools/bench/distributions_bench.cpp`
and `elementary_bench.cpp` from this branch's tip, built with the
library's own flags against a Release build of `main`'s tip and one of
this branch (`tools/bench/README.md`). Three passes, `main` first on
passes 1 and 3 and this branch first on pass 2. The benches must be
`8c55bfc` or later (the sustained timer, #320). State the result on
#156 as the other machines have: ns per element, median of three,
`main` → dev with the ratio, for batch CDF, scalar CDF and quantile of
gamma, beta, poisson, student_t and binomial, and what moved elsewhere.
Evidence under `docs/bench-evidence/<date>-<machine>-…/`.

**B — corvus per call against span length.**
`tools/bench/corvus_scaling_bench.cpp` at `4db8257` or later (it gained
`erfinv` and `erfcinv` columns), built against this branch's corvus;
three passes. Record the tier it prints on its first line. It tells
whether a one-element call costs a whole vector, which decides what a
scalar entry point could recover. Z reads half a vector to a whole one
at 8 lanes; a 2-lane tier may read differently, which is why M and I
matter here.

| Machine | A — bench against `main` | B — per-call scaling |
|---|---|---|
| **M** (NEON) | **Done 2026-11-24** at `92ec8e1`, quiet, sustained timer (`docs/bench-evidence/2026-11-24-m1-v300-cost/`): batch CDF Gamma 2.5×, Poisson 5.1×, Binomial 6.6×, Beta 12.4×; scalar CDF up to 14.4×, quantile up to 12.5× (both Beta). Stated on #156. | **Done 2026-11-24**, same sitting, tier NEON: a one-element call costs a whole 2-lane vector for `erf`, `exp`, `lgamma` and `beta_p`, half of one for `gamma_p` and `gamma_p_inv`; 1.3–2.0 elements at the long-span rate. A scalar entry point recovers 1.3–1.6× at most. Stated on #156. |
| **I** (AVX) | **Done 2026-11-24** at `92ec8e1`, quiet, sustained timer (`docs/bench-evidence/2026-11-24-ivybridge-v300-bench-scaling-pool/`): batch CDF 3.4–7.0× for gamma, beta, poisson and binomial and 0.67× for student_t; scalar CDF 4.0–9.8×; quantile 2.0–11.4×. Replaces the statement of 2026-11-22. Stated on #156. | **Done 2026-11-24**, same sitting, tier SSE4 (2 lanes): a whole vector per one-element call for `erf` and `exp`; 1.3–1.9 elements at the long-span rate for the costly kernels, against K's 2.2–3.5. At that rate Gamma's scalar CDF, Beta's scalar CDF and Gamma's quantile would still be 3.7×, 6.3× and 8.7× above `main`. Stated on #156. |
| **K** (AVX2) | Done: the sustained re-run of 2026-11-23, stated on #156. | **Done 2026-11-24** at `d23d353`, quiet, tier AVX2 (`docs/bench-evidence/2026-11-24-kabylake-corvus-scaling/`): a one-element call costs a whole 4-lane vector for `erf`, `exp`, `gamma_p` and `beta_p` and 0.5–0.8 of one for the rest; 2.2–3.5 elements at the long-span rate for the costly kernels. At that rate Beta's scalar CDF and Gamma's quantile would still be 6× and 3.7× above `main`. Stated on #156. |
| **Z** (AVX-512) | Done 2026-11-24, clang-cl and all-MSVC, both stated. | Done 2026-11-24, both compilers. |

All of it is in: the decision has what it asked for. B on M and I, the
two 2-lane tiers, reads alike (1.3–2.0 and 1.3–1.9 elements per
one-element call).

### After the legs

1. Open the dev-branch PR once one other leg is complete.
2. Remove the `PROTOTYPE` tags once all three have reported L3. **Done
   2026-11-24** [user].
3. #156: decided 2026-11-25 (the section above).
   The pool design: decided 2026-11-25 ("Pool design — decided"), which
   carries the order of work: prototype, guards, stress and
   ThreadSanitizer, one measurement sitting per machine, then the table
   capture, L1 on every machine, and L2 again on I, K and M, which they
   owe for #326 ("where it stands", "Three accuracy blocks").
4. Release docs: version 2.5.0 → 3.0.0, `CHANGELOG.md`,
   `docs/VALIDATION_HISTORY.md` (matrix, costs, the threshold tables),
   `MIGRATION_GUIDE.md`, README and AGENTS status lines.
5. Release: PR to `main` closing every issue labelled
   `fixed-on-branch`; CI green; signed tag; GitHub release; close the
   milestone. Then pylibstats and libhmm (Cross-Repo).

## State of the v3.0.0 issues [DERIVED, 2026-11-24]

Milestone #6: 33 open, 3 closed (#312; #266 and #272 as split). Issues stay open until this
branch reaches `main`; `fixed-on-branch` marks the ones that are fixed
and guarded here, each with a status comment.

- **Labelled `fixed-on-branch` (29):** #47, #103, #104, #107, #108,
  #110, #113, #126, #136, #138, #141, #156, #179, #213, #214, #221, #247, #248,
  #251, #252, #258, #262, #267, #270, #271, #287, #290, #311, #320. #107
  was labelled by M on 2026-11-24 (one trig table, `7879e9a`). #103, #104
  and #141 were labelled 2026-11-24, once all four ISA blocks were
  regenerated here: 0 contract violations each, and the large-shape
  `gamma`, `erlang` and `inverse_gamma` CDF rows at 1.4–1.8e-16 on every
  tier. #104's contract was settled 2026-09-02 (finite, best-effort); it
  was listed here as a decision owed by mistake. #47 was labelled
  2026-11-24 on K's AppleClang validation, checked against the issue's
  2026-08-20 closing conditions: the two-tier Bessel scheme is gone, the
  von Mises rows read the same on all four tiers, and the entropy guard
  misses on `main`. M made no statement of its own; its NEON block and
  its 135/135 with the guard in it cover it.
- **Wait on other machines or the pool design (2):** #52 (timings and
  rows stated per tier; done for I) and #291 (kAvx derived; provisional
  until the pool design).
- **Closed as split 2026-11-24 [user]:** #266 (CDF side resolved here;
  the density prefactors are #314) and #272 (Beta and NegativeBinomial
  CDF NaNs cleared here; the density paths are #315).
- **Decided 2026-11-25 [user]:** #156, the cost accepted ("#156 —
  decided" above); labelled `fixed-on-branch` with the others.
- **Left by #104, not blockers [OPEN]:** the 14 off-grid quantile rows
  over 100 ε (small shape, deep tail; present on `main`) are #334 in the
  patch milestone; the per-family quantile accuracy notes #104's acceptance
  asked for are not written (release docs); `truncated_normal` quantile
  on AVX-512 reads max_rel 6.3e+291 at p = 0.5, as in the v2.5.0 block:
  that row is #277 in the patch milestone (the median of a symmetric
  window is −1.4e-16, not 0, on Zen 4).
- **Pool design, decided 2026-11-25, not started (2):** #323 (slice
  arithmetic, corrected 2026-11-22) and #333 (no wake-up on submit; a
  timer tick on Windows). Both in v3.0.0.

Kept in libstats and not on corvus, by decision: the Student-t tail
kernels (`beta_continued_fraction`, `beta_i_large_a_half`) and
`inverse_t_cdf`'s solver; the density prefactor family
(`log_gamma_prefactor`, `beta_prefactor_constant`, `log1pmx`,
`stirling_remainder`); `inv_survival_normal_log`; the incomplete gamma
below `DBL_MIN` (`gamma_p_from_scaled`); libstats' own `exp`, `log`,
`cos` and `sin` SIMD kernels.

Filed from this work into the patch milestone: #314, #315, #316, #317,
#318, #319, #334. Small and unfiled:
- `scripts/analyze_crossovers.py` reports Uniform and NegBinomial as "not
  in table" (a wrapped row; a name mismatch). The derive script reads both.
- `src/simd_avx.cpp`'s cos/sin comment still reads "UNPROVEN LEG"; the
  native gate reads 1 ULP.
- Clang without an FMA tier warns that `kBudgetTight` is unused
  (`tests/test_trig_ulp_gates.cpp`).
- The Windows CI legs warn that `ilammy/msvc-dev-cmd` v1.13.0 targets
  Node 20. It is that action's latest release; harmless today. The likely
  fix is a script step calling `vcvars64.bat`, probably fleet-wide.

## Decided [DERIVED]

For v3.0.0:
- **Scope and order [user, 2026-10-26, 2026-11-19].** v3.0.0 is the
  corvus adoption plus the issues it needs. The accuracy patch
  (milestone #8) ships after it, as v3.0.1; none of its issues close as
  overtaken, since they are real defects. Then v3.1.0 New Distributions,
  then the Architecture Refactor.
- **Per-call cost does not block v3.0.0 [user, 2026-11-18].** It ships on
  corvus v1.2.0 for the accuracy. Speed is recovered on a v3.1.0 pinned
  to corvus v1.3.0 (corvus #110, #111). Speedups that need no corvus
  change may be scoped into v3.0.0. Measured on Ivy Bridge, corvus's
  worst case in the fleet (two lanes, no FMA): single calls `lgamma(2)`
  18 → 144 ns, `lbeta` 76 → 1,742, `gamma_p` 112 → 1,358, `beta_i`
  342 → 3,531 against v2.5.0.
- **No PR for the dev branch** until one other machine has run its leg
  [user, 2026-11-18].
- **Placement [user, 2026-11-19].** A fix that applies to `main` goes to
  `main` first, then `main` is merged into the dev branch. clang-cl work
  stays on the dev branch: clang-cl is needed only because of corvus and
  Highway under MSVC.
- **A fetched corvus builds as Release and uninstrumented** whatever
  libstats' configuration [user, 2026-10-25]. Instrumenting the
  dependency is noise, and corvus at -O0 starves the concurrency tests.
- **CI budgets [user, 2026-10-26; re-measured 2026-11-18].** Coverage 30
  minutes with the suite whole. cl.exe leg 40 minutes, all four Highway
  tiers, `test_snapshot_consistency` in, `test_concurrency_stress` out
  (it starves at cl.exe-built corvus speed; it runs on every other leg).
  If a slow runner pushes that leg at its budget, the snapshot test is
  the first thing to take out again.
- **`expm1`/`log1pmx` are not adopted from corvus in v3.0.0** [user,
  2026-11-18]. No accuracy to win (the rows that rest on `expm1` already
  read 1e-16), corvus exports them as spans only (56 and 105 ns for one
  call against glibc's 13), and libstats' uses are mostly scalar. On
  spans `expm1` is level with glibc, so the Exponential and Rayleigh
  batch loops are a v3.1.0 candidate.
- **kAvx takes Ivy Bridge's native values** [user, 2026-11-19], from one
  quiet capture after `src/` settles.
- **Trig and coefficient duplication stays** [user, 2026-11-21; #108,
  #221]: AGENTS.md, "SIMD kernel conventions".
- **Parallel backends for the `safe_*` wrappers** [user, 2026-11-21;
  #319]: GCD on macOS; `std::execution` on Windows and on Linux with
  TBB; the library's own pool on Linux without TBB; OpenMP not selected.
  v3.0.0 carries only the selection fixes (#179); the rest is #319.
- **#156's gates** are decided after every machine has post-bump numbers
  [user, 2026-11-18].

Standing:
- **Windows performance build is clang-cl [user, 2026-10-05]**, with
  cl.exe's `/arch:` flags (`/arch:AVX2` = AVX2 + FMA, F16C, BMI1/2,
  LZCNT, MOVBE; `/arch:AVX512` adds AVX-512 F, CD, BW, DQ, VL), so both
  compilers build the same ISA. cl.exe is the correctness build
  (standards WINDOWS-TOOLCHAIN §5); Z's timing and kAvx512 come from
  clang-cl.
- **Why the adoption is a major version [user, 2026-10-04].** The
  installed package gains `find_dependency(corvus CONFIG)`, so under
  `SameMajorVersion` a `find_package(libstats 2.x)` consumer would accept
  it and fail to configure without corvus; it also ends "zero external
  dependencies".
- **Accuracy-for-speed budget [user, 2026-10-04].** Slowdowns of 2×, 5×
  and up to ~10× are acceptable for accuracy fixes; 50–150× is not. 10×
  is a guide, not a hard limit: weigh how hot the function is for
  consumers (a Gaussian pdf matters far more than a von Mises quantile)
  against the effort, and maximise speed at the accuracy kept, within
  reasonable effort. Over-budget rows get a performance pass or a
  theoretical case that the cost is intrinsic (Known Gaps). Per-call
  corvus cost in v3.0.0 is the decided exception above.
- **Vectorized log1p and expm1 [user, 2026-10-04].** A `VectorOps`
  primitive is preferred to an inline composite. `vector_log1p` and
  `vector_expm1` are derived generically from each tier's `vector_log`
  and `vector_exp`; neither serves log1p(x) − x, whose cancellation
  amplifies their few ulp, so `detail::log1pmx_series` (atanh form) does.
  Pointing them at corvus's spans is the v3.1.0 candidate above.
- **Numerical kernel promotion [user, 2026-10-04].** A better method
  found here is fixed here, written to be moved (generated, verified
  constants; regime map; independent oracle; fail-first guards), and
  filed with corvus, which decides by its doctrine; after adoption
  libstats calls corvus and drops its copy. Distribution-specific
  kernels (the von Mises CDF) stay; their reusable machinery moves.
  Fleet rule: OldCrow/standards `NUMERICAL-KERNEL-PROMOTION.md`
  (adopted, `bf493ff`); filed: corvus #44 (the von Mises quadrature
  machinery, and fixed-cost incomplete gamma/beta), corvus #45.
- **Support-boundary rule (#161, #165) [user, 2026-10-03].** pdf/logpdf at
  a support boundary take the limit by shape: at x = 0, +inf for
  shape < 1, the finite density for shape = 1 exactly, 0 / −∞ above. Out
  of support is 0 / −∞ everywhere, Poisson included; no distribution
  returns `MIN_LOG_PROBABILITY`.
- **Special parameters [user, 2026-10-03].** A value path takes the
  shortcut for λ, α, β = 1 (or a standard parameter set) only on exact
  equality. The `isCauchy()` / `isStandard()` style queries and
  `operator==` keep `DEFAULT_TOLERANCE`: they report, and no value path
  reads them.
- **Quantile contract (#104) [user, 2026-09-02].** Finite best-effort:
  never NaN for p ∈ (0, 1), ±inf only on true overflow; discrete
  quantiles are min{k : F(k) ≥ p} on the library's own CDF.
- **±inf contract (#103) [user, 2026-09-02].** The mathematical limit
  where one exists, scalar and batch identical; von Mises keeps
  saturation as a documented exception.
- **Clean-room replacement is the remedy for a provenance defect.** An
  isolated child agent authors from a functional spec, with no access to
  the suspect implementation or its upstream. The orchestrator audits
  and integrates, and never authors. Each replacement ships a derivation
  doc and a divergence audit under `docs/` (proven on #67).
- **#172: no code with licence concerns used or derived from [user,
  2026-10-03].** It, and the review fixes after it, are built from
  published mathematics (Stirling's series, A&S 6.5.29) and the library's
  own helpers.
- **Project skills** live once, in `.claude/skills/`; `.agents/skills` is
  a tracked relative symlink. Edit only `.claude/skills/`. On Windows it
  needs Developer Mode plus a repository-local `core.symlinks=true`; Git
  for Windows defaults to false, and a clone records the value it found.
  Z is set (2026-10-03).
- **Archived branches** are signed tags: `archive/spike-corvus-bessel`
  (the only Tier 0 corvus Bessel code) and `archive/dev-v2.5.0-corvus`
  (the first adoption branch, ported onto v2.5.0 as this one).
- **CI and tooling [2026-11-11]** (rules and incidents: CI House Style §1,
  §6, §8).
  - Runner labels: jobs that install a versioned tool (gcc-13, gcc-14,
    clang-17, clang-tidy-17) are pinned to `ubuntu-24.04`, so image and
    tool move in one commit; everything else floats, macOS included
    (#294). What moving that pin needs: Known Gaps, "CI on ubuntu-26.04".
  - The canary runs weekly (Monday, 03:25–03:40 UTC across the four
    workflows; `docs.yml` gained a schedule). A `canary-report` job opens
    a "Canary failed" issue, or comments on the open one (#295).
  - Pre-commit is enforced (#296): the `precommit` job runs every hook on
    all files. Before this the config existed and nothing ran it.
  - Project hooks are Python and both can fail: `check_pragma_once.py`
    and `check_no_debug.py` (debug markers; `std::cout` in library code
    outside a DEBUG-gated block). The copyright hook (no file carried a
    header) and the include-order hook (clang-format owns include order)
    were removed. Each hook was shown to fail on a planted case.
  - gersemi replaces cmake-format; clang-format hook is v23.1.5 (#297).
    Formatter output of #296 is `39e8402`, in `.git-blame-ignore-revs`.
  - `docs/bench-evidence/` is excluded from the fixers, clang-format and
    shellcheck: it is captured output, and two files hold lone CRs.
  - `tests/tsan.supp` suppresses the exception-release false positive on
    two frames, one per libstdc++ version (#294, #298).
  - No `actions/cache` in the `precommit` job (zizmor `cache-poisoning`).
  - `reviewdog/action-actionlint` stays at v1.83.0 or newer (#292).

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-11-21. No open issue
without a milestone; no open PRs.
- GitHub is the collaborator-facing source for issues and milestones;
  this file is the agent-facing state. When creating, closing, retitling
  or moving an issue or milestone, update this file in the same change.
- Re-check when a task reads or changes the backlog, or after 7 days.

## GitHub Milestones [DERIVED]
Closed: #1 v2.2.0, #5 v2.3.0, #7 v2.3.1, #2 v2.4.0, #9 v2.4.1, #10 v2.5.0
(released 2026-10-16). Milestone numbers do not sort in version order.

- **#6 v3.0.0 — corvus adoption** (32 open, 1 closed): itemized in "State
  of the v3.0.0 issues" above.
- **#8 Accuracy, contracts & kernel hygiene patch** (33 open, 8 closed),
  v3.0.1:
  - Accuracy: #215, #243, #244, #245, #246, #249, #250, #263, #273, #277,
    #314, #315, #316, #317, #318. #245 may already be fixed (`cd9a2ab`);
    check before working it. #244's "exact path" premise looks wrong.
  - Oracle and test tooling: #259, #260, #261.
  - Concurrency: #181, #182, #217.
  - Contracts, API and docs: #220, #253, #268, #269.
  - Cleanup and CI: #114, #152, #218, #219.
  - Speed and the parallel layer: #178, #283, #284, #319.
- **#3 v3.1.0 — New Distributions (Extended)** (6 open): #58 GEV, #59
  LogLogistic, #60 Triangular, #61 Wald, #62 Hypergeometric +
  BetaBinomial + Zipf, and #307 (the fixed 4,096-element gate in
  `parallelForSlices`). Settle #62's Zipf CDF design before planning; it
  scopes corvus work.
- **#4 Post-v3 — Architecture Refactor** (8 open): #40, #41, #42, #43,
  #128, #222, #282, #285. #282 (one slice callable per batch site behind
  one scheduling seam) is the first step of #43 and is wanted whatever
  the threading direction [user, 2026-10-26]. #285 is gated on a fleet
  executor concept. #42 is on hold.

## Pool wake-up and batch spacing [OPEN; found 2026-11-22, Ivy Bridge]

Found while fixing the profiler (#320). Every figure here is from one
machine in use and is **indicative**: it says where to look, and L7 is
what measures it. Nothing below is filed or posted yet.

**What was found on I (AVX, Linux `schedutil`, 4 cores / 8 threads).**
- When the thread pool's workers sleep between short parallel sections,
  the whole package settles at its lowest clock (about 1.2 of 3.5 GHz)
  and stays there while the pattern lasts. Forced PARALLEL is then two to
  three times slower, and slower than VECTORIZED, at the sizes where the
  slices divide unevenly over four workers (5120, 6144, 7168, 10000);
  8192 and sizes from 10240 up were unaffected back to back.
- **The profiler never saw it.** The tools call
  `initialize_performance_systems()`, which starts the work-stealing
  pool, and that pool's idle workers poll (ten yields, then a 100 µs
  sleep; `src/work_stealing_pool.cpp`). The polling keeps the cores
  awake. So every table was measured with the cores held awake.
- **The polling has a cost of its own**: an idle initialised process
  burns over two cores, and serial VECTORIZED work beside it runs about
  12% slower. Same code on `main`.
- **Batch spacing decides the crossover.** With the workers awake (back
  to back, or the polling pool alive) PARALLEL wins from 4096 or below
  on Poisson PDF. With 2 ms of serial work between batches and no
  polling pool, it wins only from 100,000–200,000.
- A worker spin of 50 µs before sleeping removes the back-to-back
  slowdown and nothing else: it covers gaps up to about its own length
  (a 3 ms spin covered 2 ms gaps, partly 5 ms, not 20 ms).
- This corrects earlier notes: the "slow sizes" read off #323 and the
  "erratic between 4096 and 10000" readings on #320 and #291 were mostly
  this low-clock state. The ceil(slices / threads) cost of #323 is real
  but modest (Gamma CDF P/V 0.32 at 4 slices, 0.41 at 5, 0.30 at 7).
  Those issues, the kAvx evidence README and the `PROVISIONAL` comment
  still carry the old reading; corrections wait on the user's word.

**Not known.** Whether this is general or a property of this CPU and
governor; whether Zen 4 and Apple Silicon have it, or a different
problem with the same symptom; and whether work stealing's wins on K
(up to 7:1, the reason AUTO selects it there) are the polling keeping
cores awake rather than the stealing. L7 answers the first and third and
shows whether the second needs a trace.

**User's positions** [user, 2026-11-22; the back-port is withdrawn
2026-11-25, "Pool design — decided"]: which spacing is the primary
use cannot be known, it depends on the caller; the idle burn should go
in v2.5.x or v2.6.x if the fix is cheap, and v3.0.0 is the line it must
not cross.

**Options** (decided 2026-11-25, "Pool design — decided" below: 2 for
v3.0.0, 4 in v3.1.x, 1 and 3 rejected):
1. Keep the polling pool, add the short spin: thresholds hold for both
   spacings, back-to-back gains 12–18%, the idle burn stays.
2. Remove the polling and set thresholds for sleeping workers: no burn,
   never slower than VECTORIZED, streams lose their speedup below about
   100,000.
3. A spin of milliseconds: moves the cliff to the spin length, at the
   price of every worker spinning that long after each batch.
4. Two thresholds chosen by pool state: the low one while the workers
   are still spinning, the high one once they sleep. No burn and right
   for both spacings; costs a second threshold per row (or a factor per
   architecture) and a profiler mode for spaced batches. Favoured by the
   analysis if L7 shows the spread on more than one machine.

**Held on I, not pushed**: a branch from `main` with the portable 50 µs
spin, `initialize_performance_systems()` starting the work-stealing pool
only where AUTO selects it, and a guard (an idle initialised process
burns no CPU; fails on unfixed `main`, passes fixed). The second part
must not land alone: without the polling pool an initialised caller
with spaced batches is about 2.5× slower at 10,000. The kAvx re-capture
waits on the design. **Disposition [DERIVED from the decision below]:**
the guard is reused; the spin is dropped; the conditional start is
superseded by the wake-up on submit.

### Pool design — decided [user, 2026-11-25]

From the four L7 results and Z's `pool_wake_bench` run.

**Decided.**
- **The idle burn goes in v3.0.0**: the work-stealing pool gets a real
  wake-up on submit (#333) and its idle workers sleep.
- **Not back-ported to v2.5.x.** The fix is small in code and large in
  effect: it removes the clock-holding that v2.5 callers on Linux and
  macOS get from the polling. This replaces the 2026-11-22 position.
- **Thresholds are set for sleeping workers** (option 2): AUTO never
  picks PARALLEL where it is slower than VECTORIZED for a spaced caller.
  Back-to-back streams lose speedup below the new thresholds.
- **AUTO stops selecting work stealing.** The pool stays available on
  request.
- **Two thresholds by pool state, or a time-based gate** (option 4), is
  v3.1.x work, investigated and decided with #307.
- **No spin** (options 1 and 3).
- **Slice arithmetic (#323) goes in v3.0.0**, in the same change. It
  alters PARALLEL's time at every size between 4 and about 12 slices, so
  it must land before the tables are captured or they are captured twice.
- **A "keep warm" opt-in** for callers who block between batches is a
  v3.1.x candidate, not a commitment. v3.0.0 states the change in the
  migration guide: after a blocking gap on macOS the first milliseconds
  run about 3× slower, serial code included.

**Open, each waiting on a measurement.**
- **Thread count on hyper-threaded machines.** Coupled to retiring work
  stealing: on K its win at large sizes is its eight threads (508–574 µs
  against 847–870 µs for four at 204,800). With the pool at physical
  cores, large batches there get about 1.6× slower. Measure first.
- **What "conservative" means** [user, 2026-11-25: accepted as the
  initial definition, reassessed after the measurements]: crossover
  measured at a 2 ms busy gap, PARALLEL chosen only where it is at least
  1.25× faster than VECTORIZED (the #156 margin).
- **AVX2 under Linux is unmeasured.** Tables are per SIMD tier; the
  effect follows the operating system and its governor. kAvx2 comes from
  K under macOS and serves every AVX2 Linux machine. A virtual machine
  does not close this: the host sets the clock and schedules the virtual
  CPUs, so the guest's governor is not what gets measured (WSL2 on Z is
  the same). What can: (a) K capped to the AVX tier against I, same
  tier, two operating systems, different CPUs; (b) K or Z booted into
  Linux from external media, Z capped to AVX2. Neither is planned; if
  none is run, the release notes state the gap.

**Order of work** [steps 1–2 DERIVED, Z, 2026-11-26; the rest OPEN]:
1. **Prototype branch: built.** `proto/v3.0.0-pool-wake`, seven signed
   commits `7f2684a`..`974704a` on `b30929c`, pushed. Not merged; this
   branch is unchanged. On Z: MSVC 19.52 and clang-cl 22.1.3 build
   warning-free, correctness 137/137 on each (135, the guards, the
   override test), stress 109 passed and 27 skipped.
   - `7f2684a` wake-up on submit (#333): idle workers wait on a
     condition variable whose predicate is a pool-wide queued count.
     Also fixes `tryStealWork`, whose sweep could skip a victim.
   - `6c31612` AUTO and MAXIMIZE_THROUGHPUT select PARALLEL everywhere.
   - `a84f933` `initialize_performance_systems()` no longer starts the
     work-stealing pool.
   - `a7d7374` #323: at most one pool task per thread, each a run of
     `ceil(S / T)` whole slices. The kernel is still called once per
     slice, so results cannot move.
   - `d13af0f` `LIBSTATS_THREAD_COUNT` (1–1024) overrides the thread
     pool's size. Read once, at the pool's first use.
   - `9d404ea` `strategy_profile --spaced` (2 ms busy gap between timed
     batches) and `--gap-us N`; the CSV gains a last column `Gap_us`.
   - `974704a` the guards, `tests/test_pool_idle_wake_guards.cpp`.
2. **Guards: written, shown on Z.** `IdleProcessBurnsNoCpu` and
   `SubmitWakesAWorker` (median of 25 trials under 1 ms) fail 8/8 on a
   clang-cl build of `b30929c` and pass on the branch, 25/25 repeats on
   each compiler. Unfixed: wake medians 7–13 ms; fixed: 15–34 µs
   (indicative). **Owed on Linux and macOS:** run both on the unfixed
   head and on the branch. The idle guard should fail unfixed there;
   the latency guard targets the Windows tick and may pass unfixed.
3. **Concurrency validation [OPEN].** ThreadSanitizer is owed on a
   Linux or macOS machine: the stress suite, `test_work_stealing_pool`
   and the guards. CI was dispatched on the branch 2026-11-26 (run
   45919517500); its result is not recorded here.
   **Open points from the prototype [OPEN, user decides]:**
   - Single slow wake-ups on the fixed pool: one trial in 25 read 35 ms
     against a 22 µs median (Z, machine in use). Unverified guess: the
     woken worker is pinned to a busy core. Needs a quiet run.
   - `LIBSTATS_THREAD_COUNT` is read by the shipped library. Keep it
     documented, or remove it once the thread count is decided.
   - No batch call reaches WORK_STEALING now: the 27 cross-waiting
     stress tests skip on every machine and the profiler's
     WORK_STEALING column repeats PARALLEL. Keep, force by hint, or
     remove the path.
   - The 1.25× margin is not implemented; it belongs in
     `scripts/derive_threshold_rows.py`.
   - `MIGRATION_GUIDE.md` and `CHANGELOG.md` are untouched.
   **Z, quiet run [DERIVED, 2026-11-27; clang-cl, prototype at
   `974704a`; evidence on the prototype branch,
   `docs/bench-evidence/2026-11-27-zen4-pool-proto/`, commit
   `0b14ac9`]. This is Z's sitting of step 4, except the sizes between
   10,000 and 25,000.**
   - **6 against 12 threads** (`LIBSTATS_THREAD_COUNT`,
     `batch_gap_bench`, Poisson PDF and Gamma CDF, six sizes, two
     rounds). From 25,000 elements up 12 threads are faster than 6 at
     every gap, by 1.1–1.9×. At 10,000 back to back 6 are faster
     (1.15× on Gamma CDF, 1.39× on Poisson); with a gap the two are
     level. One thread count is not right at every size here.
   - **Sleeping workers cost little on Windows.** PARALLEL was never
     slower than VECTORIZED from 10,000 elements up, at 0, 2 and 20 ms
     gaps. Poisson at 10,000, 6 threads: 0.43, 0.71, 0.80 of
     VECTORIZED. Against the 1.25× margin at a 2 ms gap the cheap row
     qualifies at 10,000 (1.4–1.5× faster); at 20 ms it does not
     (1.16–1.25×). Milder than I under `schedutil`.
   - **Worker pinning adds a tail to the wake-up; it does not change
     the median.** One task submitted to the work-stealing pool after a
     busy gap, 7,200 trials each way: pinned (the default), median
     30–36 µs, worst 25 ms, 74 over 1 ms, 10 over 10 ms; unpinned,
     median 29–36 µs, worst 0.14 ms, none over 1 ms. An earlier
     indicative note here said unpinned was also quicker at the median;
     the quiet run does not show that. Whether the pool should pin at
     all on Windows is a design question [OPEN].
4. **Measurements on the prototype**, quiet machine, one sitting each,
   with the clock policy recorded ("Rules on every machine"). Check out
   `proto/v3.0.0-pool-wake`; commit evidence to it, not here. Thread
   counts: set `LIBSTATS_THREAD_COUNT` before the process starts.
   Spacing: `strategy_profile --spaced` for 2 ms, `--gap-us 20000` for
   20 ms; `tools/bench/batch_gap_bench` and `pool_wake_bench` as in L7.

   | Machine | Owes |
   |---|---|
   | K | Thread pool at 4 and 8 threads on SIMD kernels (cheap and expensive row), back to back and at 2 and 20 ms gaps. Decides the thread count on hyper-threaded macOS. Run first. |
   | Z | The same at 6 and 12 threads (the integer workload gave 1.6–1.9× for 12; SIMD kernels are not measured). |
   | I | Spaced crossovers at 2 and 20 ms under `schedutil` and under `performance`. Sizes the conservative thresholds on Linux. |
   | M | Spaced crossovers at 2 and 20 ms; 8 threads across 4 performance and 4 efficiency cores against 4. |

5. **Decide** thread count per platform and the definition of
   conservative; merge the prototype to dev; CI.
6. **Capture**: every table again in the spaced mode (all existing
   tables were measured with the polling alive), then L1 on every
   machine and L2 on I, K and M ("Three accuracy blocks").
7. #52, #291, #323 and #333 close with the release.

### K against I: two mechanisms, one symptom [DERIVED from the two L7 runs, K, 2026-11-23]

Read from the committed L7 CSVs (`docs/bench-evidence/2026-11-22-ivybridge-batch-gap/`,
`2026-11-23-kabylake-batch-gap/`), as extra time per forced-PARALLEL call
against the same cell back to back. Two distributions on each machine, no
frequency or idle-state data: the timing shapes are measured, **the causes
are inferred**.

| | K (macOS, AVX2) | I (Linux `schedutil`, AVX) |
|---|---|---|
| Plain run, workers asleep | a fixed 40–110 µs per call at every size (Poisson at 20 ms: 86 µs at 4096, 92 at 50,000, 41 at 200,000) | proportional: 2.4–2.9× at every size from 25,000 (Poisson at 20 ms: 1.0 ms at 50,000, 3.7 ms at 200,000) |
| Onset | between 50 and 500 µs | at 50 µs |
| Back to back, plain | no slow sizes | slow sizes (10,000: 50.6 ns/element against 18.4 after `--init`) |
| Polling pool alive, back to back | PARALLEL 1.3–1.5× slower, VECTORIZED 9% slower | PARALLEL up to 3× faster, VECTORIZED 13% slower |
| Polling pool alive, spaced | gap has no effect | gap has no effect |

- **K's penalty reads as a start delay**: the workers begin late and then
  run at full speed. A fixed cost W moves the crossover to about
  W / (VECTORIZED − PARALLEL ns per element): Poisson saves about 14 ns,
  giving about 6,400 (measured 10,000–25,000); Gaussian about 1.1 ns,
  giving about 80,000 (measured 100,000).
- **I's penalty fits neither model** [corrected 2026-11-23 by the fit
  below; this first read "a slow run"]. The loss grows with the batch,
  which a fixed W cannot give, but a single factor leaves 18–78%
  residual and the two-parameter fit 15–46%. I's back-to-back baseline
  has slow sizes of its own (10,000), so the comparison cell is not
  clean there. What I's mechanism is stays open until item 2 runs on I.
- **Work stealing on K has K's start delay from the shortest gap.** AUTO
  costs 55–110 µs more per call from 50 µs on. The idle loop is ten
  yields, then `sleep_for(MAX_NEWTON_ITERATIONS)` microseconds, that is
  100 (`src/work_stealing_pool.cpp:247`; the constant is a Newton
  iteration cap used as a duration). So an idle worker answers within a
  sleep quantum, not at once, and the pool burns CPU without being
  responsive. Its back-to-back win over a thread pool with no polling
  beside it is 7–30% from 10,000 elements (Poisson 10.3 against 11.1
  ns/element at 10,000, 6.9 against 8.3 at 50,000; Gaussian 1.02 against
  1.47 at 50,000). The 7:1 in the profile bundles was measured with the
  polling pool alive.
- **VECTORIZED on K pays about 10 µs per call after a gap of 2 ms or
  more**, Gaussian only (4096: 2.3 to 4.6 ns/element), in both pool
  states. Candidate, unverified: the CPU powers down the upper half of
  its 256-bit unit after about a millisecond without AVX use.
- The thread pool wakes its workers with one `notify_one` per task
  (`src/thread_pool.cpp:75`), so the caller pays the wake-ups in series.
  Not measured whether that is part of W.

**What it suggests for the options above** [K's reading; undecided]:
- A second table per ISA tier is the wrong axis. The effect follows the
  OS and its power management as far as two machines can show: kAvx2
  serves K under macOS and any AVX2 machine under Linux or Windows.
- On K a spaced threshold is derivable from the back-to-back profile and
  one machine constant W, with no second capture. On I that does not
  hold as measured; it needs a factor. Option 4 would then carry a
  per-platform (W, factor) pair, measured once or at
  `initialize_performance_systems()`, instead of a second threshold per
  row.
- Option 1 (50 µs spin) does nothing on K: K has no back-to-back loss
  and its penalty starts beyond 50 µs.
- Option 2 (no polling) costs K about 0.1 ms per spaced call and gains
  9% serial and 30–50% back-to-back PARALLEL; it costs I 2.7× on spaced
  parallel work.
- K's AUTO → WORK_STEALING rule holds only for back-to-back batches.

### Measurements this calls for [OPEN; proposed by K 2026-11-23, none run, user decides]

The L7 bench gives time per call. It cannot separate a late start from a
slow run, and neither machine recorded clocks or idle states. In order of
value for the pool design:

1. **Fit the two models to the L7 CSVs** — **done 2026-11-23**,
   `scripts/fit_gap_model.py` (t_gap = W + k · t_0 per distribution and
   gap; relative RMS residual over the six sizes). **K, plain, forced
   PARALLEL: a delay.** W = 61–69 µs (Gaussian) and 68–89 µs (Poisson)
   at 2 and 20 ms, residual 2–10%, against 25–30% for a factor; the
   two-parameter fit gives k = 0.97–1.07. At 500 µs W is 34 µs
   (Poisson). With `--init`: no effect (W under 12 µs). K's AUTO on
   Poisson: W = 58–70 µs with k = 0.86–0.93 from 50 µs on. **I, plain:
   neither** (above). M's and Z's L7 CSVs go through the same script.
2. **Start delay and run speed per worker, unprivileged, every OS.** A
   bench that drives the library's own pool through
   `detail::ParallelUtils::parallelFor` with a callable that records its
   entry time against the caller's submit time, runs a fixed
   integer loop, and records its exit time. Entry − submit is the wake
   latency per worker; the loop's time against its back-to-back time is
   the clock ratio (the TSC and `steady_clock` do not slow with the core).
   Same gaps and pool states as `batch_gap_bench`. This tests the
   start-delay and slow-run readings directly and gives W and k without a
   kernel in the way. It also shows whether the serial `notify_one` calls
   stagger the workers. **Written 2026-11-23**:
   `tools/bench/pool_wake_bench.cpp` (plain, `--init`, `--init --ws`;
   `--gaps`, `--sizes`, `--sleep`; compiled with AppleClang 15 only so
   far). **No quiet run yet.** A two-size smoke run on K with the machine
   in use, indicative only: at a 2 ms gap the thread pool's call costs
   60–66 µs more, the last worker starts 39–56 µs later, and the work
   time per element is unchanged (×1.00); the caller's serial work is
   unchanged too. That is the start-delay reading. **Quiet runs on K
   done 2026-11-23** at `6d95852`, six processes
   (`docs/bench-evidence/2026-11-23-kabylake-pool-wake/`, with items 5
   and 6 below); findings under "K, measured". **Since run on I**
   (2026-11-23, `docs/bench-evidence/2026-11-22-ivybridge-pool-wake/`),
   **M and Z** (2026-11-24, `…/2026-11-24-m1-pool-wake/`,
   `…/2026-11-24-zen4-pool-wake/`); owed on no machine.
3. **Clocks and idle states, privileged, once per machine, to name the
   cause.** While `batch_gap_bench` runs one plain cell at a 20 ms gap and
   one back to back: K and M `sudo powermetrics --samplers cpu_power -i
   100` (per-core frequency and C-state residency; on M also which
   cluster the workers land on); I `turbostat --interval 0.1` (Bzy_MHz,
   C-state residency) or `scaling_cur_freq`; Z the Processor Information
   counters (% Processor Performance, parking status). The user runs
   these: they need root, and the sampler itself loads the machine, so
   the result names a cause and is not a timing record.
   **Skipped on I** [user, 2026-11-24]. The governor switch already
   shows the clock is the slow run's cause, by experiment where
   `turbostat` would only confirm it. For the 55–65 µs start delay left
   under `performance`, residency averaged over 0.1 s does not separate
   idle-state exit from scheduler wake-up latency. And the design
   options (keep workers awake, accept the delay in the thresholds, more
   threads) are chosen on the measured sizes, which are in hand, not on
   the cause's name. **Revisit if** (a) the pool design leans to a short
   spin before sleeping, since a spin sized against idle-state entry
   differs from one sized against a scheduler wake-up, or (b) the other
   machines' privileged samples disagree and a third reading would
   settle it. **If revisited, prefer the direct test**: `pool_wake_bench`
   under `performance` with deep idle states disabled (`sudo cpupower
   idle-set -D 10`, restored with `-E`; the user runs it, and `cpupower`
   is not confirmed installed on I). A last-worker start falling from
   about 60 µs to about 20 names idle-state exit; no change names the
   scheduler.
4. **Whether W predicts other rows.** `batch_gap_bench` on rows that span
   the per-element saving: a cheap kernel (Exponential or Uniform PDF,
   under 1 ns), a middle one (LogNormal PDF) and a costly one (Gamma
   CDF). Needs a `--dist` option on the bench. If W / saving predicts all
   three crossovers on K, the derivable-threshold claim stands there.
   **Done on K 2026-11-24** (`--dist`, `--gaps`, `--sizes` added;
   `docs/bench-evidence/2026-11-24-kabylake-batch-gap-rows/`): **it
   does.** A gap of 2 ms or more costs forced PARALLEL mostly 56–82 µs
   on Exponential and LogNormal PDF, as on Poisson and Gaussian.
   Exponential at 50,000 saves 65 µs back to back and ties with the gap
   (ratio 0.97–0.98); LogNormal's crossover moves from 10,000 to 25,000,
   where the saving first exceeds the delay (81 against 66–68 µs); Gamma
   CDF saves 830 µs at 4096 and does not move. Forced PARALLEL below
   4096 elements times as VECTORIZED, so the row's predicted crossover
   near 400 cannot be tested. With `--init` no crossover moves.
5. **A sleeping caller.** The gap is busy serial work; a caller blocked
   on I/O lets its own core idle too. `batch_gap_bench --sleep` at 2 ms
   and 20 ms (Windows cannot sleep 50 µs). **K, 2026-11-24** (same
   directory): VECTORIZED loses the clock too, 3.7–3.8× on the two
   cheap rows at 4096 and 2.4–2.9× at 200,000, so the crossovers land
   where the busy gap put them.
6. **Finer gaps at the two onsets on K**: 5–40 µs for work stealing
   (ten yields, then the 100 µs sleep) and 0.5–2 ms for the VECTORIZED
   step, with a control that keeps a 256-bit operation in the gap loop;
   if the step vanishes, it is the AVX unit. Needs a `--gaps` option.
   **K, 2026-11-24** (`--wide-gap`, `--fma-gap`;
   `docs/bench-evidence/2026-11-24-kabylake-gap-controls/`): **it does
   not vanish, so it is not the AVX unit.** The step is 9 µs at 4096 and
   10,000 elements on Gaussian and Exponential PDF, absent up to a
   500 µs gap and complete from 1 ms; a 256-bit add or FMA in the gap
   leaves it. Ungated: `main`'s library has it; it is 3.5–4.2 µs at 1024
   elements and 1.5 µs or less on Uniform PDF; loops outside the library
   pay 1.2 µs or less. Forced PARALLEL's delay has the same 0.5–1 ms
   onset. Not identified; a counter would show it (item 3). The other
   half: `sleep_for(100 µs)` on eight threads takes 121–123 µs at the
   median whether the main thread spins or sleeps, so **timer coalescing
   does not explain** work stealing's late start after a busy gap only.
7. **Idle burn per machine**: user plus system CPU time of a process that
   calls `initialize_performance_systems()` and sleeps 10 s
   (`getrusage`). All four machines have a figure: I 24.8 CPU-s over
   two and a half cores; **M, 2026-11-24**: 5.6–6.6 (2.1 user, 3.5–4.5
   system), 0.00 with no pool started
   (`docs/bench-evidence/2026-11-24-m1-pool-wake/`); **Z, 2026-11-24**:
   0.19 (`docs/bench-evidence/2026-11-24-zen4-pool-wake/`).
   **K, 2026-11-24**: 10.5 CPU-s in 10 s, one core (5.5 user, 5.0
   system); 0.00 with only the thread pool
   (`docs/bench-evidence/2026-11-24-kabylake-pool-threads/`).

8. **The thread pool at eight threads on K** [added 2026-11-23]: the
   two pools differ in thread count there (finding 3 below), so every
   WORK_STEALING-against-PARALLEL figure compares four threads with
   eight. `pool_wake_bench` with a `ThreadPool` of
   `hardware_concurrency()` threads would separate stealing from thread
   count. **Prepared 2026-11-23**: `pool_wake_bench --local-pool 4,8`
   (a pool owned by the bench, driven by a copy of the library's slice
   submission; the 4-thread row checks the copy). **Done on K
   2026-11-24** at `164cbce` (with #324), quiet
   (`docs/bench-evidence/2026-11-24-kabylake-pool-threads/`). The copy
   is faithful (within about 5% of the library's pool in every cell);
   #324 is not separable here (2–4 µs at the small sizes). At 204,800
   elements work stealing's gain is the thread count: eight plain
   threads 499–518 µs, work stealing 508–574, four threads 847–870.
   Below that eight threads are level with four or slower, and back to
   back work stealing beats them (36 / 59 / 161 µs against 45 / 81 /
   225 at 4096 / 10,240 / 51,200): that part is its spinning workers,
   which start within 6–26 µs. With any busy gap from 50 µs work
   stealing loses to both plain pools at 4096 and 10,240 and wins at
   51,200. After a sleeping caller eight threads stay at the low clock
   through a 1.2 ms call (38–39 ns per element against 25 on four).

Items 1, 2 and 7 are cheap and decide the most: the fixed-cost and
proportional readings, and what the polling costs. Item 3 is the only
one that needs the user at the keyboard.

### K, measured: start delay, clock, and what the polling buys [DERIVED, K, 2026-11-23]

`pool_wake_bench`, quiet, integer workload of 14 ns per element
(`docs/bench-evidence/2026-11-23-kabylake-pool-wake/`). Times are read
from the workload; no frequency or idle-state counter was sampled.

1. **Busy caller, thread pool only: a late start and nothing else.** At
   a 20 ms gap the call costs 60–75 µs more than back to back at every
   size from 4096 to 204,800, the last worker starts 30–60 µs later
   (76–85 µs after the submit against 16–54), and the work per element
   is unchanged (×1.00), as is the caller's serial work. This confirms
   the fit's W on K with no kernel involved.
2. **Sleeping caller, thread pool only: the clock drops, serial work
   included.** After a 2 ms or 20 ms sleep, serial work runs 3.0× slower
   at 4096 and 10,240 (12.6 to 37.8 ns per element), recovering inside
   longer calls (20–25 ns at 204,800). Pool workers run 2.6× slower, the
   last starts 126–141 µs after the submit, and the call is 5× longer at
   4096, 2.1× at 204,800. **So K shows a slow run as well, when the
   whole process idles**; `batch_gap_bench`'s default busy gap hides it.
   The pool's advantage over serial at the same gap falls from 2.0× to
   1.4× at 10,240 and is lost at 4096.
3. **The two pools run different thread counts on K**: `ThreadPool`
   takes physical cores under hyper-threading (four,
   `src/thread_pool.cpp:131`); the work-stealing pool ran eight workers
   from 51,200 elements and finished 204,800 in 535 µs against 858
   (1.6×). Its win at large sizes is thread count. At 4096–10,240 both
   use four and it wins on the start (first worker 1.3 µs after the
   submit against 11).
4. **Work stealing's start delay after a busy gap sets in between 10 and
   20 µs** (call at 4096: 36.9, 39.0, 42.9, 58.7, 59.5, 75.1 µs at gaps
   of 0, 5, 10, 20, 40, 100 µs; 121 µs at 50 µs).
5. **What the polling pool buys on K is the clock.** With it alive, a
   sleeping caller loses nothing: serial 14.05 to 14.3 ns, thread pool
   48.6 to 54–58 µs at 4096, work stealing 36.8 to 37.5–38.9 µs with no
   start delay. Its price back to back: thread pool 27% slower at 4096,
   21% at 10,240, 9% at 51,200, 3% at 204,800; serial integer work 3%.
6. **The busy-gap delay (4) is the idle loop** [reasoned 2026-11-23, not
   traced]: an idle worker yields ten times, then calls
   `sleep_for(100 µs)`, and a submit does not interrupt that sleep
   (`src/work_stealing_pool.cpp:247`). That gives the onset at 10–20 µs
   (the ten yields), the worst case at a 50 µs gap (every worker went to
   sleep together after the last batch, so the first starts 80 µs
   late), and the milder 17–41 µs at longer gaps (the eight sleep phases
   have drifted apart). **Not explained: why a sleeping caller sees no
   delay** (first worker 2 µs after the submit). Candidate: timer
   coalescing. The caller's sleep and the workers' sleeps are all timer
   wake-ups, macOS groups nearby deadlines, and workers woken with the
   caller are in their yield window when the submit arrives. Test: the
   spread of first-worker start times per call, or a caller that blocks
   on something other than a timer.
7. Serial integer work shows no step at a 2 ms busy gap, where Gaussian
   VECTORIZED gained 10 µs per call; consistent with the AVX-unit
   candidate, which stays unverified.

**Ruled out as causes: the pipeline and the caches** [user's question,
2026-11-23]. A pipeline flush or a reorder-buffer refill costs tens of
cycles, about 10 ns; the delays here are 50–140 µs and the slow run
lasts milliseconds. `pool_wake_bench`'s workload touches no memory and
uses no SIMD and shows the full effect, and the gap loop evicts nothing.
Reorder depth, vector width and L2 associativity (8-way on I, 4-way on
K) change the steady cost per element (Poisson VECTORIZED 34 ns on I,
23 on K), which moves a back-to-back crossover; they cannot give a fixed
delay or a 3× slowdown. **The differences between K and I at the right
timescale are in power management**: K's hardware-controlled frequency
selection reacts in about a millisecond, I's is the OS governor's
(50–100 ms to recover, #320), which alone could be why a busy caller
holds K's workers at speed and not I's; the idle states and their exit
latencies (tens to over a hundred microseconds, the size of K's start
delay) differ by generation; whether I powers down its 256-bit unit as
K is suspected to is not known. All of it is confounded with the OS
(scheduler, governor, timers): two machines cannot separate silicon
from system. **One test does** [proposed]: `pool_wake_bench` on I, plain
and `--sleep`, under the `performance` governor against `schedutil`. If
the slow run goes, it is the governor's policy.

**For the design** [K's reading; undecided]:
- The caller's state between batches matters as much as the pool's. K
  has two regimes: caller busy (a delay of 60–75 µs, derivable
  thresholds) and process idle (a delay of about 130 µs plus 2.6–3×
  on everything for the first milliseconds). The second is not a pool
  defect: serial and VECTORIZED code pay it too, and no threshold
  avoids it.
- Dropping the polling (option 2) therefore costs K more than the first
  reading said: a caller that blocks between batches loses 3× on small
  serial batches as well. Keeping it costs every back-to-back caller
  3–27%.
- K's AUTO → WORK_STEALING rule compares eight threads with four. Whether
  stealing or thread count earns the selection is item 8.
- I's "neither" fit may be these two regimes mixed: on I a busy caller
  does not hold the workers' clock up as it does on K. `pool_wake_bench`
  on I, plain and `--sleep`, would show it.

### Z: on Windows the idle sleep is a timer tick, and the batch API never uses the pool [DERIVED, Z, 2026-11-24]

From Z's quiet `pool_wake_bench` runs
(`docs/bench-evidence/2026-11-24-zen4-pool-wake/`) and the code. Filed
as #333.

- **`WorkStealingPool::submit()` wakes no worker.** An idle worker finds a
  task when its loop comes round: ten yields, then `sleep_for(100 µs)`. On
  Windows that sleep runs to the next timer tick (15.6 ms by default), so
  a call into the idle pool starts 13–15 ms late at every gap from 0 to
  2 ms (the thread pool: 5–9 µs). Raising the process's timer resolution
  to 1 ms brings the start under 1 ms (checked with the machine in use;
  figures on #333). The polling loop on Windows is therefore not
  polling in the sense I, K and M measured.
- **So the polling pool buys nothing on Z and costs nothing.** Idle burn
  0.19 CPU-s per 10 s (I 24.8, M 5.6–6.6); it does not hold the clock for
  the thread pool's workers or for a sleeping caller.
- **No batch call reaches this pool on Windows.** The dispatcher returns
  PARALLEL in its `_WIN32` branch (`c9e640a`, 2026-04-12, on profiling
  that read "PARALLEL wins 3.3:1" there), and `PerformanceHint` cannot
  force work stealing. The exposure is code that drives the pool directly.
  Whether that 3.3:1 was this same delay was not established.
- **The profiler's WORK_STEALING column is PARALLEL wherever the
  dispatcher does not choose work stealing.** `strategy_profile` maps the
  column to the `MAXIMIZE_THROUGHPUT` hint (`a3bddbf`, 2026-06-21). That
  resolves to work stealing only on hyper-threaded macOS, so the column
  measures work stealing on K alone; on Z, I and M it repeats PARALLEL (Z:
  29.8 against 29.7 µs, Gaussian PDF at 50000). No PARALLEL-against-
  WORK_STEALING comparison in those three machines' bundles is one.
- **For the design:** a wake-up on submit (condition variable or
  semaphore) removes the tick on Windows and the 100 µs on the others, and
  ends the idle burn on I and M in the same change. A per-platform guard
  on the polling pool is not enough on its own.

## Design inputs from the threshold captures [OPEN]
From the v2.5.0 captures on three machines (tested 2026-10-06) and the
Ivy Bridge capture of 2026-11-18. Full analysis: `git show e1a11ec:PLAN.md`,
"Trends".
- **Crossovers are per machine; a table serves an architecture.** That,
  not run noise (a median 0–1 grid step), is why tables read off.
- **One threshold per (distribution, operation) cannot serve every
  parameter**: per-element cost varies up to 10× with the parameters, and
  AUTO is late mostly at costly ones.
- **SSE2 crosses sooner than AVX** (K 21 rows sooner, 0 later; Z 15, 1),
  so delegating kSse2 to kAvx is late on a fifth to a quarter of rows.
  No kSse2 table yet.
- **Below 4,096 elements PARALLEL does not fork** (#307), so a candidate
  there is not a crossover.
- **Thresholds follow per-element cost: supported** [DERIVED,
  2026-10-06]. log(crossover) against log(VECTORIZED ns/element) has
  slope −0.5 to −1.03 (−1 predicted; M −1.03, R² 0.85; K AVX −0.97,
  0.70; weakest Z SSE2 −0.51, 0.29), and one constant per machine and
  tier (2^16.8–2^18.2 ns) puts 69% of 648 rows within noise + 1 step,
  against 75% for the hand-tuned tables. K and Z differ by cost, not by
  noise: Z crosses later on 19–27 rows and sooner on 9–11, its
  vectorized cost about half K's, and the crossover ratio follows the
  cost ratio with slope +1.1 (AVX2, AVX; SSE2 +0.36). Exceptions are the
  two parallel-path defects fixed in v2.5.0 (#175, #176). Corvus replaced
  the kernels and invalidated every table, hence L5. Evaluate a cost model (per-element costs × one per-machine
  overhead, perhaps calibrated at startup) against the tables; it ties
  into the post-v3 Architecture Refactor.
- **Don't depend on auto-vectorization.** Beta's large-shape cost was
  6–11× under MSVC and 2.3–2.8× on M until `998d985` wrote the Horner
  out (Known Gaps, MSVC codegen). Promoted kernels use explicit SIMD, or
  must be shown to vectorize on all three compilers: a candidate
  criterion for standards `NUMERICAL-KERNEL-PROMOTION.md`.
- **One capture harness**, after v2.5.0 and written to whatever
  threshold model v3 adopts [user, 2026-10-05]. Each machine needed its
  own quiet rule (K load < 2.0, M ambient CPU, Z the EcoQoS opt-out),
  and throttling cost Z one overnight. It covers warm-up, the quiet
  gate, the throttling opt-out, noise logging and the bundle layout;
  it could be a standards convention shared with corvus and libhmm.
- **Accuracy gate for each corvus swap:** all three ISA blocks now show
  0 contract violations. Proposed acceptance: 0 violations, and no row
  worse than the v2.5.0 block, on every ISA.

## Known Gaps [OPEN]
- **CI on ubuntu-26.04** [2026-11-11]: the jobs that install versioned
  tools are pinned to `ubuntu-24.04` (#294, CI House Style §1). Moving the
  pin needs two things first: lcov 2.4 fails with `inconsistent` under
  `gcov-13` (`tests/test_discrete_quantile_bounds.cpp:57`), and the TSan
  exception-release false positive in
  `test_parallel_exception_propagation` is suppressed on two frames that
  differ between libstdc++ 14 and 15 (`tests/tsan.supp`); it depends on
  timing, so a pass proves little.
- **No CI leg runs the install-contract steps without TBB** since the GCC
  leg gained `libtbb-dev` (#290). Checked once locally; a second install
  leg would cover it at the cost of runner time.
- **`consumer_example_fetchcontent` fetches `main`**, so it cannot
  exercise this branch until it is merged.
- **Concurrency findings of 2026-10-06** are filed as #184–#191 and
  #178–#183; evidence in
  `docs/bench-evidence/2026-10-06-m1-concurrency/summary.txt`. Decided
  then: a FORCE_SCALAR batch locks per element and can mix states; one
  state under every strategy is a v3 design input (AGENTS.md, Batch API
  contracts).
- **Within the accuracy-for-speed budget, left as is**: Student-t CDF and
  quantile at large ν (5–8× against v2.4.1; the continued fraction),
  discrete log-pmfs (0.9–2×), FisherF (2–3×). The cost bench sweeps
  parameter bands at a fixed input range and does not resolve x-regions.
- **MSVC codegen behind clang-cl** [Z, 2026-10-05]: Gamma at large shapes
  1.5–1.6× behind, since MSVC does not contract FMAs; `vector_expm1`'s
  Taylor loop is scalar under MSVC, which the Pareto and Weibull batch
  CDFs use. Evidence:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/`.
- The `exp_max` clamp sits about 30 ULP below the true overflow
  threshold; in that window the kernels return `exp(exp_max)` where
  `std::exp` is still finite. A deliberate margin.
- von Mises circular variance (#93 residual): forming 1 − A costs about
  2κ × A's ULP error near the κ = 50 cut. Close to intrinsic in double;
  only a double-double complement export from corvus would remove it.

## Cross-Repo Dependencies [OPEN]
- **pylibstats** pins this repo by a `find_package` floor and a
  FetchContent `GIT_TAG` in its `CMakeLists.txt`; its pin-currency canary
  fails when it falls behind. It is at v2.5.0 (pylibstats 0.8.0). v3.0.0
  needs a major-version floor change and brings corvus into the wheels
  (pylibstats #20, #26): the Windows wheel (accept corvus's AVX2 cap
  under cl.exe, or build with clang-cl), the NOTICE and Apache-2.0 text
  for Highway, and Linux aarch64 NEON, which is unvalidated.
- **libhmm** ships the same corvus release (fleet coherence rule, libhmm
  #106) and needs the GoogleTest install fix (libhmm #146, this repo's
  #287). Shared rules and kernels: libhmm #103, #108.
- **corvus**: #110 (scalar entry points for `lgamma` and `lbeta`) and
  #111 (`LgammaDiffDd` cost) are the v1.3.0 work that v3.1.0 waits on;
  #44 (von Mises quadrature machinery, fixed-cost incomplete gamma/beta)
  and #45 (`expm1`, `log1pmx` spans, landed) came from v2.5.0. The
  corvus session is the user's own. What its v1.3.0 milestone asks of
  this repo is the subsection below.
- **standards**: the Node 20 action finding and the capture-harness
  convention (one harness for warm-up, the quiet gate, the throttling
  opt-out and the bundle layout) are candidates for the fleet repo.

### corvus v1.3.0 — what it asks of this repo [OPEN; K, 2026-11-27, from its six open issues]

corvus's v1.3.0 milestone ("Scalar paths and ISA tuning") holds #22, #97,
#110, #111, #120 and #121; v1.4.0 holds #19. Read against this branch's
code at `7c4b809`. Item 1 is done; the rest are rules for measuring
and for the bump.

1. **The overload-set break is fixed ahead of the bump [user, 2026-11-27;
   K, `05fdb41`].** `corvus_scalar(Fn, …)` in
   `src/math_utils.cpp` took a corvus function by name and deduced its
   type, which works only while the name is one function. corvus #110
   adds scalar overloads for the span-only functions, and each name then
   becomes an overload set, as `gamma_p` and the seven others did at
   v1.1.0. The template is gone: twelve named wrappers in a `one_value`
   namespace (`erf`, `erfc`, `erfinv`, `erfcinv`, `lgamma`, `lbeta`,
   `digamma`, `trigamma`, `i0`, `i1`, `i0e`, `i1e`) each call the span
   form with explicit spans, and the fifteen calls use them. Shown on K:
   with the twelve scalar overloads declared ahead of `corvus.h`, the
   old file fails to compile (15 errors, "no matching function for call
   to 'corvus_scalar'") and the new one compiles; `accuracy_sweep`'s
   26,882 rows are bit-identical before and after; 0 warnings, 135/135.
   No test guards it: the break needs a corvus that does not exist yet.
   At the v1.3.0 bump each wrapper's body becomes the scalar entry,
   which corvus promises returns the bits of a length-1 span; a sweep
   with no row moved is the proof, as for v1.2.1.
2. **A v1.3.0 that carries #22 is an accuracy bump on the tiers it
   covers, not a patch [rule for the bump].** #22 is a non-gather kernel
   for gather-weak CPUs before AVX-512 (Kaby Lake's class); it is a
   different method and corvus does not expect bit-identity. Where
   corvus dispatches it, this repo owes L2 and L2b again: the `isa=`
   block, `data/accuracy/allow_<ISA>.csv` and the row gate. #110, #111,
   #120 and #121 are performance only and need the no-row-moved sweep.
   Before that bump, check that no test on a corvus-backed route asserts
   exact bits where the published bound is what is promised [not
   audited].
3. **On K, a comparison of two builds under about 1.3× on a gather-kernel
   route is not a result until the link is checked (#120).** A load that
   follows an AVX2 gather is delayed when it touches bytes 0x7c0–0x7cf
   of its page, and the linker decides whether a kernel's literal lies
   there: 10–30 ns per one-value call on `erf`, and 19% on `erfinv` at
   batch (54.5 against 45.8 ns per element). The routes exposed are the
   ones through corvus's table kernels (`erf`, `erfinv`, `erfcinv`,
   `lgamma` and their consumers); libstats' own AVX2 kernels use no
   gather. It bears on three things here: #156's 1.25× guard read on K,
   the VECTORIZED side of every kAvx2 row captured through those
   kernels, and any A/B on K. Check with corvus's
   `docs/bench-evidence/2026-11-26-kaby-v1.2.1/n7/window_check.py`
   against the executable, or relink with the constants moved 64 bytes.
   Which of K's committed figures were taken with a literal in the
   window is not known. Nothing a consumer can do removes it; corvus's
   candidates are an aligned constant table or #22.
4. **A guard or baseline figure comes from at least three fresh
   processes (#97).** `digamma`, `trigamma` and `i0` read at one of two
   speeds, 1.50× apart, from process to process on Zen 4, cause unknown
   and suspected to follow buffer addresses. The routes here are von
   Mises (`i0`) and the entropy and fitting paths (`digamma`,
   `trigamma`). The benches' three passes are three processes already;
   a single run of one of those routes is not a regression reading.
5. **Keep one-value calls at length 1, and do not build spans of 2 to
   N−1 values in place just before a call (#121).** v1.2.1 fixed the
   one-element case only: a caller that writes N or more values and then
   calls still waits on its own stores (about 30 ns on `erfinv`), and a
   2- or 3-element tail after a full vector is 2–8 ns slower than
   before on AVX2. The batch wrappers here fill their constant-parameter
   block once before the loop, so only a batch's first block can pay it.
   #338 (a batch quantile) should hand corvus long spans, not short ones
   assembled per call.
6. **Normalising constants are the consumer side of #110 and #111.**
   `lbeta` costs 1.2–1.8 µs a call on the non-FMA tier and is 31–48% of
   four concurrency races there, most of it `LgammaDiffDd`, which a
   scalar entry does not remove. What this repo controls is how often it
   asks: once per parameter set, from the cache. Whether any setter
   recomputes a constant it did not change is not audited.
7. **#19 (v1.4.0)** is the von Mises complement 1 − I₁/I₀ in
   double-double. `bessel_i1_i0_complement` stays here until then, in
   one function, as #47's decision 5 left it.
