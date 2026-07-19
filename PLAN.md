# libstats — Plan / Status

## Decided [DERIVED]
- Layered dependency architecture (6 levels, Foundation → Complete
  Library Interface) — see AGENTS.md Architecture.
- Dual API design (auto-dispatch + explicit strategy) is the permanent
  design, not transitional.
- Deferred by design (not backlog): `vector_lgamma` (too complex,
  indefinite), SVE (no hardware in ecosystem), SSE4.1 tier (not worth
  it) — see AGENTS.md Deferred Items for rationale on each.
- Asus TUF A16 AVX-512 re-validation is complete, not pending. AGENTS.md's
  "Current status" line previously said "re-validation pending after audit
  remediation," contradicting its own current validation matrix (which
  already showed "CI validated"). Confirmed via git history: the audit-fix
  commits (`af2da57`, `9a81633`) both predate the v2.0.4 tag. Fixed
  2026-07-14.
- The "`WorkStealingPool::getOptimalThreadCount()` capped at 32 workers"
  claim previously in AGENTS.md was inaccurate, not just stale: a
  `MAX_WORKERS = 32` cap was added and then removed again during v2.0.0's
  own pre-release development (both the adding and removing commit are
  inside the v2.0.0 tag) — it never shipped in v1.5.3 or in any tagged
  v2.x release. Removed from AGENTS.md entirely rather than migrated to
  MIGRATION_GUIDE.md, since there's no real old→new behavior to document.
- CHANGELOG.md and docs/VALIDATION_HISTORY.md already contained the
  versioned changelog and historical SIMD/validation data this brief
  expected to find duplicated in AGENTS.md — confirmed 2026-07-14 that a
  prior restructuring pass (commits `8a27d6a`, `83d3cda`) had already done
  this work. AGENTS.md's Project Overview still had one leftover: an
  inline v2.0.0 breaking-changes list duplicating MIGRATION_GUIDE.md,
  several points of which weren't actually in MIGRATION_GUIDE.md yet
  (VoidResult, validate*Parameters relocation, distribution concepts,
  SIMDPolicy::Level, DistributionType relocation, legacy validation.h
  deletion). Added those to MIGRATION_GUIDE.md, then trimmed AGENTS.md to
  a one-line pointer.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-07-19.
- 2026-07-19: posted a comment on #46 (SIMD accuracy characterization) recording
  the NEON `vector_log` finding from the #33 Q1 experiment (current 2 ULP vs a
  table-based 1 ULP alternative at par speed) plus the `vector_cos` ~1e-10
  accuracy observation.
- 2026-07-19: created **#67 OPEN** — `vector_erf_neon` (shipped since v1.5.0
  Phase 3, commit `5455778`) is derived from glibc's LGPL-2.1+
  `erf_advsimd.c`, not a permissively-licensed source; found while auditing
  the #33 Q1 exp/log work for the same class of issue (that work is clean --
  MIT ARM optimized-routines). No MIT equivalent of the table+Taylor erf
  algorithm exists upstream, so this needs a dedicated remediation decision,
  not a quick citation fix.
- 2026-07-19: closed **#67** (`vector_erf_neon` LGPL-2.1+ provenance) as resolved
  — see "Issue #67" section below. Posted a comment on **#49** recording that
  the erf accuracy fix disconfirms "erf precision" as that issue's cause (see
  the updated #49 line below and the Issue #67 section).
- GitHub is the collaborator-facing source for issues and milestones; this
  PLAN.md is the agent-facing durable project state. Keep both in sync.
- When creating, closing, reopening, retitling, or moving a GitHub issue or
  milestone, update this section in the same change set or note why it could
  not be updated.
- Reconcile this section against live GitHub state when either is true:
  (a) the task at hand involves reading the backlog to decide what to work
  on next, or creating/closing/retitling/moving an issue or milestone, or
  (b) more than 7 days have passed since the "Last reconciled" date above.
  Skip the check for tasks that don't touch the backlog or this file at
  all — a per-session or per-task refresh regardless of relevance is
  wasted effort in one direction and a rubber stamp in the other. Update
  the "Last reconciled" date whenever this section is actually re-checked,
  whether or not anything had drifted.
- Convention: open (actionable) milestones/issues are fully itemized here;
  closed/historical ones are summarized as counts only.

## GitHub Milestones [DERIVED]
- v2.1.0 — Accuracy & Performance (open, #1): 6 open / 0 closed.
  - #46 OPEN — Benchmark: SIMD accuracy characterization vs arbitrary-precision reference (mpmath).
  - #47 OPEN — bessel.h Tier 2 fallback (A&S polynomial) limits VonMises accuracy to ~10⁻⁷ on macOS/AppleClang.
  - #48 OPEN — CauchyDistribution::getCumulativeProbability delegates to StudentT incomplete-beta; should use closed-form arctan.
  - #49 OPEN — LogNormalDistribution CDF accuracy 2.62×10⁻⁷. **"erf precision"
    disconfirmed as the cause (2026-07-19, see Issue #67 section)**: after
    replacing `vector_erf_neon` with a max-1-ULP implementation (was ~2.29
    ULP), the LogNormal CDF error vs scipy is unchanged at 2.62e-07. Root
    cause is elsewhere — likely the CDF's `(ln x − μ)/σ` argument transform
    and/or the `erfc`-tail cancellation path, not the `vector_erf` primitive.
  - #51 OPEN — VonMisesDistribution CDF has no SIMD/batch path — scalar integration loop is 5–10× slower than scipy.
  - #52 OPEN — BinomialDistribution CDF slower than scipy; PDF near-parity — PMF summation and scalar lgamma are limiting factors.
- v2.2.0 — New Distributions (Foundation) (open, #2): 4 open / 0 closed.
  - #54 OPEN — feat: add LogisticDistribution and GumbelDistribution — closed-form SIMD via vector_exp.
  - #55 OPEN — feat: add BernoulliDistribution and ErlangDistribution — delegation wrappers over Binomial and Gamma.
  - #56 OPEN — feat: add FDistribution and InverseGammaDistribution — Gamma/Beta family delegation wrappers.
  - #57 OPEN — feat: add HalfNormalDistribution and TruncatedNormalDistribution — erf/erfc normalisation family.
- v2.3.0 — New Distributions (Extended) (open, #3): 5 open / 0 closed.
  - #58 OPEN — feat: add GeneralizedExtremeValueDistribution (GEV) — unifies Gumbel/Fréchet/Weibull-max families (depends on #54).
  - #59 OPEN — feat: add LogLogisticDistribution (Fisk) — log-space Logistic pipeline, survival analysis (depends on #54).
  - #60 OPEN — feat: add TriangularDistribution — piecewise linear, pure arithmetic SIMD (no transcendentals).
  - #61 OPEN — feat: add WaldDistribution (Inverse Gaussian) — erfc-based CDF, first-passage time model.
  - #62 OPEN — feat: add HypergeometricDistribution, BetaBinomialDistribution, ZipfDistribution — additional discrete distributions.
- v3.0.0 — Architecture Refactor (open, #4): 4 open / 0 closed.
  - #40 OPEN — refactor: split 1773-line CMakeLists.txt into cmake/ modules.
  - #41 OPEN — refactor: unify dual SIMD namespace (stats::simd::* vs stats::arch::simd::*).
  - #42 OPEN — refactor: decompose parallel_execution.h into per-backend implementation files.
  - #43 OPEN — refactor: extract ~2000 lines of dispatch/cache boilerplate into a shared CRTP or policy helper.

## GitHub Issues Without Milestone [DERIVED]
- Open issues without milestone:
  - #33 OPEN — v1.5.x: Evaluate table+Taylor approach for NEON transcendentals — cross-architecture experiment (see Known Gaps). x86 half of Q2 fully closed null (AVX2/Kaby Lake 2026-07-18; AVX-512/Zen 4 Stage 3 2026-07-19). NEON Q1 prototyped 2026-07-19 on branch `experiment/issue-33-neon-table-transcendentals` (Mac Mini M1): **exp is a borderline win** (<1 ULP, ~+20% stream / ~+21% hot), **log is a perf null** (1 ULP — better than current's 2 ULP — but ~tied at stream). **exp productionized 2026-07-19** (47/47 ctest, `simd_verification` VectorExp 2.5-2.6x); an in-place-aliasing bug was found and fixed in the process (see "NEON Q1" below). **PR #70 opened 2026-07-19** (https://github.com/OldCrow/libstats/pull/70); issue stays OPEN until the PR merges.
  - #67 OPEN (created 2026-07-19) — `vector_erf_neon` derived from glibc's
    LGPL-2.1+ `erf_advsimd.c`, not a permissively-licensed source. No MIT
    equivalent algorithm exists upstream; needs a dedicated remediation
    decision. See `THIRD_PARTY_NOTICES.md` "KNOWN ISSUE" section.
- Closed issues without milestone: 9 as of 2026-07-14.

## In Progress [OPEN]
- Branch `feat/neon-cos-cleanroom`: **clean-room `vector_cos_neon`
  productionized 2026-07-19**, replacing the v1.4.0 Taylor kernel whose
  measured error was ~6e8 ULP inside [-2pi, 2pi] (7-term Taylor truncation
  ~6.5e-11 absolute) with sign errors near k*pi/2 — a correctness fix for
  the von Mises path. New kernel: quadrant reduction with a 4-part
  exact-product pi/2 split, compensated (r, rlo), degree-6 minimax cores,
  exact 1-u/2 head/tail; max 0.78 ULP uniform / 0.50 near-k*pi/2 for
  |x| <= 2^23, scalar fallback beyond. Speedup 1.7x vs scalar (down from the
  broken kernel's ~2.8x — correctness costs the difference; von Mises
  PDF/LogPDF at 5.7x/6.5x). Derivation + divergence audit vs ARM
  optimized-routines in docs/NEON_TRIG_DERIVATION.md /
  NEON_TRIG_DIVERGENCE_AUDIT.md; regression gate
  tests/test_simd_neon_cos_accuracy.cpp (5000-vector <1 ULP floor incl.
  stress set, even symmetry, domain fallback, aliasing). A public
  `vector_sin` API (the sin core is already computed internally) is a
  separate cross-backend decision, deferred.
- Branch `feat/issue-33-neon-log-cleanroom` (stacked on the Issue #33 branch):
  **clean-room `vector_log_neon` productionized 2026-07-19**, superseding the
  Q1 "log is a perf null" verdict — the null applied to the ARM-port design;
  an independently derived kernel (compensated (L_hi,L_lo) anchors, sqrt2
  re-centering folded into the table, division-free t = m*R-1 residual, two
  Fast2Sum steps) measures **max 0.52 ULP** (vs current 2.0) at **1.66 ns/elem
  = 1.74x scalar** (vs current 2.88 ns / 1.28x) on the M1. Derivation +
  divergence audit vs ARM optimized-routines advsimd log.c in
  `docs/NEON_LOG_DERIVATION.md` / `docs/NEON_LOG_DIVERGENCE_AUDIT.md`;
  regression test `tests/test_simd_neon_log_accuracy.cpp` (ULP floor, IEEE
  edges, in-place aliasing). Clean-room workspace:
  `~/Development/libstats-log-cleanroom/` (SPEC.md + standalone harness).
- Local branch `experiment/issue-33-neon-table-transcendentals` (pushed,
  PR #70 open against `main`): Issue #33 Q1. **Prototype complete
  (2026-07-19) + exp productionized (2026-07-19), PR opened (2026-07-19)** —
  see "Issue #33 Experiment -> NEON Q1" for full numbers. `vector_exp_neon`
  (`src/simd_neon.cpp`) now runs the table+Taylor kernel in production;
  `src/exp_ulp_vectors.inc` (renamed from `avx512_exp_ulp_vectors.inc`) backs
  a new production regression test, `tests/test_simd_neon_exp_accuracy.cpp`.
  47/47 ctest, `simd_verification` VectorExp 2.5-2.6x. Found and fixed a real
  in-place-aliasing bug in the ported kernel's edge-fixup logic (surfaced by
  `NumericalSafety.LogSpaceOperations`, which calls `vector_exp` in-place via
  `LogSpaceOps::logSumExpArrayFallback`). `vector_log_neon` unchanged
  (perf-null, not productionized). PR: https://github.com/OldCrow/libstats/pull/70.
  Internal plan artifact `8370d3c6-66f5-4814-be66-cf4b996f85fa`.
- Local/pushed branch `fix/remove-stale-vector-erfc-stub` (1 commit ahead
  of `main`, also on `origin`, no open PR yet): removes the unused
  `vector_erfc` SIMD stub since no distribution batch path actually calls
  `erfc()` (Gaussian CDF batch path uses `vector_erf` directly). Scalar
  `erfc()` is retained (used by `distribution_base.cpp`). Not yet merged.
- Local backup branch `backup/wip-sleef-avx2-gather-bench` (1 commit ahead
  of its recovered stash base, local-only as of 2026-07-14): recovered from
  stash originally on `simd-architecture-repair`. Adds opt-in SLEEF/SIMD
  development benchmark tooling for the #33 AVX2 gather-based exp/log vs.
  current SLEEF polynomial experiment. Not yet validated or merged.
- Housekeeping completed 2026-07-14: stale fully merged local branches were
  deleted, empty stash state was dropped, useful stash state was promoted to
  backup branches above, and prunable orchestration worktrees plus their
  leftover `worktree-agent-*` branches were removed.
- Deleted obsolete branch `backup/wip-dispatch-thresholds-tuning`
  2026-07-14 after confirming it was an interim manual threshold guess from
  base commit `1b564ec`, superseded by later issue #50 profiling commits on
  `main`.

## Issue #33 Experiment — gather/table exp,log [x86 (Q2) CLOSED null; NEON (Q1) exp PRODUCTIONIZED 2026-07-19 (awaiting PR approval), log perf-null (not productionized)]
Decisions locked 2026-07-18 (Q2); Q1 work started 2026-07-19. Internal plan
artifact: `8370d3c6-66f5-4814-be66-cf4b996f85fa` (covers both Q2, closed, and
Q1, open). Full write-up: `docs/SIMD_BENCHMARK_RESULTS.md` "Issue #33 —
gather-vs-polynomial exp/log experiment".

### Kaby Lake AVX2 result: null, closed
- Goal was: test whether table + short-polynomial via hardware gather
  (`_mm256_i32gather_pd`) beats the current SLEEF polynomial `vector_exp_avx2`
  (src/simd_avx2.cpp:190) / `vector_log_avx2` (src/simd_avx2.cpp:264). Both
  baselines are already <1 ULP and >1x (~3.4x exp), a higher bar than the NEON
  erf win that motivated the issue.
- First kill-gate (`tools/gather_throughput_probe.cpp`, Stage 1-2): warm
  gather already costs 7.0x a single FMA op, interleave (the agreed gate)
  8.6x, cold 1406x — more than the ~7 polynomial terms a 3-term table
  replacement would save. This is a floor, not a ceiling: index computation,
  range reduction, and edge-case handling in a real kernel only add cost on
  top of the isolated gather.
- Per the fail-forward-fast policy, this is a definitive null result: AVX2
  gather-based exp/log does not beat the current polynomial on this hardware.
  Stages 3-6 (table port, benchmarking, log, consolidation) skipped as moot.
  No further AVX2 work planned; production kernels unchanged.

### AVX-512 Zen 4 (A16) Stage 1-2 result: kill-gate CLEARED for exp, marginal for log
- Harness extension done 2026-07-18 on `experiment/issue-33-gather-transcendentals`
  (already current with `main`, no rebase needed): added an AVX-512 path to
  `tools/gather_throughput_probe.cpp` (`_mm512_i64gather_pd`, 8-wide, guarded
  by `LIBSTATS_HAS_AVX512` + runtime `supports_avx512()`), plus a matching
  CMake dev-tool flag block. Environment note: build tree was stale against a
  since-uninstalled VS 2022 Build Tools; reconfigured clean against the
  now-installed Visual Studio 2026 Community (`Visual Studio 18 2026`
  generator) — unrelated to the SIMD experiment itself.
- Measured (ns per op, this machine, Zen 4 / A16):
  | Path | FMA baseline | Warm | Interleave (gate) | Cold |
  |---|---:|---:|---:|---:|
  | AVX2 (4-wide, same box) | 0.709 ns | 0.699 ns (0.99x) | 1.027 ns (1.45x) | 248.3 ns (350x) |
  | AVX-512 (8-wide) | 0.404 ns | 0.493 ns (1.22x) | 0.688 ns (1.70x) | 268.3 ns (664x) |
- Contrast with the closed Kaby Lake result (AVX2 interleave 8.6x FMA
  baseline): even AVX2 gather on Zen 4 costs only 1.45x, and native AVX-512
  gather costs 1.70x — confirms AMD's Zen 4 gather unit is not just
  "different" from Intel's Skylake-derived one but substantially cheaper.
- Gate math (FMA baseline ≈ one 3-term-Horner-chain unit, i.e. ~2 FMAs):
  exp saves 7 terms (10→3) ≈ 2.33 baseline-units; gather costs 1.70 units —
  **cheaper than the projected savings, gate clears for exp.** log saves
  only 2 terms (7→5) ≈ 0.67 baseline-units; gather costs 1.70 units — gate
  does NOT clear for log on this simple term-count model (real range-
  reduction savings could shift this; not measured here). Same floor-not-
  ceiling caveat as Kaby Lake applies: a real kernel adds index computation,
  range reduction, and edge-case handling on top of the isolated gather.
- **Verdict: Stage 1-2 kill-gate CLEARS for exp** (unlike the Kaby Lake
  null). Per the fail-forward-fast policy this is the opposite outcome —
  proceed toward Stage 3 (actual table kernel port + ULP validation) for
  exp; log is marginal and needs empirical validation, not the term-count
  model alone. Not yet started — stopped here by agreement pending a model
  tier switch.
- Model/effort: per the plan, Stage 3 (actual table port + accuracy
  validation) should SWITCH UP to Opus 4.8 now that the kill-gate has
  cleared; this Stage 1-2 harness extension was done at Sonnet 5 tier
  (rote, mirrored existing AVX2 code).

### AVX-512 Zen 4 (A16) Stage 3 result: null — accurate table-exp fails the perf gate
Built and measured 2026-07-19 on `experiment/issue-33-gather-transcendentals`.
The experimental kernel is a faithful two-gather port of ARM optimized-routines'
scalar `exp` (MIT source, NOT the glibc LGPL copy): N=128 tail-corrected table
(`_mm512_i64gather_epi64` for the scale bits + `_mm512_i64gather_pd` for the
tail), order-5 polynomial, shift-trick index derived at runtime. Lives in the
opt-in dev tool only; production `vector_exp_avx512` is untouched and never
dispatched.
- Artifacts (all dev-only): `scripts/gen_avx512_exp_table.py` →
  `src/avx512_exp_data.inc` (table cross-checked bit-exact vs ARM's values);
  `scripts/gen_exp_ulp_vectors.py` → `src/avx512_exp_ulp_vectors.inc` (mpmath
  correctly-rounded reference, 1018 points, per issue #46); kernel + ULP/bench
  harness in `tools/gather_throughput_probe.cpp`.
- Accuracy gate PASS: table-gather exp holds core(|x|≤700) max 1 ULP, mean
  0.001 ULP; matches the current kernel's 1 ULP and is additionally correct at
  the ±inf / NaN / overflow / underflow edges the current kernel clamps.
- Performance gate FAIL: hot (cache-resident) +4.3%, stream (realistic)
  −44.5% (table-gather is ~1.8× slower under memory pressure). Needed ≥20%.
- Root cause — the tail is decisive: reaching <1 ULP requires the table's tail
  correction = a SECOND gathered value. Two 8-wide gathers cost more than the
  current 10-term SLEEF polynomial (which touches no memory), and the current
  kernel is already memory-bandwidth-bound (~0.55 ns/elem) when streaming. The
  Stage 1-2 kill-gate cleared exp only because it modeled a single-gather
  3-term replacement — but that variant is ~1.9 ULP (ARM `exp_advsimd`
  accuracy) and fails the accuracy floor. The kill-gate was necessary but not
  sufficient; the full <1 ULP kernel is what settles it.
- **Verdict: null result for the accurate table-exp on Zen 4.** exp does not
  improve, so there is nothing to solidify; per fail-forward-fast the exp
  table port is abandoned and production kernels are unchanged. log was never
  attempted (weaker candidate, same two-gather problem).
- With this the entire x86 half of Q2 is closed null (AVX2/Kaby Lake and
  AVX-512/Zen 4). Only NEON Q1 (table exp/log via software gather, needs the
  M1) remains genuinely open on issue #33.

### NEON Q1 (Mac Mini M1) — prototyped 2026-07-19 (exp WIN, log perf-null); exp PRODUCTIONIZED 2026-07-19
- Branch `experiment/issue-33-neon-table-transcendentals` cut from `main`
  (post-#66 merge, so it includes the closed Q2 work and reusable
  `LIBSTATS_BUILD_SIMD_DEV_TOOLS` infra) on the Mac Mini M1. No code changes
  yet -- only the branch exists and the plan artifact has been updated to
  cover this stage (see internal plan `8370d3c6-66f5-4814-be66-cf4b996f85fa`
  § "Approach — NEON Q1 (M1), OPEN" for the staged kill-gates).
- Goal: port ARM glibc-style table+polynomial `exp_advsimd`/`log_advsimd` to
  `vector_exp_neon`/`vector_log_neon` (src/simd_neon.cpp), reusing the
  software-gather pattern and shift-trick index derivation already proven in
  `vector_erf_neon`/`neon_erf_data.inc`. Same <1 ULP accuracy floor and ≥20%
  realistic-regime performance gate as Q2; same fail-forward-fast governance.
- Key open question carried over from the AVX-512 Stage 3 result: the naive
  single-table 3-term ARM variant is only ~1.9 ULP and likely fails the <1
  ULP floor, so a tail-corrected two-value variant (second gathered value,
  mirroring `neon_erf_data.inc`'s scale+residual pattern) may be required --
  but NEON's software-gather cost structure differs from x86's hardware
  gather, so whether the extra lookup is affordable here is a genuinely open
  empirical question, not assumed to repeat the AVX-512 null.
- **Result (2026-07-19, prototyped in `tools/neon_table_transcendental_probe.cpp`,
  opt-in dev tool; production kernels untouched):** the NEON hypothesis is
  confirmed — an Array-of-Structs table whose 16-byte entry is pulled by a single
  `vld1q` makes the tail-corrected two-value lookup nearly free, the opposite of
  the x86 two-gather penalty. Both kernels are faithful 2-wide ports of ARM
  optimized-routines (exp: tail-corrected N=128 table + order-5 poly; log: ARM's
  `{invc,logc}` N=128 tab + degree-5 poly, near-1 band + non-normal routed to
  scalar). Generators: `scripts/gen_neon_exp_table.py`, `gen_neon_log_table.py`,
  `gen_log_ulp_vectors.py` (+ reused `avx512_exp_ulp_vectors.inc`).
  - **exp — WIN (borderline).** Accuracy <1 ULP (mean ~0.001, correct at IEEE
    edges), matching current. After a 2x unroll + hoisted edge branch: stream
    ~+20% (standalone 5/5 ~20.3%; combined harness 4/5, 18–26% range), hot
    ~+21–27%. Clears the 20% gate but with a thin margin (streaming is partly
    memory-bandwidth-bound).
  - **log — perf NULL.** Accuracy <1 ULP (max 1 ULP, mean ~0.0005) — actually
    BETTER than the current SLEEF `vector_log_neon`, which measures 2 ULP on the
    3830-point set. But throughput is hot ~+10%, stream ~0% (range −4%..+1.4%):
    tied with the current 7-term atanh SLEEF log under memory pressure. The
    degree-5 table poly's savings are eaten by the hi/lo reconstruction + gather,
    and both kernels are ~1.75 ns/elem (log is costlier than exp regardless).
    Fails the 20% gate decisively.
- Per fail-forward-fast ("if only a subset improves, solidify that win and stop
  the rest"): productionize exp; log's only upside is the 2→1 ULP accuracy bump
  at no speed cost, logged on #46 for the accuracy track (cross-ref #49), not a
  performance win.
- **Decision (2026-07-19, confirmed with owner): productionize exp only.** Log is
  dropped (perf-null); its kernel/table stay on the branch as reference. Other
  transcendentals were scanned before committing: beyond erf/exp/log the only
  other vectorized transcendental is `vector_cos` (pow = exp∘log inherits; the
  `pow_elementwise` NEON path is a scalar `std::pow` loop). `cos` is NOT a
  table-speed candidate (ARM's reference cos is polynomial) but IS the least-
  accurate primitive (~1e-10 abs, 7-term Taylor) -> accuracy-only upgrade, noted
  on #46. So nothing else batches into the exp work.
- **Productionization complete 2026-07-19 (steps 1-5; step 6 DEFERRED):**
  1. Included `neon_exp_data.inc` in the ARM block of `src/simd_neon.cpp`,
     mirroring the existing `neon_erf_data.inc` include.
  2. Replaced the body of `vector_exp_neon` with the validated table+Taylor
     kernel (2x-unrolled, tail-corrected N=128 table, order-5 poly, hoisted
     `|x|>=704 -> std::exp` edge fixup). Confirmed no existing test asserted
     the old clamping behavior before finalizing.
  3. `THIRD_PARTY_NOTICES.md` updated: `src/neon_exp_data.inc` /
     `scripts/gen_neon_exp_table.py` now listed as production (not
     dev-tool-only); log/AVX-512 remain dev-tool-only; erf's separate
     "KNOWN ISSUE" section left untouched (that fix is on the unmerged #67
     branch, not this one).
  4. Renamed `src/avx512_exp_ulp_vectors.inc` -> `src/exp_ulp_vectors.inc`
     (now a production test dependency) and added
     `tests/test_simd_neon_exp_accuracy.cpp` (GTest, wired into CTest):
     core `|x|<=700` max ULP <=1.0 / mean <0.01 vs the mpmath reference, plus
     an IEEE-special-values case (±inf, NaN, ±0, overflow, underflow).
  5. Re-validated on this M1 (fresh Release build `build-exp33`):
     `ctest -LE "timing|benchmark"` **47/47** (46 existing + the new test,
     including a real bug fix — see below); `simd_verification` VectorExp
     **2.5-2.6x** (up from 2.1x baseline), full distribution suite geomeans
     unaffected/healthy (PDF 6.8x, LogPDF 8.2x, CDF 3.1x); the 5 exp-heavy
     timing-labelled enhanced tests (Gamma, LogNormal, Pareto, Weibull,
     Rayleigh) all pass.
  - **Bug found and fixed during step 5:** the ported kernel's edge-fixup
    logic re-read the input array *after* the SIMD store, a pattern that only
    breaks when called in-place (`a == result`). `LogSpaceOps::
    logSumExpArrayFallback` (`src/log_space_ops.cpp:141-161`) legitimately
    calls `vector_exp` in-place; for `-inf` input, the store overwrote the
    input with an internally-computed `NaN` before the fixup check read it
    back, so the `|x|>=704` unordered-compare check never fired and the
    wrong `NaN` survived instead of the correct `exp(-inf)=0`. This broke
    `NumericalSafety.LogSpaceOperations`. Fixed by computing the edge-fixup
    mask from the already-loaded input register (before the store), never
    re-reading `a[]` after storing into it — mirrors how the original SLEEF
    kernel never re-read the array. **General lesson for future NEON/SIMD
    kernel work in this codebase:** never re-read the input array after the
    corresponding store when in-place calls are possible; decide fixups only
    from already-loaded registers.
  Step 6 (Q3 NEON dispatch-threshold reprofiling for exp-heavy distributions)
  remains DEFERRED to a separate follow-up per decision.
- Pickup context: branch is HEAD on this M1; changes are staged but **not yet
  committed** — awaiting user approval (diff + validation summary) before
  commit/push/PR, per SOP. Regenerating any table needs mpmath in a throwaway
  venv (`python3 -m venv /tmp/v && /tmp/v/bin/pip install mpmath`).

## Issue #67 — `vector_erf_neon` LGPL provenance [RESOLVED 2026-07-19]
Branch `fix/issue-67-erf-neon-cleanroom` (off `main`, pushed to origin, commit
`5297298`), unrelated to the Issue #33 branch above.
- **Problem**: `vector_erf_neon` (shipped since v1.5.0 Phase 3, commit
  `5455778`) was found to be a near-verbatim port of glibc's LGPL-2.1+
  `sysdeps/aarch64/fpu/erf_advsimd.c` — confirmed by direct diff against the
  actual glibc source (identical hex constants, table size/shift-trick,
  gather pattern). No MIT-licensed equivalent algorithm exists upstream (ARM
  optimized-routines' `erf.c` is a different, non-table, rational/minimax
  algorithm), so this needed a genuine re-derivation, not a citation fix.
- **Process**: resolved via a clean-room methodology — a single isolated
  child agent (`erf-cleanroom`) authored a replacement in a scratch
  directory with zero access to `src/simd_neon.cpp`, `neon_erf_data.inc`,
  `gen_neon_erf_table.py`, or any existing erf implementation, working
  purely from a functional/mathematical specification. The orchestrator
  (who had read the actual glibc source) performed the divergence audit and
  integration afterward, not the authorship — structural separation rather
  than a promise not to reuse what was already seen.
- **Result**: local Taylor expansion about a uniform grid (h = 1/256) using
  the Hermite-polynomial structure of `exp(-x²)`'s derivatives, with a
  compensated (hi+lo) table for accuracy. Strictly better than the code it
  replaced: **max 1 ULP** (was ~2.29 ULP), **7.1–7.4x** in
  `simd_verification` (on par with the former ~8.0x). Full validation: 46/46
  `ctest`, `simd_verification` all-PASS across all 22 distributions, and an
  end-to-end `pylibstats` vs `scipy` rebuild+benchmark with no throughput or
  accuracy regression (Gaussian CDF 9.42e-15 vs scipy).
- **Provenance record**: `docs/NEON_ERF_DERIVATION.md` (the derivation,
  symbolically + numerically verified) and
  `docs/NEON_ERF_DIVERGENCE_AUDIT.md` (point-by-point comparison vs the
  actual glibc source — every free/expressive choice differs; only
  mathematically forced coefficient values and the generic method/NEON
  idiom coincide). `THIRD_PARTY_NOTICES.md` updated to record the
  resolution.
- **Side finding**: disconfirms "erf precision" as the cause of #49 (see
  that line above) — the more accurate erf did not change LogNormal CDF's
  error vs scipy.
- **Containment (separate from the code fix)**: the `libstats` and
  `pylibstats` GitHub repos were made private, and affected `pylibstats`
  PyPI releases (≥0.2.4) were yanked, to limit ongoing exposure from
  already-tagged releases (v1.5.3–v2.0.4) that shipped the superseded code.
  Those tags are not amended — this is a forward fix only.
- **Status**: PR #69 opened 2026-07-19. The anticipated reconciliation of
  this branch's `PLAN.md` / `THIRD_PARTY_NOTICES.md` edits against the
  Issue #33 branch's edits was performed during the 2026-07-19 rebase onto
  post-#70/#72 `main`.

## Known Gaps [OPEN]
- `vector_floor` + `vector_blend` primitives across all SIMD backends to
  enable branchless Discrete CDF and Uniform PDF/LogPDF — low priority
  given existing batch-path speedups already achieved through
  amortization, not rejected.
- Issue #33: cross-architecture experiment evaluating table-lookup vs.
  polynomial approach for exp/log (NEON's table-based `vector_erf`
  achieves 8.0x vs ~0.9x for the pure-polynomial equivalent). The x86 half
  of Q2 is now closed null on both tiers: AVX2/Kaby Lake 2026-07-18 (gather
  too expensive even warm) and AVX-512/Zen 4 Stage 3 2026-07-19 (accurate
  two-gather table-exp holds <1 ULP but is slower than the current
  polynomial — see "Issue #33 Experiment"). NEON Q1 exp is productionized
  (2026-07-19, awaiting PR approval); NEON log stays a perf-null, not
  productionized (accuracy-only finding tracked on #46).

## SIMD exp underflow fix (worktree compassionate-booth-4de55b, 2026-07-19) [DERIVED]
- The four x86 `vector_exp_*` kernels (SSE2/AVX/AVX2/AVX-512) clamped input
  at -708.0, pinning every x < -708 to ~3.3e-308 (up to ~6.7e15 ULP error in
  the subnormal range per the cleanroom baseline audit). Fixed: clamp lowered
  to -746.0 with two-step 2^n scaling (n = n1 + n2, n1 = n>>1), giving
  graceful flush through subnormals to +0. Side effect: also fixes the old
  single-step scaling returning inf for x >~ 709.44 (n = 1024 hit the inf
  exponent pattern).
- NEON needed no change after rebase: PR #70's table kernel routes |x| >= 704
  to a scalar std::exp fixup, verified bitwise-exact for
  x in {-708.5, -720, -745, -746, -800, -1e6, -1e300} on M1. The original
  NEON hunk here (written against the old SLEEF kernel) was dropped as
  obsolete during the rebase onto post-#70/#72 main.
- Regression test added
  (`BatchMathRegressions.VectorExpDeepUnderflowMatchesStdExp`); on NEON it
  exercises the table kernel's fixup path, on x86 the new two-step scaling.
- [OPEN] x86 kernels validated by cross-compile + SSE2 run under Rosetta 2
  (clean); AVX/AVX2/AVX-512 need native-machine or CI validation (Kaby Lake,
  Asus TUF A16).
- [OPEN] Pre-existing, unrelated: exp_max clamp constant is ~30 ULP (in x)
  below the true overflow threshold, so for x in that one-double window the
  kernels return exp(exp_max) (~214 ULP low) where std::exp is still finite.
  Deliberate safety margin against 1-ULP overshoot to inf; left as is.

## Next Steps
- Issue #33 x86 experiment (Q2) is fully closed null on both AVX2/Kaby Lake
  and AVX-512/Zen 4 (see "Issue #33 Experiment" and
  `docs/SIMD_BENCHMARK_RESULTS.md`); no further x86 gather-exp/log work.
  `backup/wip-sleef-avx2-gather-bench` remains salvage-reference only. The
  `experiment/issue-33-gather-transcendentals` branch holds the probe, the
  Stage 3 kernel, and the generators as reference.
- Issue #33 Q1 (NEON): exp productionization COMPLETE, **PR #70 open**
  (https://github.com/OldCrow/libstats/pull/70) against `main` from
  `experiment/issue-33-neon-table-transcendentals` (47/47 ctest,
  `simd_verification` VectorExp 2.5-2.6x; see "Issue #33 Experiment -> NEON
  Q1"). Next concrete step: await review/merge. Q3 dispatch-threshold
  reprofiling deferred to a follow-up. Log is not productionized
  (accuracy-only note lives on #46).
- Issue #67 is resolved on branch `fix/issue-67-erf-neon-cleanroom`
  (**PR #69**); the `PLAN.md` / `THIRD_PARTY_NOTICES.md` reconciliation
  against the Issue #33 branch's independent edits was performed during the
  2026-07-19 rebase onto post-#70/#72 `main`.
- Still assess `fix/remove-stale-vector-erfc-stub` (unrelated stale erfc-stub
  removal) for merge.
- Work through the v2.1.0 — Accuracy & Performance backlog (6 issues,
  mostly SIMD accuracy/perf gaps) before starting the new-distribution
  milestones (v2.2.0, v2.3.0) or the v3.0.0 architecture refactor.
