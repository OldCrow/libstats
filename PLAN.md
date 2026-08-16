# libstats — Plan / Status

## Status [DERIVED] — 2026-07-26
v2.1.0 is the current release (tag commit 2026-07-20); 19 distributions
across 7 families. `main` sits 29 commits ahead of the tag: CI hardening,
the build-stack standardization phases, and four Strict-mode source
casts — no library behavior change, so nothing here forces a release.
Working tree clean, `main` level with `origin/main`, no feature branches
local or remote (Dependabot PRs #85–#88 are the only open PRs).

Release contents live in `CHANGELOG.md`; per-version validation matrices
and SIMD speedup tables live in `docs/VALIDATION_HISTORY.md`; conventions,
build commands and architecture live in `AGENTS.md`. This file carries only
what is decided, open, or next.

## Decided [DERIVED]
- Layered dependency architecture (6 levels) and the dual API
  (auto-dispatch + explicit strategy) are permanent designs, not
  transitional. See AGENTS.md Architecture.
- Deferred by design, not backlog: `vector_lgamma` (complex, low
  distribution impact), SVE (no hardware in the fleet), an SSE4.1 tier
  (SSE2 magic-number workaround is adequate). See AGENTS.md Deferred Items.
- `WorkStealingPool`'s `MAX_WORKERS = 32` cap never shipped in any tagged
  release — it was added and removed inside v2.0.0's own pre-release
  development. Recorded because the claim outlived the code in AGENTS.md
  and could be re-derived wrongly from a partial git read.
- **Clean-room replacement is the remedy for a provenance defect**, and it
  is structural: an isolated child agent authors from a functional spec with
  no access to the suspect implementation, its tables, its generator, or the
  upstream source; the orchestrator — who has read the upstream — does the
  divergence audit and integration, never the authorship. Proven on #67.
  Every replacement ships a derivation doc plus a divergence audit under
  `docs/`.
- **A SIMD kernel must never re-read its input array after the
  corresponding store.** In-place calls are legal (`LogSpaceOps::
  logSumExpArrayFallback` calls `vector_exp` with `a == result`), so a
  post-store re-read sees internally-computed values, not the input. Decide
  every edge fixup from already-loaded registers. This cost a real
  `exp(-inf)` bug during the #33 productionization.
- **Accuracy claims hold only for tiers validated on native silicon.**
  `LIBSTATS_MAX_SIMD_TIER` (cmake/SIMDDetection.cmake) caps the highest
  compiled x86 tier so lower tiers can run natively on capable hardware;
  the first-ever native SSE2 run is what exposed #74, invisible under
  Rosetta for years.
- Gather-vs-polynomial transcendentals, settled empirically on three tiers
  (#33, full write-up in `docs/SIMD_BENCHMARK_RESULTS.md`): x86 hardware
  gather is too expensive on both Kaby Lake (interleave 8.6× an FMA) and
  Zen 4 (1.70×, cheaper but still losing once a <1 ULP kernel needs a
  second gathered tail value). NEON is the opposite — an Array-of-Structs
  table pulled by one `vld1q` makes the two-value lookup nearly free.
  Conclusion: table kernels are a NEON technique here, not an x86 one.
  Do not reopen the x86 half without new hardware.

The last three entries are conventions, not project state; they belong in
AGENTS.md Conventions on the next pass through that file.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-08-16.
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
  all. Update the date whenever this section is actually re-checked,
  whether or not anything had drifted.
- Convention: open (actionable) milestones/issues are fully itemized here;
  closed/historical ones are summarized as counts only.

## GitHub Milestones [DERIVED]
Renumbered top-down 2026-07-21 to make room for the shipped v2.1.0:
former #1/#2/#3 titles each moved up one minor version. Milestone numbers
and attached issues were unchanged; only titles moved.

- **v2.2.0 — Accuracy & Performance** (open, #1): 7 open / 5 closed
  (#83 include restructure shipped 2026-07-26; #92 and #93 closed
  2026-08-15 — see Resolved log; #95 added to the milestone 2026-08-15,
  since #51 depends on it and already sat here; #96 and #97 both added
  and closed 2026-08-16).
  - #46 — Benchmark: SIMD accuracy characterization vs mpmath.
  - #47 — bessel.h Tier 2 fallback limits VonMises accuracy to ~10⁻⁷ on
    macOS/AppleClang.
  - #48 — Cauchy CDF delegates to StudentT incomplete-beta; should use
    closed-form arctan.
  - #49 — LogNormal CDF accuracy 2.62×10⁻⁷. **"erf precision" is
    disconfirmed as the cause** (2026-07-19): replacing `vector_erf_neon`
    with a max-1-ULP kernel left the error unchanged. Suspicion moves to
    the `(ln x − μ)/σ` argument transform or the erfc-tail cancellation.
    Error is bit-identical across machines, so it is in the shared scalar
    path.
  - #51 — VonMises CDF has no SIMD/batch path; scalar integration loop is
    5–10× slower than scipy.
  - #52 — Binomial CDF slower than scipy; PMF summation and scalar lgamma
    are the limiting factors.
  - #95 — SIMD trig surface: no `vector_sin` at any tier, and the four x86
    `vector_cos` kernels measure ~6.5e-11 absolute (~2.9e5 ULP) against
    NEON's clean-room 0.50–0.78 ULP. **Gates #51** — its series recipe
    cannot reach ≤5e-16 on x86 until this is fixed. Already reaching the
    shipped VonMises batch PDF, where two test files relax to 1e-10 rather
    than fix the kernel — #47 with the platforms swapped.
  - ~~#96~~ **CLOSED 2026-08-16** — see Resolved log.
  - ~~#97~~ **CLOSED 2026-08-16** — see Resolved log.
- **v2.3.0 — New Distributions (Foundation)** (open, #2): 4 open / 0 closed
  — #54 Logistic + Gumbel, #55 Bernoulli + Erlang, #56 F + InverseGamma,
  #57 HalfNormal + TruncatedNormal.
- **v2.4.0 — New Distributions (Extended)** (open, #3): 5 open / 0 closed
  — #58 GEV (depends on #54), #59 LogLogistic (depends on #54),
  #60 Triangular, #61 Wald, #62 Hypergeometric + BetaBinomial + Zipf.
- **v3.0.0 — Architecture Refactor** (open, #4): 4 open / 0 closed —
  #40 split CMakeLists.txt into cmake/ modules, #41 unify the dual SIMD
  namespace, #42 decompose parallel_execution.h, #43 extract dispatch/cache
  boilerplate into a CRTP or policy helper.

## GitHub Issues Without Milestone [DERIVED]
- Open: **#84** — Audit compensated-summation paths for FP-contraction
  sensitivity. Filed from corvus's cross-compiler finding (GCC's default
  `-ffp-contract=fast` fused inside a compensated sequence and shifted a
  double-double result 0.6 bits vs MSVC). **Machine-independent half is
  done** (2026-08-15, written up in the issue); unmilestoned by decision —
  it rides the M1 session rather than opening a release line.
  - Inventory: three error-free transforms, all in `src/simd_neon.cpp` —
    log's two Fast2Sum steps (461-464), erf's compensated final add
    (617-619, anchors in `src/neon_erf_data.inc`), sin/cos's compensated
    `(r, rlo)` reduction (650-696). The Welford updates in `gaussian.cpp`
    and `lognormal.cpp` are the textbook recurrence with no residual term,
    so they are **in scope but not exposed** — no identity to break. No
    x86 path qualifies.
  - Exposure is one compiler, one file: the TU builds only on aarch64
    (`cmake/SIMDDetection.cmake:150`), so the set is exactly {AppleClang,
    those three sites}. The fleet's contraction-default spread produces no
    divergence surface here.
  - **No reproducibility claim exists anywhere in the repo** — every hit is
    seeded-RNG determinism. So nothing promised is at risk; the live
    concern is that the published NEON ULP bounds may silently depend on
    contraction being ON, having all been measured under AppleClang
    defaults.
  - Policy proposed: `-ffp-contract=off` scoped to that source. The
    kernels already spell every fusion they want with explicit
    `vfmaq_f64`, so contraction can only act where a separate
    `vmulq`/`vaddq` pair was deliberate. Landing site exists —
    `cmake/SIMDApplication.cmake:94-103` already sets per-source
    `COMPILE_OPTIONS` on the file (empty on aarch64), so it is a one-line
    addition.
- Closed: 14, none milestoned (#90 and #94 both closed 2026-08-15 — see
  Resolved log). Note #90 was never listed here while open; this section is
  derived from GitHub rather than maintained by hand, so re-derive it rather
  than trusting it between passes.

## In Progress [OPEN]
- **corvus adoption spike** — branch `spike/corvus-bessel`, opened 2026-08-15.
  Wires corvus `i0`/`i1`/`i0e` behind `stats::detail::bessel_i0` /
  `bessel_i1` / `log_bessel_i0` as a Tier 0 under `LIBSTATS_USE_CORVUS`
  (OFF by default), above the two existing tiers in
  `include/libstats/core/bessel.h`. Scope is deliberately ONE function path:
  three functions, eight call sites, all scalar and all in `src/von_mises.cpp`
  (parameter-cache and fit-time, none in a hot loop) — so the payoff is
  **accuracy**, retiring the Tier 2 A&S 1.6e-7 fallback behind #47, **not
  throughput**. Scalar span-of-1 wrappers are adequate at every call site.

  The deliverable is a **go/no-go adoption recommendation with evidence, not
  a merged feature.** The adoption decision itself is recorded in
  `corvus/PLAN.md` (see Cross-Repo Dependencies below); this section carries
  execution state only, and neither file restates the other.

  Stages: **S0** branch + decision record. **S1** build plumbing and the ABI
  matrix — Config A (FetchContent, all-MSVC, AVX2-capped by
  `HWY_BROKEN_MSVC`) and Config B (prebuilt clang-cl corvus consumed via
  `find_package`, the config that answers corvus's untested clang-cl→MSVC
  link question and delivers AVX-512); both assert the dispatch target via
  `CORVUS_EXPECT_TARGET` rather than assuming it, and check /MD on both
  sides. **S2** the Tier 0 wrappers (`log_bessel_i0` = `log(i0e(x)) + x` per
  corvus's documented composition; its small-x caveat is adjudicated and
  written down before implementation, since that adjudication becomes
  corvus's integration note). **S3** validation — full 49-test ctest under
  baseline / A / B, plus a wiring gate over a kappa sweep including the
  small-kappa band and kappa > 700. **S4** report, the corvus #47
  integration note, and PLAN updates on both sides.

  Out of scope, explicitly: no corvus edits (consumed at the `v0.5.0` tag,
  core/generator/test freeze in effect); no #51 Miller-recurrence CDF work;
  no macOS leg — the AppleClang Tier 2 retirement is the real-world #47
  payoff but **needs the Mac Mini M1**, same constraint as #84; no new
  oracle, because corvus's per-tier 1-ULP claims are the accuracy authority
  and what is needed here is a wiring gate, not a reference set.

  Known risk: if Config B fails to link, the spike **pivots** to "Config A
  only" adoption and the failure mode goes in the report — it does not end
  the spike.

  **S1 COMPLETE 2026-08-15 — both configs pass; the Config B risk did not
  materialise.** Config B (MSVC 19.51 consumer + clang-cl-built installed
  corvus) links, runs, and dispatches **AVX3_ZEN4**; Config A (all-MSVC
  FetchContent at the `v0.5.0` tag) dispatches **AVX2**, capped by
  `HWY_BROKEN_MSVC` as predicted. Both probes pass with zero failures
  against the consumer's own `std::cyl_bessel_i` (≤ 2.3e-16, many rows
  bit-identical), over a kappa sweep straddling corvus's x_s = 8 regime
  split. The two configs' probe output is **byte-identical except the
  `active_target` line**, so A-vs-B is a performance choice, not an
  accuracy one.

  **The tier is set by the compiler that builds corvus's TUs, not by the
  delivery mechanism** — A and B varied both at once, so a third run
  de-confounded them: FetchContent + clang-cl also gives AVX3_ZEN4, in
  177 s. Matrix: FetchContent+MSVC AVX2; FetchContent+clang-cl AVX3_ZEN4;
  installed-clang-cl + MSVC-consumer AVX3_ZEN4. What the installed path
  actually buys is **decoupling** — corvus on clang-cl while this repo
  stays pinned to `cl.exe` — which matters only to the extent libstats IS
  pinned (CI and the pylibstats wheel path are the real constraints, and
  clang-cl is MSVC-ABI-compatible, so "pinned" needs deciding, not
  assuming). An all-clang-cl libstats build would reach AVX3 from plain
  FetchContent with no prefix at all. A third lever exists for an
  MSVC-pinned build: corvus's `CORVUS_MSVC_UNBLOCK_AVX512` (measured
  working there, all gates pass, deliberately unsupported upstream).
  Secondary datum: clang-cl builds corvus 3.9x faster than cl.exe on the
  same config (177 s vs 682 s).

  **Open question for S4, not answerable from S1:** whether a Config A
  adoption would leave these kernels at AVX2 inside a binary whose own
  hand-rolled paths run AVX-512 — that depends on which compiler libstats
  commits to on Windows, which is a libstats decision this spike does not
  make. Probe sources and five re-runnable scripts are in this session's
  scratchpad; S4 folds them into the report.

  **S2 COMPLETE 2026-08-15.** Tier 0 landed in
  `include/libstats/core/bessel.h` (span-of-1 wrappers, `LIBSTATS_USE_CORVUS`
  OFF by default) plus one cohesive CMake block riding the existing
  `libstats_simd_interface` propagation that already carries
  `LIBSTATS_HAS_CXX17_BESSEL`. libstats builds clean with Tier 0 in 51 s.
  The log-composition adjudication is written at the definition site, where
  it doubles as corvus's #47 integration note: all three `log_bessel_i0`
  consumers embed the result in a sum anchored by `LN_2PI` ≈ 1.8379, so the
  governing contract is corvus's ABSOLUTE 3.3e-16, not its weaker small-κ
  relative error — which is therefore unreachable here. Verdict: compose,
  no dedicated log-I₀ kernel needed.

  **Two findings from the S2 smoke diff, both independent of adoption and
  both adjudicated against mpmath at dps 50 rather than assumed:**
  1. **Tier 1's `x > 700` fallback is inaccurate, and this is NOT a
     macOS-only problem.** Above 700 `std::cyl_bessel_i` overflows, so
     `log_bessel_i0` falls back to a hand-rolled two-term A&S asymptotic.
     At κ = 1000 it is 7.3e-11 absolute against mpmath, where corvus's
     composition is 2.2e-14 — ~3300× worse. #47 is filed as a macOS/Tier 2
     issue; this says the SAME function has a real accuracy hole on every
     platform that defines `LIBSTATS_HAS_CXX17_BESSEL`, Windows and Linux
     included. Worth widening #47's scope or filing separately.
  2. **`circularVariance_ = 1 − I₁/I₀` is ill-conditioned at large κ.**
     At κ = 200 the two builds differ by 768 ULP in the variance while
     agreeing bit-for-bit on log I₀. Cause is the formula, not either
     Bessel implementation: I₁/I₀ → 1 − 1/(2κ), so `1 − ratio` cancels ~9
     bits at κ = 200 and amplifies any last-bit difference. A dedicated
     1 − A(κ) formulation (corvus composes A exactly as i1e/i0e, the
     scalings cancel) would fix it. Independent of adoption.

  **S3 (part 1) — TIER-1 DEFECT PINNED 2026-08-15.** `core/bessel.h` is
  standalone (only `<cmath>`), so all three tiers were compiled from the SAME
  source with only the tier macros flipped and swept against mpmath at
  dps 60 — no libstats build involved, which keeps the measurement free of
  every other moving part. Error in ULP of the result (absolute alone ranks
  large-x rows wrongly, since log I₀(x) ~ x and the result's own spacing
  grows):

  | x | Tier 0 (corvus) | Tier 1 (std) | Tier 2 (A&S) |
  |---|---|---|---|
  | 0.5 | 7.4 | 10.4 | 3.6e9 |
  | 100 | 0.5 | 0.5 | 3.3e7 |
  | 700 | 0.4 | 0.4 | 1.29e6 |
  | **700.001** | **0.2** | **1881.2** | 1.29e6 |
  | 1000 | 0.2 | 644.8 | 9.4e5 |
  | 2000 | 0.0 | 40.0 | 2.5e5 |
  | 5000 | 0.3 | 0.7 | 2.6e4 |

  1. **Tier 1's defect is a STEP DISCONTINUITY at exactly x = 700**, not a
     gradual drift: 2.14e-10 absolute jump across the branch, error going
     0.4 → 1881 ULP between adjacent points. It then DECAYS with x (the
     truncation is O(1/x³)) and is back under 1 ULP by x ≈ 5000. So the
     damaged band is **κ ∈ (700, ~3000)**, worst immediately above the seam.
     A discontinuity is the stronger defect signature: any density built on
     log I₀ inherits a visible step at κ = 700. Present on every platform
     defining `LIBSTATS_HAS_CXX17_BESSEL` — Windows and Linux, not just the
     macOS path #47 names. Fixable in-repo with no corvus dependency (more
     asymptotic terms, or match the branches at the seam).
  2. **Tier 2 quantified**, since #47 asserts ~1e-7 without a measurement:
     2.5e-8 to 4.7e-7 absolute across the sweep, i.e. ~1.3e6 ULP in the
     700 band and worse below. Confirms the issue and gives it numbers.
  3. **The S2 adjudication survives measurement.** The composition's
     documented small-x relative weakness is real (7.4 ULP at x = 0.5) but
     Tier 1 is WORSE there (10.4 ULP) — both lose relative precision to
     log(1 + small), independent of tier, and both are irrelevant to this
     repo's consumers, which use the value absolutely against LN_2PI.
  4. Minor: `core/bessel.h` is not self-contained on MSVC — Tier 1 uses
     `M_PI`, supplied globally by CMakeLists.txt:168's `_USE_MATH_DEFINES`.

  **S3 (part 2) COMPLETE 2026-08-15 — both legs green, 49/49, identical
  test sets.** Baseline (Tier 1) and Tier 0 (corvus built by clang-cl,
  libstats by MSVC) both pass the full suite with timing/benchmark labels
  excluded, matching ci.yml. Suite times were 73.0 s and 52.9 s, which is
  NOT a performance result and must not be reported as one: the Tier 0 leg
  ran second with warm caches, timing tests were excluded by design, and
  the library has 8 scalar Bessel call sites, none hot. The spike measured
  accuracy, not throughput.

  Scope note: no third leg with an MSVC-built corvus (the AVX2 tier). S1
  established AVX2 and AVX3_ZEN4 corvus produce byte-identical output on
  every probe row, so that leg would exercise build plumbing, not
  behaviour, at the cost of a ~12 min Highway+corvus MSVC build. Also
  worth settling in S4: the S2 wiring only offers `find_package(corvus)`,
  so "Config A" for libstats means corvus compiled by MSVC, not a
  different delivery path — a real adoption should decide whether to
  offer FetchContent too, since that is the zero-setup path for
  contributors.

  **THIRD ADOPTION-INDEPENDENT DEFECT, found running S3: libstats cannot
  configure under `-G Ninja` on ANY platform.** `tests/CMakeLists.txt:511`
  emits a literal `$` into the `run_tests` ctest regex
  (`...^test_benchmark$`); Ninja requires `$$`, and the malformed line
  invalidates the whole `build.ninja`, so nothing builds. Never caught
  because ci.yml configures with plain `cmake -B build` and no `-G`
  (Visual Studio on Windows, Makefiles elsewhere), CMakePresets.json pins
  no generator, and the escaping is generator-specific rather than
  platform-specific. One-character fix. Pairs badly with a second Windows
  trap found the same way: MSBuild's `.tlog` file tracker breaks past
  MAX_PATH, so the only generator that tolerates long build paths is the
  one the repo cannot configure. Either fix alone removes the corner.

  **S4 COMPLETE 2026-08-15 — SPIKE CLOSED.** Report published as an artifact
  (https://claude.ai/code/artifact/ab912f14-d920-4eb2-b60f-5d92222bb33f);
  the three adoption-independent defects are now #92, #93, #94 rather than
  prose here.

  **RECOMMENDATION: adopt, but NOT on the strength of Bessel alone.** The
  spike proves the mechanism works and that corvus beats both existing tiers
  measurably. It does not by itself justify a dependency: eight scalar call
  sites, none hot, and the one real hole below corvus-grade (#92) is fixable
  in-repo with no dependency at all. The case that DOES justify adoption is
  the wider surface — #47 retired outright, #51 given the documented Miller
  recurrence, #52 given `beta_p`, plus erfinv and the incomplete gamma/beta
  family this repo has no good version of. **Decide on that basis, not on I₀.**

  Costs to price in, none of them blocking but all real: Highway becomes
  transitive into libstats and therefore pylibstats wheels; that triggers
  corvus's open NOTICE obligation (Apache-2.0 must ship with BINARY
  artifacts — source-only releases have dodged it, wheels will not);
  `libstats-config.cmake` owes a `find_dependency(corvus)` because the SIMD
  interface links PRIVATE and corvus lands in `INTERFACE_LINK_LIBRARIES` as
  `$<LINK_ONLY:...>`; and the delivery mechanism is unsettled, since the
  spike wiring offers only `find_package`.

  **Prerequisite for a real decision:** the macOS leg. Tier 2 is where #47's
  actual users are, and it was explicitly out of spike scope for want of the
  M1 — same constraint as #84.

## Known Gaps [OPEN]
- `vector_floor` + `vector_blend` primitives across all SIMD backends would
  enable a branchless Discrete CDF and Uniform PDF/LogPDF. Low priority,
  not rejected — amortization already delivers the batch-path speedups.
- The `exp_max` clamp constant sits ~30 ULP (in x) below the true overflow
  threshold, so for x in that one-double window the kernels return
  `exp(exp_max)` (~214 ULP low) where `std::exp` is still finite. This is a
  deliberate safety margin against a 1-ULP overshoot to inf; left as is.
- `detect_threading_systems()` (cmake/Threading.cmake:6) early-returns on
  its cached completion flag, but imported targets are not cache-persistent
  — so any reconfigure of an existing build dir skips
  `find_package(Threads)`, and the `if(TARGET Threads::Threads)` guard
  (CMakeLists.txt) silently drops `Threads::Threads` from the PUBLIC link
  and the installed export. Found 2026-07-26 while verifying #83's
  install-tree byte-diff (the stale-cache baseline was the side missing the
  entry). Pre-existing, orthogonal to #83; filed as #90 (also covers the
  identical TBB::tbb pattern). On Linux this can underlink
  installed-package consumers.

## Cross-Repo Dependencies [OPEN]
pylibstats consumes this repo two ways — a `find_package` version floor and
a `FetchContent` `GIT_TAG`, both in `pylibstats/CMakeLists.txt`. **That file
is the single source of truth and the version is deliberately not restated
here**; pylibstats' own `pin-currency` CI canary fails if the floor and tag
disagree with each other or fall behind libstats' newest release.

The invariant this repo owns: before cutting a release or making a breaking
API change, check pylibstats' pin and coordinate the bump.

[OPEN] **Whether libstats adopts corvus as a dependency is undecided**, and
it is tracked in `corvus/PLAN.md`, not here. It governs the real cost of at
least four v2.2.0 issues: corvus already ships 1-ULP erf/erfc with a proper
tail decomposition (#49), erfinv/erfcinv (the normal quantile), and lgamma,
and carries Bessel I0/I1 in its P1 scope (#47). Solving #47 or #49 by hand
here is plausibly wasted work if adoption is coming. Settle the direction
before starting either.

[2026-08-15] A scoped spike against this question is now running — see
**In Progress** above for its stages and branch. The decision still lands in
`corvus/PLAN.md`; nothing here changes. One correction worth carrying into
it: **#49 is not a corvus win.** This repo already disconfirmed erf precision
as its cause (a max-1-ULP kernel left the 2.62e-7 error unchanged), so
adoption will not touch it — the suspicion remains the `(ln x − μ)/σ`
transform. #47 is retired outright by corvus's `i0`/`i1`/`i0e`/`i1e`, #51 by
its documented Miller-recurrence recipe, and #52 by `beta_p`.

## Next Steps
1. **#84 is now execution-only, on the M1.** Flip `-ffp-contract=off` at
   the identified site, re-run the NEON log/erf/trig accuracy gates, and
   either confirm the published bounds or restate them. Batch it with the
   corvus spike's macOS leg and #47 — same platform, same compiler,
   adjacent code. Nothing further to decide first.
2. **#48** — Cauchy closed-form arctan CDF. Smallest measurable win in the
   backlog (3–5× on Zen 4, and the gap widens with SIMD width).
3. Settle the corvus-adoption question above, then work the rest of v2.2.0
   before starting v2.3.0/v2.4.0 or the v3.0.0 refactor.

## Resolved log
One line per closed item; detail lives in `CHANGELOG.md`, `docs/`, and this
file's git history.
- 2026-08-15 **#90 closed** — `detect_threading_systems()` and
  `detect_tbb_unified()` returned early on a cached completion flag, but
  **cache variables persist across configure passes and imported targets do
  not**. Any reconfigure of an existing build dir therefore left
  `Threads::Threads` undefined, the consuming `if(TARGET ...)` went quiet
  instead of failing, and the PUBLIC link — which is exactly what
  `install(EXPORT)` writes into `libstats-targets.cmake` — vanished. Same
  commit, same prefix, different installed package. Fixed by hoisting the
  `find_package()` calls above each guard; everything below stays put, being
  cache-setting and status output that correctly runs once. TBB was exposed
  the same way and worse (neither the target nor the pkg-config
  PARENT_SCOPE vars nor the directory-scope paths are cached, while
  `LIBSTATS_HAS_TBB` is — so a pass could believe TBB was available and link
  nothing). Both consuming sites now fail loudly rather than silently.
  Guarded by a new configure-only CI job that configures, captures the
  generated export, reconfigures, captures again, and diffs — Linux leg,
  since macOS hides pthreads in libSystem and Windows never takes the path.
  Every other job configures exactly once and was blind to this class.
- 2026-08-16 **#97 closed** — `LIBSTATS_HAS_CXX17_BESSEL` never reached
  consumers, because it sat on `libstats_simd_interface` and the exported
  library targets reference that as `$<LINK_ONLY:...>`, which propagates the
  link and strips usage requirements. Consumers therefore compiled Tier 2 on
  every platform (not just macOS, as #47 assumes) and, since the helpers are
  `inline` in an installed header while the library's TUs compiled Tier 1, it
  was also an ODR violation. Fixed by moving the probe result into a generated
  `libstats_config.h`, reusing the mechanism this repo already had for
  `libstats_version.h`. Measured on MSVC against a clean install tree: a
  `find_package` consumer goes from Tier 2 at 1.3e-08 relative on
  `bessel_i0(10)` to Tier 1 at 1.5e-16.
  - **A second defect had to be fixed in the same change**, or #97 would have
    promoted it from latent to fatal. Tier 1 called `std::cyl_bessel_i` raw
    while Tier 2 folded through `std::fabs`, so the tiers disagreed across the
    whole negative axis — and `std::cyl_bessel_i`'s domain is x ≥ 0, which
    libstdc++ enforces by throwing `std::domain_error` through these `noexcept`
    frames, i.e. `std::terminate`. MSVC does not throw, which is why Windows
    never saw it. Tier 1 now takes `|x|` and restores the symmetry itself.
    Found by watching libhmm's CI fail on exactly this after its own #75 fix
    let its tests reach Tier 1 for the first time.
  - **Two guards that could not fail, both mine, both caught late.** The
    lesson is now in AGENTS.md → CI/Validation → Test Labels. First, a
    one-sided assertion ("Tier 2 is within 1.6e-7") passes on a Tier 1 build
    too; the shipped canary is two-sided, deciding from
    `__cpp_lib_math_special_functions` and requiring libstats to agree.
    Second — and this survived a full green CI run — the guard was appended to
    a `timing`-labelled binary, and CI's correctness run is
    `-LE "timing|benchmark"`, so it executed on no runner. It now lives in an
    unlabelled `tests/test_bessel_tier.cpp`. Confirmed running as Test #58 on
    gcc-14 and AppleClang, test count 48 → 49.
- 2026-08-16 **#96 closed** — #93's complement series had c8, c9 and c10
  wrong, c10 by 0.199. Report verified independently before acting, by two
  routes: exact rational series division of the two Hankel expansions
  (A&S 9.7.1, ν = 0 and ν = 1, prefactor cancelling) reproduces c1–c7 and
  yields c8 = 375733/32768, c9 = 23797/512, c10 = 55384775/262144; a dps-220
  extraction peeling the exact low terms off the true 1 − I₁/I₀ converges on
  the same c10 from κ = 1e12 to 1e16.
  **Method lesson, which outlives the constants**: Vandermonde solves are
  ill-conditioned and degrade at HIGH order while staying exact at low order,
  so #93's stated validation — the low orders coming out dyadic — confirmed
  exactly the half a bad solve gets right. Raising precision moved the answer
  without fixing it, the other tell. Every coefficient here is an exact
  rational, so derive by series division and the question stops existing.
  **Impact was ~1.2 ULP at the κ = 50 cut against a ~110 ULP total**, because
  above the cut the series error is dominated by truncation rather than by the
  terms carried — the header now says so explicitly, since a re-measurement
  would otherwise suggest the fix achieved nothing. It earns its place in the
  other direction: the same error is ~91 ULP at κ = 30, so lowering the cut or
  extending the series would have hit it. That is how libhmm found it, at 17
  terms cutting at 30 (OldCrow/libhmm#73).
  Cut re-derived from the compiled header against mpmath and unchanged: worst
  over κ ∈ [30, 90] is 128.6 ULP at 50, versus 320 at 45 and 200.7 at 55. The
  ACCURACY.md claim stands as written; both von Mises test binaries pass.
- 2026-08-15 **#92 closed** — Tier 1's log I₀ asymptotic carried only two
  terms, truncating at O(x⁻³). The shipped c₁ = 1/8 and c₂ = 9/128 match the
  exact c_k = ((2k−1)!!)²/(k! 8^k), so the FORM was right and only the length
  was wrong; c₃ = 225/3072 evaluates to 0.0732/700³ = 2.13e-10, exactly the
  observed step. Extended to five terms: seam goes 0.4 → 1881 ULP down to
  0.4 → 0.8, discontinuity 2.139e-10 → 1.376e-13 (~1.2 ULP of the result,
  i.e. rounding). Five not four because at four the asymptotic is 0.009 ULP
  but the seam is already floored by the direct path's own error — five puts
  the asymptotic below it so it is never the limiting side. Also replaced
  `M_PI` with a local `kTwoPi` (numerically identical — doubling is exact),
  making the header self-contained instead of depending on the build's
  global `_USE_MATH_DEFINES`.
- 2026-08-15 **#93 closed, with a documented residual.** Computing A(κ)
  better cannot fix the circular variance: A → 1 − 1/(2κ), so forming 1 − A
  in double discards ~log₂(2κ) bits regardless of A's accuracy. The
  complement needs its own route — and symmetrically A does too, since
  1 − complement cancels as κ → 0. Added `bessel_i1_i0_complement` and
  `bessel_i1_over_i0`, each direct in its own regime, split at
  `kBesselRatioAsymptoticCut = 50`. Coefficients derived by Vandermonde
  solve against mpmath at dps 220, not quoted: the low orders come out
  exactly dyadic (1/2, 1/8, 1/8, 25/128, 13/32, 1073/1024), which is what
  validates the solve — a first pass at dps 60 gave c₁₀ = 213.72 where the
  truth is 211.47. The κ ≈ 713 overflow NaN is fixed as a side effect, since
  every κ in that region now takes the series branch and never evaluates
  either Bessel function. Applied at all three i1/i0 sites: `getEntropy()`
  and the MLE fit loop carried the identical latent NaN. Measured: complement
  ≤ 110 ULP at the crossover, sub-ULP for κ ≳ 80 and κ ≲ 20; ratio ≤ 1.23
  ULP; no NaN at 700/712/713/714. **RESIDUAL, and it is close to intrinsic:**
  the complement error is ≈ 2κ × (A's error in ULP), and A is already ~1.3
  ULP here, so even a correctly-rounded A leaves a floor near κ ULP at the
  cut. More series terms make it worse below the cut (asymptotic, diverges —
  ten terms give 1050 ULP at κ = 40). Closing that band needs extended
  precision inside the complement itself, which this library has no layer
  for. NOTE this is NOT remediated by adopting corvus: its exports return
  doubles, so the cancelling subtraction still happens here, and its A is the
  same ~1 ULP. It would take a corvus export that forms the complement in
  double-double internally — which does not exist and would need an unfreeze
  plus a full family pipeline. Recorded as a corvus candidate, not an
  adoption benefit.
- 2026-08-15 **#94 closed, and it was hiding a bigger bug.** The reported
  defect was real — a literal `$` in `run_tests`' ctest regex invalidated the
  whole `build.ninja`, so `-G Ninja` could not configure on any platform.
  Fixing it surfaced that **both filters on that target had never worked**:
  `-LE "timing\|benchmark"` and the `-E` list used backslash-escaped pipes,
  and in a CMake regex `\|` is an escaped LITERAL pipe, so each matched only
  the literal text `timing|benchmark`. Measured with `ctest -N`: the target
  selected **72 tests — the entire suite — where it describes 41**, i.e. it
  ran every timing and benchmark test despite its own comment. Fix: plain
  pipes, plus `LABELS "benchmark"` on `test_benchmark` so the `$` anchor is
  no longer needed at all (it was load-bearing — `test_benchmark_basic` is a
  correctness test and must stay). Verified: Ninja configures, `build.ninja`
  parses (203 targets), filter now selects 41/72, and the
  benchmark-vs-benchmark_basic distinction is preserved. Guarded the same day
  by a `ninja-generator` CI job: configure with `-G Ninja`, then
  `ninja -t targets` and `ninja -n`. It compiles NOTHING on purpose —
  configure alone does not catch this class (#94 configured cleanly and only
  failed when ninja parsed the result), so forcing a parse plus a graph
  resolution covers it exactly, in seconds. Negative-controlled against the
  pre-fix build tree: `ninja -t targets` exits 1 there with the original
  `bad $-escape`, exits 0 on the fixed tree. NOTE the job must keep tests
  ENABLED — `run_tests` lives in tests/CMakeLists.txt, so a tests-off
  configure would omit the target carrying the hazard and the guard would
  pass while guarding nothing.
- 2026-07-26 #83 include restructure: `include/` → `include/libstats/`,
  shim machinery deleted; install tree byte-identical, 135/135 TUs show
  only the predicted include-dir change, 49/49 tests + both consumers pass.
- 2026-07-24 CI lint hardening: zizmor gated on medium+ severity, latent
  shellcheck findings cleared, `lint-workflows` job green.
- 2026-07-21/23 Build-stack standardization Phases 0–4 (cross-repo effort in
  the fleet standards repo:
  [record](https://github.com/OldCrow/standards/blob/main/records/BUILD-STANDARDIZATION-PLAN.md),
  [house style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)):
  TBB block dedup, GNUInstallDirs install contract
  + installed-package CI smoke test, CMakePresets.json and CMake minimum
  3.25, `cmake/Threading.cmake` + `cmake/CompilerFlags.cmake` extraction,
  tests/tools subdirectory CMakeLists, per-target `libstats_apply_warnings()`,
  dead flag-variable cleanup, SIMD flag application split out of
  SIMDDetection.cmake. AGENTS.md's CMake-standard section describes the
  post-Phase-3 structure.
- 2026-07-21 CI matrix reshaped: a Strict `-Werror` gate replaces the
  Debug/Release matrix bloat; Strict/Sanitizers/AVX-512 build parallelism
  bounded against runner OOM; four Strict-mode source casts landed with no
  behavior change.
- 2026-07-21 Repo returned to public. The "recent account payments have
  failed" Actions failures were metered minutes during the private
  containment window, not a billing misconfiguration. The Documentation
  workflow's "Get Pages site failed" was Pages never having been enabled;
  fixed via the API with `build_type: workflow`, deployed to
  https://oldcrow.github.io/libstats/.
- 2026-07-20 **v2.1.0 tagged** — clean-room NEON erf/exp/log/cos, x86
  exp/log subnormal fixes, AVX-512 roundscale parity, the Windows tier-cap
  link fix, and the stale `vector_erfc` stub removal. See CHANGELOG [2.1.0].
- 2026-07-20 AVX-512 natively validated on the Asus TUF A16, closing the
  last unvalidated tier; all four x86 tiers pass natively there. Two fixes
  found doing it: the Windows/MSVC global `/arch:` block ignored the tier
  cap (LNK2019 on capped builds, invisible on Unix per-file flags), and
  `_mm512_roundscale_pd` omitted `_MM_FROUND_NO_EXC` where the other three
  tiers suppress the precision exception.
- 2026-07-19 **#74 closed** — `vector_log_sse2` lacked the subnormal 2^54
  scaling its AVX/AVX2/AVX-512 siblings have, returning log values up to
  ~35 natural-log-units off. Surfaced by the first-ever native SSE2 run.
- 2026-07-19 **#67 closed** — `vector_erf_neon` was a near-verbatim port of
  glibc's LGPL-2.1+ `erf_advsimd.c`. Replaced clean-room; the result is
  strictly better (max 1 ULP, was ~2.29). Provenance in
  `docs/NEON_ERF_DERIVATION.md` and `docs/NEON_ERF_DIVERGENCE_AUDIT.md`.
  Containment: both repos went temporarily private and pylibstats releases
  ≥0.2.4 were yanked. Tags v1.5.3–v2.0.4 are not amended — forward fix only.
- 2026-07-19 **#33 closed** — x86 half null on both tiers; NEON exp
  productionized; NEON log was a perf null whose only upside (2→1 ULP) is
  logged on #46. See the Decided entry above for the durable conclusion.
- 2026-07-19 x86 exp underflow and edge-case fixes: clamp lowered to −746
  with two-step 2^n scaling, plus a branchless post-clamp fixup so `+inf`,
  `NaN`, and true overflow match `std::exp` instead of collapsing to a
  finite value.
- 2026-07-14/19 Branch and stash housekeeping; `backup/wip-sleef-avx2-gather-bench`
  and `backup/wip-dispatch-thresholds-tuning` deleted after confirming both
  were superseded by work already on `main`.
