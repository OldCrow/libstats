# libstats — Plan / Status

## Status [DERIVED] — 2026-08-20
v2.3.0 is the current release (tagged 2026-08-20); 19 distributions across
7 families, API unchanged from v2.1.0. Milestone #5 (v2.3.0) closed with
all 5 issues shipped (#48, #95, #51, #49, #46).

**Why 2.2.0 and not 2.1.1.** The Bessel work (#92/#93/#96/#97) and the
export fix (#90) are patch-shaped, but three things landed alongside them
that break drop-in: the CMake minimum rises 3.20 → 3.25, install paths move
to GNUInstallDirs, and `libstats/libstats_config.h` joins the installed
header set. A patch number promises a swap-in; this is not one.

The release was cut with 7 issues still open on the milestone formerly
titled v2.2.0, because #97 is a live correctness defect for every consumer
of the installed package — silently Tier 2 Bessel, 1.3e-08 where the
library measures 1.5e-16 — and the remainder of that milestone is gated on
#95, a from-scratch x86 trig kernel. The open work moved to a new v2.3.0
milestone rather than holding the fix behind it.

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
- The three SIMD conventions formerly listed here (no re-read after store;
  accuracy claims only for natively validated tiers / `LIBSTATS_MAX_SIMD_TIER`;
  gather-vs-polynomial settled) moved to AGENTS.md "SIMD kernel conventions"
  on 2026-08-21.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-08-21.
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
Renumbered twice, and the second time is not the same operation as the
first. **2026-07-21**: nothing had shipped out of milestone #1, so the
former #1/#2/#3 titles each moved up one minor version — numbers and
attached issues unchanged, only titles.

**2026-08-16**: five issues *had* shipped out of milestone #1, so a title
cascade would have relabelled closed work with a version it did not ship
in. Instead milestone #1 keeps the title v2.2.0 and its 5 closed issues,
becoming the release record; its 7 open issues moved to a **new** milestone
#5 titled v2.3.0, and #2/#3 moved up one minor version to make room.
Milestone numbers therefore no longer sort in version order — #5 sits
between #1 and #2 — which is cosmetic and is the price of not rewriting
history.

- **v2.2.0 — Accuracy & Performance** (closed, #1): 0 open / 5 closed —
  shipped 2026-08-16. #83 include restructure, #92 and #93 (log I0
  continuity and circular variance), #96 (complement-series coefficients),
  #97 (installed export dropped the Bessel tier). See Resolved log.
- **v2.3.0 — Accuracy & Performance** (open, #5): 0 open / 5 closed —
  ALL WORK MERGED 2026-08-20; milestone close + release cut pending
  (user decision). Original scope note:
  the v2.2.0 milestone's unshipped remainder, moved 2026-08-16. #47 and
  #52 moved out 2026-08-20 (see "Parked on corvus adoption" below).
  Working order decided 2026-08-20: #48 (done) → #95 → #51 → #49
  (interleavable) → #46 last, so the mpmath characterization measures the
  final surface and reuses #95's gate infrastructure.
  - #95 — **CLOSED 2026-08-20** via PR #98 (merged, CI green incl. the
    NEON leg on the ARM runners). Max 1 ULP, mean 0.022–0.028 ULP, all
    tiers, cos and sin; sin(−0) sign defect found by the gate and fixed
    (upstream twin filed as libhmm#81); dispatched-entry gate added. See
    Resolved log and CHANGELOG [Unreleased].
  - #51 — **implemented and verified on `feature/v2.3-vonmises-cdf`
    (2026-08-20), awaiting commits/PR/CI.** Bessel-series CDF, scalar and
    batch: Miller backward recurrence normalized by f_j/f₀ (ratios only —
    no Bessel evaluated, no #47 exposure), j_max = ⌈10 + 8.5√κ⌉
    per-instance coefficients, vector_sin per term. Measured vs a 40-digit
    mpmath quadrature oracle: scalar ≤ 2.2e-16, batch ≤ 8.9e-16 absolute
    through κ = 1000; budgets pinned 2e-15/4e-15. Also a BEHAVIOR FIX:
    the old CDF wrapped into absolute (−π, π] regardless of μ,
    contradicting the documented F(μ) = 0.5 invariant and disagreeing
    with the μ-centered quantile grid; the series CDF wraps t = x−μ.
    Two oracle-side wrap seams found and fixed in verification (threshold
    at double π, then the double x−μ subtraction — the reference is F at
    the library-wrapped t). κ > 1000 keeps the wrapped-normal fallback.
    Follow-ups noted in-tree: quantile grid still trapezoid-built;
    CDF dispatch thresholds provisional pending benchmark.
  - #49 — **implemented and verified on `feature/v2.3-lognormal-cdf-tail`
    (2026-08-20), awaiting PR/CI.** Root cause was the FORMULATION, as the
    2026-07-19 disconfirmation predicted: every path computed
    0.5·(1+erf(z/√2)), whose lower tail hits the 1+erf cancellation floor
    and collapses to exact 0 once erf saturates — true max relative error
    on the benchmark grid was 1.0; the filed 2.62e-7 was the benchmark
    metric flooring its denominator. Fixed by tail-branching to
    0.5·erfc(−z/√2) (scalar normal_cdf + both batch paths; SIMD keeps
    vector_erf with per-lane erfc below w=−1). Gate budget is the
    achievable-accuracy LAW rel(F) ~ |ln F|·2⁻⁵² with headroom — a flat
    deep-tail budget is mathematically unachievable in double for this
    formulation (the original flat 1e-13 spec tripped exactly that).
    Measured: max 0.49 of law budget, scalar and batch, refs to
    F ≈ 1.9e-307. A vectorized erfc (corvus adoption) is the eventual
    clean batch answer for both fix sites.
    - Gaussian instance — **DONE 2026-08-20** (spawned session, same
      branch): GaussianDistribution never routed through normal_cdf and
      reproduced the defect in all five of its own CDF sites (scalar,
      three parallel lambdas, SIMD batch impl). Same tail-branched fix;
      the batch per-lane fixup recomputes w with the scalar path's exact
      expression, so fixed-up lanes are bit-identical to scalar (batch
      vs scalar now differs only in the −1≤w<0 plain-erf band, ≤1 ulp).
      Gate test_gaussian_cdf_accuracy + gaussian_cdf_vectors.inc
      (gen_gaussian_cdf_vectors.py, erfc oracle at dps=40; (0,1) bucket
      covers the isStandardNormal_ path): fail-first max_rel 1.0
      pre-fix; post-fix max 0.287 of the same law budget, batch-vs-
      scalar abs ≤ 1.11e-16. Correctness suite 53/53 on Zen 4 MSVC.
  - #46 — **CLOSED 2026-08-20** via PR #101 (squash 57da00a, CI green;
      a GCC 13 strict-overflow false positive in the p-grid sort was
      fixed by replacing the defensive sort with an ascending assert,
      CSV verified byte-identical). **v2.3.0 milestone now 0 open /
      5 closed — complete; closing the milestone and cutting the
      release are [OPEN] user decisions.** Follow-up issues filed
      2026-08-20: #102 batch NaN propagation (v2.4.0, bug), #103
      +-inf input contract (unmilestoned, decision-gated), #104
      quantile extreme-p contract (unmilestoned, decision-gated).
      [DERIVED] detail:
      replaced the issue's pylibstats route (pins to released v2.2.0,
      would characterize the wrong code) with tools/accuracy_sweep.cpp
      (bit-exact deterministic CSV, 19 dists × 3 instances, scalar +
      FORCE_VECTORIZED batch, tails to p=1e-300; two-process determinism
      verified) + tools/accuracy_vs_mpmath.py (dps-50 oracle, 42
      self-checks) + docs/ACCURACY_CHARACTERIZATION.md. Oracle needed
      substantial large-parameter hardening beyond the agents' build —
      mpmath betainc/gammainc hang or raise for min(a,b) ≳ 5e3 even in
      the central region: clean-room Lentz CF incomplete beta, upper-
      gamma complement, far-tail lead guards (exp(lead) below every
      double), safeguarded log-log false-position quantile solver
      (secant plateaus and pure bisection both failed), asymptotic
      normal seeds. CF/quadrature cross-validation at 1e-44. Gate
      cross-check: gaussian law_frac 1.319 ↔ pinned 0.287·(1e-15 budget)
      ≈ 1.29 bare-law equivalent; lognormal 1.784 ≲ 2.2 equivalent;
      von Mises max_abs 1.7e-16/5.1e-16 vs gates 2.2e-16/8.9e-16.
      Findings (doc Findings section + generated appendix): 86 contract
      violations — batch NaN propagation inconsistent (8 dists), NaN at
      ±inf where limits exist, batch logpdf(+inf) clamp −4605.0,
      quantile NaN/saturation at extreme p, large-param CDF limits
      (binomial n=1e6: 1.3e-2 at the mean → corvus #47/#52 remedy
      class). [OPEN] follow-up issue candidates, on user's word:
      (a) batch NaN propagation, (b) ±inf limit returns, (c) quantile
      extreme-p contract.
- **corvus adoption staged as v2.5.0** (decided 2026-08-21 [user],
  milestone #6 created; the former v2.5.0 Extended milestone renumbered
  to v2.6.0 — the THIRD renumbering, same title-cascade shape as
  2026-07-21). Staging rationale: v2.4.0 Foundation is adoption-tolerant
  (#54 closed-form exp; #55/#56 thin delegation wrappers that inherit
  core upgrades without re-authoring; #57 erf-based on the #49-hardened
  normal_cdf), while v2.6.0 Extended is dense with heavy consumers that
  should be built on corvus cores the FIRST time (#61 Wald erfc-tail
  CDF, #62 Hypergeometric/BetaBinomial incomplete-beta CDFs and the
  Zipf closed-form-CDF-vs-summation decision, which corvus's plan
  requires settled before any Hurwitz zeta work starts there).
  Acceptance evidence: before/after #46 characterization sweeps.
  Prerequisites, tracked on the milestone description: (a) M1 + Kaby
  Lake native validation legs run DURING v2.4.0 (both projects need the
  same machine time; corvus per-tier claims and this repo's PROVISIONAL
  characterization share the gap); (b) corvus API stable — pair with
  corvus v1.0.0 if near; (c) #103/#104 contract decisions settled
  before v2.4.0 starts authoring distributions (cheap policy calls,
  not corvus-gated — every distribution written first multiplies the
  inconsistency #46 found).
  #47 and #52 moved from parked/unmilestoned into milestone #6 with
  un-parking comments, 2026-08-21. Each still closes with `Fixes #NN`
  from the adoption change set: #47 the bessel.h rewire (and whether
  the tier scheme survives), #52 the `beta_p` closed-form CDF rewrite
  plus the before/after scipy benchmark. (Historical: parked
  unmilestoned 2026-08-20; libhmm v4.4.0 had no adoptable interim #47
  fix — its Tier 2 is the same A&S polynomial.)
- **Order of release: v2.3.1 → v2.3.2 → v2.4.0 → v2.5.0 → v2.6.0 →
  v3.0.0.** Decided 2026-08-21 from the defensive review, on the libhmm
  pattern: fix-now candidates grouped into two PATCH milestones (bug fixes,
  no API change); the second exists because its items change numbers
  (caps/tolerance, thresholds) or edge-case policy, so they must not gate
  the correctness patch. Structural items go to the existing major.
- **v2.3.1 — Correctness patch** (open, #7): 6 open / 0 closed.
  - #105 OPEN — `vector_log` NaN → 710.188 on every x86 tier (`cmpunord`
    blend, four sites); LogNormal batch cdf(NaN) = 1 today.
  - #102 OPEN — batch NaN propagation (moved from v2.4.0): re-run
    `accuracy_sweep` with the specials in-vector FIRST, then re-scope the
    victim list — the #46 list is structurally incomplete.
  - #106 OPEN — von Mises κ > 1000 fallback wraps x, not x − μ; add
    κ = 2000/10000 gate rows.
  - #116 OPEN — NegBin/Geometric quantile returns 0 past INT_MAX.
  - #115 OPEN — `operator>>` round-trip broken for Discrete/Uniform/Beta.
  - #112 OPEN — batch aliasing contract: central `autoDispatch` check or
    documented no-aliasing (the doc half landed 2026-08-21).
  Exit: regression tests from the issues in place; sweep regenerated and
  #102 re-scoped from it; AVX-512 native correctness suite green.
- **v2.3.2 — Accuracy, contracts & kernel hygiene** (open, #8): 10 open /
  0 closed.
  - #113 OPEN — incomplete-gamma/beta iteration caps and Lentz tolerance
    (corrects the accuracy premise recorded against #47/#52).
  - #104 OPEN — quantile contract at extreme p (+ the verified Cauchy
    split-form fix, on the issue); #103 OPEN — ±inf input contract.
  - #109 OPEN — re-profile the Cauchy CDF thresholds (rows marked STALE).
  - #111 OPEN — von Mises batch CDF blocking + the noexcept/allocation
    policy; #110 OPEN — one erfc tail-branch helper (bit-neutral).
  - #107 OPEN — one clean-room trig table; #117 OPEN — CPUID DQ/FMA/XCR0
    gates; #118 OPEN — `parallelFor` exception contract; #114 OPEN —
    review backlog.
  Exit: `docs/ACCURACY_CHARACTERIZATION.md` attribution corrected and the
  sweep regenerated; per-tier accuracy gates for #113.
- **v2.4.0 — New Distributions (Foundation)** (open, #2): 4 open / 0 closed
  (#102 moved to v2.3.1 on 2026-08-21)
  — #54 Logistic + Gumbel, #55 Bernoulli + Erlang, #56 F + InverseGamma,
  #57 HalfNormal + TruncatedNormal.
- **v2.5.0 — corvus adoption** (open, #6): 2 open / 0 closed — #47
  bessel.h rewire, #52 Binomial beta_p CDF rewrite; the core-swap work
  itself plus the before/after characterization sweeps. See the staging
  entry above for rationale and prerequisites. Both annotated 2026-08-21:
  scope against the cores' REAL accuracy once #113 (v2.3.2) lands — the
  large-parameter rows that motivated them are iteration-cap artefacts.
- **v2.6.0 — New Distributions (Extended)** (open, #3, renumbered from
  v2.5.0 on 2026-08-21): 5 open / 0 closed
  — #58 GEV (depends on #54), #59 LogLogistic (depends on #54),
  #60 Triangular, #61 Wald, #62 Hypergeometric + BetaBinomial + Zipf.
  #62's Zipf CDF design (summation vs Hurwitz-zeta closed form) must be
  settled before this milestone's planning — it scopes corvus P3 work.
- **v3.0.0 — Architecture Refactor** (open, #4): 5 open / 0 closed —
  #40 split CMakeLists.txt into cmake/ modules, #41 unify the dual SIMD
  namespace, #42 decompose parallel_execution.h, #43 extract dispatch/cache
  boilerplate into a CRTP or policy helper, #108 trig-kernel duplication
  (record the `simd_neon.cpp:761` decision or a per-tier traits layer).

## GitHub Issues Without Milestone [DERIVED]
- Open: none — #103/#104 and the 2026-08-21 review set #105–#118 are all
  milestoned (v2.3.1, v2.3.2, v3.0.0; see GitHub Milestones above).
- #84 closed 2026-08-16 — see Resolved log.
- Closed: 15, none milestoned (#84 closed 2026-08-16; #90 and #94 2026-08-15 — see
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
  payoff but **needs the Mac Mini M1**; no new
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
  M1.

## Known Gaps [OPEN]
- `vector_floor` + `vector_blend` primitives across all SIMD backends would
  enable a branchless Discrete CDF and Uniform PDF/LogPDF. Low priority,
  not rejected — amortization already delivers the batch-path speedups.
- The `exp_max` clamp constant sits ~30 ULP (in x) below the true overflow
  threshold, so for x in that one-double window the kernels return
  `exp(exp_max)` (~214 ULP low) where `std::exp` is still finite. This is a
  deliberate safety margin against a 1-ULP overshoot to inf; left as is.
- [2026-08-16] **v2.2.0's validation matrix covers one machine of three.**
  The documented release practice (`docs/CI_CD_GUIDE.md`, "Release
  validation practice") supplements CI with native runs on AVX2+FMA, NEON
  and AVX-512. Only the Asus TUF A16 AVX-512 leg has a v2.2.0 run
  (49/49 correctness, 2026-08-16); the M1 NEON and Kaby Lake AVX2 legs are
  outstanding. Judgement call whether that gates the tag: the release
  carries no SIMD kernel change — the Bessel work is scalar header code and
  the rest is build plumbing — so the tiers are not the risk surface here,
  and the one platform-dependent behaviour (#97's libstdc++ `domain_error`
  path) is covered by `test_bessel_tier` on the Linux and macOS runners.
- [2026-08-16] **`UniformEnhancedTest.SIMDAndParallelBatchImplementations`
  is flaky on the AVX-512 validation machine** — 2 failures in 3
  back-to-back runs, measuring 1.5x against a 1.8x adaptive threshold at
  5000 elements. Pre-existing: nothing in v2.2.0 touches uniform or the
  dispatch thresholds. Same class as the Poisson assertion that v2.2.0
  excluded from the AVX-512 workflow, and it is `timing`-labelled so CI
  never runs it. Not yet filed — worth an issue that either widens the
  margin for cheap-PDF distributions or drops the assertion, since a gate
  that fails two thirds of the time on the reference machine is not
  measuring what it claims.

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
least four v2.3.0 issues: corvus already ships 1-ULP erf/erfc with a proper
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

## Defensive Review 2026-08-21 [DERIVED]
Between-milestone review of v2.3.0 (metrics, architecture, numerical,
type/input safety; every finding adversarially verified against the shipped
`stats_static.lib` — 63 findings, 4 refuted, 15 downgraded). Ledger in the
session artifact; the issues carry the detail.
- **Landed at HEAD:** two small behaviour changes — Gaussian's standard-normal
  fast path now requires exactly (0, 1) rather than a 1e-8 band (a 2e-9
  discontinuity in cdf(0); regression test shown to fail first), and the SSE2
  `vector_log` scalar tail is plain `std::log` like the other tiers (it mapped
  NaN/negatives to −inf by lane position). Test/tooling: trig ULP-gate specials
  lead with ±inf/NaN so the 4/8-wide tiers evaluate them in-vector; an
  in-place aliasing test for the dispatched trig entry points; `accuracy_sweep`
  puts its specials first (this is how LogNormal's batch cdf(NaN) = 1 escaped
  #102); the Gaussian/LogNormal CDF generators invert deep-tail targets by a
  root solve (the 1e-320/1e-300/1e-100 rows now exist); `run_tests` /
  `run_tests_timing` / `run_all_tests` pass `-C $<CONFIG>` (on the VS
  generator the timing target ran zero tests and exited 0, and `run_tests`
  ran the timing suite it excludes); the accuracy gates join `run_all_tests`.
  Doc/contract: no-aliasing stated for the batch overloads (AGENTS.md, batch
  guide); the three SIMD conventions moved here → AGENTS.md; test counts
  53/77; object libraries are groupings, not a chain; Cauchy is a PDF/LogPDF-
  only delegate with STALE-marked CDF thresholds; vcpkg optional; Windows
  toolchain text version-generic.
- **Ranking for triage:** #105 (vector_log NaN → finite plausible values
  through the public batch API), #116 (quantile returns 0), #106 (CDF 0 where
  truth is 1 across the seam at κ > 1000 — reachable from `fit()`), #112
  (decide the aliasing contract centrally), #115, #113 (correct the accuracy
  premise before #47/#52 are scoped), the #104 Cauchy split form. The
  libhmm-shaped answer is a v2.3.1 patch milestone for #105/#106/#116/#104.
- **Held up, recorded so nobody re-reviews it:** every batch overload throws
  on a size mismatch (57/57); every validator rejects every non-finite
  parameter (66/66); all x86 `vector_log`s have the 2^54 subnormal prescale;
  the −0 sign blend is on all five tiers with signbit asserted; the #95
  exact-product lemma holds for parts 0–2 (part 3 rounds at ≤ 2^-124, below
  the error floor); the NEON compensated sequences (#84) are still safe by
  construction and no x86 TU carries one; all four v2.3.0 gates have
  fail-first records; `LIBSTATS_MAX_SIMD_TIER`, the dispatch table, the
  install tree and `libstats.h` are complete; the cppcheck `error` in
  `NegativeBinomial::trySetParameters` is a false positive (both arguments
  are validated before the throwing call).
- **Corrections to the record:** `docs/ACCURACY_CHARACTERIZATION.md`'s
  "large-parameter CDF → corvus" rows reproduce from the iteration caps, not
  the incomplete-gamma/beta cores (#113) — the corvus adoption decision stands
  on provenance grounds, its accuracy premise does not; the #46 sweep's
  victim list for #102 is structurally incomplete (specials in the scalar
  tail); the von Mises fallback error is ≈ 0.04/κ, not O(1/κ²).

## Next Steps
1. **v2.3.1 — Correctness patch** is next: start with the sweep re-run
   (specials now in-vector) to re-scope #102, then #105, #116, #106, #115,
   #112. Bump `[Unreleased]` in CHANGELOG to 2.3.1 at release and coordinate
   the pylibstats pin.
2. v2.3.2 after it (#113 first — it corrects the record #47/#52 rest on).
3. Settle the corvus-adoption question (corvus/PLAN.md) with #113's
   correction in hand, then v2.4.0/v2.5.0 or the v3.0.0 refactor.
4. ~~Bump pylibstats' pin to v2.2.0~~ **DONE 2026-08-16** — pylibstats 0.5.0
   is on PyPI against v2.2.0; both problems it surfaced were pre-existing
   pylibstats packaging defects. Detail lives in `pylibstats/PLAN.md`.

## Resolved log
One line per closed item; detail lives in `CHANGELOG.md`, `docs/`, and this
file's git history.
- 2026-08-20 **#95 closed** — clean-room quadrant-reduction cos/sin at every
  SIMD tier (PR #98): max 1 ULP all tiers, per-tier ULP gates + dispatched-entry
  gate checked in; sin(−0) sign defect caught by the gate, fixed, and filed
  upstream as libhmm#81. See CHANGELOG [Unreleased].
- 2026-08-20 **#48 closed** — Cauchy CDF closed-form arctan (scalar + batch
  autoDispatch), ~2 ULP with a cancellation-free lower-tail branch;
  mpmath-referenced test the old delegation fails at 2.7e-10. See
  CHANGELOG [Unreleased].
- 2026-08-16 **v2.2.0 tagged** — the Bessel set (#92 log I0 continuity, #93
  circular variance, #96 complement-series coefficients, #97 installed
  export dropping the tier), the #90 export fix, #94's Ninja unbreak, and
  the build-stack standardization. Numbered 2.2.0 rather than 2.1.1 because
  the CMake floor, install paths and installed header set all changed; see
  Status. Milestone #1 closed with 5 issues, its 7 open ones moved to the
  new v2.3.0. See CHANGELOG [2.2.0].
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
- 2026-08-16 **#84 closed — NO EXPOSURE**, reversing the 2026-08-15 verdict
  recorded on the issue and here. That pass inventoried *where* the compensated
  sequences are and then declared exposure without checking whether any of them
  contains an operation a compiler could actually contract. None does.
  Contraction needs a ROUNDED multiply adjacent to an add; all three sites fail
  that test, for two different reasons.
  - sin/cos reduction and erf's final add: every multiply is already inside an
    explicit `vfmsq_f64`/`vfmaq_f64`, including the error-recovery step itself.
    The only unfused ops have no multiply beside them, and a multiply feeding an
    FMA's *product* operand cannot be folded further. erf is arguably not the
    hazard class at all — `E`/`El` compensate a tabulated constant's
    representation error, not an operation's rounding, exactly the distinction
    libhmm #70 drew for its Cody-Waite splits.
  - log's Fast2Sum is the one genuine rounded-multiply-into-add, and its product
    is **exact**: `kLogNeonLn2Hi` carries 42 significant bits and `|ed| ≤ 1074`
    needs 11, so 53 total. Verified over the rationals, not assumed — exact for
    every e in [−2954, 2954]. Fusing skips a rounding that never happens.
  - Policy: `-ffp-contract=off` **withdrawn**. It would cost FMA throughout the
    NEON kernels, where fusion is deliberate and accuracy-positive, to defend
    against something provably inert.
  - Same conclusion as libhmm #70 by a different route, and worth keeping the
    distinction: libhmm has ZERO instances of the hazard class, so nothing can
    break. libstats has three real compensated sequences that are safe because
    of how they were WRITTEN — a stronger property and a more fragile one, since
    it lives in the source rather than the build. Hence the contraction-proofing
    rule now in AGENTS.md.
  - Also dissolves the M1 dependency the earlier write-up recorded: no kernel
    change means no ULP re-measurement, so nothing here waits on hardware.
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
