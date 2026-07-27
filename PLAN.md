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
Last reconciled against live GitHub state: 2026-07-26.
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

- **v2.2.0 — Accuracy & Performance** (open, #1): 6 open / 1 closed
  (#83 include restructure shipped 2026-07-26).
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
  double-double result 0.6 bits vs MSVC). Needs a milestone decision.
  **Exposure is confirmed and narrow**: the only error-free transforms in
  this repo are the clean-room NEON kernels in `src/simd_neon.cpp` (log's
  two Fast2Sum steps, erf's compensated final add, cos's compensated
  reduction) plus `src/neon_erf_data.inc`. AppleClang contracts by default
  and builds exactly those, and their published ULP bounds depend on the
  identities holding as written.
- Closed: 12, none milestoned.

## In Progress [OPEN]
- None. Every branch tracked here before 2026-07-21 has merged or been
  deleted; see the Resolved log.

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
  entry). Pre-existing, orthogonal to #83; needs an issue. On Linux this
  can underlink installed-package consumers.

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

## Next Steps
1. **#84** — run the FP-contraction inventory against `src/simd_neon.cpp`
   and decide the policy. corvus's resolution was `-ffp-contract=off`
   scoped PRIVATE to the affected targets, fusion requested only in source,
   at ≤~8% measured cost. Authoring is machine-independent; **re-measuring
   the NEON ULP bounds needs the Mac Mini M1.**
2. **#48** — Cauchy closed-form arctan CDF. Smallest measurable win in the
   backlog (3–5× on Zen 4, and the gap widens with SIMD width).
3. Settle the corvus-adoption question above, then work the rest of v2.2.0
   before starting v2.3.0/v2.4.0 or the v3.0.0 refactor.

## Resolved log
One line per closed item; detail lives in `CHANGELOG.md`, `docs/`, and this
file's git history.
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
