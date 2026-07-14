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
Last reconciled against live GitHub state: 2026-07-14.
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
  - #49 OPEN — LogNormalDistribution CDF accuracy 2.62×10⁻⁷ — likely scalar erfc vs vector_erf path divergence.
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
  - #33 OPEN — v1.5.x: Evaluate table+Taylor approach for NEON transcendentals — cross-architecture experiment (see Known Gaps).
- Closed issues without milestone: 9 as of 2026-07-14.

## In Progress [OPEN]
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

## Known Gaps [OPEN]
- `vector_floor` + `vector_blend` primitives across all SIMD backends to
  enable branchless Discrete CDF and Uniform PDF/LogPDF — low priority
  given existing batch-path speedups already achieved through
  amortization, not rejected.
- Issue #33: cross-architecture experiment evaluating table-lookup vs.
  polynomial approach for exp/log (NEON's table-based `vector_erf`
  achieves 8.0x vs ~0.9x for the pure-polynomial equivalent — open
  question whether this generalizes to exp/log on other architectures).

## Next Steps
- Assess the two unmerged branches now on deck:
  `fix/remove-stale-vector-erfc-stub` and
  `backup/wip-sleef-avx2-gather-bench`.
- Work through the v2.1.0 — Accuracy & Performance backlog (6 issues,
  mostly SIMD accuracy/perf gaps) before starting the new-distribution
  milestones (v2.2.0, v2.3.0) or the v3.0.0 architecture refactor.
