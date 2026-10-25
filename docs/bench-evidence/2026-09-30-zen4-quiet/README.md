# 2026-09-30 Zen 4 re-pass — QUIET (5% gate held), frequency regime still mixed

Same machine, binaries, prefixes and runner as
`../2026-09-29-zen4-indicative/` (see its README for the configuration and
the two-compiler setup). Launched by `repass.ps1`: every target listed twice
so it runs back to back, with the second run's output kept. Gate passed at
3.66%; per-target noise 2.20–5.15% (avg 3.33%) across all 16 runs
(`runner_log.txt`). Screen locked, no user activity, `InventorySvc` stopped.

**Reproducibility.** Every row agrees with the 2026-09-29 pass to within
1–3% — including the rows that pass showed as noise-affected — so the
2026-09-29 numbers stand, and the INDICATIVE label there was about the
evidence chain, not the values.

**Regime.** The back-to-back warm-up did not hold the sustained regime: the
runner's 3 s + 3 s noise samples between the two runs are enough for boost
to recover, so the second run shows the same pattern as the first
(`corvus_scaling` n = 1 row boosted, later rows ~1.5× slower; `dist_v241`
gamma–student_t boosted, binomial onward sustained; `dist_v250` gamma
boosted, beta onward sustained). The step lands at a reproducible point
~8–15 s into each process, and the rows are deterministic across passes. The
ratios under ~2× in the distribution tables remain unreliable for the same
reason as before; the large ratios and the elementary rows (4 s runs, no
step) are the record.

Holding one regime needs the warm-up inside the process (a spin before the
first measurement), not between processes — done the same day:
`../2026-09-30-zen4-quiet-warm/` is the record.
