#!/usr/bin/env python3
"""Fit a start-delay and a slow-run model to batch_gap_bench CSVs (PLAN.md, "Pool wake-up and
batch spacing").

For each distribution, strategy and gap, the time of one call at size n is compared with the same
cell back to back (gap 0):

    t_gap(n) = W + k * t_0(n)

W is a fixed cost per call (the workers start late), k a factor on the whole call (the workers run
slowly). Three fits per row, each scored by its relative RMS residual over the sizes:

    delay   k = 1,  W = median(t_gap - t_0)
    factor  W = 0,  k = median(t_gap / t_0)
    both    W and k by least squares on relative error

"reads as" names the one-parameter model with the smaller residual; when both leave more than
--tolerance (default 0.10) it is "delay+factor" if the two-parameter fit is within tolerance and
"neither" if not; "none" when the gap costs under 5% at every size.

Usage: fit_gap_model.py gap_plain.csv [gap_init.csv ...] [--strategy PARALLEL] [--tolerance 0.10]
Standard library only.
"""
import argparse, csv, statistics, sys
from collections import defaultdict


def load(path):
    """{(pool, dist, strategy): {gap_us: {n: ns_per_call}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    with open(path, newline="") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            n = int(r["BatchSize"])
            key = (r["Pool"], r["Distribution"], r["Strategy"])
            out[key][float(r["Gap_us"])][n] = float(r["MedianNsPerElement"]) * n
    return out


def rel_rms(pred, obs):
    return (sum(((p - o) / o) ** 2 for p, o in zip(pred, obs)) / len(obs)) ** 0.5


def fit_both(t0, t):
    """Least squares on relative error: minimise sum(((W + k*t0 - t)/t)^2)."""
    a = [1.0 / y for y in t]            # coefficient of W
    b = [x / y for x, y in zip(t0, t)]  # coefficient of k
    saa, sab, sbb = sum(x * x for x in a), sum(x * y for x, y in zip(a, b)), sum(y * y for y in b)
    sa, sb = sum(a), sum(b)
    det = saa * sbb - sab * sab
    if abs(det) < 1e-30:
        return 0.0, 1.0
    return (sa * sbb - sb * sab) / det, (saa * sb - sab * sa) / det


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--strategy", default="PARALLEL")
    ap.add_argument("--tolerance", type=float, default=0.10)
    args = ap.parse_args()

    print(f"{'pool':6} {'dist':9} {'strategy':10} {'gap_us':>7}  {'delay W_us':>10} {'resid':>6}  "
          f"{'factor k':>8} {'resid':>6}  {'both W_us':>9} {'k':>5} {'resid':>6}  reads as")
    for path in args.csv:
        for (pool, dist, strategy), by_gap in sorted(load(path).items()):
            if strategy != args.strategy or 0.0 not in by_gap:
                continue
            base = by_gap[0.0]
            for gap in sorted(g for g in by_gap if g > 0):
                sizes = sorted(set(base) & set(by_gap[gap]))
                t0 = [base[n] for n in sizes]
                t = [by_gap[gap][n] for n in sizes]
                w_delay = statistics.median(y - x for x, y in zip(t0, t))
                r_delay = rel_rms([x + w_delay for x in t0], t)
                k_factor = statistics.median(y / x for x, y in zip(t0, t))
                r_factor = rel_rms([k_factor * x for x in t0], t)
                w_both, k_both = fit_both(t0, t)
                r_both = rel_rms([w_both + k_both * x for x in t0], t)
                if max(y / x for x, y in zip(t0, t)) < 1.05:
                    verdict = "none"
                elif min(r_delay, r_factor) > args.tolerance:
                    verdict = "delay+factor" if r_both <= args.tolerance else "neither"
                else:
                    verdict = "delay" if r_delay < r_factor else "factor"
                print(f"{pool:6} {dist:9} {strategy:10} {gap:7.0f}  {w_delay / 1e3:10.1f} "
                      f"{r_delay:6.2f}  {k_factor:8.2f} {r_factor:6.2f}  {w_both / 1e3:9.1f} "
                      f"{k_both:5.2f} {r_both:6.2f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
