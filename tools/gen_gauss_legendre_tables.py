#!/usr/bin/env python3
"""Generate (or check) the Gauss-Legendre tables in src/von_mises.cpp.

The von Mises CDF and quantile integrate with fixed n-point Gauss-Legendre rules,
n = 6, 8, ..., 36. Each rule's n/2 positive nodes (largest first) and weights are
stored as doubles, and each node also as a two-double pair (kGlNode + kGlNodeLo):
the steep exponentials take x - 1 and x + 1 to full precision. Those constants are
regenerated here, never pasted: a hand-pasted table once carried four node pairs
off by up to 1.1e-16.

Nodes come from Newton's method on the Legendre recurrence at 80 digits; every
double is the correctly rounded value (converted through a 60-digit decimal
string; float(mpf) gives the same, checked on 2e5 values with mpmath 1.4.1), and
each low part is the correctly rounded x - hi.

Usage:
  gen_gauss_legendre_tables.py            rewrite the generated block in place
  gen_gauss_legendre_tables.py --check    exit 1 if any table value differs
Requires mpmath (the oracle venv, as tools/accuracy_vs_mpmath.py does).
"""

import argparse
import os
import re
import sys

from mpmath import cos, mp, mpf, pi

RULES = range(6, 37, 2)
BEGIN = "// BEGIN GENERATED gauss-legendre (tools/gen_gauss_legendre_tables.py)"
END = "// END GENERATED gauss-legendre"
SOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "src", "von_mises.cpp")


def to_double(x):
    return float(mp.nstr(x, 60))


def legendre(n, x):
    p0, p1 = mpf(1), x
    for k in range(2, n + 1):
        p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
    return p1, n * (x * p1 - p0) / (x * x - 1)


def rule(n):
    """The n/2 positive nodes, largest first, with their weights."""
    out = []
    for i in range(1, n // 2 + 1):
        x = cos(pi * (4 * i - 1) / (4 * n + 2))
        for _ in range(200):
            p, dp = legendre(n, x)
            step = p / dp
            x -= step
            if abs(step) < mpf(10) ** -75:
                break
        _, dp = legendre(n, x)
        out.append((x, 2 / ((1 - x * x) * dp * dp)))
    return out


def tables():
    mp.dps = 80
    nodes, lows, weights, offsets = [], [], [], []
    for n in RULES:
        offsets.append(len(nodes))
        for x, w in rule(n):
            hi = to_double(x)
            nodes.append(hi)
            lows.append(to_double(x - mpf(hi)))
            weights.append(to_double(w))
    return nodes, lows, weights, offsets


def fmt_array(decl, values, per_line, fmt):
    lines = [decl + " = {"]
    for i in range(0, len(values), per_line):
        lines.append("    " + ", ".join(fmt(v) for v in values[i:i + per_line]) + ",")
    lines.append("};")
    return lines


def block():
    nodes, lows, weights, offsets = tables()
    g = lambda v: repr(v)  # shortest round-trip form
    out = [BEGIN,
           "// Gauss-Legendre rules on [-1, 1], n = 6, 8, ..., 36 nodes: the n/2 positive nodes,",
           "// largest first, and their weights; rule n starts at kGlOffset[(n - 6) / 2]. Each node",
           "// also as a pair: kGlNodeLo is the correctly rounded x - kGlNode. The steep exponentials",
           "// take x - 1 and x + 1 to full precision with it, which removes an error of up to",
           "// ~A eps / sqrt(n) (4 ulp at A = 35) from rounded nodes."]
    out += fmt_array("constexpr double kGlNode[]", nodes, 4, g)
    out += fmt_array("constexpr double kGlWeight[]", weights, 4, g)
    out += fmt_array("constexpr int kGlOffset[]", offsets, 16, str)
    out += fmt_array("constexpr double kGlNodeLo[]", lows, 3, g)
    out.append(END)
    return "\n".join(out), (nodes, lows, weights, offsets)


def parse(text, name, cast):
    body = re.search(rf"constexpr \w+ {name}\[\] = \{{(.*?)\}};", text, re.S).group(1)
    return [cast(v) for v in re.findall(r"[-+]?\d[\d.eE+-]*", re.sub(r"//.*", "", body))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on any difference")
    args = ap.parse_args(argv)
    text = open(SOURCE).read()
    new_block, want = block()
    if args.check:
        names = [("kGlNode", float), ("kGlNodeLo", float), ("kGlWeight", float), ("kGlOffset", int)]
        expected = dict(zip(["kGlNode", "kGlNodeLo", "kGlWeight", "kGlOffset"],
                            [want[0], want[1], want[2], want[3]]))
        bad = 0
        for name, cast in names:
            got = parse(text, name, cast)
            if got != expected[name]:
                diffs = [i for i, (a, b) in enumerate(zip(got, expected[name])) if a != b]
                bad += max(len(diffs), abs(len(got) - len(expected[name])))
                print(f"{name}: {len(diffs)} values differ (first indices {diffs[:6]})")
        print("tables match" if bad == 0 else f"{bad} table values differ")
        return 1 if bad else 0
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0:
        sys.exit(f"markers not found in {SOURCE}")
    text = text[:start] + new_block + text[end + len(END):]
    open(SOURCE, "w").write(text)
    print(f"rewrote the generated block in {SOURCE}; run clang-format on it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
