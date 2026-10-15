#!/usr/bin/env python3
"""Derive dispatch-threshold table rows from the committed R8 bundles (PLAN.md R8, decided 2026-10-14).

One script for every ArchTable, so the rules cannot drift between tables. Per table it reads the
three-run bundle at the final head, the previous capture (the "moved for a code reason" test), the
current table in dispatch_thresholds.h, and for K's tables Z's capped same-tier bundle (direction
check only: Z never supplies a value).

Per (distribution, operation) row, in order:
  1. KEEP (ok)        the K-method candidate (median of three, or the max when the runs spread > 2x)
                      is within one grid step of the table value.
  2. KEEP (override)  the candidate is within one grid step of the previous capture's candidate: the
                      measurement did not move, so the table value is a standing deliberate override.
                      Its comment is listed for the user to judge stale or not. Not applied when the
                      table value is below the fork value (PARALLEL does not fork there, so such a value
                      was never a measured crossover), nor to rows in DROP_OVERRIDE (user's decisions,
                      with the evidence recorded). Rows in KEEP_ROW keep the table value by the user's
                      judgment against the rules, with the evidence recorded. Rows in TAKE_CANDIDATE
                      take the candidate although rule 1 would keep them (one grid step away), where
                      the loaded A/B agrees with the quiet captures; evidence recorded.
  3. SKIP (spread)    the three runs spread more than 2x and the maximum (the conservative pick) is
                      within one grid step of the table: too noisy to change. If the maximum still
                      clears the table by more than a grid step, the maximum is the candidate.
  4. CHANGE (fork)    the candidate is at or below the table's fork zone: not a measured crossover,
                      PARALLEL wins as soon as it forks. Encode the fork value; KEEP if the table
                      already holds it.
  5. CHANGE           otherwise, encode the candidate.
For kAvx2 and kAvx, each CHANGE row reports whether Z's capped same-tier candidate moves in the same
direction relative to the table (earlier, later, NEVER); rows where Z disagrees are flagged.

Usage: derive_threshold_rows.py [--header PATH] [--out DIR] [--tables kNeon,kAvx2,...]
Writes <out>/rows_<table>.txt (every row and its decision), <out>/summary.txt and
<out>/dispatch_thresholds.h.proposed (the header with the CHANGE rows applied, comments untouched).
"""
import argparse, importlib.util, os, re, statistics, sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PROF = os.path.join(REPO, "data", "profiles", "dispatcher")
HEADER = os.path.join(REPO, "include", "libstats", "core", "dispatch_thresholds.h")

spec = importlib.util.spec_from_file_location("ac", os.path.join(HERE, "analyze_crossovers.py"))
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
NEVER, OPS, INF = ac.NEVER, ac.OPS, float("inf")

# table -> (new bundle, previous bundle, fork value to encode, fork zone upper bound, Z direction bundle)
TABLES = OrderedDict([
    ("kAvx512", dict(new="2026-10-14T02-41-05Z_windows-x86_64_dev-v2.4.2_sha-df78931",
                     prev="2026-10-07T02-10-06Z_windows-x86_64_dev-v2.4.2_sha-8271f6e",
                     fork=8192, fork_zone=8192, confirm=None, machine="Z (Zen 4, clang-cl)")),
    ("kAvx2",   dict(new="2026-10-13T13-11-55Z_darwin-x86_64_dev-v2.4.2_sha-d1adc95",
                     prev="2026-10-05T01-50-52Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
                     fork=4096, fork_zone=6144,
                     confirm="2026-10-14T03-05-23Z_windows-x86_64_dev-v2.4.2_sha-df78931",
                     machine="K (Kaby Lake, AppleClang 15)")),
    ("kAvx",    dict(new="2026-10-13T13-58-00Z_darwin-x86_64_dev-v2.4.2_sha-d1adc95",
                     prev="2026-10-05T03-09-27Z_darwin-x86_64_dev-v2.4.2_sha-6f86996",
                     fork=4096, fork_zone=6144,
                     confirm="2026-10-14T03-29-48Z_windows-x86_64_dev-v2.4.2_sha-df78931",
                     machine="K (Kaby Lake, AppleClang 15), AVX-capped tree")),
    ("kNeon",   dict(new="2026-10-14T04-39-37Z_darwin-arm64_dev-v2.4.2_sha-d1adc95",
                     prev="2026-10-05T04-36-25Z_darwin-arm64_dev-v2.4.2_sha-2c5230a",
                     fork=2048, fork_zone=2048, confirm=None, machine="M (M1, AppleClang 21)")),
])
NAME_MAP = {"negbinomial": "negativebinomial"}  # CSV name -> header row name, after norm()
# Rows exempted from rule 2 (KEEP override) by the user's decision, with the reason recorded.
DROP_OVERRIDE = {
    ("kNeon", "VonMises", "PDF"): "user 2026-10-14: the 100k override argued from the pre-#111 CDF; 25k measured 2026-10-05 and 2026-10-14; AUTO 1.4x behind PARALLEL at 1e5 (AB_RESULTS)",
    ("kNeon", "VonMises", "LogPDF"): "user 2026-10-14: the 500k override argued from the pre-#111 CDF; 25k measured 2026-10-05 and 2026-10-14; AUTO 2.8-3x behind PARALLEL at 1e5 (AB_RESULTS)",
    ("kNeon", "Gaussian", "CDF"): "table_ab ABAB 2026-10-14: PARALLEL beats VECTORIZED 1.25x at 10k, 2.1x at 25k, 4.7x at 500k under both tables; candidate 10k measured three times",
    ("kNeon", "Discrete", "CDF"): "table_ab ABAB 2026-10-14: AUTO 1.7-3.7x behind PARALLEL at 100k-500k under both tables; candidate 75k measured three times",
    ("kNeon", "Exponential", "PDF"): "table_ab ABAB 2026-10-14: at 25k-50k VECTORIZED (1.6 ns) beats PARALLEL in every pass, so the 25k row is early; candidate 75k measured three times",
    ("kNeon", "Gumbel", "PDF"): "table_ab ABAB 2026-10-14: AUTO 1.45x behind PARALLEL at 50k under both tables; candidate 25k measured three times",
    ("kNeon", "FisherF", "PDF"): "table_ab ABAB 2026-10-14: PARALLEL 1.52x ahead at 4096 under both tables; the 8192 row has disagreed with the 2048 measurement since 2026-10-05",
}
# Rows kept at their table value by the user's judgment although the rules would change them.
KEEP_ROW = {
    ("kNeon", "Exponential", "CDF"): "user 2026-10-14: the quiet profile's 8192 is marginal for a 2.5 ns/element kernel; on the loaded machine PARALLEL at 8192-10k loses 1.6-2x to VECTORIZED and only matches it at 25k (table_ab ABAB); 25k kept",
    ("kNeon", "Rayleigh", "CDF"): "user 2026-10-14: as Exponential CDF; PARALLEL at 8192 loses 1.8x, wins at 25k (table_ab ABAB); 25k kept",
}
# Rows within one grid step of the table (rule 1 would keep them) where the user takes the candidate: the
# quiet captures and the loaded A/B agree on the direction at that size.
TAKE_CANDIDATE = {
    ("kNeon", "Gamma", "CDF"): "user 2026-10-14: candidate 2048 in both captures; table_ab round 3: PARALLEL 1.5-1.7x ahead at 2048 in both passes (ChiSquared and Erlang CDF follow this row)",
    ("kNeon", "Beta", "CDF"): "user 2026-10-14: candidate 2048 three times in both captures; PARALLEL 1.6x ahead at 2048 in both passes",
    ("kNeon", "Weibull", "LogPDF"): "user 2026-10-14: candidate 25k; PARALLEL 1.5-1.8x ahead at 25k in both passes",
    ("kNeon", "Logistic", "PDF"): "user 2026-10-14: candidate 25k; PARALLEL 1.5x ahead at 25k in both passes",
    ("kNeon", "Logistic", "LogPDF"): "user 2026-10-14: candidate 25k; PARALLEL 1.45x ahead at 25k in both passes",
    ("kNeon", "LogNormal", "PDF"): "user 2026-10-14: candidate 25k; PARALLEL 1.25-1.4x ahead at 25k in both passes",
    ("kNeon", "Pareto", "PDF"): "user 2026-10-14: candidate 50k; PARALLEL 1.5-1.6x ahead at 50k in both passes",
    ("kNeon", "Gaussian", "PDF"): "user 2026-10-14: candidate 75k three times; VECTORIZED 1.45x ahead of PARALLEL at 50k in both passes (TruncatedNormal LogPDF's AUTO tracks this row)",
}
REDERIVE_DATE = "2026-10-14"


def fmt(x):
    return "NEVER" if x == NEVER else (f"{x // 1000}k" if x >= 10000 and x % 1000 == 0 else str(x))


def runs_of(bundle):
    return [ac.load(os.path.join(PROF, bundle, f"strategy_profile_run{i}.csv")) for i in (1, 2, 3)]


def row_values(runs, dist, op):
    return [ac.sustained(r[(dist, op)]) for r in runs if (dist, op) in r]


def spread(vals):
    v = [INF if x == NEVER else x for x in vals]
    lo, hi = min(v), max(v)
    if lo == INF and hi == INF:
        return 1.0
    return INF if hi == INF else hi / lo


def maxv(vals):
    return NEVER if any(x == NEVER for x in vals) else max(vals)


def direction(cand, table, grid):
    if ac.within_one_step(grid, cand, table):
        return "same"
    if cand == NEVER:
        return "later"
    if table == NEVER:
        return "earlier"
    return "earlier" if cand < table else "later"


ROW_OPEN = re.compile(r"/\*\s*([A-Z_0-9]+?)\((\d+)\)\s*\*/\s*\{")


def table_span(lines, table):
    start = next(i for i, l in enumerate(lines) if re.match(r"\s*constexpr ArchTable %s\b" % table, l))
    end = next(i for i in range(start, len(lines)) if lines[i].strip().startswith("}};"))
    return start, end


def row_spans(lines, start, end):
    """-> {norm_name: (first_line, last_line, values_text, has_comment)}; braces may span lines."""
    rows, i = {}, start + 1
    while i < end:
        m = ROW_OPEN.search(lines[i])
        if not m:
            i += 1; continue
        j, body = i, lines[i][m.end():]
        while "}" not in body:
            j += 1; body += lines[j]
        body = body[:body.index("}")]
        rest = lines[j][lines[j].index("}") + 1:]
        has_comment = "//" in rest
        k = j + 1
        while k < end and lines[k].strip().startswith("//") and (len(lines[k]) - len(lines[k].lstrip())) >= 40:
            has_comment = True; k += 1
        rows[ac.norm(m.group(1))] = (i, j, body, has_comment)
        i = j + 1
    return rows


def parse_table(header, table):
    lines = open(header, encoding="utf-8").read().splitlines(keepends=True)
    start, end = table_span(lines, table)
    out = {}
    for key, (i, j, body, cmt) in row_spans(lines, start, end).items():
        vals = tuple(NEVER if t.strip() == "NEVER" else int(t) for t in body.split(","))
        out[key] = (vals, cmt)
    return out


def derive(table, cfg, header, grid_hint=None):
    new, prev = runs_of(cfg["new"]), runs_of(cfg["prev"])
    conf = runs_of(cfg["confirm"]) if cfg["confirm"] else None
    cur = parse_table(header, table)
    grid = sorted({s for r in new for k in r for s in r[k]})
    dists = sorted({k[0] for r in new for k in r})
    rows = []
    for dist in dists:
        key = NAME_MAP.get(ac.norm(dist), ac.norm(dist))
        if key not in cur:
            sys.exit(f"{table}: no header row for {dist} ({key})")
        tvals, has_comment = cur[key]
        for oi, op in enumerate(OPS):
            vals = row_values(new, dist, op)
            t = tvals[oi]
            cand = ac.candidate(vals)
            pcand = ac.candidate(row_values(prev, dist, op))
            sp = spread(vals)
            zdir = direction(ac.candidate(row_values(conf, dist, op)), t, grid) if conf else "-"
            note = ""
            if (table, dist, op) in TAKE_CANDIDATE:
                pick = cand
                if pick != NEVER and pick <= cfg["fork_zone"]:
                    pick = cfg["fork"]
                decision, proposed = "CHANGE taken", pick
                note = "candidate taken: " + TAKE_CANDIDATE[(table, dist, op)]
            elif ac.within_one_step(grid, cand, t):
                decision, proposed = "KEEP ok", t
            elif (table, dist, op) in KEEP_ROW:
                decision, proposed = "KEEP judgment", t
                note = KEEP_ROW[(table, dist, op)]
            elif ac.within_one_step(grid, cand, pcand) and (table, dist, op) not in DROP_OVERRIDE \
                    and not (t != NEVER and t < cfg["fork"]):
                decision, proposed = "KEEP override", t
                note = "measures as before; table comment stands unless stale" if has_comment else "measures as before; no comment on the row"
            else:
                pick = cand
                if (table, dist, op) in DROP_OVERRIDE:
                    note = "override dropped: " + DROP_OVERRIDE[(table, dist, op)]
                elif t != NEVER and t < cfg["fork"] and ac.within_one_step(grid, cand, pcand):
                    note = f"table {fmt(t)} is below the fork value {cfg['fork']}: not a measured crossover, so not an override"
                if sp > ac.SPREAD_LIMIT:
                    mx = maxv(vals)
                    if ac.within_one_step(grid, mx, t):
                        decision, proposed = "SKIP spread", t
                        note = f"runs spread {sp:.1f}x; max {fmt(mx)} within a step of the table"
                        rows.append((dist, op, t, vals, cand, pcand, decision, proposed, has_comment, zdir, note)); continue
                    pick = mx
                    note = f"runs spread {sp:.1f}x; max {fmt(mx)} still clears the table"
                if pick != NEVER and pick <= cfg["fork_zone"]:
                    if t == cfg["fork"]:
                        decision, proposed = "KEEP ok", t
                        note = "candidate in the fork zone; table already at the fork value"
                    else:
                        decision, proposed = "CHANGE fork", cfg["fork"]
                        note = (note + "; " if note else "") + f"candidate {fmt(pick)} in the fork zone: parallel once it forks"
                else:
                    decision, proposed = "CHANGE", pick
                if decision.startswith("CHANGE") and conf and zdir != direction(proposed, t, grid):
                    note = (note + "; " if note else "") + f"Z direction {zdir}, DIFFERS"
            if not ac.within_one_step(grid, pcand, t) and decision.startswith("CHANGE"):
                note = (note + "; " if note else "") + "table already overrode the previous capture"
            rows.append((dist, op, t, vals, cand, pcand, decision, proposed, has_comment, zdir, note))
    return rows, grid


def write_rows(path, table, cfg, rows, grid):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{table}: rows derived from {cfg['new']} ({cfg['machine']})\n"
                f"previous capture {cfg['prev']}; fork value {cfg['fork']}, fork zone <= {cfg['fork_zone']}"
                + (f"; Z direction from {cfg['confirm']}" if cfg["confirm"] else "") + "\n"
                f"grid: {' '.join(fmt(g) for g in grid)}\n\n")
        f.write(f"{'Distribution':16}{'Op':7}{'table':>7}  {'run1/run2/run3':>22}  {'cand':>6}  {'prev':>6}  {'decision':14}{'proposed':>9}  {'cmt':3}  {'Zdir':7} note\n")
        for dist, op, t, vals, cand, pcand, dec, prop, cmt, zdir, note in rows:
            f.write(f"{dist:16}{op:7}{fmt(t):>7}  {'/'.join(fmt(v) for v in vals):>22}  {fmt(cand):>6}  {fmt(pcand):>6}  "
                    f"{dec:14}{fmt(prop):>9}  {'yes' if cmt else '-':3}  {zdir:7} {note}\n")
        counts = {}
        for r in rows:
            counts[r[6]] = counts.get(r[6], 0) + 1
        f.write("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())) + f"  total={len(rows)}\n")


def apply(header_text, table, cfg, rows):
    """Replace the braces of CHANGE rows inside `table`; prefix each changed row's trailing comment with a
    re-derivation marker naming the old values, keep the existing prose after it, and add a note block
    above the table."""
    lines = header_text.splitlines(keepends=True)
    start, end = table_span(lines, table)
    changes = {}
    for dist, op, t, vals, cand, pcand, dec, prop, cmt, zdir, note in rows:
        if dec.startswith("CHANGE"):
            changes.setdefault(NAME_MAP.get(ac.norm(dist), ac.norm(dist)), {})[OPS.index(op)] = (t, prop)
    n = 0
    for key, (i, j, body, _) in sorted(row_spans(lines, start, end).items(), key=lambda kv: -kv[1][0]):
        if key not in changes:
            continue
        vals = [v.strip() for v in body.split(",")]
        marks = []
        for oi in sorted(changes[key]):
            t, prop = changes[key][oi]
            vals[oi] = "NEVER" if prop == NEVER else str(prop); n += 1
            marks.append(f"{OPS[oi]} {fmt(t)}\u2192{fmt(prop)}")
        m = ROW_OPEN.search(lines[i])
        rest = lines[j][lines[j].index("}") + 1:]
        marker = f"// {REDERIVE_DATE}: " + ", ".join(marks) + "."
        existing = re.search(r"// %s: (.*?)\.(\s|$)" % re.escape(REDERIVE_DATE), rest)
        prefix = f"// {REDERIVE_DATE}: "
        if existing:  # a marker from an earlier run of this script: merge the new marks into it
            rest = rest[:existing.start(1)] + existing.group(1) + ", " + ", ".join(marks) + rest[existing.end(1):]
        elif prefix in rest:  # an earlier marker wrapped onto the next line by clang-format
            i0 = rest.index(prefix) + len(prefix)
            rest = rest[:i0] + ", ".join(marks) + ", " + rest[i0:]
        elif "//" in rest:
            before, after = rest.split("//", 1)
            rest = before + marker + " " + after.lstrip()
        else:
            rest = rest.rstrip("\n").rstrip() + "  " + marker + "\n"
        lines[i:j + 1] = [lines[i][:m.end()] + ", ".join(vals) + "}" + rest]
    note = (f"// {REDERIVE_DATE}: rows re-derived at the v2.5.0 head from the R8 bundle\n"
            f"// {cfg['new']}\n"
            f"// by scripts/derive_threshold_rows.py (PLAN.md R8, the 2026-10-14 rules; fork value {cfg['fork']}).\n"
            f"// A row marked \"{REDERIVE_DATE}:\" names its old and new values; the prose after the marker\n"
            f"// predates the re-derivation and describes the older measurement.\n")
    if n and not any(note.splitlines()[0].strip() in l for l in lines[max(0, start - 8):start]):
        lines.insert(start, note)
    return "".join(lines), n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--header", default=HEADER)
    ap.add_argument("--out", default=os.path.join(REPO, "build-rows"))
    ap.add_argument("--tables", default=",".join(TABLES))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    text = open(a.header, encoding="utf-8").read()
    summary = []
    total = 0
    for table in a.tables.split(","):
        cfg = TABLES[table]
        rows, grid = derive(table, cfg, a.header)
        write_rows(os.path.join(a.out, f"rows_{table}.txt"), table, cfg, rows, grid)
        text, n = apply(text, table, cfg, rows)
        total += n
        counts = {}
        for r in rows:
            counts[r[6]] = counts.get(r[6], 0) + 1
        diff = [r for r in rows if r[6].startswith("CHANGE") and "DIFFERS" in r[10]]
        cmt = [r for r in rows if r[6].startswith("CHANGE") and r[8]]
        ovr = [r for r in rows if "already overrode" in r[10]]
        summary.append(f"{table:8} {cfg['machine']}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                       + f"; {n} values written" + (f"; Z direction differs on {len(diff)}" if cfg["confirm"] else "")
                       + f"; {len(cmt)} changed rows carry a comment to re-read; {len(ovr)} changed rows already overrode the previous capture")
    with open(os.path.join(a.out, "dispatch_thresholds.h.proposed"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary) + f"\ntotal values changed: {total}\n")
    print("\n".join(summary)); print(f"total values changed: {total}")


if __name__ == "__main__":
    main()
