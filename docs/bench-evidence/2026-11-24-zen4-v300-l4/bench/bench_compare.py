"""Median of three passes, main -> dev, for distributions_bench and elementary_bench output.

Run in the directory holding bench_{dist,elem}_{main,dev}_pass{1,2,3}.txt.
"""
import statistics as st


def dist(side):
    d = {}
    for p in (1, 2, 3):
        for line in open(f"bench_dist_{side}_pass{p}.txt"):
            t = line.split()
            if len(t) == 8 and t[0] != "ns/elem":
                d.setdefault(t[0], []).append([float(x) for x in t[1:]])
    return d


def elem(side):
    d = {}
    for p in (1, 2, 3):
        for line in open(f"bench_elem_{side}_pass{p}.txt"):
            t = line.split()
            if t and t[0].startswith("n="):
                d.setdefault(t[0], []).append([float(t[i]) for i in (2, 4, 6, 8, 10)])
    return d


cols = ["pdf@1e6", "lpdf@1e6", "cdf@1e6", "pdf-sc", "lpdf-sc", "cdf-sc", "quantile"]
m, d = dist("main"), dist("dev")
spread = 0.0
print("| ns/elem | " + " | ".join(cols) + " |")
print("|" + "---|" * 8)
for k in m:
    cells = []
    for i in range(7):
        a = [r[i] for r in m[k]]
        b = [r[i] for r in d[k]]
        for v in (a, b):
            if min(v) > 0:
                spread = max(spread, max(v) / min(v))
        ma, mb = st.median(a), st.median(b)
        cells.append(f"{ma:.1f} → {mb:.1f} ({mb / ma:.2f}×)" if ma > 0 else f"{ma} → {mb}")
    print(f"| {k} | " + " | ".join(cells) + " |")
print(f"largest max/min over the three passes of any distributions cell: {spread:.2f}")

m, d = elem("main"), elem("dev")
spread = 0.0
print("\n| ns/elem | exp | log | cos | sin | erf |")
print("|" + "---|" * 6)
for k in m:
    cells = []
    for i in range(5):
        a = [r[i] for r in m[k]]
        b = [r[i] for r in d[k]]
        for v in (a, b):
            spread = max(spread, max(v) / min(v))
        ma, mb = st.median(a), st.median(b)
        cells.append(f"{ma:.2f} → {mb:.2f} ({mb / ma:.2f}×)")
    print(f"| {k} | " + " | ".join(cells) + " |")
print(f"largest max/min over the three passes of any elementary cell: {spread:.2f}")
