import re,sys
def block(path,isa="NEON"):
    L=open(path).read().split("\n")
    s=next(i for i,l in enumerate(L) if l.strip()==f"<!-- BEGIN GENERATED isa={isa} -->")
    e=next(i for i,l in enumerate(L) if l.strip()==f"<!-- END GENERATED isa={isa} -->")
    rows={}; dist=None
    for l in L[s:e]:
        if l.startswith("### "): dist=l[4:].strip(); continue
        if l.startswith("|") and dist:
            c=[x.strip() for x in l.strip().strip("|").split("|")]
            if len(c)>=5 and c[0] in ("pdf","logpdf","cdf","quantile","survival") and c[1] in ("scalar","batch"):
                rows[(dist,c[0],c[1])]=c
    return rows
o=block(sys.argv[1], sys.argv[3] if len(sys.argv)>3 else "NEON"); n=block(sys.argv[2], sys.argv[4] if len(sys.argv)>4 else "NEON")
print(f"old rows {len(o)} new rows {len(n)}; only-old {sorted(set(o)-set(n))[:5]} only-new {sorted(set(n)-set(o))[:5]}")
def f(x):
    try: return float(x)
    except: return None
names=["max_abs","max_rel","p99_rel"]
cnt=0; worse=0
for k in sorted(set(o)&set(n)):
    for col,nm in zip((2,3,4),names):
        a,b=f(o[k][col]),f(n[k][col])
        if a is not None and b is not None and a>0 and b>0 and (a/b>2 or b/a>2):
            cnt+=1; worse+= b>a
            print(f"  {k[0]:18}{k[1]:9}{k[2]:7} {nm:8} {o[k][col]:>11} -> {n[k][col]:>11}  {'WORSE' if b>a else 'better'} {max(a/b,b/a):.1f}x")
print(f"{cnt} cells moved > 2x ({worse} worse)")
