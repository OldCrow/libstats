# Generates docs/UML/class-hierarchy.svg in the corvid-cartography style (STYLE.md).
INK, BAND, CARD, GOLD, MUTED_C, MUTED_I = "#161b23", "#232a36", "#e7e0cf", "#c8a24e", "#6b6455", "#9a9382"
W = 860
CW, CH = 76, 58          # card size
out = []
def t(x, y, s, size=9.5, fill=INK, weight=None, anchor="middle", extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    out.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}"{w}{extra}>{s}</text>')
def card(x, y, title, detail, tag, w=CW, h=CH, italic=False, stereo=None):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
    cx = x + w/2
    if stereo:
        t(cx, y+13, stereo, 8.5, MUTED_C)
        t(cx, y+27, title, 11, INK, "bold", extra=' font-style="italic"' if italic else "")
        t(cx, y+42, detail, 9.5, INK)
        if tag: t(cx, y+54, tag, 8.5, MUTED_C)
    else:
        t(cx, y+19, title, 11 if (w > CW or len(title) <= 11) else (9.5 if len(title) <= 14 else 8.5), INK, "bold", extra=' font-style="italic"' if italic else "")
        t(cx, y+34, detail, 9.5, INK)
        if tag: t(cx, y+48, tag, 8.5, MUTED_C)
def band(x, y, w, h, heading):
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{BAND}" stroke="{GOLD}" stroke-width="1.5"/>')
    t(x+14, y+18, heading, 11.5, GOLD, "bold", "start", ' letter-spacing="0.5"')
def inh(d):  out.append(f'<path d="{d}" fill="none" stroke="{GOLD}" stroke-width="1.6" marker-end="url(#inh)"/>')
def dele(d): out.append(f'<path d="{d}" fill="none" stroke="{GOLD}" stroke-width="1.6" stroke-dasharray="5 3" marker-end="url(#del)"/>')

# ---- header ----
t(20, 30, "libstats class hierarchy", 15, GOLD, "bold", "start", ' letter-spacing="0.5"')
t(20, 46, "DistributionInterface → DistributionBase (+ ThreadSafeCacheManager mixin) → 27 concrete distributions in 7 families · v2.4.0", 10.5, MUTED_I, None, "start")

# ---- top classes ----
card(300, 60, "DistributionInterface", "", "", w=260, h=62, italic=True, stereo="«interface»")
out.pop()   # drop the empty detail line
t(430, 99, "getProbability · getLogProbability · getCumulativeProbability", 8.5, INK)
t(430, 112, "getQuantile · getMean · getVariance · fit · …", 8.5, INK)
card(600, 60, "ThreadSafeCacheManager", "cache_mutex_ · cacheValidAtomic_", "", w=210, h=62, stereo="«mixin»")
out.append(f'<rect x="280" y="150" width="300" height="96" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
t(430, 168, "DistributionBase", 11, INK, "bold")
t(430, 182, "getSurvival · getHazard · fitWithDiagnostics", 8.5, INK)
t(430, 194, "shouldUseSIMDBatch · isApproximatelyEqual", 8.5, INK)
t(430, 211, "span batch API — (span in, span out, PerformanceHint):", 8.5, MUTED_C)
t(430, 223, "getProbability · getLogProbability · getCumulativeProbability", 8.5, MUTED_C)
t(430, 235, "static parallelBatchFit(datasets, results)", 8.5, MUTED_C)
inh("M430 150 V124")                 # Base -> Interface
inh("M580 190 H705 V124")            # Base -> mixin

# ---- band grid ----
LX, RX, BW = 20, 460, 380
TRUNK = 430
def cols(bx): return [bx+23 + i*86 for i in range(4)]
LC, RC = cols(LX), cols(RX)

# Positive-support (left) 250..528 — 3 rows with a 30 px delegation corridor between rows 1 and 2
PS_Y = 254; r1, r2, r3 = PS_Y+30, PS_Y+30+CH+30, PS_Y+30+CH+30+CH+12
band(LX, PS_Y, BW, r3+CH+14-PS_Y, "POSITIVE-SUPPORT CONTINUOUS  (x ≥ 0)")
card(LC[0], r1, "Gamma", "Γ(α, β)", "continuous"); card(LC[1], r1, "Exponential", "Exp(λ)", "continuous")
card(LC[2], r1, "LogNormal", "LogN(μ, σ)", "continuous"); card(LC[3], r1, "Weibull", "W(k, λ)", "continuous")
card(LC[0], r2, "ChiSquared", "χ²(ν)", "delegate"); card(LC[1], r2, "Erlang", "Erl(k, λ)", "delegate")
card(LC[2], r2, "InverseGamma", "IG(α, β)", "1/x transform"); card(LC[3], r2, "Rayleigh", "R(σ)", "continuous")
card(LC[0], r3, "FisherF", "F(d₁, d₂)", "continuous"); card(LC[1], r3, "HalfNormal", "HN(σ)", "erf/erfc")
PS_BOT = r3+CH+14
gb = r1+CH                       # Gamma bottom
dele(f"M{LC[0]+16} {r2} V{gb}")                       # ChiSquared -> Gamma   (30 px)
dele(f"M{LC[1]+38} {r2} V{gb+15} H{LC[0]+60} V{gb}")   # Erlang -> Gamma
dele(f"M{LC[2]+38} {r2} V{gb+21} H{LC[0]+38} V{gb}")   # InverseGamma -> Gamma (pdf/logpdf)

# Bounded (left)
BD_Y = PS_BOT+20; bc = BD_Y+30
band(LX, BD_Y, BW, CH+44, "BOUNDED CONTINUOUS")
card(LC[0], bc, "Uniform", "U(a, b)", "continuous"); card(LC[1], bc, "Beta", "Beta(α, β)", "continuous"); card(LC[2], bc, "TruncatedNormal", "TN(μ, σ, a, b)", "erfc-normalised")
BD_BOT = BD_Y+CH+44
# Power-law + Circular half bands (left)
HB_Y = BD_BOT+20; hc = HB_Y+30
band(LX, HB_Y, 180, CH+44, "POWER-LAW"); card(LX+52, hc, "Pareto", "Pareto(xₘ, α)", "continuous")
band(LX+200, HB_Y, 180, CH+44, "CIRCULAR"); card(LX+252, hc, "VonMises", "VM(μ, κ)", "circular")
HB_BOT = HB_Y+CH+44

# Discrete (right) — 2 rows with corridor
DS_Y = 254; d1, d2 = DS_Y+30, DS_Y+30+CH+30
band(RX, DS_Y, BW, d2+CH+14-DS_Y, "DISCRETE")
card(RC[0], d1, "Binomial", "B(n, p)", "discrete"); card(RC[1], d1, "NegativeBinomial", "NB(r, p)", "discrete")
card(RC[2], d1, "Poisson", "Pois(λ)", "discrete"); card(RC[3], d1, "Discrete", "PMF table", "discrete")
card(RC[0], d2, "Bernoulli", "Bern(p)", "delegate"); card(RC[1], d2, "Geometric", "Geo(p)", "delegate")
DS_BOT = d2+CH+14
dele(f"M{RC[0]+38} {d2} V{d1+CH}"); dele(f"M{RC[1]+38} {d2} V{d1+CH}")
# Symmetric (right)
SY_Y = DS_BOT+20; sc = SY_Y+30
band(RX, SY_Y, BW, CH+44, "SYMMETRIC, UNBOUNDED CONTINUOUS")
card(RC[1], sc, "Gaussian", "N(μ, σ²)", "continuous"); card(RC[2], sc, "Logistic", "L(μ, s)", "continuous"); card(RC[3], sc, "StudentT", "t(ν)", "continuous")
SY_BOT = SY_Y+CH+44
# Real-line asymmetric (right)
RL_Y = SY_BOT+20; rc = RL_Y+30
band(RX, RL_Y, BW, CH+44, "REAL-LINE, ASYMMETRIC")
card(RC[1], rc, "Laplace", "Laplace(μ, b)", "continuous"); card(RC[2], rc, "Gumbel", "G(μ, β)", "max-stable"); card(RC[3], rc, "Cauchy", "Cauchy(x₀, γ)", "PDF delegate")
RL_BOT = RL_Y+CH+44
dele(f"M{RC[3]+38} {rc} V{sc+CH}")   # Cauchy -> StudentT (through the inter-band gap)

# ---- inheritance trunk: Base -> every band (all classes in a band inherit DistributionBase) ----
last = max(HB_Y+ (CH+44)//2, RL_Y+(CH+44)//2)
inh_base_y = 246
out.append(f'<path d="M{TRUNK} {inh_base_y} V{last}" fill="none" stroke="{GOLD}" stroke-width="1.6"/>')
def stub_left(y):  inh(f"M{TRUNK} {y} H{LX+BW}")
def stub_right(y): inh(f"M{TRUNK} {y} H{RX}")
stub_left(PS_Y + (PS_BOT-PS_Y)//2); stub_left(BD_Y + (BD_BOT-BD_Y)//2); stub_left(HB_Y+(CH+44)//2)   # Circular right edge
inh(f"M{TRUNK} {BD_BOT+10} H{LX+90} V{HB_Y}")                                                        # Power-law via the gap
stub_right(DS_Y+(DS_BOT-DS_Y)//2); stub_right(SY_Y+(CH+44)//2); stub_right(RL_Y+(CH+44)//2)

# ---- analysis band ----
AN_Y = max(HB_BOT, RL_BOT)+22; ac = AN_Y+30
band(LX, AN_Y, 820, CH+44, "STATS::ANALYSIS NAMESPACES")
card(LX+23, ac, "stats::analysis", "kolmogorovSmirnovTest · andersonDarlingTest", "informationCriteria · bootstrapMeanCI · kFoldCrossValidation", w=290)
for i,n in enumerate(["gaussian","exponential","gamma","poisson","discrete","binomial"]):
    card(LX+330+i*82, ac, f"::{n}", "per-distribution", "tests · CIs", w=76)
AN_BOT = AN_Y+CH+44

# ---- legend + footer ----
LG_Y = AN_BOT+16
out.append(f'<rect x="{LX}" y="{LG_Y}" width="470" height="44" rx="7" fill="{CARD}" stroke="{GOLD}" stroke-width="1.2"/>')
inh(f"M{LX+14} {LG_Y+16} H{LX+70}"); t(LX+80, LG_Y+20, "inherits (solid, hollow head) — a band arrow means every class in the band", 8.5, INK, None, "start")
dele(f"M{LX+14} {LG_Y+34} H{LX+70}"); t(LX+80, LG_Y+38, "delegates internally (dashed, open head) — PDF/CDF forwarded to the target class", 8.5, INK, None, "start")
H = LG_Y+44+26
t(W-20, H-10, "libstats v2.4.0  ·  docs/UML/class-hierarchy.svg", 9.5, MUTED_I, None, "end")

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<title>libstats class hierarchy — v2.4.0</title>
<defs>
 <marker id="inh" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="11" refY="6" orient="auto"><polygon points="1 1,11 6,1 11" fill="{INK}" stroke="{GOLD}" stroke-width="1.4"/></marker>
 <marker id="del" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="11" refY="6" orient="auto"><path d="M1 1 L11 6 L1 11" fill="none" stroke="{GOLD}" stroke-width="1.6"/></marker>
</defs>
<rect width="{W}" height="{H}" fill="{INK}"/>
<g font-family="system-ui,-apple-system,Helvetica,Arial,sans-serif">
{chr(10).join(out)}
</g>
</svg>
'''
open(__import__("os").path.join(__import__("os").path.dirname(__file__), "class-hierarchy.svg"),"w").write(svg); print("height", H)
