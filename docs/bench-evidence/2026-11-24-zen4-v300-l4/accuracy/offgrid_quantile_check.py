import struct, math, mpmath as mp
mp.mp.dps = 130
R='C:/Users/gdwol/Development/libstats-z-v300/sweep/'
f=lambda h: struct.unpack('<d',struct.pack('<Q',int(h,16)))[0]
b=lambda x: struct.unpack('<Q',struct.pack('<d',x))[0]
# library values
lib={}
for l in open(R+'offgrid.csv'):
    if l[0]=='#' or l.startswith('dist'): continue
    s=l.strip().split(',')
    if s[1]=='quantile': lib[(s[0],)+tuple(int(v,16) for v in s[2:5])]=s[5]
rows=[]
for l in open(R+'offgrid_rows.csv'):
    if l[0]=='#' or l.startswith('dist'): continue
    s=l.strip().split(',')
    if s[1]!='quantile' or s[0] not in('student_t','fisher_f','inverse_gamma','weibull'): continue
    sc=[float(v) for v in s[5:7] if v!='-']
    if any(v>100 for v in sc): rows.append((s,sc[0]))
half=mp.mpf(1)/2
def tails(dist,a,c,x):
    """return (lower, upper) tail probabilities at x; the small one computed directly"""
    if dist=='student_t':
        nu=a; z=nu/(nu+x*x)
        t=mp.betainc(nu/2,half,0,z,regularized=True)/2   # P(T<-|x|)
        return (t,1-t) if x<0 else (1-t,t)
    if dist=='fisher_f':
        z=a*x/(a*x+c)
        lo=mp.betainc(a/2,c/2,0,z,regularized=True)
        up=mp.betainc(c/2,a/2,0,1-z,regularized=True) if False else mp.betainc(c/2,a/2,0,c/(a*x+c),regularized=True)
        return lo,up
    if dist=='inverse_gamma':
        y=c/x
        up_=mp.gammainc(a,0,y,regularized=True)   # P(alpha,y)= survival of IG
        lo=mp.gammainc(a,y,mp.inf,regularized=True)
        return lo,up_
    if dist=='weibull':
        e=(x/c)**a
        return -mp.expm1(-e), mp.exp(-e)
def tail_of(dist,a,c,x,p,side):
    lo,up=tails(dist,a,c,x); return lo if side=='lo' else up
def solve(dist,a,c,p,x0):
    p=mp.mpf(p)
    side='lo' if p<half else 'up'
    target=p if side=='lo' else 1-p   # 1-p exact in 130 digits
    sgn=1 if side=='lo' else -1
    if dist=='student_t':
        # solve for |x| on the small tail; x<0 if lo
        def g(u):
            x=-mp.exp(u) if side=='lo' else mp.exp(u)
            return mp.log(tail_of(dist,a,c,x,p,side))-mp.log(target)
        sg=-1 if side=='lo' else 1
        mk=lambda u: -mp.exp(u) if side=='lo' else mp.exp(u)
    else:
        def g(u): return mp.log(tail_of(dist,a,c,mp.exp(u),p,side))-mp.log(target)
        mk=lambda u: mp.exp(u)
    u0=mp.log(abs(mp.mpf(x0)))
    lo_,hi_=u0-mp.mpf('0.01'),u0+mp.mpf('0.01'); k=0
    while g(lo_)*g(hi_)>0:
        k+=1; w=mp.mpf(2)**k*mp.mpf('0.01'); lo_,hi_=u0-w,u0+w
        if k>14: raise RuntimeError('no bracket')
    # bisection to ~1e-100 in u, then it's enough
    glo=g(lo_)
    for _ in range(360):
        mid=(lo_+hi_)/2; gm=g(mid)
        if (gm>0)==(glo>0): lo_,glo=mid,gm
        else: hi_=mid
        if hi_-lo_<mp.mpf(10)**-110: break
    return mk((lo_+hi_)/2),side
def nxt(x,d): return math.nextafter(x,math.inf if d>0 else -math.inf)
def Q(dist,a,c,p):
    if dist=='weibull':
        return c*(-mp.log1p(-p))**(1/a)
for s,orc in rows:
    dist=s[0]; a=f(s[2]); c=f(s[3]); p=f(s[4]); xl=f(lib[(s[0],)+tuple(int(v,16) for v in s[2:5])])
    if dist=='student_t': pass
    pm=mp.mpf(p); am=mp.mpf(a); cm=mp.mpf(c)
    if dist=='weibull':
        ref=Q(dist,am,cm,pm); side='cf'
    else:
        ref,side=solve(dist,am,cm,pm,xl)
    err=abs(mp.mpf(xl)-ref)/abs(ref)/mp.mpf(2)**-52
    r=float(ref); ud=b(xl)-b(r) if (xl>0)==(r>0) else None
    if xl<0: ud=-(b(xl)-b(r)) if r<0 else None
    print(f"{dist} a={a!r} c={c!r} p={p!r} side={side}")
    print(f"  lib={xl!r}  ref={mp.nstr(ref,19)}  err_eps={float(err):.3f} oracle={orc}  ulps_from_RN={ud}")
    if dist!='weibull':
        target=pm if side=='lo' else 1-pm
        for nm,x in (('prev',nxt(xl,-1)),('lib',xl),('next',nxt(xl,1))):
            t=tail_of(dist,am,cm,mp.mpf(x),pm,side)
            print(f"   {nm} x={x!r} tail/target-1={mp.nstr(t/target-1,6)}")
        # CDF relative sensitivity per ulp of x
        x1=nxt(r,1); t1=tail_of(dist,am,cm,mp.mpf(x1),pm,side); t0=tail_of(dist,am,cm,mp.mpf(r),pm,side)
        print(f"   rel dF per ulp x: {mp.nstr(abs(t1/t0-1),5)}")
