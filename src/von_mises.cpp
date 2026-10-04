#include "libstats/distributions/von_mises.h"

#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
using stats::detail::validateNonNegativeParameter;
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;

#include "libstats/common/cpu_detection_fwd.h"
#include "libstats/core/bessel.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/parallel_batch_fit.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace stats {

//==============================================================================
// Private helper: angle wrapping
//==============================================================================

// Upper bound of the validated CDF range; above it the CDF uses the
// wrapped-normal approximation and I0e the Hankel series.
constexpr double kCdfExactKappaMax = 1000.0;

double VonMisesDistribution::wrapAngle(double x) noexcept {
    if (!std::isfinite(x))
        return x;
    x = std::fmod(x, detail::TWO_PI);
    if (x <= -detail::PI)
        x += detail::TWO_PI;
    if (x > detail::PI)
        x -= detail::TWO_PI;
    return x;
}

//==============================================================================
// Private helper: κ from mean resultant length R̄
//
// Mardia–Jupp approximation (Mardia & Jupp 2000, Directional Statistics §A.2),
// refined by Newton–Raphson on A(κ) = I₁(κ)/I₀(κ) = R̄.
// Derivative: A'(κ) = 1 − A(κ)² − A(κ)/κ.
// Always converges (function is monotone increasing); 3–5 Newton steps suffice.
//==============================================================================

namespace {

[[nodiscard]] double kappa_from_r_bar(double R_bar) noexcept {
    if (R_bar <= 0.0)
        return 0.0;
    if (R_bar >= 1.0)
        return 1.0e6;  // effectively point mass

    double kappa;
    if (R_bar < 0.53) {
        kappa = detail::TWO * R_bar + R_bar * R_bar * R_bar +
                (5.0 / 6.0) * R_bar * R_bar * R_bar * R_bar * R_bar;
    } else if (R_bar < 0.85) {
        kappa = -0.4 + 1.39 * R_bar + 0.43 / (detail::ONE - R_bar);
    } else {
        const double r = R_bar;
        kappa = detail::ONE / (r * r * r - 4.0 * r * r + 3.0 * r);
    }
    if (kappa < 0.0)
        kappa = 0.0;

    for (int iter = 0; iter < 20 && kappa > 0.0; ++iter) {
        // A(κ) via the ratio helper (#93): the direct i1/i0 form returned NaN
        // above κ ≈ 713, where both values overflow and the old `i0 <= 0.0`
        // guard did not catch inf — the Newton step then propagated NaN and
        // the fit silently produced a NaN concentration.
        const double A = detail::bessel_i1_over_i0(kappa);
        const double Ap = detail::ONE - A * A - A / kappa;
        if (std::fabs(Ap) < 1e-15)
            break;
        const double dk = (A - R_bar) / Ap;
        kappa -= dk;
        if (kappa < 0.0) {
            kappa = 0.0;
            break;
        }
        if (std::fabs(dk) < 1e-12 * (detail::ONE + kappa))
            break;
    }
    return kappa;
}

//==============================================================================
// CDF and quantile by quadrature in w = kappa (1 - cos theta)
//
// For mu = 0 and t in [-pi, 0] let
//   A = kappa (1 + cos t) = 2 kappa cos^2(t/2),   B = kappa (1 - cos t) = 2 kappa sin^2(t/2),
// so A + B = 2 kappa, and the density at t is g(t) = e^-B / (2 pi I0e(kappa)), with I0e(kappa)
// = I0(kappa) e^-kappa. In w = kappa (1 - cos theta), dtheta = dw / sqrt(w (2 kappa - w)), so
// every mass is an integral of e^-w / sqrt(w (2 kappa - w)); v = sqrt(w) removes the
// singularity at w = 0. Two masses follow:
//
//   central   C(tau) = int_{-tau}^{0} e^{kappa (cos theta - 1)} dtheta
//                    = 2 int_0^{sqrt B} e^{-v^2} (2 kappa - v^2)^{-1/2} dv,      B = B(-tau),
//   tail      G(t) = int_{-pi}^{t} f = g(t) J(t),
//             J(t) = int_0^A e^{-s} ((A - s)(B + s))^{-1/2} ds          (s = w - B).
//
// Within 1/4 of the median the CDF is 1/2 -+ C / (2 pi I0e); beyond it, G, which keeps
// relative accuracy down to 1e-300 and whose logarithm ln G = ln g + ln J does not underflow.
// The upper tail is the mirror: 1 - F(t) = G(-t). Every method below sums positive terms,
// except the Taylor series (1) where B < A, whose terms alternate in sign.
//
// J by one of four methods, by region; TailMethods sets which a caller may use (the series
// are cheaper, the quadratures more accurate):
//
// (1) Taylor series, min(A, B) >= 45. h(s) = ((A - s)(B + s))^{-1/2} satisfies
//     (A - s)(B + s) h' = (B - A + 2s) h / 2, so a_k = k! [s^k] h obeys
//       a_0 = (AB)^{-1/2},  a_{k+1} = ((B - A)(k + 1/2) a_k + k^2 a_{k-1}) / (AB),
//     and J ~ sum a_k (Watson's lemma). The radius of h is min(A, B), so the terms fall like
//     k! / min(A, B)^k down to ~e^-min(A, B), below 3e-20 here.
// (2) Mode series, A >= 45, B < 45. Expanding (2 kappa - w)^{-1/2} in w / (2 kappa) and taking
//     each term from w = B to infinity,
//       J ~ (2 kappa)^{-1/2} sum c_k (2 kappa)^{-k} Q_k,   c_k = binom(2k, k) / 4^k,
//       Q_k = e^B Gamma(k + 1/2, B):  Q_0 = sqrt(pi) erfcx(sqrt B),  Q_{k+1} = (k + 1/2) Q_k +
//       B^{k+1/2}.
//     The range beyond w = 2 kappa adds O(e^-A). The terms fall like (B + k) / (2 kappa); where
//     they stop falling before 2^-56 of the sum (B near kappa), the method declines.
// (3) Edge quadrature, t < -pi/2 and A < 45. With s = A - v^2,
//       J = 2 int_0^{sqrt A} e^{v^2 - A} (2 kappa - v^2)^{-1/2} dv,
//     smooth (the singularity at v = sqrt(2 kappa) is past sqrt(2) times the upper limit), by
//     Gauss-Legendre with 18-36 nodes by A. (Above A = 42 the lower limit is sqrt(A - 42),
//     dropping less than e^-42.)
// (4) Mode quadrature, t >= -pi/2 otherwise. J = 2 e^B int_{sqrt B}^{sqrt(2 kappa)} e^{-v^2}
//     (2 kappa - v^2)^{-1/2} dv, split at v = sqrt(kappa) (theta = -pi/2): the part above it is
//     e^{-kappa cos t} J(-pi/2), with J(-pi/2) cached per kappa (tailHalfJ_); the part below by
//     Gauss-Legendre with 18-28 nodes by the exponent drop kappa cos t, capped at 42 (beyond
//     it the upper limit is sqrt(B + 42) and the rest is dropped).
//
// C(tau) = 2 sqrt(B) int_0^1 e^{-B x^2} (2 kappa - B x^2)^{-1/2} dx, by Gauss-Legendre on the
// even integrand: the singularity sits at x = 1 / sin(tau/2) >= sqrt(2), so 12-24 nodes (6-12
// evaluations) suffice.
//
// The node counts are the counts mpmath needs for 1e-17 relative error, plus 2. Below kappa
// = 1e-17, J = d = t + pi and C = tau to double precision.
//
// I0e(k) = I0(k) e^-k comes from the trapezoid rule on the periodic integrand
// exp(k (cos phi - 1)), exponentially convergent (aliasing ~ e^(-N^2/(2k)),
// N^2/(2k) >= 50 here), and not from log_bessel_i0(k) - k, which cancels
// ~log2(k) bits and is only 1.6e-7 accurate on the Tier 2 Bessel path. Above
// kappa = 1000 it comes from the Hankel series log_bessel_i0 uses there,
// without its leading k.
//==============================================================================

// pi - PI: the double PI is below pi by this much (to ~1e-32).
constexpr double kPiLo = 1.2246467991473532e-16;

// The CDF takes the central mass where F is within this of the median (and the tail mass
// beyond); the quantile, where m = min(p, 1 - p) is at least 1/2 minus this.
constexpr double kCentralHalfWidth = 0.25;

// F = 1/4 lies at B <= 0.312 for every kappa; the CDF tries the central mass only below this B.
constexpr double kCentralMaxB = 0.5;

// Below this kappa, J = d and C = tau to double precision.
constexpr double kKappaTiny = 1e-17;

// The series (1) and (2) need A >= this; the Taylor series (1) needs B >= this as well.
constexpr double kSeriesMin = 45.0;

// The quadratures stop where the integrand is e^-kExponentCut of its peak.
constexpr double kExponentCut = 42.0;

// sqrt(pi), correctly rounded (detail::SQRT_PI is one ulp below it).
constexpr double kSqrtPi = 1.7724538509055160273;

// BEGIN GENERATED gauss-legendre (tools/gen_gauss_legendre_tables.py)
// Gauss-Legendre rules on [-1, 1], n = 6, 8, ..., 36 nodes: the n/2 positive nodes,
// largest first, and their weights; rule n starts at kGlOffset[(n - 6) / 2]. Each node
// also as a pair: kGlNodeLo is the correctly rounded x - kGlNode. The steep exponentials
// take x - 1 and x + 1 to full precision with it, which removes an error of up to
// ~A eps / sqrt(n) (4 ulp at A = 35) from rounded nodes.
constexpr double kGlNode[] = {
    0.932469514203152,   0.6612093864662645,  0.2386191860831969,  0.9602898564975363,
    0.7966664774136267,  0.525532409916329,   0.1834346424956498,  0.9739065285171717,
    0.8650633666889845,  0.6794095682990244,  0.4333953941292472,  0.14887433898163122,
    0.9815606342467192,  0.9041172563704749,  0.7699026741943047,  0.5873179542866175,
    0.3678314989981802,  0.1252334085114689,  0.9862838086968123,  0.9284348836635735,
    0.827201315069765,   0.6872929048116855,  0.5152486363581541,  0.31911236892788974,
    0.10805494870734367, 0.9894009349916499,  0.9445750230732326,  0.8656312023878318,
    0.755404408355003,   0.6178762444026438,  0.45801677765722737, 0.2816035507792589,
    0.09501250983763744, 0.9915651684209309,  0.9558239495713977,  0.8926024664975557,
    0.8037049589725231,  0.6916870430603532,  0.5597708310739475,  0.41175116146284263,
    0.2518862256915055,  0.0847750130417353,  0.9931285991850949,  0.9639719272779138,
    0.912234428251326,   0.8391169718222188,  0.7463319064601508,  0.636053680726515,
    0.5108670019508271,  0.37370608871541955, 0.22778585114164507, 0.07652652113349734,
    0.9942945854823992,  0.9700604978354287,  0.926956772187174,   0.8658125777203002,
    0.7878168059792081,  0.6944872631866827,  0.5876404035069116,  0.469355837986757,
    0.34193582089208424, 0.20786042668822127, 0.06973927331972223, 0.9951872199970213,
    0.9747285559713095,  0.9382745520027328,  0.8864155270044011,  0.820001985973903,
    0.7401241915785544,  0.6480936519369755,  0.5454214713888396,  0.4337935076260451,
    0.3150426796961634,  0.1911188674736163,  0.06405689286260563, 0.9958857011456169,
    0.978385445956471,   0.9471590666617142,  0.9026378619843071,  0.845445942788498,
    0.7763859488206789,  0.6964272604199573,  0.6066922930176181,  0.5084407148245057,
    0.4030517551234863,  0.2920048394859569,  0.17685882035689018, 0.05923009342931321,
    0.9964424975739544,  0.9813031653708727,  0.9542592806289382,  0.9156330263921321,
    0.8658925225743951,  0.8056413709171791,  0.7356108780136318,  0.656651094038865,
    0.5697204718114017,  0.4758742249551183,  0.3762515160890787,  0.2720616276351781,
    0.16456928213338076, 0.05507928988403427, 0.9968934840746495,  0.9836681232797472,
    0.9600218649683075,  0.9262000474292743,  0.8825605357920527,  0.8295657623827684,
    0.7677774321048262,  0.6978504947933158,  0.6205261829892429,  0.5366241481420199,
    0.44703376953808915, 0.3527047255308781,  0.25463692616788985, 0.15386991360858354,
    0.0514718425553177,  0.9972638618494816,  0.9856115115452684,  0.9647622555875064,
    0.9349060759377397,  0.8963211557660521,  0.84936761373257,    0.7944837959679424,
    0.7321821187402897,  0.6630442669302152,  0.5877157572407623,  0.5068999089322294,
    0.42135127613063533, 0.33186860228212767, 0.23928736225213706, 0.1444719615827965,
    0.04830766568773832, 0.997571753790842,   0.9872278164063095,  0.9687082625333443,
    0.9421623974051071,  0.9078096777183244,  0.8659346383345645,  0.8168842279009336,
    0.761064876629873,   0.6989391132162629,  0.6310217270805285,  0.5578755006697467,
    0.480106545190327,   0.39835927775864594, 0.3133110813394632,  0.22566669161644948,
    0.13615235725918298, 0.04550982195310254, 0.9978304624840858,  0.9885864789022122,
    0.972027691049698,   0.9482729843995076,  0.9174977745156591,  0.8799298008903972,
    0.8358471669924753,  0.7855762301322066,  0.7294891715935565,  0.668001236585521,
    0.6015676581359806,  0.5306802859262452,  0.45586394443342027, 0.37767254711968923,
    0.29668499534402826, 0.2135008923168656,  0.1287361038093848,  0.04301819847370861,
};
constexpr double kGlWeight[] = {
    0.17132449237917036,  0.3607615730481386,   0.46791393457269104,   0.10122853629037626,
    0.22238103445337448,  0.31370664587788727,  0.362683783378362,     0.06667134430868814,
    0.1494513491505806,   0.21908636251598204,  0.26926671930999635,   0.29552422471475287,
    0.04717533638651183,  0.10693932599531843,  0.16007832854334622,   0.20316742672306592,
    0.2334925365383548,   0.24914704581340277,  0.03511946033175186,   0.08015808715976021,
    0.12151857068790319,  0.15720316715819355,  0.18553839747793782,   0.2051984637212956,
    0.2152638534631578,   0.027152459411754096, 0.062253523938647894,  0.09515851168249279,
    0.12462897125553388,  0.14959598881657674,  0.16915651939500254,   0.18260341504492358,
    0.1894506104550685,   0.02161601352648331,  0.0497145488949698,    0.07642573025488905,
    0.10094204410628717,  0.12255520671147846,  0.14064291467065065,   0.15468467512626524,
    0.16427648374583273,  0.1691423829631436,   0.017614007139152118,  0.04060142980038694,
    0.06267204833410907,  0.08327674157670475,  0.10193011981724044,   0.11819453196151841,
    0.13168863844917664,  0.14209610931838204,  0.14917298647260374,   0.15275338713072584,
    0.0146279952982722,   0.03377490158481415,  0.052293335152683286,  0.06979646842452049,
    0.08594160621706773,  0.10041414444288096,  0.11293229608053922,   0.12325237681051242,
    0.13117350478706238,  0.13654149834601517,  0.13925187285563198,   0.0123412297999872,
    0.028531388628933663, 0.04427743881741981,  0.05929858491543678,   0.0733464814110803,
    0.08619016153195327,  0.09761865210411388,  0.10744427011596563,   0.1155056680537256,
    0.12167047292780339,  0.1258374563468283,   0.12793819534675216,   0.010551372617343006,
    0.02441785109263191,  0.037962383294362766, 0.05097582529714781,   0.06327404632957484,
    0.07468414976565975,  0.08504589431348523,  0.09421380035591415,   0.10205916109442542,
    0.10847184052857659,  0.11336181654631966,  0.11666044348529658,   0.11832141527926228,
    0.009124282593094517, 0.02113211259277126,  0.03290142778230438,   0.04427293475900423,
    0.05510734567571675,  0.0652729239669996,   0.07464621423456878,   0.08311341722890121,
    0.09057174439303284,  0.09693065799792992,  0.10211296757806076,   0.10605576592284642,
    0.10871119225829413,  0.1100470130164752,   0.007968192496166605,  0.01846646831109096,
    0.02878470788332337,  0.03879919256962705,  0.04840267283059405,   0.057493156217619065,
    0.06597422988218049,  0.0737559747377052,   0.08075589522942021,   0.08689978720108298,
    0.09212252223778612,  0.09636873717464425,  0.09959342058679527,   0.1017623897484055,
    0.10285265289355884,  0.007018610009470096, 0.01627439473090567,   0.02539206530926206,
    0.03427386291302143,  0.04283589802222668,  0.050998059262376175,  0.058684093478535544,
    0.06582222277636185,  0.0723457941088485,   0.07819389578707031,   0.08331192422694675,
    0.08765209300440381,  0.09117387869576389,  0.09384439908080457,   0.09563872007927486,
    0.0965400885147278,   0.006229140555908685, 0.014450162748595036,  0.02256372198549497,
    0.03049138063844613,  0.03816659379638752,  0.04552561152335327,   0.05250741457267811,
    0.059054135827524494, 0.06511152155407642,  0.07062937581425573,   0.07556197466003194,
    0.07986844433977185,  0.08351309969984566,  0.08646573974703575,   0.08870189783569386,
    0.09020304437064074,  0.09095674033025987,  0.0055657196642450455, 0.012915947284065574,
    0.020181515297735472, 0.02729862149856878,  0.03421381077030723,   0.04087575092364489,
    0.04723508349026598,  0.05324471397775992,  0.05886014424532482,   0.06403979735501548,
    0.06874532383573645,  0.07294188500565306,  0.07659841064587067,   0.0796878289120716,
    0.0821872667043397,   0.08407821897966193,  0.08534668573933862,   0.08598327567039475,
};
constexpr int kGlOffset[] = {
    0, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75, 88, 102, 117, 133, 150,
};
constexpr double kGlNodeLo[] = {
    -2.22457639152902e-17,   3.212031102377179e-17,   3.917524250872457e-18,
    -5.54885067908428e-17,   1.1625719829749724e-17,  -4.999907577775897e-18,
    -2.8967682069859046e-18, -2.3352971736535508e-17, -2.561358899462181e-17,
    -2.9354889953805544e-17, -2.2600214699526867e-17, -4.8210770585131585e-18,
    7.134192985330875e-18,   -5.209915770219317e-17,  -5.497380348312871e-18,
    -3.563183175402957e-17,  9.618137198627985e-18,   2.1901695274281555e-18,
    2.4709778500376712e-17,  -7.058645884000818e-19,  -2.6482082128493074e-17,
    -8.735001202439097e-18,  -7.602905497290265e-18,  1.5817018457098693e-17,
    -5.569183013988857e-18,  -5.914095566469922e-18,  -2.4190068142444825e-17,
    -1.1315677979849837e-17, 3.5241085894430354e-17,  -2.2123521973463665e-17,
    1.6662404170959257e-17,  -2.1958791252592132e-18, -3.275947755433097e-19,
    4.870000806634652e-17,   4.221580323615994e-17,   3.706537347391958e-17,
    -2.6752889453722812e-17, -1.5947398610890917e-17, -4.417099316759229e-18,
    1.6290180938469612e-17,  2.6451539425003548e-17,  -4.698620630419597e-18,
    4.0125692717995897e-17,  -1.8016704796146567e-17, -4.0267600310095046e-17,
    4.1065867315850824e-17,  -3.109202074074545e-18,  4.73785846574601e-19,
    -2.84952683625147e-17,   1.191005070671823e-17,   9.884156488012629e-18,
    -4.557072655796525e-18,  5.186696747055235e-17,   3.487577903103986e-17,
    1.7585344235354742e-17,  -4.395849109380292e-17,  4.962827803813215e-17,
    3.3898489128760644e-17,  -8.67988554915717e-18,   1.901010941365353e-17,
    -1.6082235459850668e-17, 1.2927941606107902e-17,  -4.10557516776595e-18,
    4.953319652513121e-17,   2.4394038263943126e-17,  -3.9371485806932436e-17,
    -3.7273784348231524e-17, -2.5126255321982337e-17, 6.326231640742655e-18,
    2.372806271361407e-17,   -2.7041126422104026e-17, 1.1230353338563027e-17,
    -2.2454009015222363e-17, -1.511796925720138e-18,  -3.8940572030843924e-18,
    9.520255665925668e-18,   -4.2645446578619344e-17, 1.7242783035381357e-17,
    8.129914062688428e-18,   -3.167109704437702e-17,  -4.9930792796313817e-17,
    -1.872431875449893e-17,  -4.042575758357977e-18,  1.593244551671629e-17,
    1.8179684189304862e-17,  -6.724944906661765e-18,  -1.999373091130135e-20,
    -4.37731727469105e-19,   2.7820620324038694e-17,  5.501388898113071e-17,
    3.041911712650031e-17,   8.295686159865963e-18,   -3.5924439538205653e-17,
    3.769966549007555e-17,   -1.3926229243773648e-17, -5.1827358803409966e-17,
    -1.0359064855595005e-17, -1.354817650810255e-17,  1.3285258387221771e-17,
    -2.7201295629040448e-17, 9.224953612809368e-18,   -2.4684805034475747e-18,
    3.5082771339583354e-17,  3.47450918343211e-17,    -3.558831088624457e-17,
    1.6805036846959143e-17,  -5.45273289997162e-17,   4.055666688840394e-17,
    9.729448743183057e-18,   -4.838875432067803e-17,  -3.0389590834717044e-17,
    3.591409801999972e-17,   2.2719754010484432e-17,  -2.486696391317402e-18,
    -7.904589832209049e-18,  5.243984899758083e-18,   -2.5314224518047992e-18,
    -6.33174860761419e-18,   -4.629527539902715e-17,  4.088711358450183e-17,
    2.2400720424980087e-17,  3.27914172951693e-17,    -2.9688180781309036e-19,
    2.1411189707536002e-17,  -3.0924248426473855e-17, -3.045078950748031e-17,
    2.4748697774509385e-17,  3.0608658253958064e-17,  1.2669012582310482e-17,
    -1.7322512507942286e-17, 9.866977009938267e-18,   5.896477317927168e-18,
    -1.0823887937355691e-18, -3.3747913797009866e-17, -3.6774163679328035e-17,
    -1.8018164930917795e-17, -7.692408148980368e-18,  3.177738832512588e-17,
    -5.223659748535799e-17,  4.0791095432544704e-17,  2.5820155385559366e-17,
    1.0700588673651796e-17,  4.029052979980266e-17,   -2.844201286600093e-17,
    6.6105815227132305e-18,  5.15710389951769e-18,    2.0877839425984293e-17,
    6.481988947244091e-18,   -6.368683824648468e-18,  2.5006275773446803e-18,
    3.5609188997690127e-17,  5.3501543513649045e-17,  -4.562713658332147e-17,
    -3.393160685516165e-17,  6.9067857212761136e-18,  -5.262709865597144e-17,
    7.499902147832018e-18,   -5.26651671138735e-17,   5.302954038400262e-17,
    4.33012612405346e-17,    -3.0168301446032566e-17, -3.843551643920972e-18,
    2.150606426856178e-18,   -1.1767330600592223e-17, 1.3267348014839387e-17,
    -8.05185204550995e-18,   -1.1148958849086981e-17, -3.8285327378751083e-19,
};
// END GENERATED gauss-legendre

// h * sum over the n-point rule of w_i f(x_i), x_i in [-1, 1]: the integral of f(c + h x)
// over [c - h, c + h] when f takes the node as x + x_lo and returns the integrand there.
template <typename F>
[[nodiscard]] double gauss_legendre(int n, double h, F&& f) noexcept {
    const int offset = kGlOffset[(n - 6) / 2];
    double sum = detail::ZERO_DOUBLE;
    for (int i = n / 2 - 1; i >= 0; --i) {  // smallest weights first
        const double x = kGlNode[offset + i];
        const double x_lo = kGlNodeLo[offset + i];
        sum += kGlWeight[offset + i] * (f(-x, -x_lo) + f(x, x_lo));
    }
    return sum * h;
}

// Node counts: mpmath's count for 1e-17 relative error, plus 2 (see the block comment).
[[nodiscard]] int edge_nodes(double A) noexcept {
    constexpr double kMaxA[] = {1.0, 4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 35.0};
    int n = 18;
    for (double a : kMaxA) {
        if (A <= a)
            return n;
        n += 2;
    }
    return n;  // 36
}

[[nodiscard]] int mode_nodes(double drop) noexcept {
    constexpr double kMaxDrop[] = {8.0, 12.0, 16.0, 25.0, 40.0};
    int n = 18;
    for (double a : kMaxDrop) {
        if (drop <= a)
            return n;
        n += 2;
    }
    return n;  // 28
}

[[nodiscard]] int central_nodes(double sin2) noexcept {
    return sin2 <= 0.07 ? 12 : sin2 <= 0.16 ? 14 : sin2 <= 0.28 ? 18 : 24;
}

// (1) Watson's lemma on the Taylor series of h; false if the terms stop falling first.
// a0 = (AB)^{-1/2} = 1 / (kappa |sin t|) leads; bma = B - A needs absolute accuracy only
// (it scales a correction).
[[nodiscard]] bool tail_j_taylor(double A, double B, double bma, double a0, double& j) noexcept {
    const double inv_ab = detail::ONE / (A * B);
    double a_prev = a0;
    double a = detail::HALF * bma * inv_ab * a_prev;
    double sum = a_prev + a;
    double floor_abs = std::fabs(a_prev);
    for (int k = 1; k < 80; ++k) {
        const double next = (bma * (static_cast<double>(k) + detail::HALF) * a +
                             static_cast<double>(k * k) * a_prev) *
                            inv_ab;
        a_prev = a;
        a = next;
        sum += a;
        // Two terms, since a_{k+1} can vanish with B = A while a_{k+2} does not.
        if (std::fabs(a) + std::fabs(a_prev) <= 0x1p-56 * std::fabs(sum)) {
            j = sum;
            return true;
        }
        if (std::fabs(a) > floor_abs && std::fabs(a_prev) > floor_abs)
            return false;
        floor_abs = std::fabs(a_prev);
    }
    return false;
}

// (2) The mode series from rB = sqrt(B); false if the terms stop falling first. erfcx is
// exp(rB^2) erfc(rB), with rB^2 kept to two doubles so that the two factors see one argument.
[[nodiscard]] bool tail_j_mode_series(double rB, double two_kappa, double& j) noexcept {
    const double B = rB * rB;
    const double B_lo = std::fma(rB, rB, -B);
    const double z = detail::ONE / two_kappa;
    double term = kSqrtPi * std::exp(B) * (detail::ONE + B_lo) * std::erfc(rB);  // c_0 Q_0
    double power = rB;  // c_k z^k B^{k+1/2}
    double sum = term;
    for (int k = 0; k < 80; ++k) {
        const double kh = static_cast<double>(k) + detail::HALF;
        const double ratio = kh / (static_cast<double>(k) + detail::ONE);  // c_{k+1} / c_k
        const double next = ratio * z * (kh * term + power);
        power *= ratio * z * B;
        if (next > term && k > 2)
            return false;
        term = next;
        sum += term;
        if (term <= 0x1p-56 * sum) {
            j = std::sqrt(z) * sum;
            return true;
        }
    }
    return false;
}

// (3) J = 2 int e^{v^2 - A} (2 kappa - v^2)^{-1/2} dv over [lo, rA], rA = sqrt(A).
// v^2 - A = (v - rA)(v + rA) = h (x - 1)(v + rA) keeps the exponent free of cancellation.
[[nodiscard]] double tail_j_edge_quadrature(double A, double rA, double two_kappa) noexcept {
    const double drop = std::min(A, kExponentCut);  // rA^2 - lo^2
    const double lo = A > kExponentCut ? std::sqrt(A - kExponentCut) : detail::ZERO_DOUBLE;
    const double h = detail::HALF * drop / (rA + lo);
    const double c = rA - h;
    return detail::TWO * gauss_legendre(edge_nodes(A), h, [&](double x, double x_lo) {
               const double v = c + h * x;
               return std::exp(h * ((x - detail::ONE) + x_lo) * (v + rA)) /
                      std::sqrt(two_kappa - v * v);
           });
}

// (4) J = 2 int e^{B - v^2} (2 kappa - v^2)^{-1/2} dv over [rB, hi], plus e^-D J(-pi/2) when
// hi = sqrt(kappa); D = kappa cos t, accurate relative to itself (the factor e^-D needs it).
[[nodiscard]] double tail_j_mode_quadrature(double rB, double D, double kappa, double two_kappa,
                                            double half_j) noexcept {
    const bool split = D <= kExponentCut;
    const double drop = split ? D : kExponentCut;  // hi^2 - rB^2
    const double hi = split ? std::sqrt(kappa) : std::sqrt(rB * rB + kExponentCut);
    const double h = detail::HALF * drop / (hi + rB);
    const double c = rB + h;
    double j = detail::TWO * gauss_legendre(mode_nodes(drop), h, [&](double x, double x_lo) {
                   const double v = c + h * x;
                   return std::exp(-h * ((x + detail::ONE) + x_lo) * (v + rB)) /
                          std::sqrt(two_kappa - v * v);
               });
    if (split)
        j += std::exp(-D) * half_j;
    return j;
}

// J(-pi/2), where A = B = kappa: the per-kappa constant of (4).
[[nodiscard]] double vonmises_half_tail_j(double kappa) noexcept {
    if (kappa < kKappaTiny)
        return detail::HALF * detail::PI;
    double j;
    if (kappa >= kSeriesMin &&
        tail_j_taylor(kappa, kappa, detail::ZERO_DOUBLE, detail::ONE / kappa, j))
        return j;
    return tail_j_edge_quadrature(kappa, std::sqrt(kappa), detail::TWO * kappa);
}

// ln(2 pi I0(kappa) e^-kappa) above kappa = 1000, by the Hankel series.
[[nodiscard]] double log_scaled_normaliser_hankel(double kappa) noexcept {
    const double t = detail::ONE / kappa;
    const double s =
        t * (0.125 + t * (0.0703125 +
                          t * (0.0732421875 + t * (0.112152099609375 + t * 0.22710800170898438))));
    return detail::LN_2PI - detail::HALF * std::log(detail::TWO_PI * kappa) + std::log1p(s);
}

// 1 / (2 pi I0(kappa) e^-kappa) to about half an ulp: the trapezoid sum (N = 2*half points
// over the full period; the two halves are mirror images) with compensated (Neumaier)
// summation, 2 pi carried to two doubles, and the division corrected by its exact residual.
// Its log is the log normaliser: forming that as ln(2 pi) + ln(sum / n) rounded three times,
// up to 1.2 ulp of the result, all of it an error in the tail mass.
[[nodiscard]] double inv_scaled_normaliser(double kappa) noexcept {
    constexpr double kTwoPiLo = 2.0 * kPiLo;
    if (kappa > kCdfExactKappaMax)
        return std::exp(-log_scaled_normaliser_hankel(kappa));
    const int half = static_cast<int>(std::ceil(detail::HALF * std::sqrt(100.0 * kappa))) + 8;
    const double n = 2.0 * static_cast<double>(half);
    double sum = std::exp(-detail::TWO * kappa);
    double comp = detail::ZERO_DOUBLE;
    auto add = [&](double v) {
        const double s = sum + v;
        comp += std::fabs(sum) >= std::fabs(v) ? (sum - s) + v : (v - s) + sum;
        sum = s;
    };
    for (int j = half - 1; j >= 1; --j) {
        const double sn = std::sin(detail::PI * static_cast<double>(j) / n);
        add(detail::TWO * std::exp(-detail::TWO * kappa * sn * sn));
    }
    add(detail::ONE);
    // D = 2 pi (sum + comp) = p + e, with p = TWO_PI * sum and e its correction.
    const double p = detail::TWO_PI * sum;
    const double e = std::fma(detail::TWO_PI, sum, -p) + kTwoPiLo * sum + detail::TWO_PI * comp;
    const double q = n / p;
    return q + (std::fma(-q, p, n) - q * e) / p;
}

// Which methods for J the tail may use: the series are 3-6 times cheaper than the quadratures
// but less accurate. Against mpmath: quadratures (3), (4) 0.6-0.7 ulp rms, at most 2.4 ulp;
// Taylor series (1) 0.8 ulp rms, at most 3.1; mode series (2) 1.1 ulp rms, at most 3.8.
enum class TailMethods {
    kAll,         ///< (1)-(4): the quantile's iterates, which only steer the solve
    kTaylor,      ///< (1), (3), (4): the CDF
    kQuadrature,  ///< (3), (4): the quantile's last evaluation
};

struct LeftTail {
    double log_mass;  ///< ln G(t)
    double j;         ///< J(t) = G(t) / g(t)
    double log_j;     ///< ln J(t)
    double b;         ///< B = kappa (1 - cos t): g(t) = e^-(b + b_lo) / (2 pi I0e(kappa))
    double b_lo;      ///< B - b: nonzero past a quarter turn, where B = 2 kappa - A is large
};

// G(t) for mu = 0 and t in [-PI, 0], in log form; see the block comment above. Since J <= d,
// ln G <= ln g + ln d; when that bound is below log_floor the bound itself is returned (with
// j = 0) and J is not evaluated -- the CDF passes a floor below which its result rounds to 0
// (or 1 - G to 1) anyway. half_j is J(-pi/2) (vonmises_half_tail_j).
[[nodiscard]] LeftTail vonmises_left_tail(double t, double kappa, double log_scaled_norm,
                                          double half_j, double log_floor,
                                          TailMethods methods) noexcept {
    const double d = (t + detail::PI) + kPiLo;
    const bool near_edge = t < -detail::HALF * detail::PI;
    // cA = cos(t/2) = sin(d/2), cB = sin(-t/2) = cos(d/2); each form where its argument is
    // accurate.
    const double cA = near_edge ? std::sin(detail::HALF * d) : std::cos(detail::HALF * t);
    const double cB = near_edge ? std::cos(detail::HALF * d) : std::sin(-detail::HALF * t);
    const double two_kappa = detail::TWO * kappa;
    const double A = two_kappa * cA * cA;
    // Past a quarter turn B nears 2 kappa, and 2 kappa cB^2 rounded to an ulp is |B| eps in
    // e^-B: ~400 eps at kappa = 200. There B = 2 kappa - A as a pair, A small and accurate:
    // B_lo is Fast2Sum's exact residual of B_hi = 2 kappa - A (|2 kappa| >= A; no product to
    // contract, and a contracted B_hi only moves B_lo by the residual it already absorbed).
    const double B = near_edge ? two_kappa - A : two_kappa * cB * cB;
    const double B_lo = near_edge ? (two_kappa - B) - A : detail::ZERO_DOUBLE;
    const double log_density = -B - log_scaled_norm;
    const double log_bound = log_density + std::log(d) - B_lo;
    if (log_bound < log_floor || kappa < kKappaTiny)
        return {log_bound, kappa < kKappaTiny ? d : detail::ZERO_DOUBLE, std::log(d), B, B_lo};

    // sqrt(2 kappa) c to about an ulp: the root carried to two doubles.
    const double root = std::sqrt(two_kappa);
    const double root_lo = std::fma(-root, root, two_kappa) / (detail::TWO * root);
    auto times_root = [&](double c) { return std::fma(root, c, root_lo * c); };
    const double bma = two_kappa * (cB - cA) * (cB + cA);  // B - A = -2 kappa cos t
    // (AB)^{-1/2} = 1 / (kappa |sin t|), from one sine: 2 kappa cA cB rounds four times.
    const double a0 = detail::ONE / (kappa * (near_edge ? std::sin(d) : std::sin(-t)));
    const bool taylor =
        methods != TailMethods::kQuadrature && A >= kSeriesMin && (near_edge || B >= kSeriesMin);
    const bool mode_series =
        methods == TailMethods::kAll && !near_edge && A >= kSeriesMin && B < kSeriesMin;
    double j;
    if (near_edge) {
        if (!(taylor && tail_j_taylor(A, B, bma, a0, j)))
            j = tail_j_edge_quadrature(A, times_root(cA), two_kappa);
    } else if (!(taylor && tail_j_taylor(A, B, bma, a0, j)) &&
               !(mode_series && tail_j_mode_series(times_root(cB), two_kappa, j))) {
        j = tail_j_mode_quadrature(times_root(cB), kappa * std::cos(t), kappa, two_kappa, half_j);
    }
    const double log_j = std::log(j);
    return {log_density + (log_j - B_lo), j, log_j, B, B_lo};
}

// C(tau) = int_{-tau}^{0} e^{kappa (cos theta - 1)} dtheta for tau in [0, PI/2], from
// s = sin(tau/2); see the block comment above.
[[nodiscard]] double vonmises_central_mass(double tau, double s, double kappa) noexcept {
    if (kappa < kKappaTiny)
        return tau;
    const double two_kappa = detail::TWO * kappa;
    const double rB = std::sqrt(two_kappa) * s;
    // The integrand is even, so its integral over [0, 1] is half the rule over [-1, 1].
    return detail::TWO * rB *
           gauss_legendre(central_nodes(s * s), detail::HALF, [&](double x, double) {
               const double v = rB * x;
               return std::exp(-v * v) / std::sqrt(two_kappa - v * v);
           });
}

// a + b + c + d with one rounding to within an ulp of a few: Neumaier summation.
[[nodiscard]] double sum_of_four(double a, double b, double c, double d) noexcept {
    double sum = a;
    double comp = detail::ZERO_DOUBLE;
    for (double v : {b, c, d}) {
        const double s = sum + v;
        comp += std::fabs(sum) >= std::fabs(v) ? (sum - s) + v : (v - s) + sum;
        sum = s;
    }
    return sum + comp;
}

// One Halley step for h(v) = 0 from the value h and its first three
// derivatives h1, h2, h3 at the current point; `predicted` is Halley's
// asymptotic error after the step, (c2^2 - c3) e^3 with c2 = h2/(2 h1) and
// c3 = h3/(6 h1), bounded here by (c2^2 + |c3|) |step|^3 so that the two terms
// cannot cancel. A Newton step is taken where the Halley denominator is far
// from 1 (the cubic model is then not trustworthy); its error is c2 step^2.
struct HalleyStep {
    double step;
    double predicted;  ///< |error| after the step, to leading order
};

[[nodiscard]] HalleyStep halley_step(double h, double h1, double h2, double h3) noexcept {
    const double newton = -h / h1;
    const double c2 = h2 / (detail::TWO * h1);
    const double c3 = h3 / (6.0 * h1);
    const double den = detail::ONE + newton * c2;
    if (den > detail::HALF && den < detail::TWO) {
        const double step = newton / den;
        const double a = std::fabs(step);
        return {step, (c2 * c2 + std::fabs(c3)) * a * a * a};
    }
    return {newton, std::fabs(c2) * newton * newton};
}

// t in [-PI, 0] with G(t) = m, for 0 < m < 1/2 (mu = 0). Halley on ln G, which
// is well scaled from 1e-300 to 1/2: in t away from the edge, in s = ln d near
// it, where G is nearly linear in d and a t-step would overshoot; bracketed,
// with bisection (geometric in d when the bracket spans decades) whenever a
// step leaves the bracket. Returns -PI when the answer is within an ulp of the
// edge.
//
// With L = ln G = ln g + ln J and g'/g = -kappa sin t, J = G/g satisfies
// J' = 1 + kappa sin(t) J, so one quadrature gives every derivative:
//   L' = 1/J,  L'' = -J'/J^2,  L''' = 2 J'^2/J^3 - J''/J^2,
//   J'' = kappa (cos(t) J + sin(t) J').
// In s (dt/ds = d2t/ds2 = d3t/ds3 = d): L_s = d L', L_ss = d^2 L'' + d L',
// L_sss = d^3 L''' + 3 d^2 L'' + d L'.
[[nodiscard]] double vonmises_left_quantile(double m, double kappa, double log_scaled_norm,
                                            double half_j) noexcept {
    const double log_m = std::log(m);
    // g is increasing on [-pi, 0], so m / g(0) <= d <= m / g(-pi).
    const double log_d_lo = log_m + log_scaled_norm;
    const double log_d_hi = log_m + detail::TWO * kappa + log_scaled_norm;
    if (log_d_hi < std::log(kPiLo + 2.3e-16))
        return -detail::PI;
    auto t_of = [](double d) { return std::max(-detail::PI, (d - kPiLo) - detail::PI); };
    double t_lo = std::max(-detail::PI, t_of(std::exp(log_d_lo)) - 1e-15);
    double t_hi =
        log_d_hi < std::log(detail::PI) ? t_of(std::exp(log_d_hi)) + 1e-15 : detail::ZERO_DOUBLE;
    t_hi = std::min(t_hi, detail::ZERO_DOUBLE);

    // Seed: the wrapped normal at moderate kappa, the uniform below it.
    double t = kappa >= detail::ONE ? detail::inverse_normal_cdf(m) / std::sqrt(kappa)
                                    : detail::TWO_PI * m - detail::PI;
    if (!(t > t_lo && t < t_hi))
        t = t_of(std::exp(detail::HALF * (log_d_lo + std::min(log_d_hi, std::log(detail::PI)))));
    if (!(t > t_lo && t < t_hi))
        t = detail::HALF * (t_lo + t_hi);

    constexpr double kEps = std::numeric_limits<double>::epsilon();
    double r_prev = std::numeric_limits<double>::infinity();
    for (int iter = 0; iter < 100; ++iter) {
        // Once the last residual is below 0.05, Halley leaves this one below ~1e-6 and the
        // step from it ends the solve: that evaluation takes the quadratures, for accuracy.
        const LeftTail tail = vonmises_left_tail(
            t, kappa, log_scaled_norm, half_j, -std::numeric_limits<double>::infinity(),
            std::fabs(r_prev) < 0.05 ? TailMethods::kQuadrature : TailMethods::kAll);
        // ln G - ln m from its four parts with one rounding: at the root the parts cancel.
        const double r = sum_of_four(-tail.b, -log_scaled_norm, tail.log_j - tail.b_lo, -log_m);
        r_prev = r;
        if (r == detail::ZERO_DOUBLE)
            return t;
        if (r < detail::ZERO_DOUBLE)
            t_lo = t;
        else
            t_hi = t;

        const double d = (t + detail::PI) + kPiLo;
        const bool near_edge = t < -detail::HALF * detail::PI;
        const double sin_t = near_edge ? -std::sin(d) : std::sin(t);
        const double cos_t = near_edge ? -std::cos(d) : std::cos(t);
        const double j = tail.j;
        const double j1 = detail::ONE + kappa * sin_t * j;
        const double j2 = kappa * (cos_t * j + sin_t * j1);
        double l1 = detail::ONE / j;
        double l2 = -j1 / (j * j);
        double l3 = (detail::TWO * j1 * j1 / j - j2) / (j * j);
        const bool log_step = d < detail::HALF;
        if (log_step) {
            l3 = d * (d * (d * l3 + 3.0 * l2) + l1);
            l2 = d * (d * l2 + l1);
            l1 = d * l1;
        }
        const HalleyStep hs = halley_step(r, l1, l2, l3);
        const double d_next = log_step ? d * std::exp(hs.step) : d + hs.step;
        double next = log_step ? t_of(d_next) : t + hs.step;
        // The predicted error, in t; trusted only once G is within 0.1% of m,
        // where the step is well inside the radius of the cubic model.
        double predicted = std::fabs(r) > 1e-3 ? std::numeric_limits<double>::infinity()
                           : log_step          ? hs.predicted * d_next
                                               : hs.predicted;
        // Converged once the step, or the error it leaves, is below the noise
        // that ln G's own error, ~|ln m| ulp absolute, leaves in t -- or below
        // half an ulp of t, which near the edge is the larger. A step that
        // small can round onto t itself, outside the open bracket.
        const double tolerance = std::max(detail::TWO * kEps * (detail::ONE - log_m) * j,
                                          detail::HALF * kEps * std::fabs(t));
        if (std::fabs(next - t) <= tolerance)
            return next;
        if (!(next > t_lo && next < t_hi)) {
            const double d_lo = (t_lo + detail::PI) + kPiLo;
            const double d_hi = (t_hi + detail::PI) + kPiLo;
            next = d_hi > 4.0 * d_lo ? t_of(std::sqrt(d_lo * d_hi)) : detail::HALF * (t_lo + t_hi);
            if (!(next > t_lo && next < t_hi))
                return t;  // the bracket is down to adjacent doubles
            predicted = std::numeric_limits<double>::infinity();
        }
        // The tolerance bounds the evaluation's own error; the error a predicted
        // step leaves adds to it, so it must sit well below.
        if (predicted <= 0.0625 * tolerance)
            return next;
        t = next;
    }
    return t;
}

// tau in (0, PI/2] with X(tau) = C(tau) / (2 pi I0e) = y, for 0 < y <= 1/4 (mu = 0; the
// quantile is -tau for p = 1/2 - y). Halley in tau with X' = f, X'' = -kappa sin(tau) f,
// X''' = kappa (kappa sin^2(tau) - cos(tau)) f; bracketed by [0, PI/2], where X(PI/2) >= 1/4
// for every kappa, with bisection whenever a step leaves the bracket. Converged once the step,
// or the error it leaves, is below X's own error (~ulp of y) mapped to tau, or half an ulp of
// tau.
[[nodiscard]] double vonmises_central_quantile(double y, double kappa, double log_scaled_norm,
                                               double inv_norm) noexcept {
    double t_lo = detail::ZERO_DOUBLE;
    double t_hi = detail::HALF * detail::PI;
    double tau = kappa >= detail::ONE
                     ? -detail::inverse_normal_cdf(detail::HALF - y) / std::sqrt(kappa)
                     : detail::TWO_PI * y;
    if (!(tau > t_lo && tau < t_hi))
        tau = detail::HALF * (t_lo + t_hi);

    constexpr double kEps = std::numeric_limits<double>::epsilon();
    for (int iter = 0; iter < 100; ++iter) {
        const double s = std::sin(detail::HALF * tau);
        const double r = vonmises_central_mass(tau, s, kappa) * inv_norm - y;
        if (r == detail::ZERO_DOUBLE)
            return tau;
        if (r < detail::ZERO_DOUBLE)
            t_lo = tau;
        else
            t_hi = tau;
        // sin(tau) and cos(tau) from the half angle: they only shape the step.
        const double c = std::cos(detail::HALF * tau);
        const double sin_t = detail::TWO * s * c;
        const double cos_t = (c - s) * (c + s);
        const double f = std::exp(-detail::TWO * kappa * s * s - log_scaled_norm);
        const HalleyStep hs =
            halley_step(r, f, -kappa * sin_t * f, kappa * (kappa * sin_t * sin_t - cos_t) * f);
        double next = tau + hs.step;
        double predicted =
            std::fabs(r) > 1e-3 * y ? std::numeric_limits<double>::infinity() : hs.predicted;
        const double tolerance = std::max(kEps * y / f, detail::HALF * kEps * tau);
        if (std::fabs(next - tau) <= tolerance)
            return next;
        if (!(next > t_lo && next < t_hi)) {
            next = detail::HALF * (t_lo + t_hi);
            if (!(next > t_lo && next < t_hi))
                return tau;
            predicted = std::numeric_limits<double>::infinity();
        }
        if (predicted <= tolerance)
            return next;
        tau = next;
    }
    return tau;
}

// F(t) for mu = 0, t in [-PI, PI] and 0 < kappa: 1/2 -+ C / (2 pi I0e) where that is within
// kCentralHalfWidth of the median, the tail mass G beyond. Floors: e^-746 is below half the
// smallest subnormal, and 1 - e^-40 rounds to 1.
[[nodiscard]] double vonmises_cdf(double t, double kappa, double log_scaled_norm, double inv_norm,
                                  double half_j) noexcept {
    const double tau = std::fabs(t);
    if (tau <= detail::HALF * detail::PI) {
        const double s = std::sin(detail::HALF * tau);
        if (detail::TWO * kappa * s * s <= kCentralMaxB) {
            const double x = vonmises_central_mass(tau, s, kappa) * inv_norm;
            if (x <= kCentralHalfWidth)
                return t < detail::ZERO_DOUBLE ? detail::HALF - x : detail::HALF + x;
        }
    }
    const bool lower = t <= detail::ZERO_DOUBLE;
    const LeftTail tail = vonmises_left_tail(lower ? t : -t, kappa, log_scaled_norm, half_j,
                                             lower ? -746.0 : -40.0, TailMethods::kTaylor);
    // G = J e^-B / (2 pi I0e) while e^-B is normal; its log otherwise, whose rounding (half an
    // ulp of ln G) the product does not carry.
    const double G = tail.j > detail::ZERO_DOUBLE && tail.b < 700.0
                         ? tail.j * (std::exp(-tail.b) * (detail::ONE - tail.b_lo)) * inv_norm
                         : std::exp(tail.log_mass);
    return lower ? G : detail::ONE - G;
}

}  // anonymous namespace

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

VonMisesDistribution::VonMisesDistribution(double mu, double kappa)
    : DistributionBase(), mu_(wrapAngle(mu)), kappa_(kappa) {
    validateParameters(mu_, kappa_);
    updateCacheUnsafe();
}

VonMisesDistribution::VonMisesDistribution(const VonMisesDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    logScaledNormaliser_ = other.logScaledNormaliser_;
    invScaledNormaliser_ = other.invScaledNormaliser_;
    tailHalfJ_ = other.tailHalfJ_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(const VonMisesDistribution& other) {
    if (this != &other) {
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);
        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        logScaledNormaliser_ = other.logScaledNormaliser_;
        invScaledNormaliser_ = other.invScaledNormaliser_;
        tailHalfJ_ = other.tailHalfJ_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

VonMisesDistribution::VonMisesDistribution(VonMisesDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    mu_ = other.mu_;
    kappa_ = other.kappa_;
    logNormaliser_ = other.logNormaliser_;
    logScaledNormaliser_ = other.logScaledNormaliser_;
    invScaledNormaliser_ = other.invScaledNormaliser_;
    tailHalfJ_ = other.tailHalfJ_;
    circularVariance_ = other.circularVariance_;
    isUniform_ = other.isUniform_;
    other.mu_ = detail::ZERO_DOUBLE;
    other.kappa_ = detail::ONE;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
}

VonMisesDistribution& VonMisesDistribution::operator=(VonMisesDistribution&& other) noexcept {
    if (this != &other) {
        mu_ = other.mu_;
        kappa_ = other.kappa_;
        logNormaliser_ = other.logNormaliser_;
        logScaledNormaliser_ = other.logScaledNormaliser_;
        invScaledNormaliser_ = other.invScaledNormaliser_;
        tailHalfJ_ = other.tailHalfJ_;
        circularVariance_ = other.circularVariance_;
        isUniform_ = other.isUniform_;
        other.mu_ = detail::ZERO_DOUBLE;
        other.kappa_ = detail::ONE;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
        atomicMu_.store(mu_, std::memory_order_release);
        atomicKappa_.store(kappa_, std::memory_order_release);
    }
    return *this;
}

//==============================================================================
// 2. PRIVATE FACTORY METHODS
//==============================================================================

VonMisesDistribution VonMisesDistribution::createUnchecked(double mu, double kappa) noexcept {
    return VonMisesDistribution(wrapAngle(mu), kappa, true);
}

VonMisesDistribution::VonMisesDistribution(double mu, double kappa,
                                           bool /*bypassValidation*/) noexcept
    : DistributionBase(), mu_(mu), kappa_(kappa) {
    updateCacheUnsafe();
}

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

void VonMisesDistribution::setMu(double mu) {
    validateParameters(mu, getKappa());
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    // Changing only μ doesn't affect logNormaliser_ or circularVariance_
    // (those depend only on κ), but cache_valid_ is reset for thread safety.
    updateCacheUnsafe();
}

void VonMisesDistribution::setKappa(double kappa) {
    validateParameters(getMu(), kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

void VonMisesDistribution::setParameters(double mu, double kappa) {
    validateParameters(mu, kappa);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
}

double VonMisesDistribution::getMean() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getVariance() const {
    double cv;
    withCacheSnapshot([&] { cv = circularVariance_; });
    return cv;
}

//==============================================================================
// 4. RESULT-BASED SETTERS
//==============================================================================

VoidResult VonMisesDistribution::trySetMu(double mu) noexcept {
    auto v = validateVonMisesParameters(mu, getKappa());
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::trySetKappa(double kappa) noexcept {
    auto v = validateVonMisesParameters(getMu(), kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::trySetParameters(double mu, double kappa) noexcept {
    auto v = validateVonMisesParameters(mu, kappa);
    if (v.isError())
        return v;
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = wrapAngle(mu);
    kappa_ = kappa;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);
    updateCacheUnsafe();
    return VoidResult::ok({});
}

VoidResult VonMisesDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateVonMisesParameters(mu_, kappa_);
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double VonMisesDistribution::getProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::ZERO_DOUBLE;  // ±inf → PDF is 0

    double k, mu, lnorm;
    withCacheSnapshot([&] {
        k = kappa_;
        mu = mu_;
        lnorm = logNormaliser_;
    });
    return std::exp(k * std::cos(x - mu) - lnorm);
}

double VonMisesDistribution::getLogProbability(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(x))
        return detail::NEGATIVE_INFINITY;  // ±inf → log PDF is -∞

    double k, mu, lnorm;
    withCacheSnapshot([&] {
        k = kappa_;
        mu = mu_;
        lnorm = logNormaliser_;
    });
    return k * std::cos(x - mu) - lnorm;
}

double VonMisesDistribution::getCumulativeProbability(double x) const {
    if (!std::isfinite(x)) {
        if (std::isnan(x))
            return std::numeric_limits<double>::quiet_NaN();
        return (x > 0 ? detail::ONE : detail::ZERO_DOUBLE);
    }

    double result = detail::ZERO_DOUBLE;
    withCacheSnapshot([&] {
        const double kappa = kappa_;
        const double mu = mu_;

        // kappa = 0 (uniform circular distribution): exact linear CDF.
        if (isUniform_) {
            const double t = wrapAngle(x - mu);
            result = std::clamp(((t + detail::PI) + kPiLo) / detail::TWO_PI, detail::ZERO_DOUBLE,
                                detail::ONE);
            return;
        }

        // kappa > 1000: unvalidated range -- use the
        // pre-#51 wrapped-normal approximation. VM(mu, kappa) ~ N(mu, 1/kappa)
        // on the circle; approximation error is ~0.043/kappa absolute (measured
        // against a quadrature oracle at kappa = 1e3, 2e3, 1e4 -- O(1/kappa),
        // not O(1/kappa^2)). The standardised argument is the WRAPPED
        // DIFFERENCE, matching the branch below: wrapping x alone and
        // subtracting mu afterwards leaves a 2*pi offset for every x on the far
        // side of the +-pi cut from mu (#106).
        if (kappa > kCdfExactKappaMax) {
            const double z = wrapAngle(x - mu) * std::sqrt(kappa);
            result = std::clamp(detail::HALF * (detail::ONE + std::erf(z * detail::INV_SQRT_2)),
                                detail::ZERO_DOUBLE, detail::ONE);
            return;
        }

        // 0 < kappa <= 1000: the central mass near the median, the tail
        // mass beyond (vonmises_cdf), at t = wrap(x - mu).
        result = vonmises_cdf(wrapAngle(x - mu), kappa, logScaledNormaliser_, invScaledNormaliser_,
                              tailHalfJ_);
    });

    return result;
}

double VonMisesDistribution::getQuantile(double p) const {
    if (std::isnan(p))
        return std::numeric_limits<double>::quiet_NaN();
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be in [0, 1]");
    }

    // Solve on the small side of the probability scale: the CDF below the
    // median, the survival above it (1 - p is exact there). Within
    // kCentralHalfWidth of the median the quantile solves on the central mass,
    // for 1/2 - m (exact); below that, on the tail mass with Halley on its
    // logarithm (vonmises_left_quantile). By symmetry the upper quantile is the
    // mirror of the lower one.
    const bool upper = p > detail::HALF;
    const double m = upper ? detail::ONE - p : p;
    double mu, kappa, log_scaled_norm, inv_norm, half_j;
    withCacheSnapshot([&] {
        mu = mu_;
        kappa = kappa_;
        log_scaled_norm = logScaledNormaliser_;
        inv_norm = invScaledNormaliser_;
        half_j = tailHalfJ_;
    });

    double t = -detail::PI;
    if (m >= detail::HALF)
        t = detail::ZERO_DOUBLE;
    else if (m >= detail::HALF - kCentralHalfWidth)
        t = -vonmises_central_quantile(detail::HALF - m, kappa, log_scaled_norm, inv_norm);
    else if (m > detail::ZERO_DOUBLE)
        t = vonmises_left_quantile(m, kappa, log_scaled_norm, half_j);

    if (upper) {
        t = -t;
    } else if (t <= -detail::PI) {
        // The left end stays at the left end. The support is (-pi, pi] and
        // wrapAngle reads -PI as +PI, so the smallest double above -PI stands
        // for a quantile within an ulp of -pi (p = 1e-300 used to return +pi).
        t = std::nextafter(-detail::PI, detail::ZERO_DOUBLE);
    }
    return wrapAngle(t + mu);
}

double VonMisesDistribution::sample(std::mt19937& rng) const {
    double kappa, mu;
    withCacheSnapshot([&] {
        kappa = kappa_;
        mu = mu_;
    });

    // Near-uniform case (κ ≈ 0): sample uniformly on the circle.
    if (kappa < 1e-9) {
        std::uniform_real_distribution<double> u(-detail::PI, detail::PI);
        return u(rng);
    }

    // Best (1979) rejection sampler for the Von Mises distribution.
    // Reference: D.J. Best and N.I. Fisher (1979). Efficient simulation of
    //            the von Mises distribution. Applied Statistics 28(2), 152–157.
    const double tau = detail::ONE + std::sqrt(detail::ONE + 4.0 * kappa * kappa);
    const double rho = (tau - std::sqrt(detail::TWO * tau)) / (detail::TWO * kappa);
    const double r = (detail::ONE + rho * rho) / (detail::TWO * rho);

    std::uniform_real_distribution<double> u01(detail::ZERO_DOUBLE, detail::ONE);

    for (;;) {
        const double u1 = u01(rng);
        const double z = std::cos(detail::PI * u1);
        const double f = (detail::ONE + r * z) / (r + z);
        const double c = kappa * (r - f);
        const double u2 = u01(rng);

        bool accept = false;
        if (c * (detail::TWO - c) > u2) {
            accept = true;
        } else if (c > detail::ZERO_DOUBLE) {
            accept = (std::log(c / u2) + detail::ONE - c >= detail::ZERO_DOUBLE);
        }

        if (accept) {
            const double u3 = u01(rng);
            const double angle = (u3 > detail::HALF) ? std::acos(f) : -std::acos(f);
            return wrapAngle(mu + angle);
        }
    }
}

std::vector<double> VonMisesDistribution::sample(std::mt19937& rng, size_t n) const {
    double kappa, mu;
    withCacheSnapshot([&] {
        kappa = kappa_;
        mu = mu_;
    });

    std::vector<double> samples;
    samples.reserve(n);

    if (kappa < 1e-9) {
        std::uniform_real_distribution<double> u(-detail::PI, detail::PI);
        for (size_t i = 0; i < n; ++i)
            samples.push_back(u(rng));
        return samples;
    }

    // Best (1979) rejection sampler — precompute constants outside the loop.
    const double tau = detail::ONE + std::sqrt(detail::ONE + 4.0 * kappa * kappa);
    const double rho = (tau - std::sqrt(detail::TWO * tau)) / (detail::TWO * kappa);
    const double r = (detail::ONE + rho * rho) / (detail::TWO * rho);
    std::uniform_real_distribution<double> u01(detail::ZERO_DOUBLE, detail::ONE);

    for (size_t i = 0; i < n; ++i) {
        for (;;) {
            const double u1 = u01(rng);
            const double z = std::cos(detail::PI * u1);
            const double f = (detail::ONE + r * z) / (r + z);
            const double c = kappa * (r - f);
            const double u2 = u01(rng);
            bool accept = (c * (detail::TWO - c) > u2) ||
                          (c > detail::ZERO_DOUBLE &&
                           std::log(c / u2) + detail::ONE - c >= detail::ZERO_DOUBLE);
            if (accept) {
                const double u3 = u01(rng);
                const double angle = (u3 > detail::HALF) ? std::acos(f) : -std::acos(f);
                samples.push_back(wrapAngle(mu + angle));
                break;
            }
        }
    }
    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void VonMisesDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit distribution to empty data");
    }

    double S = detail::ZERO_DOUBLE, C = detail::ZERO_DOUBLE;
    for (double x : values) {
        // FIT-4: NaN or Inf corrupts the sin/cos accumulation silently.
        if (!std::isfinite(x))
            throw std::invalid_argument("All values must be finite for VonMises fit");
        S += std::sin(x);
        C += std::cos(x);
    }
    const double n = static_cast<double>(values.size());
    const double mu_hat = wrapAngle(std::atan2(S / n, C / n));
    const double R_bar = std::sqrt(S * S + C * C) / n;
    const double kappa_hat = kappa_from_r_bar(R_bar);

    setParameters(mu_hat, kappa_hat);
}

void VonMisesDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<VonMisesDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void VonMisesDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    mu_ = detail::ZERO_DOUBLE;
    kappa_ = detail::ONE;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    atomicParamsValid_.store(false, std::memory_order_release);  // NEW-TS-4
    updateCacheUnsafe();
}

std::string VonMisesDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "VonMisesDistribution(mu=" << mu_ << ",kappa=" << kappa_ << ")";
    return oss.str();
}

//==============================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==============================================================================

double VonMisesDistribution::getMuAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicMu_.load(std::memory_order_acquire);
    return getMu();
}

double VonMisesDistribution::getKappaAtomic() const noexcept {
    if (atomicParamsValid_.load(std::memory_order_acquire))
        return atomicKappa_.load(std::memory_order_acquire);
    return getKappa();
}

double VonMisesDistribution::getCircularVariance() const {
    return getVariance();
}

double VonMisesDistribution::getMode() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return mu_;
}

double VonMisesDistribution::getEntropy() const {
    bool is_uniform;
    double k;
    withCacheSnapshot([&] {
        is_uniform = isUniform_;
        k = kappa_;
    });
    // H = log(2π) − log I₀(κ) + κ·I₁(κ)/I₀(κ)
    // At κ=0: H = log(2π) − log(1) + 0 = log(2π) ✓ (uniform on the circle)
    if (is_uniform)
        return detail::LN_2PI;
    const double log_i0 = detail::log_bessel_i0(k);
    // A(κ) via the ratio helper: the direct i1/i0 form returned NaN above
    // κ ≈ 713 where both values overflow (#93).
    const double A1 = detail::bessel_i1_over_i0(k);
    return detail::LN_2PI - log_i0 + k * A1;
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==============================================================================

void VonMisesDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const VonMisesDistribution& d, double x) { return d.getProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            d.getProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        return;
                    }
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        continue;
                    }
                    res[i] = std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm)
                                              : detail::ZERO_DOUBLE;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                    res[i] = x;
                    return;
                }
                res[i] =
                    std::isfinite(x) ? std::exp(k * std::cos(x - mu) - lnorm) : detail::ZERO_DOUBLE;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const VonMisesDistribution& d, double x) { return d.getLogProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            d.getLogProbabilityBatchUnsafeImpl(vals, res, count, k, mu, lnorm);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        return;
                    }
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                });
            } else {
                for (std::size_t i = 0; i < count; ++i) {
                    const double x = vals[i];
                    if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                        res[i] = x;
                        continue;
                    }
                    res[i] =
                        std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
                }
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            double k, mu, lnorm;
            d.withCacheSnapshot([&] {
                k = d.kappa_;
                mu = d.mu_;
                lnorm = d.logNormaliser_;
            });
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                const double x = vals[i];
                if (std::isnan(x)) {  // NaN propagates, as on the scalar path
                    res[i] = x;
                    return;
                }
                res[i] =
                    std::isfinite(x) ? k * std::cos(x - mu) - lnorm : detail::NEGATIVE_INFINITY;
            });
            pool.waitForAll();
        });
}

void VonMisesDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const VonMisesDistribution& d, double x) { return d.getCumulativeProbability(x); },
        [](const VonMisesDistribution& d, const double* vals, double* res, size_t count) {
            double mu, kappa, log_scaled_norm, inv_norm, half_j;
            bool uniform;
            d.withCacheSnapshot([&] {
                mu = d.mu_;
                kappa = d.kappa_;
                log_scaled_norm = d.logScaledNormaliser_;
                inv_norm = d.invScaledNormaliser_;
                half_j = d.tailHalfJ_;
                uniform = d.isUniform_;
            });
            d.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, mu, kappa, log_scaled_norm,
                                                      inv_norm, half_j, uniform);
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size())
                throw std::invalid_argument("Input and output spans must have the same size");
            const std::size_t count = vals.size();
            if (count == 0)
                return;
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    res[i] = d.getCumulativeProbability(vals[i]);
                });
            } else {
                for (std::size_t i = 0; i < count; ++i)
                    res[i] = d.getCumulativeProbability(vals[i]);
            }
        },
        [](const VonMisesDistribution& d, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            const std::size_t count = vals.size();
            pool.parallelFor(std::size_t{0}, count,
                             [&](std::size_t i) { res[i] = d.getCumulativeProbability(vals[i]); });
            pool.waitForAll();
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool VonMisesDistribution::operator==(const VonMisesDistribution& other) const {
    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);
    return std::fabs(mu_ - other.mu_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE &&
           std::fabs(kappa_ - other.kappa_) < detail::ULTRA_HIGH_PRECISION_TOLERANCE;
}

bool VonMisesDistribution::operator!=(const VonMisesDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const VonMisesDistribution& d) {
    return os << d.toString();
}

std::istream& operator>>(std::istream& is, VonMisesDistribution& d) {
    std::string token;
    is >> token;
    if (!token.starts_with("VonMisesDistribution(")) {
        is.setstate(std::ios::failbit);
        return is;
    }
    const size_t mu_pos = token.find("mu=");
    const size_t comma = token.find(",", mu_pos);
    const size_t kappa_pos = token.find("kappa=");
    const size_t close = token.find(")", kappa_pos);
    if (mu_pos == std::string::npos || comma == std::string::npos ||
        kappa_pos == std::string::npos || close == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    try {
        const double mu = std::stod(token.substr(mu_pos + 3, comma - mu_pos - 3));
        const double kappa = std::stod(token.substr(kappa_pos + 6, close - kappa_pos - 6));
        auto result = d.trySetParameters(mu, kappa);
        if (result.isError())
            is.setstate(std::ios::failbit);
    } catch (...) {
        is.setstate(std::ios::failbit);
    }
    return is;
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//
// LogPDF batch:  z[i] = x[i] − μ  |  c[i] = vector_cos(z)  |  r[i] = κ·c[i] − ln Z
// PDF batch:     same as LogPDF then r[i] = vector_exp(r)
// CDF batch:     per element, the scalar CDF's central or tail mass on the
//                snapshot (κ = 0 or κ > 1000: the scalar CDF itself)
//
// PDF/LogPDF use VectorOps::vector_cos — AVX/AVX2/SSE2/NEON/AVX-512. Non-finite
// inputs receive an exact sentinel value via a scalar fixup pass after the
// SIMD kernel.
//
// The primary performance gain over per-element calls remains avoiding the
// cache-validity check and lock acquisition on every element.
//==============================================================================

void VonMisesDistribution::getLogProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_kappa, double cached_mu,
    double cached_log_normaliser) const noexcept {
    // Step 1: z[i] = values[i] - mu  (scalar_add with -mu)
    arch::simd::VectorOps::scalar_add(values, -cached_mu, results, count);

    // Step 2: results[i] = cos(z[i])  (vectorised)
    arch::simd::VectorOps::vector_cos(results, results, count);

    // Step 3: results[i] = kappa * results[i] - log_normaliser
    arch::simd::VectorOps::scalar_multiply(results, cached_kappa, results, count);
    arch::simd::VectorOps::scalar_add(results, -cached_log_normaliser, results, count);

    // Fixup: NaN propagates; ±inf must produce -∞ regardless of the SIMD result
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = values[i];
        else if (!std::isfinite(values[i]))
            results[i] = detail::NEGATIVE_INFINITY;
    }
}

void VonMisesDistribution::getProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_kappa, double cached_mu,
    double cached_log_normaliser) const noexcept {
    // Compute log-PDF then exponentiate
    getLogProbabilityBatchUnsafeImpl(values, results, count, cached_kappa, cached_mu,
                                     cached_log_normaliser);

    // Step 4: results[i] = exp(results[i])
    arch::simd::VectorOps::vector_exp(results, results, count);

    // Fixup: NaN propagates; ±inf must produce 0
    for (std::size_t i = 0; i < count; ++i) {
        if (std::isnan(values[i]))
            results[i] = values[i];
        else if (!std::isfinite(values[i]))
            results[i] = detail::ZERO_DOUBLE;
    }
}

void VonMisesDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, double cached_mu, double cached_kappa,
    double cached_log_scaled_norm, double cached_inv_norm, double cached_half_j,
    bool cached_uniform) const noexcept {
    // κ = 0 or κ > 1000: the scalar CDF's closed form or wrapped-normal approximation.
    if (cached_uniform || cached_kappa > kCdfExactKappaMax) {
        for (std::size_t i = 0; i < count; ++i)
            results[i] = getCumulativeProbability(values[i]);
        return;
    }
    // Each element costs one central-mass rule (6-12 exp) or one tail evaluation, so the
    // element loop dominates: the snapshot only saves the per-element lock. The scalar
    // contract for non-finite input: NaN -> NaN, +inf -> 1, -inf -> 0.
    for (std::size_t i = 0; i < count; ++i) {
        const double x = values[i];
        if (std::isnan(x))
            results[i] = std::numeric_limits<double>::quiet_NaN();
        else if (std::isinf(x))
            results[i] = x > 0.0 ? detail::ONE : detail::ZERO_DOUBLE;
        else
            results[i] = vonmises_cdf(wrapAngle(x - cached_mu), cached_kappa,
                                      cached_log_scaled_norm, cached_inv_norm, cached_half_j);
    }
}

//==============================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==============================================================================

void VonMisesDistribution::updateCacheUnsafe() const noexcept {
    // logNormaliser = log(2π) + log I₀(κ)
    // When κ = 0: I₀(0) = 1, log I₀ = 0, logNormaliser = log(2π). ✓
    logNormaliser_ = detail::LN_2PI + detail::log_bessel_i0(kappa_);
    // The same constant less kappa, formed without the cancellation, for the
    // central and tail masses behind the CDF and the quantile.
    // Formed as the log of its reciprocal 1/(2π·I₀(κ)·e^−κ), which scales the
    // central and tail masses directly.
    invScaledNormaliser_ = inv_scaled_normaliser(kappa_);
    logScaledNormaliser_ = kappa_ > kCdfExactKappaMax ? log_scaled_normaliser_hankel(kappa_)
                                                      : -std::log(invScaledNormaliser_);
    // J(-pi/2), the per-kappa constant of the tail's mode quadrature.
    tailHalfJ_ = vonmises_half_tail_j(kappa_);

    isUniform_ = (kappa_ < 1e-10);

    // Circular variance = 1 − I₁(κ)/I₀(κ), via the dedicated complement helper
    // (#93). Forming it as `ONE - i1 / i0` discarded ~log₂(2κ) bits — about 9
    // at κ = 200 — and returned NaN above κ ≈ 713, where I₀ and I₁ both
    // overflow and the `i0 > 0.0` guard does not catch inf.
    if (isUniform_) {
        circularVariance_ = detail::ONE;
    } else {
        circularVariance_ = detail::bessel_i1_i0_complement(kappa_);
    }

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    atomicMu_.store(mu_, std::memory_order_release);
    atomicKappa_.store(kappa_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

//==============================================================================
// 20–24. PLACEHOLDERS (maintained for template compliance)
//==============================================================================

}  // namespace stats
