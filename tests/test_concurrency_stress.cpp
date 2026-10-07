// tests/test_concurrency_stress.cpp
//
// R9 concurrency stress test: one table row per distribution (all 27), each raced by a writer
// against every public reader, plus two-writer rounds and a strategy differential. The classes it
// targets, and the defects that defined them, are in PLAN.md, "R9".
//
// The oracle. A writer runs a program: a cycle of steps (setParameters, single setters, fit,
// stream operator>>, copy- and move-assignment) that returns to the starting state. The same
// program applied to a private instance gives the allowed states; every reader evaluated on that
// private instance at each step gives the allowed results. A raced reader must return one of
// them bit for bit: a "whole" reader (a batch, a copy) all from one state, an "elementwise" reader
// (a run of separate scalar calls) each element from some state.
//
// Driving. Each program is raced two ways: free-running loops, and aligned rounds that release
// the writer and one reader at the same instant (racedRounds, from test_concurrency_gates.cpp).
// Loops missed every one-shot race the aligned rounds caught there. Every race runs under a
// watchdog, so a deadlock fails the test instead of hanging it.
//
// LIBSTATS_STRESS_SCALE (default 1) multiplies rounds and loop time: lower it under
// ThreadSanitizer, raise it for a long machine run.
//
// This file is self-contained (it must also build on commits before test_concurrency_gates.cpp
// existed), so runWithWatchdog and racedRounds are copied from there.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

using namespace stats;

namespace {

// ---------------------------------------------------------------------------------------------
// Harness (runWithWatchdog and racedRounds copied from test_concurrency_gates.cpp)

// The reader each racing thread is inside, for the watchdog's report (a deadlock names the call).
std::atomic<const char*> gInFlight[3] = {nullptr, nullptr, nullptr};

// Runs body on its own thread; a body still running after `seconds` is reported and the process
// exits, since a deadlocked thread can be neither joined nor detached safely.
template <typename Body>
void runWithWatchdog(const std::string& what, int seconds, Body body) {
    std::atomic<bool> done{false};
    std::thread t([&] {
        body();
        done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!done.load()) {
        std::fprintf(stderr, "[  FAILED  ] %s: no progress in %d s (deadlock)\n", what.c_str(),
                     seconds);
        for (const auto& f : gInFlight)
            if (const char* r = f.load())
                std::fprintf(stderr, "             a reader was inside: %s\n", r);
        std::fflush(stderr);
        std::_Exit(1);
    }
    t.join();
}

// Rounds of two calls started together: setup() between rounds, then first() and second() on two
// threads released at the same instant, then check(). Returns the rounds that check() rejected.
template <typename Setup, typename First, typename Second, typename Check>
int racedRounds(int rounds, Setup setup, First first, Second second, Check check) {
    std::atomic<int> go{0}, done{0};
    std::atomic<bool> quit{false};
    const auto worker = [&](auto& call) {
        for (int r = 1;; ++r) {
            while (go.load(std::memory_order_acquire) < r)
                if (quit.load(std::memory_order_acquire))
                    return;
            call();
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    };
    std::thread t1([&] { worker(first); });
    std::thread t2([&] { worker(second); });
    int rejected = 0;
    for (int r = 1; r <= rounds; ++r) {
        setup();
        done.store(0, std::memory_order_release);
        go.store(r, std::memory_order_release);
        while (done.load(std::memory_order_acquire) < 2) {
        }
        rejected += !check();
    }
    quit.store(true, std::memory_order_release);
    t1.join();
    t2.join();
    return rejected;
}

double stressScale() {
    static const double scale = [] {
        const char* s = std::getenv("LIBSTATS_STRESS_SCALE");
        const double v = s ? std::atof(s) : 1.0;
        return v > 0.0 ? v : 1.0;
    }();
    return scale;
}

int scaled(int n) {
    return std::max(1, static_cast<int>(n * stressScale()));
}

constexpr int kWatchdogSeconds = 120;

// ---------------------------------------------------------------------------------------------
// Values as bits. An exception becomes a fixed sentinel, so "throws in state S" is an allowed
// result exactly when the private instance threw in S too.

using Bits = std::vector<std::uint64_t>;

template <typename D>
double invalidFlag(const D& d) {
    return d.validateCurrentParameters().isError() ? 1.0 : 0.0;
}

constexpr std::uint64_t kThrew = 0x7ff8dead0000beefULL;

std::uint64_t bitsOf(double v) {
    return std::bit_cast<std::uint64_t>(v);
}

template <typename F>
std::uint64_t guarded(F f) {
    try {
        return bitsOf(static_cast<double>(f()));
    } catch (...) {
        return kThrew;
    }
}

template <typename D>
using Step = std::function<void(D&)>;

template <typename D>
struct Reader {
    std::string name;
    std::function<Bits(const D&)> fn;
    bool whole;  // true: all elements from one state; false: each element from any state
};

template <typename D>
struct Program {
    std::string name;
    std::vector<Step<D>> steps;  // a cycle: applied from state A, the last step returns to A
};

// Two writers raced from a common start; the result must be one of the two serial orders.
template <typename D>
struct Pair {
    std::string name;
    Step<D> start, first, second;
    int weight = 1;  // rounds multiplier
};

template <typename D>
struct Row {
    std::string name;
    std::function<D()> makeA;
    Step<D> setA, setB;                    // setParameters to A and to B
    std::vector<Step<D>> singlesToB;       // single-parameter setters, A -> B in order
    std::vector<Step<D>> singlesToA;       // and back, B -> A
    std::function<Bits(const D&)> params;  // the parameter getters
    std::vector<Pair<D>> extraPairs;       // row-specific conflicting writers
    std::vector<double> probes;            // scalar probes
    std::vector<double> xs;                // batch input
    bool discrete = false;
};

template <typename D>
void quietly(D& d, const Step<D>& s) {
    try {
        s(d);
    } catch (...) {
    }
}

// The state fingerprint used for copies and for two-writer results: parameters, then pdf, logpdf
// and CDF at every probe, then the mean.
template <typename D>
Bits signature(const Row<D>& row, const D& d) {
    Bits b = row.params(d);
    for (double x : row.probes) {
        b.push_back(guarded([&] { return d.getProbability(x); }));
        b.push_back(guarded([&] { return d.getLogProbability(x); }));
        b.push_back(guarded([&] { return d.getCumulativeProbability(x); }));
    }
    b.push_back(guarded([&] { return d.getMean(); }));
    b.push_back(guarded([&] { return invalidFlag(d); }));
    return b;
}

using Hint = detail::PerformanceHint;
using Pref = detail::PerformanceHint::PreferredStrategy;

struct NamedHint {
    const char* name;
    Hint hint;
};

const std::vector<NamedHint>& batchHints() {
    static const std::vector<NamedHint> hints = {
        {"SCALAR", {Pref::FORCE_SCALAR, std::nullopt}},
        {"VECTORIZED", {Pref::FORCE_VECTORIZED, std::nullopt}},
        {"PARALLEL", {Pref::FORCE_PARALLEL, std::nullopt}},
        {"MAX_THROUGHPUT", {Pref::MAXIMIZE_THROUGHPUT, std::nullopt}},
        {"AUTO", {Pref::AUTO, std::nullopt}},
    };
    return hints;
}

template <typename D>
Bits batchBits(const D& d, const std::vector<double>& xs, int op, const Hint& hint) {
    std::vector<double> out(xs.size());
    const std::span<const double> in(xs);
    const std::span<double> res(out);
    try {
        if (op == 0)
            d.getProbability(in, res, hint);
        else if (op == 1)
            d.getLogProbability(in, res, hint);
        else
            d.getCumulativeProbability(in, res, hint);
    } catch (...) {
        return Bits{kThrew};
    }
    Bits b(out.size());
    for (std::size_t i = 0; i < out.size(); ++i)
        b[i] = bitsOf(out[i]);
    return b;
}

constexpr const char* kOpNames[] = {"pdf", "logpdf", "cdf"};

template <typename D>
std::vector<Reader<D>> makeReaders(const Row<D>& row, const D& fixedA, const D& fixedB) {
    std::vector<Reader<D>> rs;
    const auto probes = row.probes;
    rs.push_back({"scalar pdf",
                  [probes](const D& d) {
                      Bits b;
                      for (double x : probes)
                          b.push_back(guarded([&] { return d.getProbability(x); }));
                      return b;
                  },
                  false});
    rs.push_back({"scalar logpdf",
                  [probes](const D& d) {
                      Bits b;
                      for (double x : probes)
                          b.push_back(guarded([&] { return d.getLogProbability(x); }));
                      return b;
                  },
                  false});
    rs.push_back({"scalar cdf",
                  [probes](const D& d) {
                      Bits b;
                      for (double x : probes)
                          b.push_back(guarded([&] { return d.getCumulativeProbability(x); }));
                      return b;
                  },
                  false});
    rs.push_back({"quantile",
                  [](const D& d) {
                      Bits b;
                      for (double p : {0.01, 0.3, 0.5, 0.9, 0.999})
                          b.push_back(guarded([&] { return d.getQuantile(p); }));
                      return b;
                  },
                  false});
    rs.push_back({"moments",
                  [](const D& d) {
                      return Bits{guarded([&] { return d.getMean(); }),
                                  guarded([&] { return d.getVariance(); }),
                                  guarded([&] { return d.getSkewness(); }),
                                  guarded([&] { return d.getKurtosis(); }),
                                  guarded([&] { return d.getMedian(); }),
                                  guarded([&] { return d.getMode(); }),
                                  guarded([&] { return d.getEntropy(); })};
                  },
                  false});
    rs.push_back({"getters", row.params, false});
    for (int op = 0; op < 3; ++op)
        for (const auto& h : batchHints()) {
            const auto xs = row.xs;
            const Hint hint = h.hint;
            // FORCE_SCALAR calls the locking scalar method per element (executeStrategy's SCALAR
            // case); the one-snapshot contract covers the batch lambdas only. Checked per element.
            const bool whole = hint.strategy != Pref::FORCE_SCALAR;
            rs.push_back({std::string("batch ") + kOpNames[op] + " " + h.name,
                          [xs, op, hint](const D& d) { return batchBits(d, xs, op, hint); },
                          whole});
        }
    // Two draws per call, each from its own fresh generator, so each element depends only on the
    // state its call saw (a rejection loop consumes a state-dependent amount of a shared stream).
    // Two draws also exposed hidden per-thread sampler state (#188); SeededSample gates that.
    rs.push_back({"sample",
                  [](const D& d) {
                      std::mt19937 rng1(1234), rng2(5678);
                      const std::uint64_t first = guarded([&] { return d.sample(rng1); });
                      return Bits{first, guarded([&] { return d.sample(rng2); })};
                  },
                  false});
    rs.push_back({"sample n",
                  [](const D& d) {
                      std::mt19937 rng(4321);
                      Bits b;
                      try {
                          for (double v : d.sample(rng, 64))
                              b.push_back(bitsOf(v));
                      } catch (...) {
                          b.push_back(kThrew);
                      }
                      return b;
                  },
                  true});
    const Row<D>* rowp = &row;
    rs.push_back({"copy construct",
                  [rowp](const D& d) {
                      const D c(d);
                      return signature(*rowp, c);
                  },
                  true});
    rs.push_back({"copy assign",
                  [rowp](const D& d) {
                      D c = rowp->makeA();
                      c = d;
                      return signature(*rowp, c);
                  },
                  true});
    rs.push_back({"equality",
                  [&fixedA, &fixedB](const D& d) {
                      return Bits{guarded([&] { return (d == fixedA) ? 1.0 : 0.0; }),
                                  guarded([&] { return (d == fixedB) ? 1.0 : 0.0; })};
                  },
                  false});
    rs.push_back({"validate",
                  [](const D& d) { return Bits{guarded([&] { return invalidFlag(d); })}; }, false});
    return rs;
}

// expected[state][reader]
template <typename D>
std::vector<std::vector<Bits>> allowedResults(const Row<D>& row, const Program<D>& prog,
                                              const std::vector<Reader<D>>& readers,
                                              std::string& error) {
    D p = row.makeA();
    std::vector<std::vector<Bits>> expected;
    const auto snap = [&] {
        std::vector<Bits> v;
        for (const auto& r : readers)
            v.push_back(r.fn(p));
        expected.push_back(std::move(v));
    };
    snap();
    const Bits start = signature(row, p);
    for (const auto& s : prog.steps) {
        quietly(p, s);
        snap();
    }
    if (signature(row, p) != start)
        error = "program " + prog.name + " does not return to state A";
    return expected;
}

template <typename D>
bool accepted(const Reader<D>& r, std::size_t ri, const Bits& got,
              const std::vector<std::vector<Bits>>& expected) {
    if (r.whole) {
        for (const auto& st : expected)
            if (st[ri] == got)
                return true;
        return false;
    }
    if (got.size() != expected[0][ri].size())
        return false;
    for (std::size_t j = 0; j < got.size(); ++j) {
        bool ok = false;
        for (const auto& st : expected)
            if (st[ri][j] == got[j]) {
                ok = true;
                break;
            }
        if (!ok)
            return false;
    }
    return true;
}

// Defects open at the freeze head, each filed with the issue's number. A result they explain is
// printed as "[ known ]" and does not fail the test; a test whose only failures are known ones is
// skipped with the issue numbers, so it shows in the summary. The programs still run, so a fix
// shows as the "[ known ]" lines disappearing: remove its entry then (each issue's acceptance says
// so).
int knownReaderIssue(const std::string& row, const std::string& prog, const std::string& reader) {
    if (prog == "move assign")
        return 184;  // move-assignment takes no lock
    if (reader == "sample n" && (row == "InverseGamma" || row == "FisherF"))
        return 186;  // the vector sample loops the locking scalar sample
    if (reader == "sample" && row == "FisherF")
        return 182;  // sample mixes a snapshot with the resynced delegate
    return 0;
}

int knownProgramIssue(const std::string& /*row*/, const std::string& /*prog*/) {
    return 0;
}

int knownPairIssue(const std::string& row, const std::string& pair) {
    if (row == "TruncatedNormal" && pair.rfind("fit / ", 0) == 0)
        return 185;  // fit reads the bounds, then writes mu and sigma under a later lock
    return 0;
}

// Collects the known issues a test hit; skips the test at its end if nothing else failed.
struct KnownIssues {
    std::vector<std::string> lines;
    void hit(int issue, const std::string& what) {
        std::printf("[ known    ] #%d %s\n", issue, what.c_str());
        lines.push_back("#" + std::to_string(issue) + " " + what);
    }
};

#define SKIP_IF_ONLY_KNOWN(known)                                                                  \
    do {                                                                                           \
        if (!(known).lines.empty() && !::testing::Test::HasFailure()) {                            \
            std::string list_;                                                                     \
            for (const auto& l_ : (known).lines)                                                   \
                list_ += "\n  " + l_;                                                              \
            GTEST_SKIP() << "only known open defects failed:" << list_;                            \
        }                                                                                          \
    } while (0)

struct Failures {
    std::vector<std::string> lines;
    void add(const std::string& s) {
        if (lines.size() < 40)
            lines.push_back(s);
    }
};

// Data for fit: drawn from B, or from A where fit rejects B's data (TruncatedNormal holds its
// bounds fixed and rejects values outside them). Empty if neither changes the state from A.
template <typename D>
std::vector<double> fitData(const Row<D>& row) {
    const D a = row.makeA();
    const Bits sigA = signature(row, a);
    for (int fromB = 1; fromB >= 0; --fromB) {
        D src = row.makeA();
        if (fromB)
            row.setB(src);
        std::mt19937 rng(7);
        std::vector<double> data;
        try {
            data = src.sample(rng, 400);
            D t = row.makeA();
            t.fit(data);
            if (signature(row, t) != sigA)
                return data;
        } catch (...) {
        }
    }
    return {};
}

template <typename D>
std::vector<Program<D>> programsFor(const Row<D>& row) {
    std::vector<Program<D>> progs;
    progs.push_back({"setParameters", {row.setB, row.setA}});
    if (!row.singlesToB.empty()) {
        Program<D> p{"single setters", {}};
        for (const auto& s : row.singlesToB)
            p.steps.push_back(s);
        for (const auto& s : row.singlesToA)
            p.steps.push_back(s);
        progs.push_back(p);
    }
    // fit, then back to A
    {
        const auto data = fitData(row);
        if (data.empty())
            ADD_FAILURE() << row.name << ": no fit data changes the state from A";
        else
            progs.push_back({"fit", {[data](D& d) { d.fit(data); }, row.setA}});
    }
    // stream operator>> of B's text, then back to A
    {
        D b = row.makeA();
        row.setB(b);
        std::ostringstream os;
        os << b;
        const std::string text = os.str();
        progs.push_back({"stream",
                         {[text](D& d) {
                              std::istringstream is(text);
                              is >> d;
                          },
                          row.setA}});
    }
    // copy-assignment into the shared object, then move-assignment, each against setParameters
    {
        D b = row.makeA();
        row.setB(b);
        const D a = row.makeA();
        progs.push_back({"copy assign", {[b](D& d) { d = b; }, row.setA}});
        progs.push_back({"move assign", {[b](D& d) { d = D(b); }, row.setA}});
    }
    // A first step that leaves the state at A races nothing; say so rather than pass silently.
    const D a = row.makeA();
    const Bits sigA = signature(row, a);
    for (const auto& p : progs) {
        D t = row.makeA();
        quietly(t, p.steps.front());
        if (signature(row, t) == sigA && !knownProgramIssue(row.name, p.name))
            ADD_FAILURE() << row.name << " / " << p.name
                          << ": the first step does not change the state, so the program races "
                             "nothing";
    }
    return progs;
}

template <typename D>
void raceReaders(const Row<D>& row, KnownIssues& known) {
    const D fixedA = row.makeA();
    D fixedB = row.makeA();
    row.setB(fixedB);
    const auto readers = makeReaders(row, fixedA, fixedB);

    for (const auto& prog : programsFor(row)) {
        if (const int issue = knownProgramIssue(row.name, prog.name)) {
            known.hit(issue, row.name + " / " + prog.name + ": the program races nothing");
            continue;
        }
        std::string error;
        const auto expected = allowedResults(row, prog, readers, error);
        ASSERT_TRUE(error.empty()) << row.name << ": " << error;

        Failures fails;
        std::vector<int> badLoop(readers.size(), 0), badRound(readers.size(), 0);
        D d = row.makeA();

        // Free-running: one writer cycling the program, two readers cycling every reader.
        runWithWatchdog(row.name + " " + prog.name + " loops", kWatchdogSeconds, [&] {
            std::atomic<bool> stop{false};
            std::thread writer([&] {
                for (std::size_t i = 0; !stop.load(std::memory_order_relaxed); ++i)
                    quietly(d, prog.steps[i % prog.steps.size()]);
            });
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::milliseconds(scaled(150));
            std::vector<std::vector<int>> bad(2, std::vector<int>(readers.size(), 0));
            std::vector<std::thread> rt;
            for (int t = 0; t < 2; ++t)
                rt.emplace_back([&, t] {
                    for (std::size_t k = static_cast<std::size_t>(t);
                         std::chrono::steady_clock::now() < deadline; ++k) {
                        const std::size_t ri = k % readers.size();
                        gInFlight[t].store(readers[ri].name.c_str());
                        if (!accepted(readers[ri], ri, readers[ri].fn(d), expected))
                            ++bad[t][ri];
                        gInFlight[t].store(nullptr);
                    }
                });
            for (auto& t : rt)
                t.join();
            stop.store(true);
            writer.join();
            for (std::size_t ri = 0; ri < readers.size(); ++ri)
                badLoop[ri] = bad[0][ri] + bad[1][ri];
        });
        // The writer stopped at an arbitrary step; walk it back to A for the aligned rounds.
        d = row.makeA();

        // Aligned rounds: each round releases one writer step and one reader together.
        runWithWatchdog(row.name + " " + prog.name + " rounds", kWatchdogSeconds, [&] {
            const int perReader = scaled(30);
            std::size_t step = 0, ri = 0;
            Bits got;
            const int rounds = perReader * static_cast<int>(readers.size());
            int r = 0;
            racedRounds(
                rounds, [&] { ri = static_cast<std::size_t>(r++) % readers.size(); },
                [&] { quietly(d, prog.steps[step++ % prog.steps.size()]); },
                [&] {
                    gInFlight[2].store(readers[ri].name.c_str());
                    got = readers[ri].fn(d);
                    gInFlight[2].store(nullptr);
                },
                [&] {
                    const bool ok = accepted(readers[ri], ri, got, expected);
                    badRound[ri] += !ok;
                    return ok;
                });
        });

        for (std::size_t ri = 0; ri < readers.size(); ++ri) {
            if (badLoop[ri] + badRound[ri] == 0)
                continue;
            const std::string line = readers[ri].name + ": " + std::to_string(badLoop[ri]) +
                                     " loop, " + std::to_string(badRound[ri]) +
                                     " round results outside the allowed states";
            if (const int issue = knownReaderIssue(row.name, prog.name, readers[ri].name))
                known.hit(issue, row.name + " / " + prog.name + " / " + line);
            else
                fails.add(line);
        }
        std::string msg;
        for (const auto& l : fails.lines)
            msg += "\n  " + l;
        EXPECT_TRUE(fails.lines.empty()) << row.name << " / " << prog.name << msg;
    }
}

// Two writers from a common start, raced in aligned rounds. A throwing writer leaves the state
// unchanged, so the result must equal one of the two serial orders.
template <typename D>
void raceWriters(const Row<D>& row, KnownIssues& known) {
    std::vector<Pair<D>> pairs = row.extraPairs;
    pairs.push_back({"setParameters A/B from A", row.setA, row.setA, row.setB});
    // fit racing a setter: fit must not combine an estimate from one state with another's
    // parameters (TruncatedNormal reads its bounds, then writes mu and sigma under a later lock)
    if (const auto data = fitData(row); !data.empty()) {
        const Step<D> fit = [data](D& d) { d.fit(data); };
        pairs.push_back({"fit / setParameters B", row.setA, fit, row.setB});
        for (std::size_t i = 0; i < row.singlesToB.size(); ++i)
            pairs.push_back(
                {"fit / single setter " + std::to_string(i), row.setA, fit, row.singlesToB[i]});
    }
    for (std::size_t i = 0; i < row.singlesToB.size(); ++i)
        for (std::size_t j = i + 1; j < row.singlesToB.size(); ++j)
            pairs.push_back({"single setters " + std::to_string(i) + "/" + std::to_string(j),
                             row.setA, row.singlesToB[i], row.singlesToB[j]});
    // The same setter racing itself with two values, A's and B's, from B. This is the delegate
    // class (owner and delegate left on different values), whose window is a few instructions:
    // on M at 6987317, Geometric failed 1-52 and Bernoulli 1-2 rounds in 60,000, none in 3,000.
    // Hence 20x the rounds.
    for (std::size_t i = 0; i < row.singlesToB.size() && i < row.singlesToA.size(); ++i)
        pairs.push_back({"single setter " + std::to_string(i) + " A/B", row.setB, row.singlesToA[i],
                         row.singlesToB[i], 20});

    for (const auto& pr : pairs) {
        D o1 = row.makeA(), o2 = row.makeA();
        quietly(o1, pr.start);
        quietly(o1, pr.first);
        quietly(o1, pr.second);
        quietly(o2, pr.start);
        quietly(o2, pr.second);
        quietly(o2, pr.first);
        const Bits s1 = signature(row, o1), s2 = signature(row, o2);
        D d = row.makeA();
        int bad = 0;
        const int rounds = scaled(3000 * pr.weight);
        runWithWatchdog(row.name + " writers " + pr.name, kWatchdogSeconds, [&] {
            bad = racedRounds(
                rounds, [&] { quietly(d, pr.start); }, [&] { quietly(d, pr.first); },
                [&] { quietly(d, pr.second); },
                [&] {
                    const Bits s = signature(row, d);
                    return s == s1 || s == s2;
                });
        });
        const std::string what = row.name + " / writers " + pr.name + ": " + std::to_string(bad) +
                                 " of " + std::to_string(rounds) +
                                 " rounds ended outside both serial orders";
        if (const int issue = knownPairIssue(row.name, pr.name); issue && bad > 0)
            known.hit(issue, what);
        else
            EXPECT_EQ(bad, 0) << what;
    }
}

// Strategy differential (C5): no race. At sizes straddling the SIMD and fork thresholds, every
// strategy must return exactly one known kernel's bits: the batch kernel (FORCE_VECTORIZED's) or
// the scalar method's (FORCE_SCALAR's, itself gated against the scalar method element by
// element). Below the fork threshold PARALLEL must be the batch kernel. Settled on M (NEON,
// 2026-10-06): above the threshold, PARALLEL and MAXIMIZE_THROUGHPUT run the batch kernel in some
// distributions and the scalar formula in others (#175/#176 rewired only the per-element-locking
// lambdas), up to 88 ULP apart, so neither alone can be the gate. The kernel each strategy ran is
// printed per row.
template <typename D>
void differential(const Row<D>& row) {
    const D d = row.makeA();
    const std::size_t fork = arch::get_min_elements_for_parallel();
    const std::size_t simdMin = arch::simd::SIMDPolicy::getMinThreshold();
    const std::vector<std::size_t> sizes = {simdMin - 1, simdMin, simdMin + 1, 1000,
                                            fork - 1,    fork,    fork + 1,    3 * fork + 7};
    for (int op = 0; op < 3; ++op) {
        std::string ran[5];
        for (std::size_t n : sizes) {
            std::vector<double> xs(n);
            for (std::size_t i = 0; i < n; ++i)
                xs[i] = row.xs[(i * 7919) % row.xs.size()];
            const Bits vec = batchBits(d, xs, op, {Pref::FORCE_VECTORIZED, std::nullopt});
            const Bits sc = batchBits(d, xs, op, {Pref::FORCE_SCALAR, std::nullopt});
            Bits want(n);
            for (std::size_t i = 0; i < n; ++i)
                want[i] = op == 0   ? guarded([&] { return d.getProbability(xs[i]); })
                          : op == 1 ? guarded([&] { return d.getLogProbability(xs[i]); })
                                    : guarded([&] { return d.getCumulativeProbability(xs[i]); });
            EXPECT_EQ(sc, want) << row.name << " " << kOpNames[op] << " n=" << n
                                << ": FORCE_SCALAR differs from the scalar method";
            for (std::size_t h = 2; h < batchHints().size(); ++h) {
                const Bits got = batchBits(d, xs, op, batchHints()[h].hint);
                const bool isVec = got == vec, isSc = got == sc;
                EXPECT_TRUE(isVec || isSc)
                    << row.name << " " << kOpNames[op] << " n=" << n << " " << batchHints()[h].name
                    << ": neither the batch kernel's nor the scalar "
                    << "method's bits";
                if (std::string(batchHints()[h].name) == "PARALLEL" && n < fork)
                    EXPECT_TRUE(isVec)
                        << row.name << " " << kOpNames[op] << " n=" << n
                        << ": PARALLEL below the fork threshold is not the batch path";
                if (n >= fork) {
                    const char* k = isVec && isSc ? "=" : isVec ? "batch" : isSc ? "scalar" : "?";
                    if (ran[h].empty())
                        ran[h] = k;
                    else if (ran[h] != k && std::string(k) != "=")
                        ran[h] = ran[h] == "=" ? k : "mixed";
                }
            }
        }
        std::printf("[ kernel   ] %-18s %-6s n >= fork: PARALLEL %s, MAX_THROUGHPUT %s, AUTO %s\n",
                    row.name.c_str(), kOpNames[op], ran[2].c_str(), ran[3].c_str(), ran[4].c_str());
    }
}

// ---------------------------------------------------------------------------------------------
// Rows

std::vector<double> batchInput(double lo, double hi, bool discrete) {
    const std::size_t n = 2 * arch::get_min_elements_for_parallel() + 37;
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        const double x = (lo - 1.0) + t * (hi - lo + 2.0);
        xs[i] = discrete ? std::floor(x) : x;
    }
    // exact support edges, scattered
    for (std::size_t i = 0; i < n; i += 97)
        xs[i] = (i / 97) % 2 ? lo : hi;
    return xs;
}

std::vector<double> probesFor(double lo, double hi, bool discrete) {
    std::vector<double> p;
    for (double t : {-0.1, 0.0, 0.13, 0.37, 0.5, 0.71, 0.94, 1.0, 1.1}) {
        const double x = lo + t * (hi - lo);
        p.push_back(discrete ? std::floor(x) : x);
    }
    return p;
}

template <typename D>
Bits getters(std::initializer_list<std::function<double(const D&)>> gs, const D& d) {
    Bits b;
    for (const auto& g : gs)
        b.push_back(guarded([&] { return g(d); }));
    return b;
}

// One parameter.
template <typename D>
Row<D> row1(const char* name, double a, double b, void (D::*set)(double), double (D::*get)() const,
            double lo, double hi, bool discrete) {
    Row<D> r;
    r.name = name;
    r.makeA = [a] { return D::create(a).unwrap(); };
    r.setA = [set, a](D& d) { (d.*set)(a); };
    r.setB = [set, b](D& d) { (d.*set)(b); };
    r.singlesToB = {r.setB};
    r.singlesToA = {r.setA};
    r.params = [get](const D& d) { return Bits{guarded([&] { return (d.*get)(); })}; };
    r.probes = probesFor(lo, hi, discrete);
    r.xs = batchInput(lo, hi, discrete);
    r.discrete = discrete;
    return r;
}

// Two parameters; P1/P2 are the setter argument types (int for counts and bounds).
template <typename D, typename P1, typename P2, typename G1, typename G2>
Row<D> row2(const char* name, P1 a1, P2 a2, P1 b1, P2 b2, void (D::*setBoth)(P1, P2),
            void (D::*set1)(P1), void (D::*set2)(P2), G1 (D::*get1)() const, G2 (D::*get2)() const,
            double lo, double hi, bool discrete) {
    Row<D> r;
    r.name = name;
    r.makeA = [a1, a2] { return D::create(a1, a2).unwrap(); };
    r.setA = [setBoth, a1, a2](D& d) { (d.*setBoth)(a1, a2); };
    r.setB = [setBoth, b1, b2](D& d) { (d.*setBoth)(b1, b2); };
    r.singlesToB = {[set1, b1](D& d) { (d.*set1)(b1); }, [set2, b2](D& d) { (d.*set2)(b2); }};
    r.singlesToA = {[set1, a1](D& d) { (d.*set1)(a1); }, [set2, a2](D& d) { (d.*set2)(a2); }};
    r.params = [get1, get2](const D& d) {
        return Bits{guarded([&] { return static_cast<double>((d.*get1)()); }),
                    guarded([&] { return static_cast<double>((d.*get2)()); })};
    };
    r.probes = probesFor(lo, hi, discrete);
    r.xs = batchInput(lo, hi, discrete);
    r.discrete = discrete;
    return r;
}

// Bounded rows: the two bound setters raced against each other into a >= b (the TOCTOU class).
template <typename D, typename T>
void addBoundPairs(Row<D>& r, T a0, T b0, T newA, T newB) {
    r.extraPairs.push_back({"bounds cross", [a0, b0](D& d) { d.setBounds(a0, b0); },
                            [newA](D& d) { d.setLowerBound(newA); },
                            [newB](D& d) { d.setUpperBound(newB); }});
}

// Seeded reproducibility (no race): a draw from a freshly seeded generator depends only on that
// generator and the parameters, not on what the thread drew before (#188: Gaussian kept a
// thread_local Box-Muller spare, so alternate calls returned the previous generator's spare).
template <typename D>
void seededSample(const Row<D>& row) {
    const D d = row.makeA();
    const auto draw = [&] {
        std::mt19937 rng(42);
        const double one = d.sample(rng);
        std::mt19937 rng2(42);
        const auto three = d.sample(rng2, 3);
        Bits b{bitsOf(one)};
        for (double v : three)
            b.push_back(bitsOf(v));
        return b;
    };
    // Hidden per-thread state can depend on the parity of earlier draws, so compare the draw with
    // itself both immediately repeated and after one unrelated draw.
    const Bits first = draw();
    EXPECT_EQ(draw(), first) << row.name << ": a repeated freshly seeded draw differs";
    std::mt19937 other(7);
    (void)d.sample(other);
    EXPECT_EQ(draw(), first) << row.name << ": a freshly seeded draw depends on an earlier draw";
}

// What this machine reaches. physical_cores is logical / 2 on every platform today (#190), so
// MAXIMIZE_THROUGHPUT reaches WORK_STEALING on every Mac.
template <typename D>
void runRow(const Row<D>& row) {
    std::printf(
        "[ reach    ] %s: logical %zu physical %zu, fork %zu, batch n %zu, "
        "MAX_THROUGHPUT -> %s, scale %.3g\n",
        row.name.c_str(), detail::SystemCapabilities::current().logical_cores(),
        detail::SystemCapabilities::current().physical_cores(),
        arch::get_min_elements_for_parallel(), row.xs.size(),
        detail::PerformanceDispatcher::selectMultiThreadedStrategy(
            D::kDistributionType, detail::SystemCapabilities::current()) ==
                detail::Strategy::WORK_STEALING
            ? "WORK_STEALING"
            : "PARALLEL",
        stressScale());
    std::fflush(stdout);
}

}  // namespace

#define STRESS_ROW(Name, MakeRow)                                                                  \
    TEST(ConcurrencyStress_##Name, Readers) {                                                      \
        const auto row = MakeRow;                                                                  \
        runRow(row);                                                                               \
        KnownIssues known;                                                                         \
        raceReaders(row, known);                                                                   \
        SKIP_IF_ONLY_KNOWN(known);                                                                 \
    }                                                                                              \
    TEST(ConcurrencyStress_##Name, Writers) {                                                      \
        KnownIssues known;                                                                         \
        raceWriters(MakeRow, known);                                                               \
        SKIP_IF_ONLY_KNOWN(known);                                                                 \
    }                                                                                              \
    TEST(ConcurrencyStress_##Name, SeededSample) {                                                 \
        seededSample(MakeRow);                                                                     \
    }                                                                                              \
    TEST(ConcurrencyStress_##Name, StrategyDifferential) {                                         \
        differential(MakeRow);                                                                     \
    }

// clang-format off
STRESS_ROW(Gaussian, (row2<GaussianDistribution, double, double>("Gaussian", 0.0, 1.0, 0.5, 2.0,
    &GaussianDistribution::setParameters, &GaussianDistribution::setMean,
    &GaussianDistribution::setStandardDeviation, &GaussianDistribution::getMean,
    &GaussianDistribution::getStandardDeviation, -4.0, 4.0, false)))
STRESS_ROW(Exponential, (row1<ExponentialDistribution>("Exponential", 1.0, 2.5,
    &ExponentialDistribution::setLambda, &ExponentialDistribution::getLambda, 0.0, 6.0, false)))
STRESS_ROW(Uniform, ([] {
    auto r = row2<UniformDistribution, double, double>("Uniform", 0.0, 3.0, 1.0, 4.0,
        &UniformDistribution::setParameters, &UniformDistribution::setLowerBound,
        &UniformDistribution::setUpperBound, &UniformDistribution::getLowerBound,
        &UniformDistribution::getUpperBound, 0.0, 4.0, false);
    addBoundPairs<UniformDistribution, double>(r, 0.0, 3.0, 2.0, 1.0);
    return r;
}()))
STRESS_ROW(Discrete, ([] {
    auto r = row2<DiscreteDistribution, int, int>("Discrete", 0, 3, 1, 4,
        &DiscreteDistribution::setParameters, &DiscreteDistribution::setLowerBound,
        &DiscreteDistribution::setUpperBound, &DiscreteDistribution::getLowerBound,
        &DiscreteDistribution::getUpperBound, 0.0, 4.0, true);
    addBoundPairs<DiscreteDistribution, int>(r, 0, 3, 2, 1);
    return r;
}()))
STRESS_ROW(Poisson, (row1<PoissonDistribution>("Poisson", 3.0, 4.5,
    &PoissonDistribution::setLambda, &PoissonDistribution::getLambda, 0.0, 15.0, true)))
STRESS_ROW(Gamma, (row2<GammaDistribution, double, double>("Gamma", 2.0, 1.0, 3.5, 0.5,
    &GammaDistribution::setParameters, &GammaDistribution::setAlpha,
    &GammaDistribution::setBeta, &GammaDistribution::getAlpha, &GammaDistribution::getBeta,
    0.0, 12.0, false)))
STRESS_ROW(Bernoulli, (row1<BernoulliDistribution>("Bernoulli", 0.2, 0.7,
    &BernoulliDistribution::setP, &BernoulliDistribution::getP, 0.0, 1.0, true)))
STRESS_ROW(Beta, (row2<BetaDistribution, double, double>("Beta", 2.0, 3.0, 0.7, 5.0,
    &BetaDistribution::setParameters, &BetaDistribution::setAlpha,
    &BetaDistribution::setBeta, &BetaDistribution::getAlpha, &BetaDistribution::getBeta,
    0.0, 1.0, false)))
STRESS_ROW(Binomial, (row2<BinomialDistribution, int, double>("Binomial", 10, 0.3, 14, 0.6,
    &BinomialDistribution::setParameters, &BinomialDistribution::setN,
    &BinomialDistribution::setP, &BinomialDistribution::getN, &BinomialDistribution::getP,
    0.0, 14.0, true)))
STRESS_ROW(StudentT, (row1<StudentTDistribution>("StudentT", 3.0, 7.5,
    &StudentTDistribution::setNu, &StudentTDistribution::getNu, -5.0, 5.0, false)))
STRESS_ROW(ChiSquared, (row1<ChiSquaredDistribution>("ChiSquared", 3.0, 6.5,
    &ChiSquaredDistribution::setK, &ChiSquaredDistribution::getK, 0.0, 15.0, false)))
STRESS_ROW(VonMises, (row2<VonMisesDistribution, double, double>("VonMises", 0.0, 2.0, 0.7, 5.0,
    &VonMisesDistribution::setParameters, &VonMisesDistribution::setMu,
    &VonMisesDistribution::setKappa, &VonMisesDistribution::getMu,
    &VonMisesDistribution::getKappa, -3.14159, 3.14159, false)))
STRESS_ROW(LogNormal, (row2<LogNormalDistribution, double, double>("LogNormal", 0.0, 0.5, 0.4, 0.9,
    &LogNormalDistribution::setParameters, &LogNormalDistribution::setMu,
    &LogNormalDistribution::setSigma, &LogNormalDistribution::getMu,
    &LogNormalDistribution::getSigma, 0.0, 6.0, false)))
STRESS_ROW(Pareto, (row2<ParetoDistribution, double, double>("Pareto", 1.0, 2.5, 1.5, 4.0,
    &ParetoDistribution::setParameters, &ParetoDistribution::setScale,
    &ParetoDistribution::setAlpha, &ParetoDistribution::getScale,
    &ParetoDistribution::getAlpha, 0.5, 8.0, false)))
STRESS_ROW(Rayleigh, (row1<RayleighDistribution>("Rayleigh", 1.0, 2.2,
    &RayleighDistribution::setSigma, &RayleighDistribution::getSigma, 0.0, 6.0, false)))
STRESS_ROW(Weibull, (row2<WeibullDistribution, double, double>("Weibull", 1.5, 1.0, 3.0, 2.0,
    &WeibullDistribution::setParameters, &WeibullDistribution::setShape,
    &WeibullDistribution::setScale, &WeibullDistribution::getShape,
    &WeibullDistribution::getScale, 0.0, 5.0, false)))
STRESS_ROW(Cauchy, (row2<CauchyDistribution, double, double>("Cauchy", 0.0, 1.0, 0.8, 2.5,
    &CauchyDistribution::setParameters, &CauchyDistribution::setX0,
    &CauchyDistribution::setGamma, &CauchyDistribution::getX0,
    &CauchyDistribution::getGamma, -8.0, 8.0, false)))
STRESS_ROW(Laplace, (row2<LaplaceDistribution, double, double>("Laplace", 0.0, 1.0, 0.6, 1.8,
    &LaplaceDistribution::setParameters, &LaplaceDistribution::setMu,
    &LaplaceDistribution::setB, &LaplaceDistribution::getMu, &LaplaceDistribution::getB,
    -6.0, 6.0, false)))
STRESS_ROW(Logistic, (row2<LogisticDistribution, double, double>("Logistic", 0.0, 1.0, 0.6, 1.8,
    &LogisticDistribution::setParameters, &LogisticDistribution::setMu,
    &LogisticDistribution::setS, &LogisticDistribution::getMu, &LogisticDistribution::getS,
    -6.0, 6.0, false)))
STRESS_ROW(Gumbel, (row2<GumbelDistribution, double, double>("Gumbel", 0.0, 1.0, 0.6, 1.8,
    &GumbelDistribution::setParameters, &GumbelDistribution::setMu,
    &GumbelDistribution::setBeta, &GumbelDistribution::getMu, &GumbelDistribution::getBeta,
    -4.0, 8.0, false)))
STRESS_ROW(HalfNormal, (row1<HalfNormalDistribution>("HalfNormal", 1.0, 2.2,
    &HalfNormalDistribution::setSigma, &HalfNormalDistribution::getSigma, 0.0, 6.0, false)))
STRESS_ROW(InverseGamma, (row2<InverseGammaDistribution, double, double>("InverseGamma", 3.0, 2.0,
    5.0, 1.0, &InverseGammaDistribution::setParameters, &InverseGammaDistribution::setAlpha,
    &InverseGammaDistribution::setBeta, &InverseGammaDistribution::getAlpha,
    &InverseGammaDistribution::getBeta, 0.0, 4.0, false)))
STRESS_ROW(Erlang, (row2<ErlangDistribution, int, double>("Erlang", 2, 1.0, 4, 2.5,
    &ErlangDistribution::setParameters, &ErlangDistribution::setK,
    &ErlangDistribution::setLambda, &ErlangDistribution::getK, &ErlangDistribution::getLambda,
    0.0, 8.0, false)))
STRESS_ROW(FisherF, (row2<FDistribution, double, double>("FisherF", 5.0, 10.0, 8.0, 20.0,
    &FDistribution::setParameters, &FDistribution::setD1, &FDistribution::setD2,
    &FDistribution::getD1, &FDistribution::getD2, 0.0, 5.0, false)))
STRESS_ROW(Geometric, (row1<GeometricDistribution>("Geometric", 0.2, 0.7,
    &GeometricDistribution::setP, &GeometricDistribution::getP, 0.0, 12.0, true)))
STRESS_ROW(NegativeBinomial, (row2<NegativeBinomialDistribution, double, double>(
    "NegativeBinomial", 3.0, 0.4, 5.5, 0.7, &NegativeBinomialDistribution::setParameters,
    &NegativeBinomialDistribution::setR, &NegativeBinomialDistribution::setP,
    &NegativeBinomialDistribution::getR, &NegativeBinomialDistribution::getP, 0.0, 20.0, true)))
STRESS_ROW(TruncatedNormal, ([] {
    using T = TruncatedNormalDistribution;
    Row<T> r;
    r.name = "TruncatedNormal";
    r.makeA = [] { return T::create(0.0, 1.0, -1.0, 2.0).unwrap(); };
    r.setA = [](T& d) { d.setParameters(0.0, 1.0, -1.0, 2.0); };
    r.setB = [](T& d) { d.setParameters(0.5, 1.5, -0.5, 3.0); };
    r.singlesToB = {[](T& d) { d.setMu(0.5); }, [](T& d) { d.setSigma(1.5); },
                    [](T& d) { d.setLowerBound(-0.5); }, [](T& d) { d.setUpperBound(3.0); }};
    r.singlesToA = {[](T& d) { d.setMu(0.0); }, [](T& d) { d.setSigma(1.0); },
                    [](T& d) { d.setLowerBound(-1.0); }, [](T& d) { d.setUpperBound(2.0); }};
    r.params = [](const T& d) {
        return getters<T>({[](const T& x) { return x.getMu(); },
                           [](const T& x) { return x.getSigma(); },
                           [](const T& x) { return x.getLowerBound(); },
                           [](const T& x) { return x.getUpperBound(); }}, d);
    };
    r.extraPairs.push_back({"bounds cross", r.setA, [](T& d) { d.setLowerBound(1.5); },
                            [](T& d) { d.setUpperBound(0.5); }});
    r.probes = probesFor(-1.0, 3.0, false);
    r.xs = batchInput(-1.0, 3.0, false);
    return r;
}()))
// clang-format on
