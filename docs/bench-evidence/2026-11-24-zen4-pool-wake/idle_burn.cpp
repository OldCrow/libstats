// CPU time of a process idle for 10 s, with or without initialize_performance_systems()
// (libstats PLAN.md, "Measurements this calls for", item 7). Usage: idle_burn [--plain]
// Windows version of ../2026-11-24-m1-pool-wake/idle_burn.cpp: GetProcessTimes for getrusage.
#include "libstats/libstats.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <windows.h>

static double seconds(const FILETIME& t) {
    ULARGE_INTEGER u;
    u.LowPart = t.dwLowDateTime;
    u.HighPart = t.dwHighDateTime;
    return static_cast<double>(u.QuadPart) * 1e-7;
}

int main(int argc, char** argv) {
    const bool plain = argc > 1 && std::strcmp(argv[1], "--plain") == 0;
    if (!plain) {
        stats::initialize_performance_systems();
    }
    FILETIME c, e, k0, u0, k1, u1;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k0, &u0);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k1, &u1);
    std::printf("idle 10 s %s: user %.2f s, sys %.2f s\n",
                plain ? "with no pool started" : "after initialize_performance_systems",
                seconds(u1) - seconds(u0), seconds(k1) - seconds(k0));
    return 0;
}
