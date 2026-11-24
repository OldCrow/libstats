// CPU time of a process idle for 10 s, with or without initialize_performance_systems()
// (libstats PLAN.md, "Measurements this calls for", item 7). Usage: idle_burn [--plain]
#include "libstats/libstats.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <thread>

int main(int argc, char** argv) {
    const bool plain = argc > 1 && std::strcmp(argv[1], "--plain") == 0;
    if (!plain) {
        stats::initialize_performance_systems();
    }
    rusage a{}, b{};
    getrusage(RUSAGE_SELF, &a);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    getrusage(RUSAGE_SELF, &b);
    auto s = [](const timeval& x, const timeval& y) {
        return static_cast<double>(y.tv_sec - x.tv_sec) +
               1e-6 * static_cast<double>(y.tv_usec - x.tv_usec);
    };
    std::printf("idle 10 s %s: user %.2f s, sys %.2f s, voluntary switches %ld\n",
                plain ? "with no pool started" : "after initialize_performance_systems",
                s(a.ru_utime, b.ru_utime), s(a.ru_stime, b.ru_stime), b.ru_nvcsw - a.ru_nvcsw);
    return 0;
}
