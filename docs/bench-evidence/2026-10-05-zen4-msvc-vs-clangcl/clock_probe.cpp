// Per-pass timing over N seconds: prints elapsed_s and ms per 1e8-iteration dependent chain, averaged per 5 s.
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc, char** argv) {
    const int secs = argc > 1 ? std::atoi(argv[1]) : 90;
    volatile double sink = 0;
    const auto start = std::chrono::steady_clock::now();
    double acc = 0; int n = 0; int bucket = 0;
    for (;;) {
        auto t0 = std::chrono::steady_clock::now();
        double x = 1.0;
        for (int i = 0; i < 100000000; ++i) x = x * 1.0000001 + 1e-9;
        sink = x;
        auto t1 = std::chrono::steady_clock::now();
        double el = std::chrono::duration<double>(t1 - start).count();
        acc += std::chrono::duration<double, std::milli>(t1 - t0).count(); ++n;
        if (int(el / 5) != bucket) { std::printf("%5.0f s  %.1f ms\n", el, acc / n); std::fflush(stdout); acc = 0; n = 0; bucket = int(el / 5); }
        if (el > secs) break;
    }
}
