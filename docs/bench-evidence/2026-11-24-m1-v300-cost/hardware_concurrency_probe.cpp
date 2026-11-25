#include <chrono>
#include <cstdio>
#include <thread>
int main() {
    for (int rep = 0; rep < 5; ++rep) {
        unsigned s = 0; const int n = 2000000;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) s += std::thread::hardware_concurrency();
        auto t1 = std::chrono::steady_clock::now();
        std::printf("hardware_concurrency: %.1f ns per call (sum %u)\n",
                    std::chrono::duration<double, std::nano>(t1 - t0).count() / n, s);
    }
}
