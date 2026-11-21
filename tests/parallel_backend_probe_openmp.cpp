// #179 probe: this file alone is compiled with -fopenmp where the compiler takes it, as a
// consumer's own sources might be. _OPENMP is then defined here and nowhere else in the test.
#include "include/parallel_backend_probe.h"

namespace libstats_test {
int backendWithOpenMpFlag() noexcept {
#if defined(_OPENMP)
    return LIBSTATS_TEST_SELECTED_BACKEND;
#else
    return kBackendNotProbed;
#endif
}
}  // namespace libstats_test
