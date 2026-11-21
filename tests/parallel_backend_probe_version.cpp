// #179 probe: <version> comes first, so every feature macro is defined before
// parallel_execution.h is read. Before the fix, this translation unit chose std::execution on
// libstdc++ while the other two probes chose pthreads.
// clang-format off: the include order is what this file tests.
#include <version>

#include "include/parallel_backend_probe.h"
// clang-format on

namespace libstats_test {
int backendAfterVersion() noexcept {
    return LIBSTATS_TEST_SELECTED_BACKEND;
}
}  // namespace libstats_test
