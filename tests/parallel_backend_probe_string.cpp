// #179 probe: an ordinary standard header comes first, as in most translation units.
// clang-format off: the include order is what this file tests.
#include <string>

#include "include/parallel_backend_probe.h"
// clang-format on

namespace libstats_test {
int backendAfterString() noexcept {
    return LIBSTATS_TEST_SELECTED_BACKEND;
}
}  // namespace libstats_test
