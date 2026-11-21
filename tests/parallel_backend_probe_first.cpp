// #179 probe: parallel_execution.h is the first header this translation unit includes. Before
// the fix, libstdc++ had not yet defined __cpp_lib_execution here and the header chose pthreads.
#include "include/parallel_backend_probe.h"

namespace libstats_test {
int backendWithNothingBefore() noexcept {
    return LIBSTATS_TEST_SELECTED_BACKEND;
}
}  // namespace libstats_test
