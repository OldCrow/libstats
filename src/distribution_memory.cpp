#include "../include/core/distribution_memory.h"

namespace libstats {

// Thread-local memory pool instance
thread_local MemoryPool thread_pool_;

}  // namespace libstats
