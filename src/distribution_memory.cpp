#include "../include/core/distribution_memory.h"

namespace stats {

// Thread-local memory pool instance
thread_local MemoryPool thread_pool_;

}  // namespace stats
