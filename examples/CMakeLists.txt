# Examples CMakeLists.txt

# Function to create example executables with proper configuration
function(create_example EXAMPLE_NAME SOURCE_FILE)
    add_executable(${EXAMPLE_NAME} ${SOURCE_FILE})
    
    # Link against static library by default for examples
    target_link_libraries(${EXAMPLE_NAME} PRIVATE libstats_static)
    
    # Add include directories
    target_include_directories(${EXAMPLE_NAME} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    
    # Apply same compiler flags as main library
    if(APPLE)
        target_compile_options(${EXAMPLE_NAME} PRIVATE -stdlib=libc++)
        target_link_options(${EXAMPLE_NAME} PRIVATE
            -L${LLVM_ROOT}/lib/c++
            -Wl,-rpath,${LLVM_ROOT}/lib/c++
        )
    endif()
    
    # Add SIMD compile definitions
    if(LIBSTATS_HAS_SSE2)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE LIBSTATS_HAS_SSE2=1)
    endif()
    if(LIBSTATS_HAS_AVX)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE LIBSTATS_HAS_AVX=1)
    endif()
    if(LIBSTATS_HAS_AVX2)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE LIBSTATS_HAS_AVX2=1)
    endif()
    if(LIBSTATS_HAS_AVX512)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE LIBSTATS_HAS_AVX512=1)
    endif()
    if(LIBSTATS_HAS_NEON)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE LIBSTATS_HAS_NEON=1)
    endif()
    
    # Set output directory
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endfunction()

# Basic usage example (temporarily disabled due to missing distributions)
# create_example(basic_usage basic_usage.cpp)

# Platform parallel demo (temporarily disabled due to missing libstats.h)
# create_example(platform_parallel_demo platform_parallel_demo.cpp)

# Comprehensive Gaussian performance benchmark
create_example(gaussian_performance_benchmark gaussian_performance_benchmark.cpp)

# Phase 3 Update Demo - showcases new goodness-of-fit tests and cross-validation methods
create_example(gaussian_phase3_update_demo gaussian_phase3_update_demo.cpp)

# Future examples (to be implemented)
# add_executable(parameter_fitting parameter_fitting.cpp)
# target_link_libraries(parameter_fitting libstats)
# 
# add_executable(validation_demo validation_demo.cpp)
# target_link_libraries(validation_demo libstats)
# 
# add_executable(performance_demo performance_demo.cpp)
# target_link_libraries(performance_demo libstats)
