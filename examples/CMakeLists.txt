# Examples CMakeLists.txt

# Function to create example executables with proper configuration
function(create_example EXAMPLE_NAME SOURCE_FILE)
    add_executable(${EXAMPLE_NAME} ${SOURCE_FILE})

    # Link against static library by default for examples
    target_link_libraries(${EXAMPLE_NAME} PRIVATE libstats_static)

    # Add include directories
    target_include_directories(${EXAMPLE_NAME} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR}
                                                       ${CMAKE_CURRENT_SOURCE_DIR}/../src)

    # Apply same compiler flags as main library
    if(APPLE)
        target_compile_options(${EXAMPLE_NAME} PRIVATE -stdlib=libc++)
        # Rpath inherited from libstats_static via PUBLIC link options
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
    set_target_properties(${EXAMPLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# Quick start and core usage examples
create_example(quick_start_tutorial quick_start_tutorial.cpp)
create_example(basic_usage basic_usage.cpp)
create_example(distribution_families_demo distribution_families_demo.cpp)

# Distribution and scenario examples
create_example(error_handling_demo error_handling_demo.cpp)

# Advanced examples
create_example(statistical_validation_demo statistical_validation_demo.cpp)
create_example(parallel_execution_demo parallel_execution_demo.cpp)
create_example(performance_learning_demo performance_learning_demo.cpp)

# Note: gaussian_performance_benchmark.cpp and exponential_performance_benchmark.cpp are not built
# here — benchmark-style tools belong in tools/, not examples/.
