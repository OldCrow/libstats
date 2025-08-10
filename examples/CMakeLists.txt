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
        # Only apply Homebrew LLVM linking if it's being used
        if(USING_HOMEBREW_LLVM AND LLVM_ROOT)
            target_link_options(${EXAMPLE_NAME} PRIVATE
                -L${LLVM_ROOT}/lib/c++
                -Wl,-rpath,${LLVM_ROOT}/lib/c++
            )
        endif()
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

# Quick start and core usage examples
create_example(quick_start_tutorial quick_start_tutorial.cpp)
create_example(basic_usage basic_usage.cpp)

# Distribution-specific examples
create_example(uniform_usage_demo uniform_usage_demo.cpp)
create_example(poisson_usage_demo poisson_usage_demo.cpp)
create_example(comparative_distributions_demo comparative_distributions_demo.cpp)
create_example(error_handling_demo error_handling_demo.cpp)

# Advanced examples
create_example(statistical_validation_demo statistical_validation_demo.cpp)
create_example(parallel_execution_demo parallel_execution_demo.cpp)
create_example(performance_learning_demo performance_learning_demo.cpp)

# Performance benchmarks
create_example(gaussian_performance_benchmark gaussian_performance_benchmark.cpp)
create_example(exponential_performance_benchmark exponential_performance_benchmark.cpp)
