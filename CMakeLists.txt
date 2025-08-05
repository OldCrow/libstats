cmake_minimum_required(VERSION 3.15)

# Set CMake policies
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)  # find_package uses upper-case <PACKAGENAME>_ROOT variables
endif()

# Detect Homebrew LLVM and fallback to system compiler (macOS/Linux only)
set(HOMEBREW_LLVM_ROOT "/usr/local/opt/llvm")
set(HOMEBREW_LLVM_ROOT_ARM "/opt/homebrew/opt/llvm")

# Homebrew LLVM detection (Intel Mac path first, then ARM Mac path)
if(APPLE OR (UNIX AND NOT WIN32))
    if(EXISTS "${HOMEBREW_LLVM_ROOT}/bin/clang++")
        set(LLVM_ROOT "${HOMEBREW_LLVM_ROOT}")
        set(CMAKE_C_COMPILER "${LLVM_ROOT}/bin/clang")
        set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang++")
        set(CMAKE_PREFIX_PATH "${LLVM_ROOT}")
        set(USING_HOMEBREW_LLVM TRUE)
        message(STATUS "Using Homebrew LLVM from ${LLVM_ROOT}")
    elseif(EXISTS "${HOMEBREW_LLVM_ROOT_ARM}/bin/clang++")
        set(LLVM_ROOT "${HOMEBREW_LLVM_ROOT_ARM}")
        set(CMAKE_C_COMPILER "${LLVM_ROOT}/bin/clang")
        set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang++")
        set(CMAKE_PREFIX_PATH "${LLVM_ROOT}")
        set(USING_HOMEBREW_LLVM TRUE)
        message(STATUS "Using Homebrew LLVM from ${LLVM_ROOT}")
    else()
        # Fallback to system compiler
        find_program(CMAKE_C_COMPILER clang)
        find_program(CMAKE_CXX_COMPILER clang++)
        set(USING_HOMEBREW_LLVM FALSE)
        message(STATUS "Homebrew LLVM not found, using system compiler: ${CMAKE_CXX_COMPILER}")
    endif()
endif()

project(libstats VERSION 0.7.1 LANGUAGES CXX)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-specific configurations
if(APPLE)
    # macOS specific configuration
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    
    if(USING_HOMEBREW_LLVM)
        # Homebrew LLVM specific configuration - Force LLVM libc++ to get C++20 execution policies
        # Note: rpath will be set per-target to avoid duplication warnings
        
        # CRITICAL: Include LLVM libc++ headers BEFORE system headers to get C++20 execution policies
        include_directories(BEFORE ${LLVM_ROOT}/include/c++/v1)
        include_directories(BEFORE ${LLVM_ROOT}/include)
        
        # Add compiler flags to ensure we use LLVM headers
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${LLVM_ROOT}/include/c++/v1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${LLVM_ROOT}/include")
        
        # Enable experimental PSTL support in LLVM libc++
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_HAS_EXPERIMENTAL_PSTL=1")
    else()
        # System compiler configuration - use system libc++
        message(STATUS "Using system libc++ with Apple Clang")
    endif()
    
    # Optional TBB support for parallel execution policies
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(TBB QUIET tbb)
        if(TBB_FOUND)
            include_directories(${TBB_INCLUDE_DIRS})
            link_directories(${TBB_LIBRARY_DIRS})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_CFLAGS_OTHER}")
            set(LIBSTATS_HAS_TBB TRUE)
            message(STATUS "TBB found - parallel execution policies enabled")
        else()
            set(LIBSTATS_HAS_TBB FALSE)
            message(STATUS "TBB not found - parallel execution policies may not work")
        endif()
    else()
        set(LIBSTATS_HAS_TBB FALSE)
        message(STATUS "PkgConfig not found - cannot detect TBB")
    endif()

elseif(UNIX AND NOT APPLE)
    # Linux specific configuration
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-soname,libstats.so.${PROJECT_VERSION_MAJOR}")

elseif(WIN32)
    # Windows specific configuration
    # Enable Windows compatibility macros
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    
    # Find TBB and GTest using find_package (no Homebrew or pkg-config on Windows)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        set(LIBSTATS_HAS_TBB TRUE)
        message(STATUS "TBB found - parallel execution policies enabled (Windows)")
    else()
        set(LIBSTATS_HAS_TBB FALSE)
        message(STATUS "TBB not found - parallel execution policies may not work (Windows)")
    endif()

    find_package(GTest QUIET)
    if(GTest_FOUND)
        set(GTEST_FOUND TRUE)
        message(STATUS "GTest found via find_package (Windows)")
    else()
        set(GTEST_FOUND FALSE)
        message(STATUS "GTest not found - GTest-based tests will be skipped (Windows)")
    endif()
endif()

# SIMD feature detection and compilation flags
# Use our comprehensive SIMD detection system
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SIMDDetection.cmake")

# Set options for SIMD detection behavior
option(LIBSTATS_ENABLE_RUNTIME_CHECKS "Enable runtime CPU checks even when cross-compiling" OFF)
option(LIBSTATS_CONSERVATIVE_SIMD "Use conservative SIMD settings (disable newer instruction sets)" OFF)

# Perform comprehensive SIMD detection
detect_simd_features()

# Compiler flags
if(MSVC)
    # MSVC-specific compiler flags for warnings and optimization
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/W4 /O2)
    else()
        add_compile_options(/W4)
    endif()
    # MSVC-specific compiler definitions for Windows compatibility
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)  # Allow standard C functions
    add_compile_definitions(_CRT_NONSTDC_NO_DEPRECATE)  # Allow POSIX function names
    # SIMD flags for MSVC (x64 only)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        add_compile_options(/arch:AVX2)
        add_compile_definitions(LIBSTATS_HAS_AVX2=1)
        add_compile_definitions(LIBSTATS_HAS_AVX=1)
        add_compile_definitions(LIBSTATS_HAS_SSE2=1)
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (MSVC x64)")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # ClangCL or Clang on Windows
    add_compile_options(-Wall -Wextra -O2)
    if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        add_compile_options(-mavx2)
        add_compile_definitions(LIBSTATS_HAS_AVX2=1)
        add_compile_definitions(LIBSTATS_HAS_AVX=1)
        add_compile_definitions(LIBSTATS_HAS_SSE2=1)
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (ClangCL x64)")
    endif()

elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-Wall -Wextra -O3)
    # CONSERVATIVE SIMD approach: Only enable what we're confident will work
    # Since we use runtime dispatch, we don't need to enable all SIMD at compile time
    # We'll just enable SSE2 (universal on x86_64) and let runtime dispatch handle the rest
    if(LIBSTATS_HAS_SSE2)
        add_compile_options(-msse2)
        add_compile_definitions(LIBSTATS_HAS_SSE2=1)
        message(STATUS "SIMD: SSE2 enabled (safe baseline for x86_64)")
    endif()
    if(LIBSTATS_HAS_AVX)
        add_compile_definitions(LIBSTATS_HAS_AVX=1)
        message(STATUS "SIMD: AVX compilation enabled (runtime-dispatched)")
    endif()
    if(LIBSTATS_HAS_AVX2)
        add_compile_definitions(LIBSTATS_HAS_AVX2=1)
        message(STATUS "SIMD: AVX2 compilation enabled (runtime-dispatched)")
    endif()
    if(LIBSTATS_HAS_AVX512)
        add_compile_definitions(LIBSTATS_HAS_AVX512=1)
        message(STATUS "SIMD: AVX-512 compilation enabled (runtime-dispatched)")
    endif()
    if(LIBSTATS_HAS_NEON)
        add_compile_definitions(LIBSTATS_HAS_NEON=1)
        message(STATUS "SIMD: NEON enabled")
    endif()
    # Disable -march=native to avoid generating unsupported instructions
    # if(NOT CMAKE_BUILD_TYPE STREQUAL "Distribution")
    #     add_compile_options(-march=native)
    # endif()
endif()

# =============================================================================
# DEPENDENCY-AWARE OBJECT LIBRARIES FOR OPTIMAL COMPILATION
# =============================================================================
# Organize sources into object libraries that align with header dependency levels
# This enables:
# - Parallel compilation of independent components
# - Minimal recompilation when headers change
# - Better incremental build performance
# - Clear architectural boundaries

# Level 0-1: Platform Foundation and Constants (No internal dependencies)
set(LIBSTATS_FOUNDATION_SOURCES
    src/cpu_detection.cpp              # Runtime CPU feature detection
    src/simd_policy.cpp               # SIMD policy and feature detection
)

# Level 2: Core Utilities (Depends on Level 0-1)
set(LIBSTATS_CORE_UTILITIES_SOURCES
    src/safety.cpp                    # Safe numerical operations
    src/validation.cpp                # Parameter validation and constraint checking
    src/math_utils.cpp                # Mathematical utilities and special functions
    src/log_space_ops.cpp             # Log-space arithmetic for numerical stability
)

# Level 2: Platform Capabilities (Depends on Level 0-1)
set(LIBSTATS_PLATFORM_SOURCES
    src/parallel_thresholds.cpp       # Architecture-specific parallel thresholds
    src/thread_pool.cpp               # Thread pool implementation
    src/work_stealing_pool.cpp        # Advanced work-stealing thread pool
)

# Level 3: Advanced Infrastructure (Depends on Level 0-2)
set(LIBSTATS_INFRASTRUCTURE_SOURCES
    src/adaptive_cache.cpp            # Memory-aware adaptive caching
    src/distribution_cache.cpp        # Distribution-specific cache optimization
    src/benchmark.cpp                 # Performance benchmarking utilities
    src/performance_history.cpp       # Performance data collection and analysis
    src/performance_dispatcher.cpp    # Smart auto-dispatch system
    src/system_capabilities.cpp       # System capability detection
)

# Level 4: Distribution Framework (Depends on Level 0-3)
set(LIBSTATS_FRAMEWORK_SOURCES
    src/distribution_memory.cpp       # Memory management for distributions (if exists)
    src/distribution_base.cpp         # Complete distribution base framework
)

# Level 5: Distribution Implementations (Depends on Level 0-4)
set(LIBSTATS_DISTRIBUTIONS_SOURCES
    src/gaussian.cpp                  # Gaussian distribution implementation
    src/exponential.cpp               # Exponential distribution implementation
    src/uniform.cpp                   # Uniform distribution implementation
    src/poisson.cpp                   # Poisson distribution implementation
    src/discrete.cpp                  # Discrete distribution implementation
    src/gamma.cpp                     # Gamma distribution implementation
)

# SIMD sources are added separately by the SIMD detection system
# These compile with specific architecture flags and are platform-dependent

# =============================================================================
# CREATE OBJECT LIBRARIES FOR PARALLEL COMPILATION
# =============================================================================
# Object libraries allow parallel compilation and better dependency tracking

# Foundation object library (Level 0-1)
add_library(libstats_foundation_obj OBJECT ${LIBSTATS_FOUNDATION_SOURCES})
target_include_directories(libstats_foundation_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_foundation_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})

# Core utilities object library (Level 2)
add_library(libstats_core_utilities_obj OBJECT ${LIBSTATS_CORE_UTILITIES_SOURCES})
target_include_directories(libstats_core_utilities_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_core_utilities_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_core_utilities_obj libstats_foundation_obj)

# Platform capabilities object library (Level 2)
add_library(libstats_platform_obj OBJECT ${LIBSTATS_PLATFORM_SOURCES})
target_include_directories(libstats_platform_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_platform_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_platform_obj libstats_foundation_obj)

# Advanced infrastructure object library (Level 3)
add_library(libstats_infrastructure_obj OBJECT ${LIBSTATS_INFRASTRUCTURE_SOURCES})
target_include_directories(libstats_infrastructure_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_infrastructure_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_infrastructure_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj)

# Distribution framework object library (Level 4)
add_library(libstats_framework_obj OBJECT ${LIBSTATS_FRAMEWORK_SOURCES})
target_include_directories(libstats_framework_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_framework_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_framework_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj libstats_infrastructure_obj)

# Distribution implementations object library (Level 5)
add_library(libstats_distributions_obj OBJECT ${LIBSTATS_DISTRIBUTIONS_SOURCES})
target_include_directories(libstats_distributions_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_distributions_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_distributions_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj libstats_infrastructure_obj libstats_framework_obj)

# SIMD object library (Platform-dependent, can compile in parallel)
add_library(libstats_simd_obj OBJECT ${LIBSTATS_SIMD_SOURCES})
target_include_directories(libstats_simd_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(libstats_simd_obj PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
add_dependencies(libstats_simd_obj libstats_foundation_obj)  # SIMD depends on CPU detection

# =============================================================================
# COMBINED SOURCE LIST FOR FINAL LIBRARIES
# =============================================================================
# Collect all object library outputs for the final static/shared libraries
set(LIBSTATS_ALL_OBJECTS
    $<TARGET_OBJECTS:libstats_foundation_obj>
    $<TARGET_OBJECTS:libstats_core_utilities_obj>
    $<TARGET_OBJECTS:libstats_platform_obj>
    $<TARGET_OBJECTS:libstats_infrastructure_obj>
    $<TARGET_OBJECTS:libstats_framework_obj>
    $<TARGET_OBJECTS:libstats_distributions_obj>
    $<TARGET_OBJECTS:libstats_simd_obj>
)

# Header-only library (for header-only components)
add_library(libstats_headers INTERFACE)
target_include_directories(libstats_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)

# Static library using object libraries for better build performance
add_library(libstats_static STATIC ${LIBSTATS_ALL_OBJECTS})
target_include_directories(libstats_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)
if(WIN32)
    set_target_properties(libstats_static PROPERTIES
        OUTPUT_NAME stats_static
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
else()
    set_target_properties(libstats_static PROPERTIES
        OUTPUT_NAME stats
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Dynamic/shared library using object libraries for better build performance
add_library(libstats_shared SHARED ${LIBSTATS_ALL_OBJECTS})
target_include_directories(libstats_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)
set_target_properties(libstats_shared PROPERTIES
    OUTPUT_NAME "stats"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Platform-specific shared library properties
if(APPLE)
    set_target_properties(libstats_shared PROPERTIES
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
elseif(UNIX AND NOT APPLE)
    set_target_properties(libstats_shared PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Configure SIMD compilation for all object libraries and final targets
# This ensures SIMD-specific compile flags are applied correctly to each component
configure_simd_target(libstats_foundation_obj)
configure_simd_target(libstats_core_utilities_obj)
configure_simd_target(libstats_platform_obj)
configure_simd_target(libstats_infrastructure_obj)
configure_simd_target(libstats_framework_obj)
configure_simd_target(libstats_distributions_obj)
configure_simd_target(libstats_simd_obj)

# Configure SIMD compilation for final targets
configure_simd_target(libstats_static)
configure_simd_target(libstats_shared)

# Platform-specific library linking configuration
if(APPLE AND USING_HOMEBREW_LLVM)
    # Apply Homebrew LLVM libc++ linking to the shared library
    target_link_options(libstats_shared PRIVATE
        -L${LLVM_ROOT}/lib/c++
        -Wl,-rpath,${LLVM_ROOT}/lib/c++
    )
endif()

# Create aliases for easier consumption
add_library(libstats::headers ALIAS libstats_headers)
add_library(libstats::static ALIAS libstats_static)
add_library(libstats::shared ALIAS libstats_shared)

# Test infrastructure
option(LIBSTATS_BUILD_TESTS "Build tests" ON)

if(LIBSTATS_BUILD_TESTS)
    # Create tests directory in build folder
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
    
    # Find GTest (Homebrew installation)
    # First, try to use Homebrew's GTest installation directly
    # Check both Intel and ARM64 paths
    set(GTEST_ROOT_INTEL "/usr/local/opt/googletest")
    set(GTEST_ROOT_ARM "/opt/homebrew/opt/googletest")
    
    # Determine which Homebrew path to use
    if(EXISTS "${GTEST_ROOT_ARM}/lib/cmake/GTest")
        set(GTEST_ROOT "${GTEST_ROOT_ARM}")
        message(STATUS "Using ARM64 Homebrew GTest path")
    elseif(EXISTS "${GTEST_ROOT_INTEL}/lib/cmake/GTest")
        set(GTEST_ROOT "${GTEST_ROOT_INTEL}")
        message(STATUS "Using Intel Homebrew GTest path")
    else()
        set(GTEST_ROOT "${GTEST_ROOT_ARM}")
        message(STATUS "Using default ARM64 Homebrew GTest path (may not exist)")
    endif()
    
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${GTEST_ROOT}")
    
    # Try find_package first with explicit path
    find_package(GTest QUIET HINTS ${GTEST_ROOT}/lib/cmake/GTest)
    
    if(GTest_FOUND OR TARGET GTest::gtest)
        set(GTEST_FOUND TRUE)
        message(STATUS "GTest found via find_package")
    else()
        # Fallback to pkg-config
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            # Set PKG_CONFIG_PATH to include Homebrew's pkgconfig directory
            set(ENV{PKG_CONFIG_PATH} "${GTEST_ROOT}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
            pkg_check_modules(GTEST QUIET gtest)
            pkg_check_modules(GTEST_MAIN QUIET gtest_main)
            if(GTEST_FOUND)
                message(STATUS "GTest found via pkg-config")
            endif()
        endif()
        
        # Manual fallback if both methods fail
        if(NOT GTEST_FOUND)
            # Check if the Homebrew installation exists manually
            if(EXISTS "${GTEST_ROOT}/lib/libgtest.a" AND EXISTS "${GTEST_ROOT}/include/gtest/gtest.h")
                set(GTEST_FOUND TRUE)
                set(GTEST_INCLUDE_DIRS "${GTEST_ROOT}/include")
                set(GTEST_LIBRARIES "${GTEST_ROOT}/lib/libgtest.a")
                set(GTEST_MAIN_LIBRARIES "${GTEST_ROOT}/lib/libgtest_main.a")
                message(STATUS "GTest found manually at ${GTEST_ROOT}")
            else()
                message(STATUS "GTest not found - GTest-based tests will be skipped")
            endif()
        endif()
    endif()
    
    # Function to create a test executable with proper flags
    function(create_libstats_test TEST_NAME SOURCE_FILE)
        add_executable(${TEST_NAME} ${SOURCE_FILE})
        
        # Set output directory to build/tests
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tests
        )
        
        # Link against static library by default for tests
        target_link_libraries(${TEST_NAME} PRIVATE libstats_static)
        
        # Link TBB if available
        if(LIBSTATS_HAS_TBB)
            target_link_libraries(${TEST_NAME} PRIVATE ${TBB_LIBRARIES})
        endif()
        
        # Apply same compiler flags as main library
        if(APPLE)
            target_compile_options(${TEST_NAME} PRIVATE -stdlib=libc++)
        endif()
        
        # Add include directories
        target_include_directories(${TEST_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        
        # Platform-specific linking
        if(APPLE AND USING_HOMEBREW_LLVM)
            target_link_options(${TEST_NAME} PRIVATE
                -L${LLVM_ROOT}/lib/c++
                -Wl,-rpath,${LLVM_ROOT}/lib/c++
            )
        endif()
        
        # Add SIMD compile definitions
        if(LIBSTATS_HAS_SSE2)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_SSE2=1)
        endif()
        if(LIBSTATS_HAS_AVX)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX=1)
        endif()
        if(LIBSTATS_HAS_AVX2)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX2=1)
        endif()
        if(LIBSTATS_HAS_AVX512)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX512=1)
        endif()
        if(LIBSTATS_HAS_NEON)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_NEON=1)
        endif()
        
        # Add to test suite
        add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tests/${TEST_NAME})
        
        # Set working directory for tests
        set_tests_properties(${TEST_NAME} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endfunction()
    
    # Function to create a test executable linked to dynamic library
    function(create_libstats_test_dynamic TEST_NAME SOURCE_FILE)
        add_executable(${TEST_NAME} ${SOURCE_FILE})
        
        # Set output directory to build/tests
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tests
        )
        
        # Link against shared library instead of static
        target_link_libraries(${TEST_NAME} PRIVATE libstats_shared)
        
        # Apply same compiler flags as main library
        if(APPLE)
            target_compile_options(${TEST_NAME} PRIVATE -stdlib=libc++)
        endif()
        
        # Add include directories
        target_include_directories(${TEST_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        
        # Platform-specific linking
        if(APPLE AND USING_HOMEBREW_LLVM)
            target_link_options(${TEST_NAME} PRIVATE
                -L${LLVM_ROOT}/lib/c++
                -Wl,-rpath,${LLVM_ROOT}/lib/c++
            )
        endif()
        
        # Add SIMD compile definitions
        if(LIBSTATS_HAS_SSE2)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_SSE2=1)
        endif()
        if(LIBSTATS_HAS_AVX)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX=1)
        endif()
        if(LIBSTATS_HAS_AVX2)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX2=1)
        endif()
        if(LIBSTATS_HAS_AVX512)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX512=1)
        endif()
        if(LIBSTATS_HAS_NEON)
            target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_NEON=1)
        endif()
        
        # Add to test suite
        add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tests/${TEST_NAME})
        
        # Set working directory for tests
        set_tests_properties(${TEST_NAME} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
    endfunction()
    
    # Function to create a GTest-based test executable
    function(create_libstats_gtest TEST_NAME SOURCE_FILE)
        if(GTEST_FOUND)
            add_executable(${TEST_NAME} ${SOURCE_FILE})
            
            # Set output directory to build/tests
            set_target_properties(${TEST_NAME} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
                RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tests
                RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tests
            )
            
            # Link against static library and GTest
            target_link_libraries(${TEST_NAME} PRIVATE libstats_static)
            
            # Link GTest libraries (prefer modern CMake targets to avoid duplicate warnings)
            if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
                # Modern CMake targets - these handle dependencies correctly
                target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main GTest::gtest)
            elseif(TARGET GTest::GTest AND TARGET GTest::Main)
                # Alternative modern target name
                target_link_libraries(${TEST_NAME} PRIVATE GTest::Main GTest::GTest)
            else()
                # Fallback to manual linking - be more careful about duplicates
                if(GTEST_MAIN_LIBRARIES)
                    target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_MAIN_LIBRARIES})
                    # Only add gtest if it's not already included in main libraries
                    if(GTEST_LIBRARIES AND NOT "${GTEST_MAIN_LIBRARIES}" MATCHES "gtest")
                        target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_LIBRARIES})
                    endif()
                elseif(GTEST_LIBRARIES)
                    target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_LIBRARIES})
                endif()
                target_include_directories(${TEST_NAME} PRIVATE ${GTEST_INCLUDE_DIRS})
            endif()
            
            # Apply same compiler flags as main library
            if(APPLE)
                target_compile_options(${TEST_NAME} PRIVATE -stdlib=libc++)
            endif()
            
            # Add include directories
            target_include_directories(${TEST_NAME} PRIVATE 
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src
            )
            
            # Platform-specific linking
            if(APPLE AND USING_HOMEBREW_LLVM)
                target_link_options(${TEST_NAME} PRIVATE
                    -L${LLVM_ROOT}/lib/c++
                    -Wl,-rpath,${LLVM_ROOT}/lib/c++
                )
            endif()
            
            # Add SIMD compile definitions
            if(LIBSTATS_HAS_SSE2)
                target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_SSE2=1)
            endif()
            if(LIBSTATS_HAS_AVX)
                target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX=1)
            endif()
            if(LIBSTATS_HAS_AVX2)
                target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX2=1)
            endif()
            if(LIBSTATS_HAS_AVX512)
                target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_AVX512=1)
            endif()
            if(LIBSTATS_HAS_NEON)
                target_compile_definitions(${TEST_NAME} PRIVATE LIBSTATS_HAS_NEON=1)
            endif()
            
            # Add to test suite
            add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tests/${TEST_NAME})
            
            # Set working directory for tests
            set_tests_properties(${TEST_NAME} PROPERTIES
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        else()
            message(WARNING "GTest not found - skipping GTest-based test: ${TEST_NAME}")
        endif()
    endfunction()
    
    # Enable testing
    enable_testing()
    
    # =======================================================================
    # LEVEL 0 TESTS: Foundational Headers (Standard Library Only)
    # =======================================================================
    # These test headers with no internal dependencies - only standard library
    
    # C++20 features and basic constants (no internal dependencies)
    create_libstats_test(test_cpp20_features tests/test_cpp20_features.cpp)
    create_libstats_test(test_constants tests/test_constants.cpp)
    
    # =======================================================================
    # LEVEL 1 TESTS: Platform Foundation and Aggregated Constants
    # =======================================================================
    # Tests for runtime capabilities and aggregated constant headers
    
    # cpu_detection.h - Runtime CPU feature detection
    create_libstats_test(test_cpu_detection tests/test_cpu_detection.cpp)
    
    # simd.h - SIMD compile-time detection and basic operations
    create_libstats_test(test_simd_integration_simple tests/test_simd_integration_simple.cpp)
    create_libstats_test(test_simd_integration tests/test_simd_integration.cpp)
    create_libstats_test(test_simd_operations tests/test_simd_operations.cpp)
    create_libstats_test(test_platform_optimizations tests/test_platform_optimizations.cpp)
    
    # =======================================================================
    # LEVEL 2 TESTS: Core Utilities and Platform Capabilities
    # =======================================================================
    # Test core utilities (Level 2a) and platform capabilities (Level 2b)
    
    # LEVEL 2a: Core Utilities (Platform-Independent)
    # safety.h - Safe numerical operations
    create_libstats_test(test_safety tests/test_safety.cpp)
    
    # math_utils.h - Mathematical utilities and special functions
    create_libstats_gtest(test_math_utils tests/test_math_utils.cpp)
    create_libstats_test(test_vectorized_math tests/test_vectorized_math.cpp)
    
    # validation.h - Parameter validation and constraint checking
    create_libstats_test(test_goodness_of_fit tests/test_goodness_of_fit.cpp)
    create_libstats_test(test_validation_enhanced tests/test_validation_enhanced.cpp)
    
    # error_handling.h - Exception-free error handling
    create_libstats_test(test_safe_factory tests/test_safe_factory.cpp)
    create_libstats_test(test_dual_api tests/test_dual_api.cpp)
    
    # LEVEL 2b: Platform Capabilities
    # thread_pool.h and work_stealing_pool.h - Thread pool implementations  
    create_libstats_test(test_thread_pool tests/test_thread_pool.cpp)
    create_libstats_test(test_work_stealing_pool tests/test_work_stealing_pool.cpp)
    
    # =======================================================================
    # LEVEL 3 TESTS: Advanced Infrastructure
    # =======================================================================
    # Test advanced infrastructure components that depend on Level 0-2
    
    # Performance and Caching Infrastructure
    # adaptive_cache.h - Memory-aware adaptive caching
    create_libstats_test(test_adaptive_cache tests/test_adaptive_cache.cpp)
    
    # parallel_execution.h - C++20 parallel execution with automatic fallback
    create_libstats_test(test_parallel_execution_integration tests/test_parallel_execution_integration.cpp)
    create_libstats_test(test_parallel_execution_comprehensive tests/test_parallel_execution_comprehensive.cpp)
    
    # benchmark.h - Performance measurement utilities
    create_libstats_test(test_benchmark_basic tests/test_benchmark_basic.cpp)
    
    # PHASE 3 PERFORMANCE OPTIMIZATION FRAMEWORK (NEW TESTS)
    # performance_history.h - Performance data collection and analysis
    create_libstats_gtest(test_performance_history tests/test_performance_history.cpp)
    
    # performance_dispatcher.h - Smart auto-dispatch system  
    create_libstats_gtest(test_performance_dispatcher tests/test_performance_dispatcher.cpp)
    
    # system_capabilities.cpp - System capability detection and integration
    create_libstats_gtest(test_system_capabilities tests/test_system_capabilities.cpp)
    
    # =======================================================================
    # LEVEL 4 TESTS: Complete Distribution Framework
    # =======================================================================
    # Test the integrated distribution framework (distribution_base.h)
    
    # distribution_base.h - Complete base class implementation
    # (Tested indirectly through distribution implementations, but could add direct tests)
    
    # =======================================================================
    # LEVEL 5 TESTS: Concrete Distribution Implementations
    # =======================================================================
    # Test concrete distribution implementations using the complete framework
    
    # gaussian.h - Gaussian/Normal distribution implementation (CONSOLIDATED)
    create_libstats_test(test_gaussian_basic tests/test_gaussian_basic.cpp)
    create_libstats_gtest(test_gaussian_enhanced tests/test_gaussian_enhanced.cpp)
    
    # exponential.h - Exponential distribution implementation (CONSOLIDATED)
    create_libstats_test(test_exponential_basic tests/test_exponential_basic.cpp)
    create_libstats_gtest(test_exponential_enhanced tests/test_exponential_enhanced.cpp)
    
    # discrete.h - Discrete uniform distribution implementation (CONSOLIDATED)
    create_libstats_test(test_discrete_basic tests/test_discrete_basic.cpp)
    create_libstats_gtest(test_discrete_enhanced tests/test_discrete_enhanced.cpp)
    
    # Atomic parameters test - Test lock-free parameter access and invalidation for all distributions
    create_libstats_test(test_atomic_parameters tests/test_atomic_parameters.cpp)
    
    # uniform.h - Uniform distribution implementation
    create_libstats_test(test_uniform_basic tests/test_uniform_basic.cpp)
    create_libstats_gtest(test_uniform_enhanced tests/test_uniform_enhanced.cpp)
    
    # poisson.h - Poisson distribution implementation
    create_libstats_test(test_poisson_basic tests/test_poisson_basic.cpp)
    create_libstats_gtest(test_poisson_enhanced tests/test_poisson_enhanced.cpp)
    
    # gamma.h - Gamma distribution implementation
    create_libstats_test(test_gamma_basic tests/test_gamma_basic.cpp)
    create_libstats_gtest(test_gamma_enhanced tests/test_gamma_enhanced.cpp)
    
    # =======================================================================
    # ADDITIONAL TESTS: Cross-cutting concerns and integration
    # =======================================================================
    # These test cross-cutting functionality and integration scenarios
    
    # Copy/move semantics stress tests
    # TODO: Temporarily commented out due to exponential/uniform distribution dependencies
    # if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_stress.cpp)
    #     create_libstats_test(test_copy_move_stress tests/test_copy_move_stress.cpp)
    #     set(HAS_COPY_MOVE_STRESS TRUE)
    # else()
    #     set(HAS_COPY_MOVE_STRESS FALSE)
    # endif()
    set(HAS_COPY_MOVE_STRESS FALSE)
    
    # if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_fix.cpp)
    #     create_libstats_test(test_copy_move_fix tests/test_copy_move_fix.cpp)
    #     set(HAS_COPY_MOVE_FIX TRUE)
    # else()
    #     set(HAS_COPY_MOVE_FIX FALSE)
    # endif()
    set(HAS_COPY_MOVE_FIX FALSE)
    
    # Dynamic library linking tests
    create_libstats_test_dynamic(test_dynamic_linking tests/test_dynamic_linking.cpp)
    
    # Dynamic versions of all _simple tests
    create_libstats_test_dynamic(test_simd_integration_simple_dynamic tests/test_simd_integration_simple.cpp)
    create_libstats_test_dynamic(test_gaussian_basic_dynamic tests/test_gaussian_basic.cpp)
    create_libstats_test_dynamic(test_exponential_basic_dynamic tests/test_exponential_basic.cpp)
    # create_libstats_test_dynamic(test_uniform_simple_dynamic tests/test_uniform_simple.cpp)  # TODO: uncomment after fixing uniform.cpp
    
    # TODO: Add these tests once distributions are implemented
    # create_libstats_test(test_exponential tests/test_exponential.cpp)
    # create_libstats_test(test_uniform tests/test_uniform.cpp)
    # create_libstats_test(test_poisson tests/test_poisson.cpp)
    # create_libstats_test(test_gamma tests/test_gamma.cpp)
    # create_libstats_test(test_validation tests/test_validation.cpp)
    
    # Future GTest-based tests can be added here using create_libstats_gtest()
    # Example: create_libstats_gtest(test_gaussian_gtest tests/test_gaussian_gtest.cpp)
    
    # Test target for running all tests (organized by dependency hierarchy)
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS
            # Level 0: Foundational tests (most critical)
            test_cpp20_features
            test_constants
            test_cpu_detection
            test_adaptive_cache
            test_simd_integration_simple
            test_simd_integration
            test_simd_operations
            test_platform_optimizations
            test_safe_factory
            
            # Level 1: Core infrastructure tests
            test_dual_api
            test_safety
            test_math_utils
            test_vectorized_math
            test_goodness_of_fit
            test_validation_enhanced
            
            # Level 2: Framework tests (no direct tests - covered by Level 4)
            
            # Level 3: Parallel infrastructure tests
            test_thread_pool
            test_work_stealing_pool
            test_parallel_execution_integration
            test_parallel_execution_comprehensive
            test_benchmark_basic
            
            # Level 4: Distribution implementation tests (CONSOLIDATED)
            test_gaussian_basic
            test_gaussian_enhanced
            test_exponential_basic
            test_exponential_enhanced
            test_discrete_basic
            test_discrete_enhanced
            test_atomic_parameters
            test_uniform_basic
            test_uniform_enhanced
            test_poisson_basic
            test_poisson_enhanced
            test_gamma_basic
            test_gamma_enhanced
            
            # Additional tests: Cross-cutting concerns
            test_dynamic_linking
            test_simd_integration_simple_dynamic
            test_gaussian_basic_dynamic
            test_exponential_basic_dynamic
            # test_uniform_simple_dynamic      # TODO: uncomment after fixing uniform.cpp
            
            # Conditional tests (added if they exist)
            $<$<BOOL:${HAS_COPY_MOVE_STRESS}>:test_copy_move_stress>
            $<$<BOOL:${HAS_COPY_MOVE_FIX}>:test_copy_move_fix>
        COMMENT "Running all libstats tests in dependency order (Level 0-4)"
    )
    
    message(STATUS "Test infrastructure enabled")
else()
    message(STATUS "Test infrastructure disabled")
endif()

# Tools and utilities
option(LIBSTATS_BUILD_TOOLS "Build utility tools" ON)

if(LIBSTATS_BUILD_TOOLS)
    # Helper function to create tool executables with consistent configuration
    function(add_libstats_tool tool_name source_file)
        add_executable(${tool_name} tools/${source_file})
        target_link_libraries(${tool_name} PRIVATE libstats_static)
        target_include_directories(${tool_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        
        # Apply same compiler flags as main library
        if(APPLE)
            target_compile_options(${tool_name} PRIVATE -stdlib=libc++)
            if(USING_HOMEBREW_LLVM)
                target_link_options(${tool_name} PRIVATE
                    -L${LLVM_ROOT}/lib/c++
                    -Wl,-rpath,${LLVM_ROOT}/lib/c++
                )
            endif()
        endif()
        
        # Add SIMD compile definitions
        target_compile_definitions(${tool_name} PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
        
        set_target_properties(${tool_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tools
        )
    endfunction()
    
    # System Information & Diagnostics Tools
    add_libstats_tool(cpu_info cpu_info.cpp)
    add_libstats_tool(constants_inspector constants_inspector.cpp)
    
    # Performance & Benchmarking Tools
    add_libstats_tool(performance_benchmark performance_benchmark.cpp)
    add_libstats_tool(parallel_threshold_benchmark parallel_threshold_benchmark.cpp)
    add_libstats_tool(grain_size_optimizer grain_size_optimizer.cpp)
    add_libstats_tool(performance_dispatcher_tool performance_dispatcher_tool.cpp)
    add_libstats_tool(system_capabilities_inspector system_capabilities_inspector.cpp)
    add_libstats_tool(threshold_learning_demo threshold_learning_demo.cpp)
    
    message(STATUS "Tools enabled:")
    message(STATUS "  - cpu_info: System CPU capability detection")
    message(STATUS "  - constants_inspector: Platform-specific constants analysis with Phase 3 integration")
    message(STATUS "  - performance_benchmark: Enhanced parallel operation benchmarks with PerformanceHistory integration")
    message(STATUS "  - parallel_threshold_benchmark: Enhanced distribution-specific threshold optimization with adaptive learning")
    message(STATUS "  - grain_size_optimizer: Comprehensive grain size optimization")
    message(STATUS "  - performance_dispatcher_tool: Interactive Phase 3 performance framework demonstration")
    message(STATUS "  - system_capabilities_inspector: Detailed system capability inspection and performance analysis")
    message(STATUS "  - threshold_learning_demo: Adaptive threshold learning demonstration and performance optimization showcase")
endif()

# Examples
add_subdirectory(examples)

# Install targets
install(TARGETS libstats_headers libstats_static libstats_shared
    EXPORT libstats-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/ DESTINATION include/libstats)

# Install CMake config files for find_package support
install(EXPORT libstats-targets
    FILE libstats-targets.cmake
    NAMESPACE libstats::
    DESTINATION lib/cmake/libstats
)

# Create and install config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    INSTALL_DESTINATION lib/cmake/libstats
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    DESTINATION lib/cmake/libstats
)

# Print summary
message(STATUS "libstats Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(LIBSTATS_BUILD_TESTS)
    message(STATUS "  Testing: Enabled")
    if(GTEST_FOUND)
        message(STATUS "  GTest: Available")
    else()
        message(STATUS "  GTest: Not found - using standalone tests")
    endif()
else()
    message(STATUS "  Testing: Disabled")
endif()

