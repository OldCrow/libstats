# =============================================================================
# SHARED COMPILER FLAG DEFINITIONS
# =============================================================================
# Define reusable flag sets to reduce repetition and improve maintainability

# Common warning flag sets
set(LIBSTATS_COMMON_WARNINGS_UNIX -Wall -Wextra)
set(LIBSTATS_COMMON_WARNINGS_MSVC /W3)

# Strict warning flag sets (shared between compiler-specific modes)
set(LIBSTATS_CLANG_STRICT_WARNINGS
    -Wall
    -Wextra
    -pedantic
    # Type conversion warnings (strict)
    -Wconversion
    -Wfloat-conversion
    -Wimplicit-int-conversion
    -Wshorten-64-to-32
    -Wdouble-promotion
    -Wsign-conversion
    # Additional strict warnings
    -Wshadow
    -Wunused
    -Wcast-align
    -Wcast-qual
    -Wold-style-cast
    -Woverloaded-virtual
    -Wextra-semi
    -Wmissing-declarations
    # Clang-specific warnings
    -Wloop-analysis
    -Wlogical-op-parentheses
    -Wbool-conversion
    -Wint-conversion
    -Wnull-dereference
    -Wuninitialized
    -Wconditional-uninitialized
    -Wstring-conversion
    -Wdeprecated-volatile # Catch deprecated volatile usage (C++20)
    # ODR (One Definition Rule) violation detection - Enhanced
    -Wduplicate-decl-specifier
    -Wodr
    -fno-common
    -Winconsistent-missing-override
    -Wmultiple-move-vbase
    # Allow some warnings that are too strict for practical development
    -Wno-unused-parameter # Common in template/callback code
    -Wno-padded # Not practical for cross-platform development
    -Wno-c++98-compat # We're using C++20
    -Wno-c++98-compat-pedantic)

set(LIBSTATS_GCC_STRICT_WARNINGS
    -Wall
    -Wextra
    -pedantic
    # GCC-specific conversion warnings
    -Wconversion
    -Wfloat-conversion
    -Wdouble-promotion
    -Wsign-conversion
    -Wold-style-cast
    # GCC-specific diagnostic warnings
    -Wlogical-op
    -Wduplicated-cond
    # NOTE: -Wduplicated-branches deliberately omitted. The per-CPU-tier dispatch functions in
    # platform_constants_impl.cpp intentionally keep distinct branches (e.g. Intel_Legacy vs generic
    # AVX) that currently return the same tuning constant but are kept separate for future per-tier
    # divergence. The warning fires on this by-design structure and catches no real bug here.
    -Wrestrict
    -Wnull-dereference
    -Wjump-misses-init
    # Template and name resolution differences
    -Wtrampolines
    -Wunsafe-loop-optimizations
    -Wvector-operation-performance
    -Wsuggest-override
    -Wsuggest-final-types
    -Wsuggest-final-methods
    # Memory and alignment warnings
    -Wcast-align=strict
    -Wstrict-overflow=2
    -Wformat=2
    -Wformat-overflow=2
    -Wformat-truncation=2
    # GCC's stricter undefined behavior detection
    -Wshift-overflow=2
    -Wstringop-overflow=4
    # ODR (One Definition Rule) violation detection - Enhanced
    -fno-common
    -Wodr
    # NOTE: -Wredundant-decls deliberately omitted. The library uses forward-declaration headers
    # (e.g. common/platform_constants_fwd.h) by design to decouple includes; those symbols are then
    # fully declared in their platform/ headers. -Wredundant-decls flags this intentional fwd-decl
    # idiom as an error under -Werror. -Wodr + -fno-common above still cover the actual
    # ODR-violation class this section targets. Allow some warnings that are too pedantic
    -Wno-inline
    -Wno-padded
    -Wno-unused-parameter
    -Wno-sign-conversion # Too noisy with standard library
)

# MSVC-compatible warning flags for cross-compiler testing
set(LIBSTATS_MSVC_COMPAT_WARNINGS
    -Wall
    -Wextra
    # Type conversion warnings (MSVC-like strictness)
    -Wconversion
    -Wfloat-conversion
    -Wimplicit-int-conversion
    -Wshorten-64-to-32
    -Wdouble-promotion
    # Additional strict warnings
    -Wshadow
    -Wunused
    -Wcast-align
    -Wcast-qual
    -Wold-style-cast
    -Woverloaded-virtual
    -Wextra-semi
    -Wmissing-declarations
    # Allow some warnings that are too strict for practical development
    -Wno-sign-conversion # Too many false positives in standard library interactions
    -Wno-unused-parameter # Common in template/callback code
    -Wno-padded # Not practical for cross-platform development
)

# Enhanced MSVC warning sets
set(LIBSTATS_MSVC_ENHANCED_WARNINGS
    /w14242
    /w14254
    /w14263
    /w14265
    /w14287
    /w14289
    /w14296
    /w14311
    /w14545
    /w14546
    /w14547
    /w14549
    /w14555
    /w14619
    /w14640
    /w14826
    /w14905
    /w14906
    /w14928
    # Additional warnings that MSVC shows but our builds miss
    /w14267 # size_t to smaller integer conversion
    /w14244 # possible loss of data (narrowing conversion)
    /w14101 # unreferenced local variable
    # Code analysis warnings
    /analyze:WX- # Don't treat code analysis warnings as errors (for MSVCWarn)
    # ODR (One Definition Rule) violation detection - Enhanced
    /we4020 # Too many actual parameters for function-like macro
    /we4002 # Too many actual parameters for macro
    /we4005 # Macro redefinition (can catch ODR issues)
    /we4229 # Anachronism used: modifiers on data are ignored
    /we4239 # Nonstandard extension used (can catch duplicate definitions)
)

# Optimization levels for different build types
set(LIBSTATS_OPT_NONE_UNIX -O0)
set(LIBSTATS_OPT_LIGHT_UNIX -O1)
set(LIBSTATS_OPT_FULL_UNIX -O3)
set(LIBSTATS_OPT_NONE_MSVC /Od)
set(LIBSTATS_OPT_LIGHT_MSVC /O1)
set(LIBSTATS_OPT_FULL_MSVC /O2)

# Debug information flags
set(LIBSTATS_DEBUG_INFO_UNIX -g)
set(LIBSTATS_DEBUG_INFO_MSVC /Zi)

# =============================================================================
# COMPILER-SPECIFIC BUILD CONFIGURATION WITH GENERATOR EXPRESSIONS
# =============================================================================
# Use generator expressions to reduce repetitive if/elseif chains Standard CMake Build Types
# (Release, Debug) - Cross-platform Custom Compiler-Specific Build Types (ClangStrict, ClangWarn,
# MSVCStrict, MSVCWarn, GCCStrict, GCCWarn)
# =============================================================================

# =============================================================================
# PROVEN TRADITIONAL CMAKE COMPILER CONFIGURATION
# =============================================================================
# Using traditional conditionals that are guaranteed to work reliably Generator expressions have
# proven problematic with complex nested conditions

# MSVC-specific compiler definitions (Windows compatibility)
add_compile_definitions($<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
                        $<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_DEPRECATE>)

# Apply build type and compiler-specific flags using traditional CMake approach This maintains
# reliability while ensuring cross-compiler compatibility

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC compiler flags /utf-8: interpret source files as UTF-8 and emit UTF-8 output. Silences
    # C4566 (universal-character-name cannot be represented in current code page) for Unicode
    # characters in string literals across all tool sources.
    add_compile_options(/utf-8)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/W3 /O2 /DNDEBUG)
        message(STATUS "Applied MSVC Release flags: /W3 /O2 /DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/W3 /Od /Zi /RTC1 /MDd)
        message(STATUS "Applied MSVC Debug flags: /W3 /Od /Zi /RTC1 /MDd")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(/W3 /O1 /Zi /MD /wd4996)
        message(STATUS "Applied MSVC Dev flags: /W3 /O1 /Zi /MD /wd4996")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(/W4 /WX /permissive-)
        add_compile_options(${LIBSTATS_MSVC_ENHANCED_WARNINGS})
        message(STATUS "Applied Strict flags (MSVC): /W4 /WX /permissive-")
    else()
        add_compile_options(/W3)
        message(STATUS "Applied MSVC default flags: /W3")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    # Clang/AppleClang compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied Clang Release flags: ${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied Clang Debug flags: ${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
        message(
            STATUS
                "Applied Clang Dev flags: ${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
        message(STATUS "Applied Strict flags (Clang/AppleClang): strict warnings + -Werror")
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} -O2)
        message(STATUS "Applied Clang default flags: ${LIBSTATS_COMMON_WARNINGS_UNIX} -O2")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied GCC Release flags: ${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -fstack-protector-strong)
        message(
            STATUS
                "Applied GCC Debug flags: ${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -fstack-protector-strong"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
        message(
            STATUS
                "Applied GCC Dev flags: ${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(${LIBSTATS_GCC_STRICT_WARNINGS} -Werror)
        message(STATUS "Applied Strict flags (GCC): strict warnings + -Werror")
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} ${LIBSTATS_OPT_FULL_UNIX})
        message(
            STATUS
                "Applied GCC default flags: ${LIBSTATS_COMMON_WARNINGS_UNIX} ${LIBSTATS_OPT_FULL_UNIX}"
        )
    endif()
else()
    # Unknown compiler - use sensible defaults
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        message(STATUS "Applied unknown compiler Release flags: -O3 -DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
        message(STATUS "Applied unknown compiler Debug flags: -O0 -g")
    else()
        add_compile_options(-O2)
        message(STATUS "Applied unknown compiler default flags: -O2")
    endif()
endif()
