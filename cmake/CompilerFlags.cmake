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
# PER-CONFIG OPTIMIZATION/DEBUG-INFO FLAGS FOR THE CUSTOM BUILD TYPES (Phase 3B/B5)
# =============================================================================
# CMake's standard per-config mechanism (CMAKE_CXX_FLAGS_<CONFIG>, applied to every target in every
# directory -- including FetchContent dependencies such as GTest) replaces the
# optimization/debug-info portion of the old global add_compile_options() chains below for Dev and
# Strict, exactly the way it already handles the built-in Release/Debug/RelWithDebInfo/MinSizeRel
# configs.
#
# DEVIATION from a literal "set(... CACHE STRING ...) -- no FORCE" reading, verified empirically and
# flagged for review: CMake auto-creates CMAKE_CXX_FLAGS_<CONFIG> as an EMPTY cache STRING the
# moment CXX is enabled in project(), for ANY CMAKE_BUILD_TYPE spelling -- including custom ones
# like "Dev" that CMake itself does not otherwise know about. That auto-created (empty) entry
# already exists in the cache by the time this file is include()'d, so a plain
# `set(CMAKE_CXX_FLAGS_DEV "-O1 -g" CACHE STRING ...)` with no FORCE is a silent no-op:
# CMAKE_CXX_FLAGS_DEV would stay "" forever, and Dev builds would get NO optimization/debug-info
# flags at all -- a real regression, not the "zero behavior change" this batch requires. Confirmed
# with a bare two-line test project (CMAKE_BUILD_TYPE=Dev, no other configuration) before writing
# this comment.
#
# Fix (the standard, documented CMake idiom for exactly this situation): guard the FORCE-set on the
# variable currently being empty/falsy. A user's own command-line override
# (-DCMAKE_CXX_FLAGS_DEV="...") or a parent project's pre-set value is truthy, so the guard skips
# the FORCE and the override survives untouched; CMake's own auto-created blank placeholder is
# falsy, so we do overwrite that one time with our real default. This preserves the
# "user-overridable, no repeated stomping" intent the no-FORCE instruction was going for, while
# actually working.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    if(NOT CMAKE_CXX_FLAGS_DEV)
        set(CMAKE_CXX_FLAGS_DEV
            "-O1 -g"
            CACHE STRING "Flags used by the CXX compiler during Dev builds." FORCE)
    endif()
    # Strict has never carried optimization/debug-info flags -- it exists solely to enable strict
    # warnings + -Werror/-WX (see libstats_apply_warnings below). Preserve that: empty string. (The
    # guard is applied here too, for symmetry/correctness, even though CMake's own auto-created
    # default is already "" -- so this branch is observably a no-op today, same as an unguarded
    # set() would be, but it correctly leaves room for a future non-empty Strict default without
    # silently re-stomping a user override the way an unguarded FORCE would.)
    if(NOT CMAKE_CXX_FLAGS_STRICT)
        set(CMAKE_CXX_FLAGS_STRICT
            ""
            CACHE STRING "Flags used by the CXX compiler during Strict builds." FORCE)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(NOT CMAKE_CXX_FLAGS_DEV)
        set(CMAKE_CXX_FLAGS_DEV
            "/O1 /Zi /MD"
            CACHE STRING "Flags used by the CXX compiler during Dev builds." FORCE)
    endif()
    if(NOT CMAKE_CXX_FLAGS_STRICT)
        set(CMAKE_CXX_FLAGS_STRICT
            ""
            CACHE STRING "Flags used by the CXX compiler during Strict builds." FORCE)
    endif()
endif()

# GCC's own Debug default (CMAKE_CXX_FLAGS_DEBUG = "-g") does not include -fstack-protector-strong,
# which the old global chain added for GCC only. Re-set the SAME (non-cache) variable, appending to
# whatever it currently holds, rather than a CACHE write: a plain set() creates a directory-scope
# variable that shadows the cache entry's value for the rest of *this* configure run only -- it
# never touches the CACHE entry itself. Each fresh `cmake` invocation re-reads the untouched cache
# value ("-g") as the right-hand side here, so re-running configure on an already-configured build
# directory reproduces "-g -fstack-protector-strong" once, not an ever-growing accumulation.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fstack-protector-strong")
endif()

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
#
# Phase 3B/B5: optimization/debug-info flags for Release/Debug/Dev/Strict no longer come from this
# chain -- Release/Debug rely on CMake's own CMAKE_CXX_FLAGS_<CONFIG> defaults (previously
# duplicated here); Dev/Strict rely on the CMAKE_CXX_FLAGS_DEV/_STRICT cache variables defined
# above. WARNING flags no longer come from this chain either -- they are applied PRIVATE,
# per-target, by libstats_apply_warnings() (defined below) so that FetchContent dependencies (GTest)
# do not receive them. What remains in this chain: the MSVC /utf-8 global flag (documented
# exception, must cover GTest), and -- preserved verbatim, unreachable on every supported flow --
# the "unknown compiler" branch and the Clang/GNU "unmatched build type" shadow-appends (see below).
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC compiler flags /utf-8: interpret source files as UTF-8 and emit UTF-8 output. Silences
    # C4566 (universal-character-name cannot be represented in current code page) for Unicode
    # characters in string literals across all tool sources. Stays global (unlike the warning/opt
    # flags below) so it also reaches FetchContent'd GTest translation units.
    add_compile_options(/utf-8)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    # Clang/AppleClang: Release/Debug/Dev/Strict flags now applied via CMAKE_CXX_FLAGS_<CONFIG>
    # (opt/ debug-info) and libstats_apply_warnings() (warnings) -- nothing left to add here for
    # those four build types. The "unmatched build type" branch below is unreachable on every
    # supported flow (the top level of CMakeLists.txt forces CMAKE_BUILD_TYPE to "Dev" when unset
    # for a single-config generator) but its non-warning flag is preserved verbatim via a
    # CMAKE_CXX_FLAGS shadow append (see the GCC Debug comment above for why this idiom does not
    # accumulate across reconfigures); the warning part of that same original branch moved into
    # libstats_apply_warnings' own else-branch.
    if(CMAKE_BUILD_TYPE STREQUAL "Release")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")

    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC: same restructuring as Clang/AppleClang above.
    if(CMAKE_BUILD_TYPE STREQUAL "Release")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")

    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")

    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LIBSTATS_OPT_FULL_UNIX}")
    endif()
else()
    # Unknown compiler - use sensible defaults. Untouched by Phase 3B/B5: this branch never applied
    # any LIBSTATS_* warning flags (nothing for libstats_apply_warnings to mirror here), and is
    # unreachable in this project regardless (the compiler-baseline check near the top of
    # CMakeLists.txt rejects any CXX_COMPILER_ID that is not AppleClang/Clang/GNU/MSVC via
    # FATAL_ERROR before this file is even included). Preserved verbatim.
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

# =============================================================================
# PER-TARGET WARNING APPLICATION (Phase 3B/B5)
# =============================================================================
# Applies this project's warning flags PRIVATE to a single target. Branches on compiler ID and
# CMAKE_BUILD_TYPE exactly mirroring the build-type dispatch the old global chain used above
# (including its else/default branches) -- only the WARNING flags live here now;
# optimization/debug-info flags are handled by CMake's standard per-config CMAKE_CXX_FLAGS_<CONFIG>
# mechanism above, which -- unlike the old global add_compile_options() chain -- also reaches
# FetchContent dependencies (GTest). Dev's -Wno-deprecated-declarations and MSVC Dev's /wd4996 are
# warning-control, so they live here rather than in the per-config cache variables.
#
# Do NOT apply this function to GTest targets (gtest/gtest_main or GTest::* imported targets) or to
# the INTERFACE libstats_headers target (INTERFACE libraries have no compilation of their own to
# apply PRIVATE flags to).
function(libstats_apply_warnings target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target} PRIVATE /W3)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE /W3)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
            target_compile_options(${target} PRIVATE /W3 /wd4996)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
            target_compile_options(${target} PRIVATE /W4 /WX /permissive-)
            target_compile_options(${target} PRIVATE ${LIBSTATS_MSVC_ENHANCED_WARNINGS})
        else()
            target_compile_options(${target} PRIVATE /W3)
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX}
                                                     -Wno-deprecated-declarations)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
            target_compile_options(${target} PRIVATE ${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
        else()
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_BUILD_TYPE STREQUAL "Release")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX}
                                                     -Wno-deprecated-declarations)
        elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
            target_compile_options(${target} PRIVATE ${LIBSTATS_GCC_STRICT_WARNINGS} -Werror)
        else()
            target_compile_options(${target} PRIVATE ${LIBSTATS_COMMON_WARNINGS_UNIX})
        endif()
    else()
        # Unknown compiler: the old global chain never applied any LIBSTATS_* warning flags for this
        # branch (see the untouched "unknown compiler" block above) -- nothing to mirror here.
    endif()
endfunction()
