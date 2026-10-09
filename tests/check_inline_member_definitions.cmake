# AR D19 / #189 class: an `inline` member-function definition in a .cpp file is ill-formed (no
# diagnostic required) once any other translation unit calls it directly; Poisson's four virtual
# overrides linked only because the vtable forced their emission. Fails on any such definition in
# src/*.cpp outside the allowlist below.
#
# Allowlist: private static helpers whose only callers are in their own .cpp.
set(_allowed "DiscreteDistribution::roundToInt" "DiscreteDistribution::isValidIntegerValue"
             "PoissonDistribution::roundToNonNegativeInt" "PoissonDistribution::isValidCount")
file(GLOB _sources "${SOURCE_DIR}/src/*.cpp")
set(_bad "")
foreach(_file IN LISTS _sources)
    file(STRINGS "${_file}" _lines REGEX "^inline [^(=]*[A-Za-z_]::[A-Za-z_~]+\\(")
    foreach(_line IN LISTS _lines)
        string(REGEX MATCH "[A-Za-z_][A-Za-z_0-9]*::[A-Za-z_~][A-Za-z_0-9]*\\(" _name "${_line}")
        string(REGEX REPLACE "\\($" "" _name "${_name}")
        if(NOT _name IN_LIST _allowed)
            get_filename_component(_base "${_file}" NAME)
            list(APPEND _bad "${_base}: ${_line}")
        endif()
    endforeach()
endforeach()
if(_bad)
    list(JOIN _bad "\n  " _msg)
    message(FATAL_ERROR "inline member definitions in src/*.cpp:\n  ${_msg}")
endif()
message(STATUS "no inline member definitions outside the allowlist")
