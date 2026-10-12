cmake_minimum_required(VERSION 3.25) # IN_LIST (CMP0057) in script mode
# AR D19 / #189 class: an `inline` member-function definition in a .cpp file is ill-formed (no
# diagnostic required) once any other translation unit calls it directly; Poisson's four virtual
# overrides linked only because the vtable forced their emission. Fails on any such definition in
# src/*.cpp outside the allowlist below.
#
# Allowlist: empty. The four private static helpers once listed here (Discrete's roundToInt and
# isValidIntegerValue, Poisson's roundToNonNegativeInt and isValidCount) dropped `inline`
# (2026-10-11): it bought nothing inside one TU and made a later call from another TU ill-formed.
# Add an entry only for a helper that must stay inline; prefer dropping the keyword.
set(_allowed "")
file(GLOB _sources "${SOURCE_DIR}/src/*.cpp")
set(_bad "")
# Matched over the whole file, not line by line: `inline` may sit on its own line above the
# qualified name, follow an attribute such as [[nodiscard]], or be indented. The excluded characters
# keep a match inside one declaration; a line comment never starts a match.
set(_pattern "(^|\n)[ \t]*(\\[\\[[^]\n]*\\]\\][ \t]*)*inline[ \t\n]+[^(;={}]*")
string(APPEND _pattern "[A-Za-z_][A-Za-z_0-9]*::[A-Za-z_~][A-Za-z_0-9]*\\(")
foreach(_file IN LISTS _sources)
    file(READ "${_file}" _text)
    string(REGEX MATCHALL "${_pattern}" _lines "${_text}")
    foreach(_line IN LISTS _lines)
        string(REGEX REPLACE "[ \t\n]+" " " _line "${_line}")
        string(STRIP "${_line}" _line)
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
