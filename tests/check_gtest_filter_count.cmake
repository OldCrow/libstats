cmake_minimum_required(VERSION 3.25)
# #212 split guard: every case name given to libstats_split_timing_cases must select exactly one
# test. The _timing test's "Running 0 tests" check catches only a filter where every name is wrong;
# one misspelled name among several passed, and its case then ran in neither half. Lists the tests
# the filter selects and requires their count to equal the number of names. Inputs: -DBINARY=<test
# executable> -DFILTER=<a:b:c> -DEXPECTED=<number of names>
execute_process(
    COMMAND "${BINARY}" --gtest_list_tests "--gtest_filter=${FILTER}"
    OUTPUT_VARIABLE _out
    RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "${BINARY} --gtest_list_tests failed (${_rc})")
endif()
string(REGEX MATCHALL "\n  [A-Za-z_0-9/]+" _cases "\n${_out}")
list(LENGTH _cases _count)
if(NOT _count EQUAL EXPECTED)
    message(FATAL_ERROR "filter selects ${_count} of ${EXPECTED} named cases: ${FILTER}\n${_out}")
endif()
message(STATUS "filter selects all ${EXPECTED} named cases")
