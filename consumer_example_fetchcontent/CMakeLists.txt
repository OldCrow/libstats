cmake_minimum_required(VERSION 3.20)
project(libstats_fetchcontent_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch libstats from GitHub (or use a local path for development)
FetchContent_Declare(
    libstats
    GIT_REPOSITORY https://github.com/OldCrow/libstats.git
    GIT_TAG main)

# To use a local checkout instead of fetching from GitHub, replace the FetchContent_Declare above
# with: FetchContent_Declare(libstats SOURCE_DIR /path/to/libstats)

FetchContent_MakeAvailable(libstats)

add_executable(consumer_demo main.cpp)
target_link_libraries(consumer_demo PRIVATE libstats_static)
