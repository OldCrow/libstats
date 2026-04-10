cmake_minimum_required(VERSION 3.20)
project(libstats_consumer_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the installed libstats package
find_package(libstats REQUIRED)

add_executable(consumer_demo main.cpp)
target_link_libraries(consumer_demo PRIVATE libstats::libstats_static)
