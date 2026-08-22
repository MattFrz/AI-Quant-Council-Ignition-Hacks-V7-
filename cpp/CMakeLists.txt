# Step 4.1. Used for the standalone test binary; the Python extension is built
# through cpp/bindings/setup.py instead, which is the path that matters.
cmake_minimum_required(VERSION 3.15)
project(aqc_exec CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/O2 /W4)
else()
  add_compile_options(-O3 -Wall -Wextra)
endif()

add_library(aqc_core STATIC
  orderbook/orderbook.cpp
  orderbook/itch_parser.cpp
  execution/execution_sim.cpp
)
target_include_directories(aqc_core PUBLIC include orderbook execution)

# Standalone assertions, no test framework needed.
add_executable(test_orderbook ../tests/cpp/test_orderbook.cpp)
target_link_libraries(test_orderbook PRIVATE aqc_core)

enable_testing()
add_test(NAME orderbook COMMAND test_orderbook)
