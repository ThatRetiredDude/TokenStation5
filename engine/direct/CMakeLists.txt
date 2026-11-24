# The direct engine: ts5 driving the PS5 GPU itself, with no Vulkan, Mesa
# or compiler. Built on its own for now (it joins the ts5 binary once it
# runs models):
#   cmake -S engine/direct -B build/direct && cmake --build build/direct
#   python3 engine/direct/tests/asm_check.py build/direct/ts5-asm
cmake_minimum_required(VERSION 3.16)
project(ts5direct CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

add_library(ts5gfx10 STATIC gfx10_asm.cpp)
target_include_directories(ts5gfx10 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ts5-asm ts5_asm_main.cpp)
target_link_libraries(ts5-asm ts5gfx10)
