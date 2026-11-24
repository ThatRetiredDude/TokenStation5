# The direct engine: ts5 driving the PS5 GPU itself, with no Vulkan, Mesa
# or compiler. Built on its own for now (it joins the ts5 binary once it
# runs models):
#   cmake -S engine/direct -B build/direct && cmake --build build/direct
#   python3 engine/direct/tests/asm_check.py build/direct/ts5-asm
cmake_minimum_required(VERSION 3.21)
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

# ts5-direct: the runtime test tool (info, fill, read-bw). Linked with a
# static C++ library so the CI build runs on the PS5's Debian as it is.
file(GLOB kernel_srcs "${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.s")
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.h
  COMMAND ${CMAKE_COMMAND} -DSRC_DIR=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${CMAKE_CURRENT_BINARY_DIR}/kernels.h
          -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_kernels.cmake
  DEPENDS ${kernel_srcs} ${CMAKE_CURRENT_SOURCE_DIR}/embed_kernels.cmake)
# ts5's model reader, quantization, tokenizer and CPU backend: the token
# command loads models with them and checks its logits against the CPU.
set(ENGINE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../src)
find_package(Threads REQUIRED)
add_library(ts5model STATIC ${ENGINE_SRC}/gguf.cpp ${ENGINE_SRC}/model.cpp ${ENGINE_SRC}/log.cpp ${ENGINE_SRC}/quants.cpp
            ${ENGINE_SRC}/cpu_backend.cpp ${ENGINE_SRC}/tokenizer.cpp)
target_include_directories(ts5model PUBLIC ${ENGINE_SRC})
target_compile_options(ts5model PRIVATE -w)
target_link_libraries(ts5model PUBLIC Threads::Threads)
add_executable(ts5-direct ts5_direct_main.cpp token.cpp runner.cpp kfd.cpp ${CMAKE_CURRENT_BINARY_DIR}/kernels.h)
target_include_directories(ts5-direct PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ts5-direct ts5gfx10 ts5model -static-libstdc++ -static-libgcc)
