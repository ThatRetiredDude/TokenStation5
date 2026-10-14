cmake_minimum_required(VERSION 3.16)
project(tokenstation5 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TS5_VULKAN "Build the Vulkan backend" ON)

add_library(ts5core STATIC
  src/gguf.cpp src/quants.cpp src/model.cpp src/cpu_backend.cpp)
target_include_directories(ts5core PUBLIC src)
find_package(Threads REQUIRED)
target_link_libraries(ts5core PUBLIC Threads::Threads)

if(TS5_VULKAN)
  find_package(Vulkan REQUIRED)
  find_program(GLSLANG glslangValidator REQUIRED)
  file(GLOB SHADERS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.comp)
  set(SPV_HEADERS)
  foreach(src ${SHADERS})
    get_filename_component(stem ${src} NAME_WE)
    set(hdr ${CMAKE_CURRENT_BINARY_DIR}/shaders/${stem}.spv.h)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLANG} -V --target-env vulkan1.2 -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
              --vn spv_${stem} -o ${hdr} ${src}
      DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/get_rows.glsl
      COMMENT "glslang ${stem}.comp")
    list(APPEND SPV_HEADERS ${hdr})
  endforeach()
  add_custom_target(ts5shaders DEPENDS ${SPV_HEADERS})
  target_sources(ts5core PRIVATE src/vk_backend.cpp)
  add_dependencies(ts5core ts5shaders)
  target_include_directories(ts5core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  target_include_directories(ts5core PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(ts5core PUBLIC ${CMAKE_DL_LIBS})
  target_compile_definitions(ts5core PUBLIC TS5_VULKAN=1)
endif()

add_executable(ts5 src/main.cpp src/cli.cpp src/tokenizer.cpp src/server.cpp src/engine.cpp)
target_link_libraries(ts5 PRIVATE ts5core)
