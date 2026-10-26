cmake_minimum_required(VERSION 3.16)
project(tokenstation5 C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(TS5_VULKAN "Build the Vulkan backend" ON)
option(TS5_HTTPS "HTTPS for the web search tool (Mbed TLS, fetched by deps.sh)" ON)

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
      DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_tiled.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_big.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mv16_q4_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/get_rows.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn_pf.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn_dec.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qkv_post.glsl
      COMMENT "glslang ${stem}.comp")
    list(APPEND SPV_HEADERS ${hdr})
  endforeach()
  add_custom_target(ts5shaders DEPENDS ${SPV_HEADERS})
  target_sources(ts5core PRIVATE src/vk_backend.cpp src/vk_context.cpp)
  add_dependencies(ts5core ts5shaders)
  target_include_directories(ts5core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  target_include_directories(ts5core PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(ts5core PUBLIC ${CMAKE_DL_LIBS})
  target_compile_definitions(ts5core PUBLIC TS5_VULKAN=1)
endif()

add_executable(ts5 src/main.cpp src/cli.cpp src/checks.cpp src/tokenizer.cpp src/server.cpp src/engine.cpp
  src/tools.cpp src/net.cpp src/websearch.cpp src/log.cpp)
target_link_libraries(ts5 PRIVATE ts5core)

if(TS5_HTTPS)
  # Mbed TLS and the CA bundle, pinned and hash-checked (deps.sh), built in.
  set(TS5_DEPS ${CMAKE_CURRENT_SOURCE_DIR}/../.deps CACHE PATH "where deps.sh puts what it fetches")
  execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/deps.sh ${TS5_DEPS} RESULT_VARIABLE deps_rc)
  if(NOT deps_rc EQUAL 0)
    message(FATAL_ERROR "engine/deps.sh failed; or configure with -DTS5_HTTPS=OFF")
  endif()
  set(MBEDTLS ${TS5_DEPS}/mbedtls-3.6.7)
  set(CA_PEM ${TS5_DEPS}/cacert-2026.07.22.pem)
  file(GLOB MBEDTLS_SOURCES ${MBEDTLS}/library/*.c)
  add_library(ts5mbedtls STATIC ${MBEDTLS_SOURCES})
  target_include_directories(ts5mbedtls PUBLIC ${MBEDTLS}/include PRIVATE ${MBEDTLS}/library src)
  target_compile_definitions(ts5mbedtls PUBLIC MBEDTLS_USER_CONFIG_FILE="ts5_mbedtls_config.h")
  target_sources(ts5 PRIVATE src/ca_bundle.cpp)
  set_source_files_properties(src/ca_bundle.cpp PROPERTIES
    COMPILE_DEFINITIONS TS5_CA_PEM="${CA_PEM}" OBJECT_DEPENDS ${CA_PEM})
  target_compile_definitions(ts5 PRIVATE TS5_HTTPS=1)
  target_include_directories(ts5 PRIVATE src)
  target_link_libraries(ts5 PRIVATE ts5mbedtls)
endif()
