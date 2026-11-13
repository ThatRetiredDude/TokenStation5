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
  # Extra preprocessor names for every shader, to build a variant of a
  # kernel for a comparison on the device (-DTS5_GLSL_DEFINES="A;B").
  set(TS5_GLSL_DEFINES "" CACHE STRING "names #defined in every compute shader")
  set(GLSL_DEFS)
  foreach(d ${TS5_GLSL_DEFINES})
    list(APPEND GLSL_DEFS -D${d})
  endforeach()
  set(SPV_HEADERS)
  foreach(src ${SHADERS})
    get_filename_component(stem ${src} NAME_WE)
    set(hdr ${CMAKE_CURRENT_BINARY_DIR}/shaders/${stem}.spv.h)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
      COMMAND ${GLSLANG} -V --target-env vulkan1.2 ${GLSL_DEFS} -I${CMAKE_CURRENT_SOURCE_DIR}/shaders
              --vn spv_${stem} -o ${hdr} ${src}
      DEPENDS ${src} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_tiled.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matmul_big.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/matvec_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mv16_q4_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q4k_sum.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mv16_q6_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mv16_q8_0.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvt_q4_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvtx_q4_k.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvt_q8_0.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mv16_q5_0.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/mvt_q5_0.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/q5_0.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/get_rows.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn_pf.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/attn_dec.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/qkv_post.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/kv_word.glsl ${CMAKE_CURRENT_SOURCE_DIR}/shaders/peak_fma.glsl
      COMMENT "glslang ${stem}.comp")
    list(APPEND SPV_HEADERS ${hdr})
  endforeach()
  add_custom_target(ts5shaders DEPENDS ${SPV_HEADERS})
  target_sources(ts5core PRIVATE src/vk_backend.cpp src/vk_context.cpp src/peak.cpp)
  add_dependencies(ts5core ts5shaders)
  target_include_directories(ts5core PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/shaders)
  target_include_directories(ts5core PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_link_libraries(ts5core PUBLIC ${CMAKE_DL_LIBS})
  target_compile_definitions(ts5core PUBLIC TS5_VULKAN=1)
endif()

add_executable(ts5 src/main.cpp src/cli.cpp src/checks.cpp src/tokenizer.cpp src/server.cpp src/engine.cpp
  src/tools.cpp src/known_good.cpp src/net.cpp src/websearch.cpp src/log.cpp src/browser.cpp src/access.cpp
  src/github.cpp)
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

  # Tailscale (tailnet_serve.h): the TV app's client and its WireGuard
  # tunnel (src/tailnet) over lwIP and wireguard-lwip (third_party/, BSD-3),
  # in user space.
  set(TS5_THIRD ${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
  file(GLOB TS5_LWIP ${TS5_THIRD}/lwip/src/core/*.c ${TS5_THIRD}/lwip/src/core/ipv4/*.c)
  list(REMOVE_ITEM TS5_LWIP ${TS5_THIRD}/lwip/src/core/udp.c)  # src/tailnet/wg_udp.c stands in
  file(GLOB TS5_WGL ${TS5_THIRD}/wireguard-lwip/src/*.c ${TS5_THIRD}/wireguard-lwip/src/crypto/refc/*.c)
  add_library(ts5tunnel STATIC ${TS5_LWIP} ${TS5_WGL} src/tailnet/wg_udp.c)
  set_target_properties(ts5tunnel PROPERTIES C_STANDARD 11 C_EXTENSIONS ON)
  target_compile_options(ts5tunnel PRIVATE -w)  # vendored, unchanged
  target_include_directories(ts5tunnel PUBLIC src/tailnet ${TS5_THIRD}/lwip/src/include ${TS5_THIRD}/wireguard-lwip/src)
  target_sources(ts5 PRIVATE src/tailnet_serve.cpp src/tailnet/wg.cpp src/tailnet/tailscale.cpp
    src/tailnet/noise.cpp src/tailnet/h2.cpp src/tailnet/nacl.cpp)
  target_compile_definitions(ts5 PRIVATE TS5_TAILNET=1 TS5_LOG_NAME="ts5")
  target_link_libraries(ts5 PRIVATE ts5tunnel)

  # The pages ts5 serve shows a browser (web_serve.h): the chat page, the
  # browser terminal, and xterm.js (third_party/xterm, MIT), built in.
  enable_language(ASM)
  file(GLOB TS5_WEB_FILES ${CMAKE_CURRENT_SOURCE_DIR}/web/*.html ${CMAKE_CURRENT_SOURCE_DIR}/web/*.png ${CMAKE_CURRENT_SOURCE_DIR}/web/*.js ${TS5_THIRD}/xterm/*)
  set_source_files_properties(web/web_assets.S PROPERTIES
    COMPILE_OPTIONS "-Wa,-I${CMAKE_CURRENT_SOURCE_DIR}/web;-Wa,-I${TS5_THIRD}/xterm"
    OBJECT_DEPENDS "${TS5_WEB_FILES}")
  target_sources(ts5 PRIVATE src/web_serve.cpp web/web_assets.S)
  target_link_libraries(ts5 PRIVATE util)  # forkpty
endif()
