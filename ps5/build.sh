#!/usr/bin/env bash
# Builds ts5 as a PS5 title, ready for ShadowMountPlus:
#
#   build/ps5/PPSA99555/eboot.bin               ts5 with RADV linked in
#   build/ps5/PPSA99555/sce_sys/param.json      title metadata, icon0.png
#   build/ps5/PPSA99555/sce_module/libc.prx     the runtime module titles carry
#   build/ps5/PPSA99555/LICENSE, NOTICE.md      the GPL and the third-party notices
#   build/ps5/PPSA99555/SOURCE.txt              the commits it was built from
#
# Needs ps5/setup.sh run once. The engine sources are the same ones the Linux
# build compiles; ps5/title.cpp is the title's main, with its jobs (runner,
# library) and its dashboard on the TV (ps5/ui).
#
#   ps5/build.sh
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vk="${PS5_VULKAN_ROOT:-${TS5_PS5_DEPS:-$root/.deps/ps5}/PS5_Vulkan}"
sdk="$vk/.deps/native/ps5-payload-sdk"
radv="$vk/.deps/native/radv-release"
archive="$radv/lib/libvulkan_radeon.ps5.a"
tool="$vk/build/host/ps5-native-tool"
param="$root/ps5/sce_sys/param.json"
out="$root/build/ps5"
for f in "$archive" "$tool" "$vk/runtime/libc.prx" "$sdk/bin/prospero-lld"; do
    [ -e "$f" ] || { echo "build: missing $f (run ps5/setup.sh)" >&2; exit 2; }
done
command -v glslangValidator > /dev/null || { echo "build: glslangValidator was not found" >&2; exit 2; }
version=$(git -C "$root" describe --always --dirty 2>/dev/null || echo dev)
title_id=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["titleId"])' "$param")
mkdir -p "$out/obj" "$out/shaders" "$out/stubs"

cc() { PS5_PAYLOAD_SDK="$sdk" sh "$vk/tooling/prospero-clang18" "$@"; }

# The compute shaders, as SPIR-V in C headers, as the CMake build makes them.
for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    hdr="$out/shaders/$stem.spv.h"
    if [ ! -f "$hdr" ] || [ -n "$(find "$root/engine/shaders" -newer "$hdr" -print -quit)" ]; then
        glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" -o "$hdr" "$src" > /dev/null
    fi
done

# Mbed TLS and the CA bundle for the web search tool's HTTPS (engine/deps.sh).
deps=${TS5_DEPS:-$root/.deps}
bash "$root/engine/deps.sh" "$deps"
mbedtls="$deps/mbedtls-3.6.7"
ca_pem="$deps/cacert-2026.07.22.pem"
mbedflags=(-I "$mbedtls/include" -I "$root/engine/src" "-DMBEDTLS_USER_CONFIG_FILE=\"ts5_mbedtls_config.h\"")

# The engine and the title, with exceptions and RTTI (the PS5 target turns
# them off by default; the engine's C++ library code uses both).
w="$root/third_party/whisper.cpp"
k="$root/third_party/kokoro.cpp"
cxxflags=(-std=c++17 -O2 -Wall -ffunction-sections -fdata-sections -fexceptions -fcxx-exceptions -frtti
    -DTS5_PS5=1 -DTS5_VULKAN=1 -DTS5_HTTPS=1 "-DTS5_VERSION=\"$version\"" "-DTS5_TITLE_ID=\"$title_id\""
    "-DTS5_CA_PEM=\"$ca_pem\"" "${mbedflags[@]}"
    -I "$root/engine/src" -I "$root/ps5" -I "$root/ps5/ui" -I "$out/shaders" -I "$radv/include"
    -I "$w/include" -I "$w/ggml/include" -I "$k/include" -I "$root/third_party/qrcodegen")
objects=()
pids=()
mkdir -p "$out/obj/mbedtls"
mbed_objects=()
for src in "$mbedtls"/library/*.c; do
    o="$out/obj/mbedtls/$(basename "$src" .c).o"
    mbed_objects+=("$o")
    if [ ! -f "$o" ] || [ "$src" -nt "$o" ] || [ "$root/engine/src/ts5_mbedtls_config.h" -nt "$o" ]; then
        cc -std=c11 -O2 -ffunction-sections -fdata-sections "${mbedflags[@]}" -I "$mbedtls/library" -c "$src" -o "$o" &
        pids+=($!)
    fi
done
for src in "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_context,vk_backend,tokenizer,engine,server,checks}.cpp \
        "$root"/engine/src/{tools,net,websearch,ca_bundle,log}.cpp \
        "$root"/ps5/{title,library,runner,audio_probe,chat,chat_store,stt,voice,settings,speak,text_input,usage_store,downloads,keys,hf_search,backup}.cpp \
        "$root"/ps5/{wg_devices,wg_page}.cpp "$root"/third_party/qrcodegen/qrcodegen.cpp \
        "$root"/ps5/ui/{canvas,display,pad,dashboard,chat_page,settings_page,home_page,logs_page}.cpp "$root"/ps5/tts/{g2p,text,numbers,spelling,synth}.cpp; do
    o="$out/obj/$(basename "$src" .cpp).o"
    objects+=("$o")
    cc "${cxxflags[@]}" -c "$src" -o "$o" &
    pids+=($!)
done
# whisper.cpp (third_party/whisper.cpp), CPU only, for speech to text
# (ps5/stt.cpp): its x86 code at the target's own znver2 (AVX2, FMA, F16C),
# no OpenMP (ggml's own threads), and CMake's defaults for the rest
# (third_party/whisper.cpp/VENDOR.md). kokoro.cpp (third_party/kokoro.cpp)
# on the same ggml, for speech out (ps5/tts/synth.cpp), reading its models
# with stdio (KOKORO_NO_MMAP).
wflags=(-O2 -ffunction-sections -fdata-sections -DNDEBUG -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE
    -DGGML_SCHED_MAX_COPIES=4 "-DWHISPER_VERSION=\"$(sed -n 's/^version=//p' "$root/scripts/vendor-whisper.sh")\""
    -DKOKORO_NO_MMAP -I "$w/include" -I "$w/ggml/include" -I "$w/ggml/src" -I "$w/ggml/src/ggml-cpu"
    -I "$k/include" -I "$k/src")
mkdir -p "$out/obj/speech"
wobjects=()
for src in "$w"/src/whisper.cpp "$w"/src/aneforge/whisper-aneforge.cpp \
        "$w"/ggml/src/{ggml.c,ggml.cpp,ggml-alloc.c,ggml-quants.c,ggml-threading.cpp,gguf.cpp} \
        "$w"/ggml/src/{ggml-backend,ggml-backend-dl,ggml-backend-meta,ggml-backend-reg}.cpp \
        "$w"/ggml/src/ggml-cpu/{ggml-cpu.c,ggml-cpu.cpp,quants.c,ops.cpp,binary-ops.cpp,unary-ops.cpp,vec.cpp,repack.cpp,traits.cpp} \
        "$w"/ggml/src/ggml-cpu/{tiled/tiled.cpp,tiled/tiled-kernel.cpp,llamafile/sgemm.cpp,arch/x86/quants.c,arch/x86/repack.cpp} \
        "$k"/src/{kokoro.cpp,core/gguf_loader.cpp,core/simd_math.cpp}; do
    rel=${src#"$root"/third_party/}
    o="$out/obj/speech/${rel//\//_}.o"
    wobjects+=("$o")
    # Clang 18 warns about the SDK's own isinf and isnan, each time ggml uses them.
    case $src in
        *.c) cc -std=c11 -Wno-unreachable-code-generic-assoc "${wflags[@]}" -c "$src" -o "$o" & ;;
        # kokoro.cpp asks for F32 matrix products with a call ggml 0.26 deprecates.
        "$k"/*) cc -std=c++17 -fexceptions -fcxx-exceptions -frtti -Wno-deprecated-declarations "${wflags[@]}" -c "$src" -o "$o" & ;;
        *) cc -std=c++17 -fexceptions -fcxx-exceptions -frtti "${wflags[@]}" -c "$src" -o "$o" & ;;
    esac
    pids+=($!)
done
# WireGuard inside the title (ps5/wg/wg.h): wireguard-lwip and lwIP
# (third_party/, scripts/vendor-wireguard.sh) with ps5/wg's settings, on the
# title's own UDP socket (ps5/wg/wg_udp.c stands in for lwIP's udp.c).
lw="$root/third_party/lwip/src"
wgl="$root/third_party/wireguard-lwip/src"
wgflags=(-O2 -ffunction-sections -fdata-sections -I "$root/ps5/wg" -I "$lw/include" -I "$wgl")
mkdir -p "$out/obj/wg"
wgobjects=()
for src in "$lw"/core/*.c "$lw"/core/ipv4/*.c "$wgl"/*.c "$wgl"/crypto/refc/*.c "$root"/ps5/wg/wg_udp.c "$root"/ps5/wg/wg.cpp \
        "$root"/ps5/ts/*.cpp; do
    [ "$src" = "$lw/core/udp.c" ] && continue
    rel=${src#"$root"/}
    o="$out/obj/wg/${rel//\//_}.o"
    wgobjects+=("$o")
    case $src in
        # gnu11: wg_udp.c reads the clocks, and lwIP wants ssize_t from the system.
        *.c) cc -std=gnu11 "${wgflags[@]}" -c "$src" -o "$o" & ;;
        *) cc "${cxxflags[@]}" "${wgflags[@]}" -c "$src" -o "$o" & ;;
    esac
    pids+=($!)
done
cc -std=c11 -O2 -Wall -ffunction-sections -fdata-sections -c "$root/ps5/compat.c" -o "$out/obj/compat.o" &
pids+=($!)
# The dashboard's fonts and logo, as they are (ps5/ui/make_font.py, make_logo.py).
objects+=("$out/obj/font_data.o")
cc -x assembler-with-cpp -Wa,-I"$root/ps5/ui" -c "$root/ps5/ui/font_data.S" -o "$out/obj/font_data.o" &
pids+=($!)
# The G2P's lexicon, as it is (scripts/pack-g2p-lexicon.py).
objects+=("$out/obj/lexicon_data.o")
cc -x assembler-with-cpp -Wa,-I"$root/ps5/tts" -c "$root/ps5/tts/lexicon_data.S" -o "$out/obj/lexicon_data.o" &
pids+=($!)
# PS5_Vulkan's title start code: IEEE floating point, the platform runtime,
# static constructors, then main.
cc -std=c++20 -O2 -fno-exceptions -fno-rtti -c "$vk/tooling/native/app_crt.cpp" -o "$out/obj/app_crt.o" &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
# In an archive, so --exclude-libs keeps its names local: the SDK's stub
# modules define some of them too, and the title converter refuses exports.
# whisper.cpp and kokoro.cpp likewise: their names stay inside the title.
rm -f "$out/obj/libts5compat.a" "$out/obj/libts5speech.a" "$out/obj/libts5wg.a"
"$sdk/bin/llvm-ar" rcs "$out/obj/libts5compat.a" "$out/obj/compat.o"
"$sdk/bin/llvm-ar" rcs "$out/obj/libts5speech.a" "${wobjects[@]}"
"$sdk/bin/llvm-ar" rcs "$out/obj/libts5wg.a" "${wgobjects[@]}"
rm -f "$out/obj/libts5mbedtls.a"
"$sdk/bin/llvm-ar" rcs "$out/obj/libts5mbedtls.a" "${mbed_objects[@]}"

# The AGC modules RADV's winsys submits through, named for the import tables.
stub() {
    cc -std=c11 -O2 -fPIC -c "$vk/vendor/ps5/sdk/stubs/$2" -o "$out/obj/$1_stub.o"
    "$sdk/bin/prospero-lld" --shared -soname "$1.prx" -o "$out/stubs/$1.so" "$out/obj/$1_stub.o"
}
stub libSceAgc agc_canary_link_stub.c
stub libSceAgcDriver agc_driver_canary_link_stub.c
# libSceAudioIn and libSceCommonDialog, which the SDK has no stubs for, named
# as the SDK names its own.
for m in libSceAudioIn libSceCommonDialog; do
    cc -std=c11 -O2 -fPIC -c "$root/ps5/stubs/$m.c" -o "$out/obj/${m}_stub.o"
    "$sdk/bin/prospero-lld" --shared -soname "$m.sprx" -o "$out/stubs/$m.so" "$out/obj/${m}_stub.o"
done

# The link PS5_Vulkan's RADV titles use: RADV whole, libc++, the platform
# layer's heap and threads, and its stand-ins for libc functions titles lack.
# shellcheck source=/dev/null
source "$vk/tools/radv-link.sh"
radv_link_recipe "$vk" "$sdk" "$archive" || exit 2
# Mesa names every Vulkan entry point through a weak reference that must read
# as NULL, and the title converter takes every undefined dynamic symbol for an
# SDK import: --no-dynamic-linker keeps the weak ones out of the dynamic table
# on LLD 18, -z nodynamic-undefined-weak on newer LLD.
weak=(--no-dynamic-linker)
"$sdk/bin/ld.lld" -z nodynamic-undefined-weak --version 2>&1 | grep -q "unknown -z" ||
    weak+=(-z nodynamic-undefined-weak)
"$sdk/bin/prospero-lld" "${radv_linker_script[@]}" --eh-frame-hdr "${weak[@]}" "${radv_link_flags[@]}" \
    --version-script "$vk/tooling/native/app-symbols.map" --exclude-libs=ALL \
    -e _start -o "$out/llvm-pie.elf" \
    "$out/obj/app_crt.o" "${objects[@]}" "$out/obj/libts5speech.a" "$out/obj/libts5wg.a" "$out/obj/libts5compat.a" "$out/obj/libts5mbedtls.a" \
    "$out/stubs/libSceAgc.so" "$out/stubs/libSceAgcDriver.so" "$out/stubs/libSceAudioIn.so" \
    "$out/stubs/libSceCommonDialog.so" "${radv_link_inputs[@]}" \
    --as-needed "$sdk"/target/lib/*.so
# An import that only a module titles get nothing from provides is a call
# through NULL at run time: refuse to ship one.
python3 "$root/ps5/check-imports.py" "$sdk" "$out/llvm-pie.elf" "$out/stubs"
"$tool" link --in "$out/llvm-pie.elf" --out "$out/eboot.elf" \
    --stub-dir "$sdk/target/lib" --stub "$out/stubs/libSceAgc.so" \
    --stub "$out/stubs/libSceAgcDriver.so" --stub "$out/stubs/libSceAudioIn.so" \
    --stub "$out/stubs/libSceCommonDialog.so" --module-sdk 0x02000009 \
    --companion-sdk 0x08050001 --file-name eboot.elf > /dev/null

app="$out/$title_id"
rm -rf -- "$app"
mkdir -p "$app/sce_sys" "$app/sce_module"
"$tool" self --sign --in "$out/eboot.elf" --out "$app/eboot.bin" --magic 0x1D3D154F > /dev/null
cp "$param" "$root/ps5/sce_sys/icon0.png" "$app/sce_sys/"
cp "$vk/runtime/libc.prx" "$app/sce_module/libc.prx"
# What every copy of the title must carry (LICENSING.md): its license, the
# notices, and where its source is.
cp "$root/ps5/LICENSE" "$root/NOTICE.md" "$app/"
commit() { git -C "$1" rev-parse HEAD 2> /dev/null || echo "$2"; }
# CI names a pull request's two commits: it builds their merge, which only the
# pull request's ref keeps.
ts5_source=${TS5_SOURCE_COMMITS:-$(commit "$root" "$version")}
cat > "$app/SOURCE.txt" << EOF
TokenStation5 $version is GPL-3.0-or-later as a whole (LICENSE). NOTICE.md
lists the third-party code in it. Its source is these repositories at these
commits:

https://github.com/ThatRetiredDude/TokenStation5  $ts5_source
https://github.com/mihawk-99/PS5_Vulkan  $(commit "$vk" "$(sed -n 's/^vulkan_rev=//p' "$root/ps5/setup.sh")")
https://github.com/mihawk-99/PS5_PayloadSDK  $(sed -n 's/^sdk_revision=//p' "$vk/tools/setup-native-dependencies.sh")
https://github.com/mihawk-99/PS5_Mesa  $(sed -n 's/^mesa_revision=//p' "$vk/tools/build-radv.sh")
EOF
"$tool" self --inspect --file "$app/eboot.bin" > /dev/null
echo "build: $app (ts5 $version, eboot.bin $(stat -c %s "$app/eboot.bin") bytes)"
