#!/usr/bin/env bash
# Builds TokenStation5 for Linux: the PS5 title's own code (ps5/) with the
# console's functions from SDL2 and the system (linux/platform.cpp), the
# engine's command line (ts5) beside it, and the terminal app (ts5-tui).
#
#   linux/build.sh            # build/linux/tokenstation, ts5 and ts5-tui
#
# Needs: a C++17 compiler, glslang-tools, libvulkan-dev, libsdl2-dev, curl
# (engine/deps.sh fetches Mbed TLS and the CA bundle, hash-checked).
#
# TS5_MARCH picks the instructions it uses: x86-64-v3 (AVX2 and FMA, which
# the PS5's Zen 2 has, as has any x86 PC from the last ten years) by default,
# znver2 for the PS5 alone, native for this machine. The code is scheduled
# for Zen 2 (-mtune=znver2) either way, unless native.
# Third-party objects (Mbed TLS, whisper.cpp and kokoro.cpp, lwIP and
# WireGuard) are kept and compiled again only when their sources or the
# flags change; TokenStation5's own files are compiled every time.
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
out="${TS5_LINUX_OUT:-$root/build/linux}"
march="${TS5_MARCH:-x86-64-v3}"
jobs="${TS5_JOBS:-$(nproc)}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"
for tool in glslangValidator pkg-config "$CC" "$CXX"; do
    command -v "$tool" > /dev/null || { echo "build: $tool was not found (see linux/README.md)" >&2; exit 2; }
done
pkg-config --exists sdl2 || { echo "build: SDL2 was not found: sudo apt install libsdl2-dev" >&2; exit 2; }
[ -f /usr/include/vulkan/vulkan.h ] || pkg-config --exists vulkan ||
    { echo "build: the Vulkan headers were not found: sudo apt install libvulkan-dev" >&2; exit 2; }
version=${TS5_VERSION:-$(git -C "$root" describe --tags --always --dirty 2>/dev/null || echo dev)}
mkdir -p "$out/obj" "$out/shaders"

for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    hdr="$out/shaders/$stem.spv.h"
    if [ ! -f "$hdr" ] || [ -n "$(find "$root/engine/shaders" -newer "$hdr" -print -quit)" ]; then
        glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" -o "$hdr" "$src" > /dev/null
    fi
done

deps=${TS5_DEPS:-$root/.deps}
bash "$root/engine/deps.sh" "$deps"
mbedtls="$deps/mbedtls-3.6.7"
ca_pem="$deps/cacert-2026.07.22.pem"
mbedflags=(-I "$mbedtls/include" -I "$root/engine/src" "-DMBEDTLS_USER_CONFIG_FILE=\"ts5_mbedtls_config.h\"")

w="$root/third_party/whisper.cpp"
k="$root/third_party/kokoro.cpp"
lw="$root/third_party/lwip/src"
wgl="$root/third_party/wireguard-lwip/src"
opt=(-O3 "-march=$march" -pthread -fno-plt)
[ "$march" = native ] || opt+=(-mtune=znver2)
# whisper.cpp's AMX kernels aren't vendored: never build for them.
[ "$march" = native ] && opt+=(-mno-amx-tile -mno-amx-int8 -mno-amx-bf16)
sdl_cflags=($(pkg-config --cflags sdl2))
cxxflags=(-std=c++17 "${opt[@]}" -Wall -DTS5_LINUX=1 -DTS5_VULKAN=1 -DTS5_HTTPS=1
    "-DTS5_VERSION=\"$version\"" "-DTS5_HOME=\".\"" "-DTS5_DATA=\"./data\"" "-DTS5_STORE=\"./store\""
    "-DTS5_CA_PEM=\"$ca_pem\"" "${mbedflags[@]}" "${sdl_cflags[@]}"
    -I "$root/engine/src" -I "$root/ps5" -I "$root/ps5/ui" -I "$root/linux" -I "$out/shaders"
    -I "$w/include" -I "$w/ggml/include" -I "$k/include" -I "$root/third_party/qrcodegen")

# Runs up to $jobs commands at once; `wait_all` fails if any did.
pids=()
run() {
    while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do wait -n || true; done
    "$@" &
    pids+=($!)
}
wait_all() {
    local bad=0
    for p in "${pids[@]}"; do wait "$p" || bad=1; done
    pids=()
    [ "$bad" = 0 ] || { echo "build: a compile failed (above)" >&2; exit 1; }
}

# A kept object is compiled again when its source is newer or the flags
# changed (a stamp per group of objects).
stale() {  # object source stamp
    [ ! -f "$1" ] || [ "$2" -nt "$1" ] || [ "$3" -nt "$1" ]
}
stamp() {  # group flags...
    local f="$out/obj/$1.flags"
    shift
    if [ "$(cat "$f" 2> /dev/null)" != "$*" ]; then printf '%s' "$*" > "$f"; fi
    echo "$f"
}

t0=$(date +%s)
mkdir -p "$out/obj/mbedtls" "$out/obj/speech" "$out/obj/wg" "$out/obj/own"
st=$(stamp mbedtls "$CC" "${opt[@]}" "${mbedflags[@]}")
mbed_objects=()
for src in "$mbedtls"/library/*.c; do
    o="$out/obj/mbedtls/$(basename "$src" .c).o"
    mbed_objects+=("$o")
    stale "$o" "$src" "$st" || [ "$root/engine/src/ts5_mbedtls_config.h" -nt "$o" ] || continue
    run "$CC" -std=c11 "${opt[@]}" "${mbedflags[@]}" -I "$mbedtls/library" -c "$src" -o "$o"
done

wflags=("${opt[@]}" -D_GNU_SOURCE -DNDEBUG -DGGML_USE_CPU -DGGML_USE_CPU_REPACK -DGGML_USE_LLAMAFILE
    -DGGML_SCHED_MAX_COPIES=4 "-DWHISPER_VERSION=\"$(sed -n 's/^version=//p' "$root/scripts/vendor-whisper.sh")\""
    -DKOKORO_NO_MMAP -I "$w/include" -I "$w/ggml/include" -I "$w/ggml/src" -I "$w/ggml/src/ggml-cpu"
    -I "$k/include" -I "$k/src")
st=$(stamp speech "$CC" "$CXX" "${wflags[@]}")
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
    stale "$o" "$src" "$st" || continue
    case $src in
        *.c) run "$CC" -std=c11 "${wflags[@]}" -c "$src" -o "$o" ;;
        "$k"/*) run "$CXX" -std=c++17 -Wno-deprecated-declarations "${wflags[@]}" -c "$src" -o "$o" ;;
        *) run "$CXX" -std=c++17 "${wflags[@]}" -c "$src" -o "$o" ;;
    esac
done

wgflags=("${opt[@]}" -I "$root/engine/src/tailnet" -I "$lw/include" -I "$wgl")
st=$(stamp wg "$CC" "${wgflags[@]}")
wgobjects=()
for src in "$lw"/core/*.c "$lw"/core/ipv4/*.c "$wgl"/*.c "$wgl"/crypto/refc/*.c "$root"/engine/src/tailnet/wg_udp.c; do
    [ "$src" = "$lw/core/udp.c" ] && continue
    rel=${src#"$root"/}
    o="$out/obj/wg/${rel//\//_}.o"
    wgobjects+=("$o")
    stale "$o" "$src" "$st" || continue
    run "$CC" -std=gnu11 "${wgflags[@]}" -c "$src" -o "$o"
done

# TokenStation5's own files, every time.
title_objects=()
engine_objects=()
for src in "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_context,vk_backend,peak,tokenizer,engine,server,checks}.cpp \
        "$root"/engine/src/{tools,known_good,net,websearch,ca_bundle,log}.cpp; do
    o="$out/obj/own/engine_$(basename "$src" .cpp).o"
    engine_objects+=("$o")
    run "$CXX" "${cxxflags[@]}" -c "$src" -o "$o"
done
for src in "$root"/ps5/{title,library,runner,audio_probe,chat,chat_store,stt,voice,settings,speak,text_input,usage_store,downloads,keys,hf_search,backup,local_time,keep_models}.cpp \
        "$root"/engine/src/tailnet/wg.cpp "$root"/engine/src/tailnet/{tailscale,noise,h2,nacl}.cpp "$root"/third_party/qrcodegen/qrcodegen.cpp \
        "$root"/ps5/ui/{canvas,pad,dashboard,chat_page,settings_page,home_page,logs_page}.cpp \
        "$root"/ps5/tts/{g2p,text,numbers,spelling,synth}.cpp "$root"/linux/{platform,display,hardware}.cpp; do
    rel=${src#"$root"/}
    o="$out/obj/own/${rel//\//_}.o"
    title_objects+=("$o")
    run "$CXX" "${cxxflags[@]}" -I "$root/engine/src/tailnet" -I "$lw/include" -I "$wgl" -c "$src" -o "$o"
done
for src in "$root"/linux/tui/*.cpp; do
    o="$out/obj/own/tui_$(basename "$src" .cpp).o"
    run "$CXX" "${cxxflags[@]}" -I "$root/linux/tui" -c "$src" -o "$o"
done
for src in "$root"/engine/src/{main,cli,browser,access,tailnet_serve}.cpp; do
    o="$out/obj/own/cli_$(basename "$src" .cpp).o"
    run "$CXX" "${cxxflags[@]}" -DTS5_TAILNET=1 -c "$src" -o "$o"
done
# ts5's own copy of the Tailscale client and its tunnel (tailnet_serve.h),
# its log lines starting "ts5:".
for src in "$root"/engine/src/tailnet/{wg,tailscale,noise,h2,nacl}.cpp; do
    o="$out/obj/own/ts5net_$(basename "$src" .cpp).o"
    run "$CXX" "${cxxflags[@]}" "-DTS5_LOG_NAME=\"ts5\"" -I "$root/engine/src/tailnet" -I "$lw/include" -I "$wgl" \
        -c "$src" -o "$o"
done
run "$CC" -x assembler-with-cpp -Wa,-I"$root/ps5/ui" -c "$root/ps5/ui/font_data.S" -o "$out/obj/own/font_data.o"
run "$CC" -x assembler-with-cpp -Wa,-I"$root/ps5/tts" -c "$root/ps5/tts/lexicon_data.S" -o "$out/obj/own/lexicon_data.o"
wait_all
echo "== compiled in $(($(date +%s) - t0)) s"

ar_if_stale() {  # archive objects...
    local a=$1
    shift
    if [ ! -f "$a" ] || [ -n "$(find "$@" -newer "$a" -print -quit)" ]; then
        rm -f "$a"
        ar rcs "$a" "$@"
    fi
}
ar_if_stale "$out/obj/libts5mbedtls.a" "${mbed_objects[@]}"
ar_if_stale "$out/obj/libts5speech.a" "${wobjects[@]}"
ar_if_stale "$out/obj/libts5wg.a" "${wgobjects[@]}"

"$CXX" "${opt[@]}" -o "$out/tokenstation" "${engine_objects[@]}" "${title_objects[@]}" \
    "$out/obj/own/font_data.o" "$out/obj/own/lexicon_data.o" \
    "$out/obj/libts5wg.a" "$out/obj/libts5speech.a" "$out/obj/libts5mbedtls.a" \
    $(pkg-config --libs sdl2) -ldl
"$CXX" "${opt[@]}" -o "$out/ts5" "$out/obj/own/cli_main.o" "$out/obj/own/cli_cli.o" "$out/obj/own/cli_browser.o" \
    "$out/obj/own/cli_access.o" "$out/obj/own/cli_tailnet_serve.o" "$out"/obj/own/ts5net_*.o "${engine_objects[@]}" \
    "$out/obj/libts5wg.a" "$out/obj/libts5mbedtls.a" -ldl
"$CXX" "${opt[@]}" -o "$out/ts5-tui" "$out"/obj/own/tui_*.o "$out/obj/own/third_party_qrcodegen_qrcodegen.cpp.o"
echo "build: $out/tokenstation, $out/ts5 and $out/ts5-tui ($version, tuned for $march)"
