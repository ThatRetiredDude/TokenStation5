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
#
# TS5_RUNTIME picks the title runtime (start code, libc stand-ins, heap,
# threads, AGC stubs, libc.prx):
#   gpl  PS5_Vulkan's and the payload SDK's (the default)
#   own  ours, in ps5/rt (clean-room; not finished yet, see ps5/rt/README.md)
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
vk="${PS5_VULKAN_ROOT:-${TS5_PS5_DEPS:-$root/.deps/ps5}/PS5_Vulkan}"
sdk="$vk/.deps/native/ps5-payload-sdk"
radv="$vk/.deps/native/radv-release"
archive="$radv/lib/libvulkan_radeon.ps5.a"
tool="$vk/build/host/ps5-native-tool"
param="$root/ps5/sce_sys/param.json"
out="$root/build/ps5"
runtime="${TS5_RUNTIME:-gpl}"
case "$runtime" in
    gpl) needs=("$vk/runtime/libc.prx") ;;
    own) # shellcheck source=ps5/rt/runtime.sh
        source "$root/ps5/rt/runtime.sh"
        needs=() ;;
    *) echo "build: TS5_RUNTIME must be gpl or own, not $runtime" >&2; exit 2 ;;
esac
for f in "$archive" "$tool" "${needs[@]}" "$sdk/bin/prospero-lld"; do
    [ -e "$f" ] || { echo "build: missing $f (run ps5/setup.sh)" >&2; exit 2; }
done
command -v glslangValidator > /dev/null || { echo "build: glslangValidator was not found" >&2; exit 2; }
version=$(git -C "$root" describe --always --dirty 2>/dev/null || echo dev)
title_id=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["titleId"])' "$param")
mkdir -p "$out/obj" "$out/shaders" "$out/stubs"

cc() { PS5_PAYLOAD_SDK="$sdk" sh "$vk/tooling/prospero-clang18" "$@"; }
[ "$runtime" = gpl ] || rt_check

# The compute shaders, as SPIR-V in C headers, as the CMake build makes them.
for src in "$root"/engine/shaders/*.comp; do
    stem=$(basename "$src" .comp)
    hdr="$out/shaders/$stem.spv.h"
    if [ ! -f "$hdr" ] || [ -n "$(find "$root/engine/shaders" -newer "$hdr" -print -quit)" ]; then
        glslangValidator -V --target-env vulkan1.2 -I"$root/engine/shaders" --vn "spv_$stem" -o "$hdr" "$src" > /dev/null
    fi
done

# The engine and the title, with exceptions and RTTI (the PS5 target turns
# them off by default; the engine's C++ library code uses both).
cxxflags=(-std=c++17 -O2 -Wall -ffunction-sections -fdata-sections -fexceptions -fcxx-exceptions -frtti
    -DTS5_PS5=1 -DTS5_VULKAN=1 "-DTS5_VERSION=\"$version\"" "-DTS5_TITLE_ID=\"$title_id\""
    -I "$root/engine/src" -I "$root/ps5" -I "$root/ps5/ui" -I "$out/shaders" -I "$radv/include")
objects=()
pids=()
for src in "$root"/engine/src/{gguf,quants,model,cpu_backend,vk_context,vk_backend,tokenizer,engine,server,checks}.cpp \
        "$root"/ps5/{title,library,runner}.cpp "$root"/ps5/ui/{canvas,display,pad,dashboard}.cpp; do
    o="$out/obj/$(basename "$src" .cpp).o"
    objects+=("$o")
    cc "${cxxflags[@]}" -c "$src" -o "$o" &
    pids+=($!)
done
cc -std=c11 -O2 -Wall -ffunction-sections -fdata-sections -c "$root/ps5/compat.c" -o "$out/obj/compat.o" &
pids+=($!)
# The dashboard's fonts and logo, as they are (ps5/ui/make_font.py, make_logo.py).
objects+=("$out/obj/font_data.o")
cc -x assembler-with-cpp -Wa,-I"$root/ps5/ui" -c "$root/ps5/ui/font_data.S" -o "$out/obj/font_data.o" &
pids+=($!)
# The title start code: IEEE floating point, the platform runtime, static
# constructors, then main. PS5_Vulkan's, or ours.
if [ "$runtime" = gpl ]; then
    cc -std=c++20 -O2 -fno-exceptions -fno-rtti -c "$vk/tooling/native/app_crt.cpp" -o "$out/obj/app_crt.o" &
    pids+=($!)
else
    rt_crt "$out/obj/app_crt.o"
fi
for p in "${pids[@]}"; do wait "$p"; done
# In an archive, so --exclude-libs keeps its names local: the SDK's stub
# modules define some of them too, and the title converter refuses exports.
rm -f "$out/obj/libts5compat.a"
"$sdk/bin/llvm-ar" rcs "$out/obj/libts5compat.a" "$out/obj/compat.o"

# The AGC modules RADV's winsys submits through, named for the import tables.
stub() {
    cc -std=c11 -O2 -fPIC -c "$vk/vendor/ps5/sdk/stubs/$2" -o "$out/obj/$1_stub.o"
    "$sdk/bin/prospero-lld" --shared -soname "$1.prx" -o "$out/stubs/$1.so" "$out/obj/$1_stub.o"
}
if [ "$runtime" = gpl ]; then
    stub libSceAgc agc_canary_link_stub.c
    stub libSceAgcDriver agc_driver_canary_link_stub.c
else
    rt_stubs "$out/stubs"
fi

# The link RADV titles need: RADV whole, libc++, the platform layer's heap
# and threads, and stand-ins for libc functions titles lack. PS5_Vulkan's
# recipe, or ours; either sets the radv_link_* arrays and symbols_map.
if [ "$runtime" = gpl ]; then
    # shellcheck source=/dev/null
    source "$vk/tools/radv-link.sh"
    radv_link_recipe "$vk" "$sdk" "$archive" || exit 2
    symbols_map="$vk/tooling/native/app-symbols.map"
else
    rt_link_recipe "$archive" || exit 2
fi
# Mesa names every Vulkan entry point through a weak reference that must read
# as NULL, and the title converter takes every undefined dynamic symbol for an
# SDK import: --no-dynamic-linker keeps the weak ones out of the dynamic table
# on LLD 18, -z nodynamic-undefined-weak on newer LLD.
weak=(--no-dynamic-linker)
"$sdk/bin/ld.lld" -z nodynamic-undefined-weak --version 2>&1 | grep -q "unknown -z" ||
    weak+=(-z nodynamic-undefined-weak)
"$sdk/bin/prospero-lld" "${radv_linker_script[@]}" --eh-frame-hdr "${weak[@]}" "${radv_link_flags[@]}" \
    --version-script "$symbols_map" --exclude-libs=ALL \
    -e _start -o "$out/llvm-pie.elf" \
    "$out/obj/app_crt.o" "${objects[@]}" "$out/obj/libts5compat.a" \
    "$out/stubs/libSceAgc.so" "$out/stubs/libSceAgcDriver.so" \
    "${radv_link_inputs[@]}" \
    --as-needed "$sdk"/target/lib/*.so
# An import that only a module titles get nothing from provides is a call
# through NULL at run time: refuse to ship one.
python3 "$root/ps5/check-imports.py" "$sdk" "$out/llvm-pie.elf" "$out/stubs"
"$tool" link --in "$out/llvm-pie.elf" --out "$out/eboot.elf" \
    --stub-dir "$sdk/target/lib" --stub "$out/stubs/libSceAgc.so" \
    --stub "$out/stubs/libSceAgcDriver.so" --module-sdk 0x02000009 \
    --companion-sdk 0x08050001 --file-name eboot.elf > /dev/null

app="$out/$title_id"
rm -rf -- "$app"
mkdir -p "$app/sce_sys" "$app/sce_module"
"$tool" self --sign --in "$out/eboot.elf" --out "$app/eboot.bin" --magic 0x1D3D154F > /dev/null
cp "$param" "$root/ps5/sce_sys/icon0.png" "$app/sce_sys/"
if [ "$runtime" = gpl ]; then
    cp "$vk/runtime/libc.prx" "$app/sce_module/libc.prx"
else
    rt_libc_prx "$app/sce_module/libc.prx"
fi
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
echo "build: $app (ts5 $version, $runtime runtime, eboot.bin $(stat -c %s "$app/eboot.bin") bytes)"
