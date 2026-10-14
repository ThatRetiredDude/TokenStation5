#!/usr/bin/env bash
# Sets up everything the PS5 build links against, in .deps/ps5:
#
#   PS5_Vulkan      Mihawk-99's PS5 Vulkan tree: title link recipe, CRT,
#                   libc.prx generator, FSELF signer
#   PS5_PayloadSDK  the payload SDK fork PS5_Vulkan pins (clang target, stubs,
#                   libc++, the platform layer)
#   PS5_Mesa        the Mesa 26.2 fork with RADV's PS5 winsys
#
# then builds RADV (release) and the title tools. Takes a while: RADV is a
# full Mesa cross build. Run once; ps5/build.sh does the rest.
#
# Ubuntu 24.04 packages:
#   clang-18 lld-18 llvm-18-dev libclang-18-dev libclc-18-dev
#   libllvmspirvlib-18-dev ninja-build rsync wget unzip glslang-tools
#   python3-mako python3-yaml bison flex
# plus meson 1.4 or newer (pip install --user meson).
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
deps="${TS5_PS5_DEPS:-$root/.deps/ps5}"
# PS5_Vulkan's revision; it pins the SDK fork and Mesa revisions itself.
vulkan_rev=252bf912a745e3566a93b69b098bdc8ec4048837
mkdir -p "$deps"

# A pinned revision of a GitHub repo, fetched alone (no history).
fetch() {
    local repo=$1 dir=$2 rev=$3
    if [ ! -d "$dir/.git" ]; then
        git init -q "$dir"
        git -C "$dir" remote add origin "https://github.com/$repo.git"
    fi
    if ! git -C "$dir" cat-file -e "$rev^{commit}" 2>/dev/null; then
        echo "setup: fetching $repo at ${rev:0:12}"
        git -C "$dir" fetch -q --depth 1 origin "$rev"
    fi
}
fetch mihawk-99/PS5_Vulkan "$deps/PS5_Vulkan" "$vulkan_rev"
git -C "$deps/PS5_Vulkan" checkout -q "$vulkan_rev"
vk="$deps/PS5_Vulkan"
sdk_rev=$(sed -n 's/^sdk_revision=//p' "$vk/tools/setup-native-dependencies.sh")
mesa_rev=$(sed -n 's/^mesa_revision=//p' "$vk/tools/build-radv.sh")
fetch mihawk-99/PS5_PayloadSDK "$deps/PS5_PayloadSDK" "$sdk_rev"
# Mesa's source is only needed until RADV is built at its revision.
grep -qs "^revision: $mesa_rev$" "$vk/.deps/native/radv-release/PROVENANCE.txt" ||
    fetch mihawk-99/PS5_Mesa "$deps/PS5_Mesa" "$mesa_rev"

# zlib.net and Meson's WrapDB are not always reachable; GitHub carries the
# same files, and the scripts that use them check their hashes.
zlib_dir="$vk/.deps/native/zlib"
if [ ! -f "$zlib_dir/zlib-1.3.2.tar.gz" ]; then
    mkdir -p "$zlib_dir"
    curl -fsSL -o "$zlib_dir/zlib-1.3.2.tar.gz" \
        https://github.com/madler/zlib/releases/download/v1.3.2/zlib-1.3.2.tar.gz
fi
wraps="$vk/.deps/work/radv-src/subprojects/packagecache"
mkdir -p "$wraps"
for f in zlib-1.3.1.tar.gz zlib_1.3.1-1_patch.zip; do
    [ -f "$wraps/$f" ] ||
        curl -fsSL -o "$wraps/$f" "https://github.com/mesonbuild/wrapdb/releases/download/zlib_1.3.1-1/$f"
done

# Mesa 26.2 wants meson 1.4 or newer; Ubuntu 24.04's is 1.3.
meson=${MESON:-}
if [ -z "$meson" ]; then
    for m in "$HOME/.local/bin/meson" "$(command -v meson || true)"; do
        [ -x "$m" ] || continue
        v=$("$m" --version)
        if [ "$(printf '1.4.0\n%s\n' "$v" | sort -V | head -1)" = 1.4.0 ]; then
            meson=$m
            break
        fi
    done
fi
[ -n "$meson" ] || { echo "setup: meson 1.4 or newer is needed (pip install --user meson)" >&2; exit 2; }
(
    cd "$vk"
    bash tools/setup-native-dependencies.sh > /dev/null
    MESON=$meson bash tools/build-radv.sh release
    # libc.prx, the runtime module a title carries, checked against its hash.
    make --no-print-directory runtime/libc.prx > /dev/null
    (cd runtime && sha256sum --check --strict --quiet libc.prx.sha256)
    # ps5-native-tool: turns the linked ELF into a PS5 module and signs it.
    zlib="$PWD/.deps/native/zlib/root/usr"
    mkdir -p build/host
    clang++ -std=c++20 -O2 -I "$zlib/include" tooling/native/native_app_builder.cpp \
        tooling/native/self_container.cpp tooling/native/elf_object.cpp \
        tooling/native/sce_module_writer.cpp "$zlib/lib/libz.a" -o build/host/ps5-native-tool
)
[ -x "$vk/build/host/ps5-native-tool" ] || { echo "setup: ps5-native-tool was not built" >&2; exit 1; }
echo "setup: ready. Next: ps5/build.sh"
