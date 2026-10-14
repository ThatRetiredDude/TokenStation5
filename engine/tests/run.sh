#!/usr/bin/env bash
# Correctness tests for the engine. Runs anywhere with a Vulkan driver
# (lavapipe is enough: apt install mesa-vulkan-drivers glslang-tools
# libvulkan-dev; pip install gguf numpy).
#
#   engine/tests/run.sh                       # Vulkan vs ts5's CPU reference
#   LLAMA_CPP=~/llama.cpp engine/tests/run.sh # also vs llama.cpp, K-quants, tokenizer
#
# LLAMA_CPP must point at a llama.cpp checkout built with llama-tokenize and
# llama-quantize (cmake --build build --target llama-tokenize llama-quantize).
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
engine="$here/.."
work=${TS5_TEST_DIR:-$engine/build/test-models}
mkdir -p "$work"

cmake -S "$engine" -B "$engine/build" > /dev/null
cmake --build "$engine/build" -j"$(nproc)" > /dev/null
ts5="$engine/build/ts5"

# Vocab-only GGUFs from llama.cpp, pinned.
rev=6868a52cb337450658006ad2f9281c2eb63f1666
for v in qwen2 llama-spm llama-bpe; do
    f="$work/ggml-vocab-$v.gguf"
    [ -s "$f" ] || curl -fsSL -o "$f" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$rev/models/ggml-vocab-$v.gguf"
done

models=()
mk() {  # arch vocab quant [name]
    local out="$work/${4:-$1-$3}.gguf"
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch "$1" --quant "$3" --vocab "$work/ggml-vocab-$2.gguf" "$out" > /dev/null
    models+=("$out")
}
for q in F32 F16 Q8_0 Q4_0; do
    mk qwen2 qwen2 $q
    mk qwen3 qwen2 $q
    mk llama llama-spm $q
done

echo "== Vulkan vs CPU reference"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${models[@]}"
echo "== one token per call (KV cache across calls)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --step "$work/qwen3-Q4_0.gguf" "$work/llama-Q8_0.gguf"
echo "== 300 tokens (prefill chunking, >128 keys per attention block)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$work/qwen2-Q8_0.gguf"
echo "== ts5 check (GPU vs CPU in one command, as run on the console)"
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null
echo "== the PS5 title's main on this machine (ps5/host-test.sh)"
"$engine/../ps5/host-test.sh" "$work/qwen2-Q4_0.gguf"

if [ -n "${LLAMA_CPP:-}" ]; then
    bin="$LLAMA_CPP/build/bin"
    ref="$engine/build/ref_logits"
    g++ -O2 -std=c++17 "$here/ref_logits.cpp" -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
        -L"$bin" -lllama -lggml -lggml-base -Wl,-rpath,"$bin" -o "$ref"
    kq=()
    for a in qwen2 qwen3 llama; do
        for t in Q4_K_M Q5_K_M; do
            out="$work/$a-$t.gguf"
            [ -s "$out" ] || "$bin/llama-quantize" "$work/$a-F32.gguf" "$out" $t > /dev/null 2>&1
            kq+=("$out")
        done
    done
    echo "== K-quants: Vulkan vs CPU reference"
    python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${kq[@]}"
    echo "== vs llama.cpp (llama.cpp quantizes activations, so quantized models differ a little)"
    python3 "$here/compare.py" --ts5 "$ts5" --ref "$ref" --backend vk "$work/qwen2-F32.gguf" "$work/qwen3-F32.gguf" \
        "$work/llama-F32.gguf" "$work/qwen3-Q4_K_M.gguf"
    echo "== tokenizer vs llama-tokenize"
    mk llama llama-bpe Q8_0 llama3-Q8_0
    python3 "$here/tokenizer_test.py" --ts5 "$ts5" --ref "$bin/llama-tokenize" "$work/qwen2-Q4_0.gguf" \
        "$work/llama-Q4_0.gguf" "$work/llama3-Q8_0.gguf"
fi
echo "all tests passed"
