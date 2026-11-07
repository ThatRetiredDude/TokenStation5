#!/usr/bin/env bash
# Correctness tests for the engine. Runs anywhere with a Vulkan driver
# (lavapipe is enough: apt install mesa-vulkan-drivers glslang-tools
# libvulkan-dev; pip install gguf numpy jinja2).
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
# The 32-value formats llama.cpp falls back to where K-quant blocks don't
# fit a row (Qwen2.5 0.5B's 896-wide embeddings in a Q4_K_M file are Q5_0).
for q in Q4_1 Q5_0 Q5_1; do
    mk qwen2 qwen2 $q
done
mk llama llama-spm Q5_0

g4=()
for q in F32 Q8_0; do
    out="$work/gemma4-$q.gguf"
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch gemma4 --quant $q --layers 3 --global-hd 512 \
        --vocab "$work/ggml-vocab-qwen2.gguf" "$out" > /dev/null
    g4+=("$out")
done
echo "== Gemma 4: at once, one token per call and 72 tokens across its 8-token window"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${g4[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 24 --step "${g4[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 72 "${g4[@]}"
echo "== Vulkan vs CPU reference"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${models[@]}"
echo "== one token per call (KV cache across calls)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --step "$work/qwen3-Q4_0.gguf" "$work/llama-Q8_0.gguf"
echo "== 300 tokens (prefill chunking, >128 keys per attention block)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$work/qwen2-Q8_0.gguf"
echo "== 72 tokens per call (the tiled matmul: a full and a partial 64-token tile)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 72 "${models[@]}"
# K-quants as Q4_K_M files have them (Q4_K, with Q6_K for attn_v, ffn_down
# and output), quantized by make_test_model.py: the 128 x 128 matmul blocks
# for prompts (TS5_MMB_MIN=1 uses them even for these small matrices; one
# model's attn_k has only 64 rows; they split their columns where there
# are few blocks), the 64 x 64 ones and the one-token mat-vec kernels.
kmix=()
mk_k() {  # name arch vocab [make_test_model.py options]
    local out="$work/$1.gguf" arch=$2 vocab=$3
    shift 3
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch "$arch" --quant Q4_K_M \
        --vocab "$work/ggml-vocab-$vocab.gguf" "$@" "$out" > /dev/null
    kmix+=("$out")
}
mk_k qwen2-kmix qwen2 qwen2
mk_k qwen3-kmix qwen3 qwen2
mk_k llama-kmix llama llama-spm
mk_k llama-kmix-kv1 llama llama-spm --kv-heads 1 --ff 768
# Gemma 4 as the 12B's file has it: every 3rd layer here has heads of 512
# on one K/V head with V from K, the rest 2 heads of 64 and an 8-token
# window; V normalized, scaled layer outputs, soft-capped logits.
mk_k gemma4-kmix gemma4 qwen2 --layers 3 --global-hd 512
echo "== K-quants: 300 tokens in 128 x 128 blocks, 72 in 64 x 64 blocks, one token per call"
TS5_MMB_MIN=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${kmix[@]}"
# The columns split in 5 slices, which don't start on a block, and not at all.
TS5_MMB_MIN=1 TS5_MMB_SLICES=5 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
TS5_MMB_MIN=1 TS5_MMB_SPLIT=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
# The f32 sums throughout (where the device has f16, mmb sums each step in f16).
TS5_MMB_MIN=1 TS5_MMB_F16=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
# The prompt kernel's variants: the plain kernel (the f16 default is dq,
# double-buffered tiles with quad loads of x), quad loads alone, and quad
# loads with f32 sums.
TS5_MMB_MIN=1 TS5_MMB_VAR= python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
TS5_MMB_MIN=1 TS5_MMB_VAR=q python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
# The old double buffer (stores after the sums; Q8_0's default) everywhere.
TS5_MMB_MIN=1 TS5_MMB_VAR=dql python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
# Slices from how many workgroups run at once (the compute units on AMD
# GPUs; lavapipe doesn't tell them): 4 makes these small matrices split.
TS5_MMB_MIN=1 TS5_MMB_WGS=4 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${kmix[@]}"
TS5_MMB_MIN=1 TS5_MMB_VAR=q TS5_MMB_F16=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk \
    --ntok 300 "${kmix[@]}"
# A wider model with the minimum at 12 blocks: attn_k, attn_v, ffn_gate and
# ffn_up (4 or 8 blocks) take mmb only by splitting their columns in 4
# slices, attn_q and ffn_down have the blocks; the last 44 tokens take the
# 64 x 64 kernel for the small ones. (A Qwen3 this narrow is f32 by default,
# as Qwen3 0.6B: TS5_MMB_F16=1 keeps these on the f16 kernel.)
wide="$work/qwen3-kmix-wide.gguf"
[ -s "$wide" ] || python3 "$here/make_test_model.py" --arch qwen3 --quant Q4_K_M --vocab "$work/ggml-vocab-qwen2.gguf" \
    --embd 1024 --heads 8 --kv-heads 2 "$wide" > /dev/null
TS5_MMB_F16=1 TS5_MMB_MIN=12 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$wide"
TS5_MMB_F16=1 TS5_MMB_MIN=12 TS5_MMB_SMALL=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$wide"
TS5_MMB=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 72 "${kmix[@]}"
# One token per call: a row per workgroup (these matrices are small), 8 rows
# (TS5_MV_ROWS_MIN=1), 3 (a partial last group), and the older mat-vecs.
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 24 --step "${kmix[@]}"
TS5_MV_ROWS_MIN=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 12 --step \
    "${kmix[@]}"
TS5_MV_ROWS_MIN=1 TS5_MV_ROWS=3 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 12 \
    --step "${kmix[@]}"
TS5_MV_ROWS_MIN=1 TS5_MV16=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 12 \
    --step "${kmix[@]}"
# Attention's head layouts: query heads per KV head and head size.
shapes=()
mk_shape() {  # name embd heads kv_heads
    local out="$work/$1.gguf"
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch llama --quant Q8_0 --embd "$2" --heads "$3" \
        --kv-heads "$4" --vocab "$work/ggml-vocab-llama-spm.gguf" "$out" > /dev/null
    shapes+=("$out")
}
mk_shape heads14-kv2 896 14 2  # 7 per KV head, size 64 (Qwen 2.5 0.5B)
mk_shape heads7-kv1 896 7 1    # 7 per KV head, size 128 (Qwen 2.5 7B)
mk_shape heads8-kv2 512 8 2    # 4 per KV head, size 64
mk_shape heads2-kv2 512 2 2    # size 256
mk_shape heads4-kv1 512 4 1    # 4 per KV head, size 128 (Llama 3 8B, Qwen 3 8B)
# Rows that are not whole 256-value groups (896 values, as Qwen 2.5 0.5B;
# Gemma 3 1B has 1,152) in Q5_0, as llama.cpp stores Q4_K_M's Q4_K tensors
# there: the word-load mat-vecs take any whole number of 32-value blocks
# of Q8_0 (heads14-kv2 above) and Q5_0.
q50="$work/q50-896.gguf"
[ -s "$q50" ] || python3 "$here/make_test_model.py" --arch llama --quant Q5_0 --embd 896 --heads 14 --kv-heads 2 \
    --vocab "$work/ggml-vocab-llama-spm.gguf" "$q50" > /dev/null
echo "== Q8_0 and Q5_0 rows of 896 values: one token a call, and with TS5_MV32=0 (generic kernels)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 "$q50"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 --step "$q50" "${shapes[0]}"
TS5_MV32=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 --step "$q50" \
    "${shapes[0]}"
echo "== attention: head layouts, 300 tokens at once (attn_pf, then attn), then one per call (keys split in slices)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_PF=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_PF128=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_SPLITS=3 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step "${shapes[@]}"
# Heads of 256 and 512 in attn_dec's 4-column builds (the 8-column WIDE8
# ones are the default).
TS5_ATTN_WIDE8=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step \
    "${shapes[3]}" "$work/gemma4-kmix.gguf"
TS5_ATTN_SPLITS=7 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 --step \
    "${shapes[@]}"
# A generated token's layers submitted one at a time, without waiting in
# between, and the same with the profile's timestamps across the submits.
TS5_DEC_SUBMIT=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 --step \
    "${shapes[@]}"
TS5_DEC_SUBMIT=1 TS5_PROFILE=1 "$ts5" bench -m "${shapes[1]}" --backend vk > "$work/bench-dec-submit.txt"
grep -q "CPU per call" "$work/bench-dec-submit.txt"
# Each generated token's command buffer is recorded while the GPU runs the
# one before (the token's id is read when it runs; the step tests above go
# through it): every generated token reuses one, and with TS5_AHEAD=0 none
# is recorded and the logits stay the same.
TS5_PROFILE=2 "$ts5" bench -m "${shapes[1]}" --backend vk > "$work/bench-ahead.txt"
grep -q "(128 calls, 128 recorded ahead)" "$work/bench-ahead.txt"
TS5_AHEAD=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 --step "${shapes[@]}"
TS5_AHEAD=0 TS5_PROFILE=2 "$ts5" bench -m "${shapes[1]}" --backend vk > "$work/bench-plain.txt"
grep -q "(128 calls, 0 recorded ahead)" "$work/bench-plain.txt"
# A recording ahead that fails only stops recording ahead: the call it was
# made in, and the generation, go on with the same output.
TS5_DRAFT=0 TS5_AHEAD=0 "$ts5" run --backend vk -m "$work/qwen2-Q8_0.gguf" -p "one two three four" -n 40 --temp 0 \
    > "$work/af-off.txt" 2> /dev/null
TS5_DRAFT=0 TS5_AHEAD_FAIL=5 "$ts5" run --backend vk -m "$work/qwen2-Q8_0.gguf" -p "one two three four" -n 40 \
    --temp 0 > "$work/af-on.txt" 2> "$work/af-on.err"
cmp "$work/af-off.txt" "$work/af-on.txt"
grep -q "recording ahead stopped: TS5_AHEAD_FAIL" "$work/af-on.err"
# attn_dec (generation over a long cache; TS5_ATTN_DEC=1 from the first key),
# heads of 128: one token per call, its keys in slices of 128, then 300 at
# once without attn_pf (8 rows of 1 or 2 tokens a workgroup). By default each
# key is scored by 16 lanes side by side, 8 in lavapipe's 8-wide subgroups;
# TS5_ATTN_DEC_KEYS=0 gives each key one thread, with plain and prefetching
# key loads.
TS5_ATTN_DEC=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 --step "${shapes[@]}"
TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 \
    --backend vk --ntok 300 --step "${shapes[@]}"
TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 \
    --backend vk --ntok 300 --step "${shapes[@]}"
TS5_ATTN_DEC=1 TS5_ATTN_PF=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${shapes[@]}"
# attn_dec scoring each key with 8 or 16 lanes side by side, in lavapipe's
# subgroups made 16 wide (heads of 128 only). Wider settings report bigger
# subgroups, but shuffles still stay within 16 lanes, so 32 lanes a key is
# only checked on RADV (ts5 check with TS5_ATTN_DEC=1).
for k in 8 16; do
    LP_NATIVE_VECTOR_WIDTH=512 TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=$k python3 "$here/compare.py" --ts5 "$ts5" --ref cpu \
        --tol 0.002 --backend vk --ntok 160 --step "${shapes[1]}" "${shapes[4]}"
done
# Heads of 64 through attn_dec's _64 builds (the default; TS5_ATTN_DEC64=0
# keeps attn): 8 rows for 7 query heads a KV head, 4 for 4; lanes per key,
# a thread per key, 300 tokens at once, slices. The profile shows the build
# that ran.
h64=("${shapes[0]}" "${shapes[2]}")
TS5_ATTN_DEC64=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step "${h64[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 --step "${h64[@]}"
TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 \
    --backend vk --ntok 300 --step "${h64[@]}"
TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 \
    --backend vk --ntok 300 --step "${h64[@]}"
TS5_ATTN_DEC=1 TS5_ATTN_PF=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "${h64[@]}"
LP_NATIVE_VECTOR_WIDTH=512 TS5_ATTN_DEC_KEYS=16 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu \
    --tol 0.002 --backend vk --ntok 160 --step "${h64[@]}"
TS5_ATTN_SPLITS=7 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 40 \
    --step "${h64[@]}"
TS5_PROFILE=1 "$ts5" bench -m "${shapes[0]}" --backend vk --pp 16 --tg 8 > "$work/bench-dec64.txt"
grep -q "attn_dec_64 " "$work/bench-dec64.txt"
TS5_PROFILE=1 "$ts5" bench -m "${shapes[2]}" --backend vk --pp 16 --tg 8 > "$work/bench-dec64.txt"
grep -q "attn_dec4_64 " "$work/bench-dec64.txt"
echo "== 8-bit KV cache (TS5_KV=q8), against the CPU's fp32 cache: about 0.3% of the largest logit"
# Every attention kernel has an 8-bit build: attn_pf (and attn_pf128,
# attn_pf256; attn_pf512 below, with Gemma 4), attn_dec (lanes per key, and
# a thread per key) and attn.
TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 300 "${shapes[@]}"
TS5_KV=q8 TS5_ATTN_PF128=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 300 "${shapes[@]}"
TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 60 --step "${shapes[@]}"
TS5_KV=q8 TS5_ATTN_DEC=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 160 --step \
    "${shapes[@]}"
TS5_KV=q8 TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu \
    --tol 0.01 --backend vk --ntok 160 --step "${shapes[1]}" "${shapes[4]}"
LP_NATIVE_VECTOR_WIDTH=512 TS5_KV=q8 TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=16 python3 "$here/compare.py" --ts5 "$ts5" \
    --ref cpu --tol 0.01 --backend vk --ntok 160 --step "${shapes[1]}" "${shapes[3]}" "${shapes[4]}"
TS5_KV=q8 TS5_ATTN_SPLITS=7 TS5_ATTN_DEC=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk \
    --ntok 40 --step "${shapes[@]}"
TS5_KV=q8 TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu \
    --tol 0.01 --backend vk --ntok 160 --step "${h64[@]}"
TS5_KV=q8 TS5_ATTN_DEC64=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 60 --step \
    "${h64[@]}"
# Gemma 4's heads of 512 (attn_pf512, which needs subgroups of 16 or more,
# and attn_dec4_512) and its sliding layers, with the f16 and the 8-bit
# cache. In lavapipe's subgroups made 16 wide the f16 attn_pf512 build's
# 16-lane sums come out wrong (4.5% off, where the 8-bit build and the
# console are fine), so these run in 32-wide subgroups.
LP_NATIVE_VECTOR_WIDTH=1024 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${g4[@]}"
LP_NATIVE_VECTOR_WIDTH=1024 TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk "${g4[@]}"
LP_NATIVE_VECTOR_WIDTH=1024 TS5_KV=q8 TS5_ATTN_DEC=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 \
    --backend vk --ntok 40 --step "${g4[@]}"

echo "== speculative decoding: the small verify batch, then draft-on output == draft-off output"
mk qwen2 qwen2 Q4_K_M  # the mvt_* verify kernels serve only the K-quants
for n in 2 5 8; do
    python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok $n "$work/qwen2-Q4_K_M.gguf"
done
TS5_MVT=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 5 "$work/qwen2-Q4_K_M.gguf"
# The x-staging batch kernels (the default on matrices of mv_rows_min_ rows,
# forced onto the small test matrices) against the unstaged ones, byte for
# byte, at 4 tokens (one pass of staged x) and 7 (two): the profile line
# proves the staged kernel ran, and the Q6_K and Q8_0 ones on the mix and
# the Q8_0 file.
for n in 4 7; do
    t=$(echo 12 34 56 78 90 11 22 | cut -d' ' -f1-$n)
    TS5_MV_ROWS_MIN=1 TS5_MVTX=1 TS5_PROFILE=1 "$ts5" logits -m "$work/qwen2-kmix.gguf" -o "$work/mvtx-on.bin" $t 2> "$work/mvtx-on.err"
    grep -q "mvtx_q4_k\.$n " "$work/mvtx-on.err"
    grep -q "mvtx_q6_k\.$n " "$work/mvtx-on.err"
    TS5_MV_ROWS_MIN=1 TS5_MVTX=0 "$ts5" logits -m "$work/qwen2-kmix.gguf" -o "$work/mvtx-off.bin" $t 2> /dev/null
    cmp "$work/mvtx-on.bin" "$work/mvtx-off.bin"
    TS5_MV_ROWS_MIN=1 TS5_MVTX=1 TS5_PROFILE=1 "$ts5" logits -m "$work/qwen2-Q8_0.gguf" -o "$work/mvtx-on.bin" $t 2> "$work/mvtx-on.err"
    grep -q "mvtx_q8_0\.$n " "$work/mvtx-on.err"
    TS5_MV_ROWS_MIN=1 TS5_MVTX=0 "$ts5" logits -m "$work/qwen2-Q8_0.gguf" -o "$work/mvtx-off.bin" $t 2> /dev/null
    cmp "$work/mvtx-on.bin" "$work/mvtx-off.bin"
done
# The row-tiled variant (big matrices on the console), forced onto the small
# test matrices; 3 leaves row tails.
for r in 3 4; do
    TS5_MV_ROWS_MIN=1 TS5_MVT_ROWS=$r python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk \
        --ntok 7 "$work/qwen2-Q4_K_M.gguf"
done
# The fused gate+up verify kernel (off by default) against the split one,
# byte for byte; its profile line proves the fused kernel is what ran, one
# row per workgroup and capped-tiled (TS5_MVT_ROWS=4 still runs it at its
# 2-row spill cap, so the non-".1" pipeline).
TS5_MVT_GLU=1 TS5_PROFILE=1 "$ts5" logits -m "$work/qwen2-Q4_K_M.gguf" -o "$work/glu-on.bin" 12 34 56 78 90 11 22 2> "$work/glu-on.err"
grep -q "mvt_glu_q4_k\.7\.1 ffn_gate+up" "$work/glu-on.err"
"$ts5" logits -m "$work/qwen2-Q4_K_M.gguf" -o "$work/glu-off.bin" 12 34 56 78 90 11 22 2> /dev/null
cmp "$work/glu-on.bin" "$work/glu-off.bin"
TS5_MVT_GLU=1 TS5_MV_ROWS_MIN=1 TS5_MVT_ROWS=4 TS5_PROFILE=1 "$ts5" logits -m "$work/qwen2-Q4_K_M.gguf" \
    -o "$work/glu-r2.bin" 12 34 56 78 90 11 22 2> "$work/glu-r2.err"
grep -q "mvt_glu_q4_k\.7 ffn_gate+up" "$work/glu-r2.err"
cmp "$work/glu-on.bin" "$work/glu-r2.bin"
TS5_MVT_GLU=1 TS5_MV_ROWS_MIN=1 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk \
    --ntok 7 "$work/qwen2-Q4_K_M.gguf"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 5 "$work/qwen2-Q8_0.gguf" "$work/llama-Q8_0.gguf"
# Every K-quant verify kernel sums a block as the one-token kernel does
# (shaders/q4k_sum.glsl): a 7-token batch's logits at position 0 (later
# positions differ through attention) equal one token's, byte for byte,
# with the 4-row, 3-row (row tails), x-staging and fused gate+up kernels.
mk qwen2 qwen2 Q5_K_M qwen2-mk-Q5_K_M
for m in qwen2-Q4_K_M qwen2-kmix qwen2-mk-Q5_K_M; do
    "$ts5" logits -m "$work/$m.gguf" -o "$work/vt-step.bin" --step 12 34 56 78 90 11 22 2> /dev/null
    n=$(( $(stat -c %s "$work/vt-step.bin") / 7 ))
    for e in "TS5_MV_ROWS_MIN=1" "TS5_MV_ROWS_MIN=1 TS5_MVT_ROWS=3" "TS5_MV_ROWS_MIN=1 TS5_MVTX=1" \
        "TS5_MV_ROWS_MIN=1 TS5_MVT_GLU=1" "TS5_MVT_GLU=1"; do
        env $e "$ts5" logits -m "$work/$m.gguf" -o "$work/vt-batch.bin" 12 34 56 78 90 11 22 2> /dev/null
        cmp -n $n "$work/vt-batch.bin" "$work/vt-step.bin"
    done
done
# And at every position, attention included: `logits --verify K` runs K
# tokens a call with every position's logits, as a helper's drafts are
# checked, and its dump must be --step's byte for byte (each token's keys
# split as one token's would be, slices merged in a fixed order): heads of
# 128 (7 a KV head), 64 and 256, Q4_K, and Gemma 4's 512 with its 8-token
# window; batches of 6 and 8, forced key slices, the 8-bit cache; Q8_0
# and Q5_0 in rows of 896 values. 140 tokens: more than one 128-key slice
# for attn_dec. (The build before this rule failed it on the heads of 256
# and with the forced slices.)
vt=$(seq 3 7 975)
for m in "${shapes[1]}" "${shapes[2]}" "${shapes[3]}" "$work/qwen2-Q4_K_M.gguf" "$work/gemma4-kmix.gguf" \
    "${shapes[0]}" "$q50"; do
    for e in "TS5_ATTN_SPLITS=0:6 8" "TS5_ATTN_SPLITS=3:8" "TS5_KV=q8:6"; do
        env ${e%%:*} "$ts5" logits -m "$m" -o "$work/vf-step.bin" --step $vt 2> /dev/null
        for k in ${e#*:}; do
            env ${e%%:*} "$ts5" logits -m "$m" -o "$work/vf-verify.bin" --verify $k $vt 2> /dev/null
            cmp "$work/vf-verify.bin" "$work/vf-step.bin"
        done
    done
done
rep="one two three one two three one two three one two three one two"
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-off.txt" 2> /dev/null
TS5_DRAFT=7 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-on.txt" 2> "$work/sd-on.err"
cmp "$work/sd-off.txt" "$work/sd-on.txt"
grep -q "lookup [0-9]*/[1-9]" "$work/sd-on.err"  # drafts were actually proposed
TS5_ADAPT=0 TS5_DRAFT=7 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-fix.txt" 2> /dev/null
cmp "$work/sd-off.txt" "$work/sd-fix.txt"  # the fixed-length policy too
# Served, the request's timings and log line count the drafts.
TS5_DRAFT=7 "$ts5" serve --backend cpu -m "$work/qwen2-Q8_0.gguf" --host 127.0.0.1 --port 18190 2> "$work/sd-serve.err" &
spid=$!
for _ in $(seq 60); do curl -sf http://127.0.0.1:18190/health | grep -q '"ok"' && break; sleep 1; done
curl -sf http://127.0.0.1:18190/v1/completions -H 'Content-Type: application/json' \
    -d "{\"prompt\": \"$rep\", \"max_tokens\": 300, \"temperature\": 0}" > "$work/sd-serve.json"
kill $spid; wait $spid || true
grep -q '"draft_n":[1-9]' "$work/sd-serve.json"
grep -q "lookup [0-9]*/[1-9][0-9]* accepted, sample [0-9]* ms" "$work/sd-serve.err"
# Drafting on the GPU, where each generated token's command buffer is
# recorded ahead (the default): verify batches and rejected drafts record
# anew, and the output is the same as without recording ahead.
TS5_DRAFT=7 "$ts5" run --backend vk -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-vk.txt" \
    2> "$work/sd-vk.err"
grep -q "lookup [0-9]*/[1-9]" "$work/sd-vk.err"
TS5_AHEAD=0 TS5_DRAFT=7 "$ts5" run --backend vk -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 \
    > "$work/sd-vk0.txt" 2> /dev/null
cmp "$work/sd-vk.txt" "$work/sd-vk0.txt"
echo "== the draft model cascade: a non-repeating prompt, the model as its own drafter"
nr="The sky turned a deep shade of violet over the quiet harbor town that evening"
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$nr" -n 200 --temp 0 > "$work/dm-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-Q8_0.gguf" "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" \
    -p "$nr" -n 200 --temp 0 > "$work/dm-on.txt" 2> "$work/dm-on.err"
cmp "$work/dm-off.txt" "$work/dm-on.txt"
grep -q "draft model .* ready" "$work/dm-on.err"
grep -q "helper [0-9]*/[1-9]" "$work/dm-on.err"  # the drafter actually proposed
echo "== a drafter whose weight rows are padded past the token list (and the reverse), as released models pad"
mk_k qwen2-pad qwen2 qwen2 --pad-vocab 128
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-pad.gguf" -p "$nr" -n 100 --temp 0 > "$work/pad-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-pad.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/pad-on.txt" 2> "$work/pad-on.err"
cmp "$work/pad-off.txt" "$work/pad-on.txt"
grep -q "draft model .* ready" "$work/pad-on.err"  # narrower drafter accepted (the console case)
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/pad-on.err"  # and it kept proposing
! grep -q "draft model \(stopped\|failed\)" "$work/pad-on.err"
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" -p "$nr" -n 100 --temp 0 > "$work/padr-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-pad.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padr-on.txt" 2> "$work/padr-on.err"
cmp "$work/padr-off.txt" "$work/padr-on.txt"
grep -q "draft model .* ready" "$work/padr-on.err"  # wider drafter accepted too
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/padr-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padr-on.err"
echo "== a target whose token list itself is padded (and the reverse), as Qwen 7B pads against the 0.5B"
mk_k qwen2-padlist qwen2 qwen2 --pad-tokens 128
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" -p "$nr" -n 100 --temp 0 > "$work/padl-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padl-on.txt" 2> "$work/padl-on.err"
cmp "$work/padl-off.txt" "$work/padl-on.txt"
grep -q "draft model .* ready" "$work/padl-on.err"  # the drafter's shorter token list is accepted
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/padl-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padl-on.err"
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-padlist.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padlr-on.txt" 2> "$work/padlr-on.err"
cmp "$work/padr-off.txt" "$work/padlr-on.txt"
grep -q "draft model .* ready" "$work/padlr-on.err"  # a longer one too
! grep -q "draft model \(stopped\|failed\)" "$work/padlr-on.err"
echo "== a drafter whose control tokens are named differently drafts (DeepSeek's Qwens, Gemma 4); one text token differing is refused"
mk_k qwen2-ctl qwen2 qwen2 --rename-controls
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-ctl.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/ctl-on.txt" 2> "$work/ctl-on.err"
cmp "$work/padr-off.txt" "$work/ctl-on.txt"
grep -q "draft model: [1-9][0-9]* control or placeholder tokens named differently" "$work/ctl-on.err"
grep -q "draft model .* ready" "$work/ctl-on.err"
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/ctl-on.err"
! grep -q "draft model \(stopped\|failed\|not used\)" "$work/ctl-on.err"
mk_k qwen2-tok qwen2 qwen2 --rename-token 1000
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-tok.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/tok-on.txt" 2> "$work/tok-on.err"
cmp "$work/padr-off.txt" "$work/tok-on.txt"
grep -q 'draft model not used: the draft model.s vocabulary differs (token 1000 is "\[RENAMED\]"' "$work/tok-on.err"
echo "== placeholders under other names (Gemma 4's <unusedN> rows, typed as text, against Gemma 3's) draft; a text token named like one doesn't"
mk_k qwen2-padnorm qwen2 qwen2 --pad-tokens 128 --pad-name "<unused" --pad-type 1
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-padnorm.gguf" "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padn-on.txt" 2> "$work/padn-on.err"
cmp "$work/padl-off.txt" "$work/padn-on.txt"
grep -q "draft model: 128 control or placeholder tokens named differently (ids 151936 to 152063)" "$work/padn-on.err"
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/padn-on.err"
mk_k qwen2-unusedtok qwen2 qwen2 --rename-token 1000 --rename-to "<unused9>"
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-unusedtok.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/unu-on.txt" 2> "$work/unu-on.err"
cmp "$work/padr-off.txt" "$work/unu-on.txt"
grep -q 'draft model not used: the draft model.s vocabulary differs (token 1000 is "<unused9>", not "atus")' "$work/unu-on.err"
echo "== a SentencePiece drafter whose files spell the space mark as a space (Gemma 4 against Gemma 3) drafts"
mk_k llama-spaces llama llama-spm --plain-spaces
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/llama-kmix.gguf" -p "$nr" -n 100 --temp 0 > "$work/sp-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/llama-spaces.gguf" "$ts5" run --backend cpu -m "$work/llama-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/sp-on.txt" 2> "$work/sp-on.err"
cmp "$work/sp-off.txt" "$work/sp-on.txt"
grep -q "draft model .* ready" "$work/sp-on.err"
! grep -q "named differently\|draft model \(stopped\|failed\|not used\)" "$work/sp-on.err"
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/sp-on.err"
echo "== the guards at the vocabulary seams, forced to fire (a boosted row the greedy target soon picks)"
mk_k qwen2-padhot qwen2 qwen2 --pad-tokens 128 --boost-token 151940
# A target that samples into its pad range stops the draft model with a
# message, and the output stays its own...
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-padhot.gguf" -p "$nr" -n 100 --temp 0 > "$work/padf-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-padhot.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padf-on.txt" 2> "$work/padf-on.err"
cmp "$work/padf-off.txt" "$work/padf-on.txt"
grep -q "draft model stopped: token 151940 is past its 151936 embedding rows" "$work/padf-on.err"
# ...and a drafter that argmaxes past the target's rows only loses that draft.
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-padhot.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padb-on.txt" 2> "$work/padb-on.err"
cmp "$work/padr-off.txt" "$work/padb-on.txt"
grep -q "draft model .* ready" "$work/padb-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padb-on.err"
echo "== ts5 check (GPU vs CPU in one command, as run on the console), then again on the saved answers"
rm -f "$work"/qwen3-Q4_0.gguf*.ts5ref
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
! "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep -q perplexity  # test tokens: no next-token measure
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
# On a text's tokens (-p): answers saved apart from the test tokens', then used again.
txt=$(head -c 3000 "$engine/DESIGN.md")
"$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "$txt" 2> /dev/null |
    grep ", over the bar at 0, on the text's next tokens perplexity [0-9.e+]* on the GPU and [0-9.e+]* on the CPU, KL [0-9.e-]* on average .*, cpu [0-9]* ms: PASS$"
"$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "$txt" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
ls "$work"/qwen3-Q4_0.gguf.t*.ts5ref > /dev/null
! "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "hi" 2>&1 | grep -q PASS
# A check whose logits and CPU reference don't fit in the RAM left beside the
# system's reserve is refused before it takes any, with how many tokens would.
{ "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 100000 2>&1 > /dev/null || true; } |
    grep "not enough memory for a 100000-token check: the two sides' logits take about [0-9.]* GiB of RAM, [0-9.]* GiB is free and [0-9.]* GiB is kept for the system (TS5_SYSTEM_RESERVE_GIB); about [0-9]* tokens would fit"
echo "== memory: a load that would leave the system less than its reserve is refused, with the numbers"
# (The refused runs exit 1, which pipefail would take for the grep's.)
{ TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null || true; } |
    grep "not enough memory: the weights and buffers take [0-9.]* GiB and the smallest context [0-9.]* GiB, [0-9.]* GiB is free"
{ TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 2048 -p hi -n 1 2>&1 > /dev/null || true; } |
    grep "and a context of 2048 tokens [0-9.]* GiB"
! TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 2048 -p hi -n 1 > /dev/null 2>&1
"$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "GiB of it kept free for the system"
echo "== -c auto with TS5_KV=q8: an f16 cache, as the test model's whole context fits in f16"
TS5_KV=q8 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "KV cache [0-9.]* GiB (f16, "
echo "== a model's own tuned settings (ts5-tune --models): where nothing set them, for that file only"
tune=$(mktemp -d)
mkdir -p "$tune/tokenstation/tune"
printf '# a test\nsize=%s\nTS5_DRAFT=2\nTS5_DRAFT_MODEL=helper.gguf\nRADV_PERFTEST=nocswave32\n' \
    "$(stat -c %s "$work/qwen3-Q4_0.gguf")" > "$tune/tokenstation/tune/qwen3-Q4_0.gguf.env"
XDG_CONFIG_HOME="$tune" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null |
    grep -x "ts5: settings for qwen3-Q4_0.gguf from $tune/tokenstation/tune/qwen3-Q4_0.gguf.env: TS5_DRAFT=2 TS5_DRAFT_MODEL=$work/helper.gguf"
XDG_CONFIG_HOME="$tune" TS5_DRAFT=4 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null |
    grep -x "ts5: settings for qwen3-Q4_0.gguf from .*: TS5_DRAFT_MODEL=$work/helper.gguf"
! XDG_CONFIG_HOME="$tune" TS5_MODEL_TUNE= "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null | grep -q "settings for"
! XDG_CONFIG_HOME="$tune" "$ts5" run -m "$work/qwen2-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null | grep -q "settings for"
sed -i 's/^size=.*/size=1/' "$tune/tokenstation/tune/qwen3-Q4_0.gguf.env"
XDG_CONFIG_HOME="$tune" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null |
    grep "qwen3-Q4_0.gguf.env is for another qwen3-Q4_0.gguf (not this size), so it isn't used"
rm -rf "$tune"
echo "== tool calls: prompts against the models' own templates, the parser, web_search over HTTP(S)"
for t in Qwen-Qwen2.5-7B-Instruct Qwen-Qwen3-0.6B meta-llama-Llama-3.1-8B-Instruct meta-llama-Llama-3.2-3B-Instruct; do
    f="$work/$t.jinja"
    [ -s "$f" ] || curl -fsSL -o "$f" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$rev/models/templates/$t.jinja"
done
python3 "$here/tools_test.py" "$ts5" "$work/qwen2-Q8_0.gguf" "$work"
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
    python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 72 "${kq[@]}"
    python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --step \
        "$work/qwen3-Q4_K_M.gguf" "$work/llama-Q4_K_M.gguf"
    TS5_MV_ROWS=3 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --step --ntok 12 \
        "$work/qwen3-Q4_K_M.gguf" "$work/llama-Q4_K_M.gguf"
    echo "== vs llama.cpp (llama.cpp quantizes activations, so quantized models differ a little)"
    python3 "$here/compare.py" --ts5 "$ts5" --ref "$ref" --backend vk "$work/qwen2-F32.gguf" "$work/qwen3-F32.gguf" \
        "$work/llama-F32.gguf" "$work/qwen3-Q4_K_M.gguf"
    echo "== tokenizer vs llama-tokenize"
    mk llama llama-bpe Q8_0 llama3-Q8_0
    python3 "$here/tokenizer_test.py" --ts5 "$ts5" --ref "$bin/llama-tokenize" "$work/qwen2-Q4_0.gguf" \
        "$work/llama-Q4_0.gguf" "$work/llama3-Q8_0.gguf"
fi
echo "all tests passed"
