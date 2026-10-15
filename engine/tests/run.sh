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
# A wider model with the minimum at 12 blocks: attn_k, attn_v, ffn_gate and
# ffn_up (4 or 8 blocks) take mmb only by splitting their columns in 4
# slices, attn_q and ffn_down have the blocks; the last 44 tokens take the
# 64 x 64 kernel for the small ones.
wide="$work/qwen3-kmix-wide.gguf"
[ -s "$wide" ] || python3 "$here/make_test_model.py" --arch qwen3 --quant Q4_K_M --vocab "$work/ggml-vocab-qwen2.gguf" \
    --embd 1024 --heads 8 --kv-heads 2 "$wide" > /dev/null
TS5_MMB_MIN=12 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$wide"
TS5_MMB_MIN=12 TS5_MMB_SMALL=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "$wide"
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
echo "== attention: head layouts, 300 tokens at once (attn_pf, then attn), then one per call (keys split in slices)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_PF=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_PF128=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
TS5_ATTN_SPLITS=3 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${shapes[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step "${shapes[@]}"
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
echo "== 8-bit KV cache (TS5_KV=q8), against the CPU's fp32 cache: about 0.3% of the largest logit"
TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 300 "${shapes[@]}"
TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 60 --step "${shapes[@]}"

echo "== speculative decoding: the small verify batch, then draft-on output == draft-off output"
mk qwen2 qwen2 Q4_K_M  # the mvt_* verify kernels serve only the K-quants
for n in 2 5 8; do
    python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok $n "$work/qwen2-Q4_K_M.gguf"
done
TS5_MVT=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 5 "$work/qwen2-Q4_K_M.gguf"
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
rep="one two three one two three one two three one two three one two"
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-off.txt" 2> /dev/null
TS5_DRAFT=7 "$ts5" run --backend cpu -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-on.txt" 2> "$work/sd-on.err"
cmp "$work/sd-off.txt" "$work/sd-on.txt"
grep -q "draft [0-9]*/[1-9]" "$work/sd-on.err"  # drafts were actually proposed
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
grep -q "draft [0-9]*/[1-9][0-9]*, sample [0-9]* ms" "$work/sd-serve.err"
# Drafting on the GPU, where each generated token's command buffer is
# recorded ahead (the default): verify batches and rejected drafts record
# anew, and the output is the same as without recording ahead.
TS5_DRAFT=7 "$ts5" run --backend vk -m "$work/qwen2-Q8_0.gguf" -p "$rep" -n 300 --temp 0 > "$work/sd-vk.txt" \
    2> "$work/sd-vk.err"
grep -q "draft [0-9]*/[1-9]" "$work/sd-vk.err"
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
grep -q "draft [0-9]*/[1-9]" "$work/dm-on.err"  # the drafter actually proposed
echo "== a drafter whose weight rows are padded past the token list (and the reverse), as released models pad"
mk_k qwen2-pad qwen2 qwen2 --pad-vocab 128
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-pad.gguf" -p "$nr" -n 100 --temp 0 > "$work/pad-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-pad.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/pad-on.txt" 2> "$work/pad-on.err"
cmp "$work/pad-off.txt" "$work/pad-on.txt"
grep -q "draft model .* ready" "$work/pad-on.err"  # narrower drafter accepted (the console case)
grep -q "draft [0-9]*/[1-9][0-9]* accepted" "$work/pad-on.err"  # and it kept proposing
! grep -q "draft model \(stopped\|failed\)" "$work/pad-on.err"
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" -p "$nr" -n 100 --temp 0 > "$work/padr-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-pad.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padr-on.txt" 2> "$work/padr-on.err"
cmp "$work/padr-off.txt" "$work/padr-on.txt"
grep -q "draft model .* ready" "$work/padr-on.err"  # wider drafter accepted too
grep -q "draft [0-9]*/[1-9][0-9]* accepted" "$work/padr-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padr-on.err"
echo "== a target whose token list itself is padded (and the reverse), as Qwen 7B pads against the 0.5B"
mk_k qwen2-padlist qwen2 qwen2 --pad-tokens 128
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" -p "$nr" -n 100 --temp 0 > "$work/padl-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padl-on.txt" 2> "$work/padl-on.err"
cmp "$work/padl-off.txt" "$work/padl-on.txt"
grep -q "draft model .* ready" "$work/padl-on.err"  # the drafter's shorter token list is accepted
grep -q "draft [0-9]*/[1-9][0-9]* accepted" "$work/padl-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padl-on.err"
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-padlist.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padlr-on.txt" 2> "$work/padlr-on.err"
cmp "$work/padr-off.txt" "$work/padlr-on.txt"
grep -q "draft model .* ready" "$work/padlr-on.err"  # a longer one too
! grep -q "draft model \(stopped\|failed\)" "$work/padlr-on.err"
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
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
echo "== -c auto with TS5_KV=q8: an f16 cache, as the test model's whole context fits in f16"
TS5_KV=q8 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "KV cache [0-9.]* GiB (f16, "
echo "== tool calls: prompts against the models' own templates, the parser, web_search over HTTP(S)"
for t in Qwen-Qwen2.5-7B-Instruct Qwen-Qwen3-0.6B meta-llama-Llama-3.1-8B-Instruct meta-llama-Llama-3.2-3B-Instruct; do
    f="$work/$t.jinja"
    [ -s "$f" ] || curl -fsSL -o "$f" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$rev/models/templates/$t.jinja"
done
python3 "$here/tools_test.py" "$ts5" "$work/qwen2-Q8_0.gguf" "$work"
echo "== usage.tsv (TS5_USAGE_LOG): a line per completions request, never its text; rotation; the parser"
usage="$work/usage.tsv"
rm -f "$usage" "$work/usage.prev.tsv"
TS5_USAGE_LOG="$usage" "$ts5" serve --backend cpu -m "$work/qwen2-Q8_0.gguf" --host 127.0.0.1 --port 18191 \
    2> "$work/usage-serve.err" &
upid=$!
api=http://127.0.0.1:18191/v1
for _ in $(seq 60); do curl -sf http://127.0.0.1:18191/health | grep -q '"ok"' && break; sleep 1; done
word=Quillfeather  # in every request, and never in the file
msg="\"messages\":[{\"role\":\"user\",\"content\":\"$word\"}],\"temperature\":0"
curl -sf $api/chat/completions -d "{$msg,\"max_tokens\":8}" > "$work/usage-1.json"
curl -sf $api/chat/completions -d "{$msg,\"max_tokens\":8,\"stream\":true}" > "$work/usage-2.txt"
curl -sf $api/completions -d "{\"prompt\":\"$word\",\"max_tokens\":8,\"temperature\":0}" > "$work/usage-3.json"
[ "$(curl -s -o /dev/null -w '%{http_code}' $api/chat/completions -d "{\"messages\": $word")" = 400 ]
# A stream the client leaves after its first chunk.
{ curl -sfN $api/chat/completions -d "{$msg,\"max_tokens\":200,\"stream\":true}" || true; } | head -n 1 > /dev/null
curl -sf http://127.0.0.1:18191/health > /dev/null
curl -sf $api/models > /dev/null
for _ in $(seq 100); do [ "$(grep -vc '^#' "$usage" || true)" -ge 5 ] && break; sleep 0.2; done
kill $upid; wait $upid || true
# Again with TS5_USAGE_MAX=600: a few lines in, the file moves to
# usage-rot.prev.tsv and a new one starts.
rm -f "$work/usage-rot.tsv" "$work/usage-rot.prev.tsv"
TS5_USAGE_MAX=600 TS5_USAGE_LOG="$work/usage-rot.tsv" "$ts5" serve --backend cpu -m "$work/qwen2-Q8_0.gguf" \
    --host 127.0.0.1 --port 18192 2> /dev/null &
upid=$!
for _ in $(seq 60); do curl -sf http://127.0.0.1:18192/health | grep -q '"ok"' && break; sleep 1; done
for _ in $(seq 10); do curl -s -o /dev/null http://127.0.0.1:18192/v1/completions -d "{"; done
kill $upid; wait $upid || true
python3 - "$ts5" "$work" "$word" <<'EOF'
import json, os, subprocess, sys
ts5, work, word = sys.argv[1:]
text = open(os.path.join(work, "usage.tsv")).read()
lines = text.splitlines()
names = lines[1][1:].split("\t")
recs = [dict(zip(names, l.split("\t"))) for l in lines if not l.startswith("#")]
bad = []
if lines[0] != "# ts5 usage v1" or len(names) != 25 or names[0] != "time_ms" or names[-1] != "why":
    bad.append("the header: " + "\n".join(lines[:2]))
if len(recs) != 5 or any(len(l.split("\t")) != 25 for l in lines if not l.startswith("#")):
    bad.append(f"{len(recs)} lines, not 5 of 25 columns")
got = [(r["status"], r["kind"], r["stream"], r["why"]) for r in recs]
if got != [("200", "chat", "0", ""), ("200", "chat", "1", ""), ("200", "completion", "0", ""),
           ("400", "chat", "0", "bad_json"), ("200", "chat", "1", "")]:
    bad.append(f"status, kind, stream, why: {got}")
if recs[4:] and recs[4]["finish"] != "left":
    bad.append("the stream left after a chunk finished " + recs[4]["finish"])
for r in recs:
    if r["client"] != "127.0.0.1" or r["key"] != "":
        bad.append(f"client {r['client']!r}, key {r['key']!r}")
    if r["status"] == "200":
        if (r["model"], r["name"]) != ("qwen2-Q8_0.gguf", "qwen2-Q8_0"):
            bad.append(f"model {r['model']!r}, name {r['name']!r}")
        if int(r["total_ms"]) < int(r["prompt_ms"]) + int(r["gen_ms"]):
            bad.append(f"total_ms {r['total_ms']} < prompt_ms {r['prompt_ms']} + gen_ms {r['gen_ms']}")
stream = [json.loads(l[6:]) for l in open(os.path.join(work, "usage-2.txt")).read().splitlines()
          if l.startswith("data: {")]
replies = [json.load(open(os.path.join(work, "usage-1.json"))), stream[-1],
           json.load(open(os.path.join(work, "usage-3.json")))]
for r, reply in zip(recs, replies):
    if int(r["gen"]) != reply["usage"]["completion_tokens"]:
        bad.append(f"gen {r['gen']}, the reply's completion_tokens {reply['usage']['completion_tokens']}")
if word in text:
    bad.append("the prompt's text is in usage.tsv")
# Rotation: both files start with the header.
for f in ("usage-rot.tsv", "usage-rot.prev.tsv"):
    p = os.path.join(work, f)
    if not os.path.exists(p) or open(p).read().splitlines()[:2] != lines[:2]:
        bad.append(f"{f} is missing or doesn't start with the header")
# The parser: the requests come back as they were written, and a later
# version's columns (one in the middle, one at the end) are skipped.
parsed = subprocess.run([ts5, "usage-parse", os.path.join(work, "usage.tsv")], capture_output=True, text=True).stdout
if parsed != "".join(l + "\n" for l in lines if not l.startswith("# start")):
    bad.append("usage-parse:\n" + parsed)
mid = names.index("client") + 1
v2 = [lines[0], "#" + "\t".join(names[:mid] + ["later_mid"] + names[mid:] + ["later_end"])]
for l in lines[2:]:
    c = l.split("\t")
    v2.append(l if l.startswith("#") else "\t".join(c[:mid] + ["x"] + c[mid:] + ["y"]))
open(os.path.join(work, "usage-v2.tsv"), "w").write("\n".join(v2) + "\n")
if subprocess.run([ts5, "usage-parse", os.path.join(work, "usage-v2.tsv")], capture_output=True,
                  text=True).stdout != parsed:
    bad.append("usage-parse doesn't skip the columns it doesn't know")
if bad:
    sys.exit("usage.tsv: " + "\n".join(bad) + "\n" + text)
print("usage.tsv: ok")
EOF
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
