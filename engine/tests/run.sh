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
# `! cmd` alone never stops a set -e script, so a negated check ends in
# `|| unexpected $LINENO`: the command that should have failed succeeded.
unexpected() { echo "run.sh line $1: succeeded where it should fail"; exit 1; }

# Vocab-only GGUFs from llama.cpp, pinned.
rev=6868a52cb337450658006ad2f9281c2eb63f1666
# Qwen3.5's came later; the .inp and .out files are llama.cpp's own
# tokenizer cases (the texts and their tokens).
rev35=954de248ac6907dd05dc19f0f511ab18f2807a54
for v in qwen2 llama-spm llama-bpe qwen35; do
    r=$rev
    [ "$v" = qwen35 ] && r=$rev35
    for x in "" .inp .out; do
        [ "$v" = llama-spm ] && [ -n "$x" ] && continue
        f="$work/ggml-vocab-$v.gguf$x"
        [ -s "$f" ] || curl -fsSL -o "$f" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$r/models/ggml-vocab-$v.gguf$x"
    done
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
# Gemma 3 as the 4B to 27B have it: every 6th layer full attention with RoPE
# scaled linearly by 8, the rest sliding (over 64 tokens here, 1,024 there),
# heads of 256. 600 tokens cross the window nine times.
g3="$work/gemma3-F32.gguf"
[ -s "$g3" ] || python3 "$here/make_test_model.py" --arch gemma3 --quant F32 --layers 6 --embd 512 --heads 2 \
    --kv-heads 1 --window 64 --rope-linear 8 --vocab "$work/ggml-vocab-llama-spm.gguf" "$g3" > /dev/null
echo "== Gemma 3: 600 tokens across its 64-token window, and one token per call"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 600 "$g3"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step "$g3"
echo "== logits --every: positions N-1, 2N-1, ... and the last of the full dump, byte for byte"
every_toks=$(seq 5 7 2400)
"$ts5" logits -m "$g3" -o "$work/every-all.bin" $every_toks 2> /dev/null
"$ts5" logits -m "$g3" -o "$work/every.bin" --every 128 $every_toks 2> /dev/null
python3 - "$work/every-all.bin" "$work/every.bin" <<'PY'
import sys
import numpy as np
a = np.fromfile(sys.argv[1], np.float32).reshape(-1, 32000)
e = np.fromfile(sys.argv[2], np.float32).reshape(-1, 32000)
pos = [p for p in range(len(a)) if (p + 1) % 128 == 0 or p == len(a) - 1]
if not np.array_equal(a[pos], e):
    sys.exit(f"FAIL: --every 128 rows differ from the full dump ({len(pos)} wanted, {len(e)} written)")
print(f"same rows: {len(pos)} of {len(a)}")
PY
rm -f "$work/every-all.bin" "$work/every.bin"
# Gemma 3 12B's attention as the real file has it: 16 heads of 256 on 8 KV
# heads, wider than the model (3,840 there), over a 1,024-token window. A
# 1,100-token prompt (past the window), then its last 8 tokens as generation
# takes them: one a call, and 4 a call as drafts are checked.
g3w="$work/gemma3-12b-shape.gguf"
[ -s "$g3w" ] || python3 "$here/make_test_model.py" --arch gemma3 --quant F16 --layers 6 --embd 384 --heads 16 \
    --kv-heads 8 --head-dim 256 --window 1024 --rope-linear 8 --vocab "$work/ggml-vocab-llama-spm.gguf" "$g3w" > /dev/null
echo "== Gemma 3 12B's heads: generation after a 1,100-token prompt, GPU vs CPU, and drafts' checks == one a call"
tail_toks=$(seq 7 13 14300)
"$ts5" logits -m "$g3w" -o "$work/tail-vk.bin" --tail 8 $tail_toks 2> /dev/null
"$ts5" logits -m "$g3w" --backend cpu -o "$work/tail-cpu.bin" --tail 8 $tail_toks 2> /dev/null
"$ts5" logits -m "$g3w" -o "$work/tail-v4.bin" --tail 8 --verify 4 $tail_toks 2> /dev/null
cmp -s "$work/tail-vk.bin" "$work/tail-v4.bin" || { echo "FAIL: --tail 8 --verify 4 differs from one token a call"; exit 1; }
python3 - "$work/tail-vk.bin" "$work/tail-cpu.bin" <<'PY'
import sys
import numpy as np
a = np.fromfile(sys.argv[1], np.float32).reshape(-1, 32000)
b = np.fromfile(sys.argv[2], np.float32).reshape(-1, 32000)
d = float((np.abs(a - b).max(axis=1) / (b.max(axis=1) - b.min(axis=1))).max())
if len(a) != 8 or d > 0.002:
    sys.exit(f"FAIL: generation rows {len(a)}, GPU vs CPU {d:.5f} of the logit range")
print(f"8 rows, GPU vs CPU {d:.6f} of the logit range, drafts' checks byte for byte")
PY
rm -f "$work/tail-vk.bin" "$work/tail-cpu.bin" "$work/tail-v4.bin"
# Qwen3.5 as the 9B's file has it: linear-attention (Gated DeltaNet) layers
# with every 4th one full attention, gated by a sigmoid, M-RoPE on a quarter
# of each head; heads of 128 in both.
q35=()
for q in F32 Q4_K_M; do
    out="$work/qwen35-$q.gguf"
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch qwen35 --quant $q --layers 8 --heads 2 --kv-heads 1 \
        --seed 3 --vocab "$work/ggml-vocab-qwen35.gguf" "$out" > /dev/null
    q35+=("$out")
done
echo "== Qwen3.5: at once, one token per call, and 300 tokens in chunks (the linear layers' state across calls)"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk "${q35[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 24 --step "${q35[@]}"
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 "${q35[@]}"
echo "== Qwen3.5: no drafts and no helper (that state only goes forward), the same text as with drafting off"
p35="the cat sat on the mat. the cat sat on the mat. the cat sat on the mat. the cat sat on the"
for be in vk cpu; do
    a35=$("$ts5" run -m "${q35[1]}" --backend $be -c 256 -p "$p35" -n 24 --temp 0 2> /dev/null)
    b35=$(TS5_DRAFT=0 "$ts5" run -m "${q35[1]}" --backend $be -c 256 -p "$p35" -n 24 --temp 0 2> /dev/null)
    [ -n "$a35" ] && [ "$a35" = "$b35" ] || { echo "FAIL qwen35 $be: the text differs with drafting allowed"; exit 1; }
done
o35=$(TS5_DRAFT_MODEL="${q35[0]}" "$ts5" run -m "${q35[1]}" -c 256 -p hi -n 4 2>&1)
grep -q "draft model not used: .*only goes forward" <<< "$o35" || { echo "FAIL qwen35: a helper was not refused"; exit 1; }
echo "PASS qwen35 run: the same text with and without drafting, the helper refused"
echo "== Qwen3.5: a conversation's next turn goes on from the state saved where the last turn started"
t35="$work/Qwen3.5-4B.jinja"
[ -s "$t35" ] || curl -fsSL -o "$t35" "https://raw.githubusercontent.com/ggml-org/llama.cpp/$rev35/models/templates/Qwen3.5-4B.jinja"
python3 "$here/resume_test.py" "$ts5" "${q35[1]}" "$t35" vk
python3 "$here/resume_test.py" "$ts5" "${q35[1]}" "$t35" cpu
echo "== tokenizer vs llama.cpp's own cases (Qwen2, Llama 3, Qwen3.5: \\p{L}, \\p{M} and \\p{N} as Unicode has them)"
l3="$work/llama3-Q8_0.gguf"
[ -s "$l3" ] || python3 "$here/make_test_model.py" --arch llama --quant Q8_0 --vocab "$work/ggml-vocab-llama-bpe.gguf" \
    "$l3" > /dev/null
python3 "$here/vocab_test.py" --ts5 "$ts5" "$work/qwen2-Q4_0.gguf" "$work/ggml-vocab-qwen2.gguf.inp" "$work/ggml-vocab-qwen2.gguf.out"
python3 "$here/vocab_test.py" --ts5 "$ts5" "$work/llama3-Q8_0.gguf" "$work/ggml-vocab-llama-bpe.gguf.inp" \
    "$work/ggml-vocab-llama-bpe.gguf.out"
python3 "$here/vocab_test.py" --ts5 "$ts5" "${q35[1]}" "$work/ggml-vocab-qwen35.gguf.inp" "$work/ggml-vocab-qwen35.gguf.out"
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
# keeps attn): 7 rows for 7 query heads a KV head (8 with
# TS5_ATTN_DEC_EXACT=0, as the store title has it), 4 for 4; lanes per key,
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
TS5_ATTN_DEC_EXACT=0 TS5_PROFILE=1 "$ts5" bench -m "${shapes[0]}" --backend vk --pp 16 --tg 8 > "$work/bench-dec64.txt"
grep -q "attn_dec_64 " "$work/bench-dec64.txt"
TS5_PROFILE=1 "$ts5" bench -m "${shapes[2]}" --backend vk --pp 16 --tg 8 > "$work/bench-dec64.txt"
grep -q "attn_dec4_64 " "$work/bench-dec64.txt"
echo "== attn_dec in as many rows as a KV head has query heads, and without the ring's modulo where a layer keeps none: the same logits as the 4- and 8-row builds with the modulo, byte for byte"
# Heads of 128 with 5 query heads a KV head (Qwen3 14B, DeepSeek R1 14B,
# Coder 14B), 3 (Llama 3.2 3B), 2 (Qwen3 0.6B), 6 and 1; the shapes above
# with 7 of 128 and of 64 and 1 of 256; Gemma 4's sliding layers past
# their 264-slot ring. One token a call against the CPU, then byte for byte
# against TS5_ATTN_DEC_EXACT=0 TS5_ATTN_DEC_NORING=0 (the builds before, the
# store title's), each cache's build, one slice over two 128-key tiles
# (TS5_ATTN_SPLITS=1: tokens 129 to 139), and the profile names the build
# that ran.
rows=()
mk_rows() {  # name embd heads kv_heads
    local out="$work/$1.gguf"
    [ -s "$out" ] || python3 "$here/make_test_model.py" --arch llama --quant Q8_0 --embd "$2" --heads "$3" \
        --kv-heads "$4" --vocab "$work/ggml-vocab-llama-spm.gguf" "$out" > /dev/null
    rows+=("$out")
}
mk_rows heads5-kv1 640 5 1
mk_rows heads3-kv1 384 3 1
mk_rows heads2-kv1 256 2 1
mk_rows heads6-kv1 768 6 1
mk_rows heads1-kv1 128 1 1
python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 100 --step "${rows[@]}"
TS5_KV=q8 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 100 --step "${rows[0]}" \
    "${rows[1]}"
xt=$(seq 3 7 975)
before=(TS5_ATTN_DEC_EXACT=0 TS5_ATTN_DEC_NORING=0)
for c in "${rows[0]}:attn_dec5:TS5_KV=f16" "${rows[0]}:attn_dec5_q8:TS5_KV=q8" "${rows[0]}:attn_dec5_q4:TS5_KV=q4" \
    "${rows[0]}:attn_dec5_k8v4:TS5_KV=k8v4" "${rows[0]}:attn_dec5:TS5_ATTN_SPLITS=1" \
    "${rows[1]}:attn_dec3_k8v4:TS5_KV=k8v4" "${rows[2]}:attn_dec2:TS5_KV=f16" "${rows[3]}:attn_dec6_q8:TS5_KV=q8" \
    "${rows[4]}:attn_dec1_q4:TS5_KV=q4" "${shapes[0]}:attn_dec7_64_k8v4:TS5_KV=k8v4" \
    "${shapes[0]}:attn_dec7_64:TS5_KV=f16" "${shapes[0]}:attn_dec7_64_q8:TS5_KV=q8" \
    "${shapes[0]}:attn_dec7_64_q4:TS5_KV=q4" "${shapes[0]}:attn_dec7_64:TS5_ATTN_SPLITS=1" \
    "${shapes[1]}:attn_dec7_q8:TS5_KV=q8" "${shapes[3]}:attn_dec1_256w:TS5_KV=f16"; do
    IFS=: read -r m k e <<< "$c"
    env $e TS5_PROFILE=1 "$ts5" logits -m "$m" -o "$work/xr-1.bin" --step $xt 2> "$work/xr-1.err"
    env $e "${before[@]}" TS5_PROFILE=1 "$ts5" logits -m "$m" -o "$work/xr-0.bin" --step $xt 2> "$work/xr-0.err"
    cmp "$work/xr-1.bin" "$work/xr-0.bin"
    grep -q "$k " "$work/xr-1.err" || { echo "FAIL: $k didn't run on $(basename "$m") ($e)"; exit 1; }
    if grep -q "$k " "$work/xr-0.err"; then echo "FAIL: $k ran with ${before[*]}"; exit 1; fi
    echo "same logits: $(basename "$m") $e ($k)"
done
g4t=$(seq 3 3 840)
for e in TS5_KV=f16 TS5_KV=k8v4; do
    env $e TS5_PROFILE=1 "$ts5" logits -m "$work/gemma4-kmix.gguf" -o "$work/xr-1.bin" --step $g4t 2> "$work/xr-1.err"
    env $e "${before[@]}" "$ts5" logits -m "$work/gemma4-kmix.gguf" -o "$work/xr-0.bin" --step $g4t 2> /dev/null
    cmp "$work/xr-1.bin" "$work/xr-0.bin"
    grep -q "attn_dec2_64[_a-z0-9]*\.ring " "$work/xr-1.err"
    grep -q "attn_dec4_512w[_a-z0-9]* " "$work/xr-1.err"
    echo "same logits: gemma4-kmix $e"
done
rm -f "$work/xr-0.bin" "$work/xr-1.bin"
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
# The steps above turn V too (the Linux default: qkv_post_q8tv, and
# attn_unturn turning each head of the output back); with V as it comes
# (TS5_KV_TURN_V=0, qkv_post_q8t) it is still within 1% of the fp32 reference.
TS5_KV=q8 TS5_KV_TURN_V=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 300 "${shapes[@]}"
TS5_KV=q8 TS5_KV_TURN_V=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 60 --step \
    "${shapes[@]}" "${h64[@]}"
LP_NATIVE_VECTOR_WIDTH=1024 TS5_KV=q8 TS5_KV_TURN_V=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 \
    --backend vk --ntok 40 --step "${g4[@]}"
echo "== 8-bit KV cache without the turn of Q and K (TS5_KV_TURN=0, as the store title has it)"
# The steps above turn Q and K (the Linux default), which keeps every Q.K, so
# the fp32 reference (which doesn't turn) holds for both: heads of 64 to 512,
# the prompt's kernels and one token a call.
TS5_KV=q8 TS5_KV_TURN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 300 "${shapes[@]}"
TS5_KV=q8 TS5_KV_TURN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 --backend vk --ntok 60 --step \
    "${shapes[@]}" "${h64[@]}"
LP_NATIVE_VECTOR_WIDTH=1024 TS5_KV=q8 TS5_KV_TURN=0 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.01 \
    --backend vk --ntok 40 --step "${g4[@]}"
echo "== 4-bit KV cache (TS5_KV=q4), against the CPU rounding K and V the same way (TS5_CPU_KV=q4): about 1% of the largest logit"
# A 4-bit step is 1/8 of its block's largest value, so the two sides round a
# few values to neighbouring steps: about 1% of the largest logit here
# (against the CPU's fp32 cache 4%, and against its 8-bit one too). Every
# attention kernel's 4-bit build, as the 8-bit steps above, turned (the
# default) and not.
q4=(env TS5_KV=q4 TS5_CPU_KV=q4)
cq4=(python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.025 --backend vk)
"${q4[@]}" "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${q4[@]}" TS5_ATTN_PF=0 "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${q4[@]}" TS5_ATTN_PF128=0 TS5_ATTN_PF_SLIM=0 "${cq4[@]}" --ntok 300 "${shapes[1]}" "${shapes[3]}"
"${q4[@]}" "${cq4[@]}" --ntok 60 --step "${shapes[@]}"
"${q4[@]}" TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 160 --step "${shapes[@]}"
"${q4[@]}" TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=0 TS5_ATTN_DEC_PLAIN=0 "${cq4[@]}" --ntok 160 --step "${shapes[1]}" \
    "${shapes[4]}" "${h64[@]}"
"${q4[@]}" LP_NATIVE_VECTOR_WIDTH=512 TS5_ATTN_DEC=1 TS5_ATTN_DEC_KEYS=16 "${cq4[@]}" --ntok 160 --step \
    "${shapes[1]}" "${shapes[3]}" "${shapes[4]}"
"${q4[@]}" TS5_ATTN_WIDE8=0 TS5_ATTN_SPLITS=7 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 100 --step "${shapes[3]}"
"${q4[@]}" TS5_ATTN_DEC64=0 "${cq4[@]}" --ntok 60 --step "${h64[@]}"
"${q4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 "${cq4[@]}" "${g4[@]}"
"${q4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 TS5_ATTN_PF_SLIM=0 "${cq4[@]}" "${g4[@]}"
"${q4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 40 --step "${g4[@]}"
"${q4[@]}" TS5_KV_TURN=0 "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${q4[@]}" TS5_KV_TURN=0 "${cq4[@]}" --ntok 60 --step "${shapes[@]}" "${h64[@]}"
"${q4[@]}" TS5_KV_TURN=0 LP_NATIVE_VECTOR_WIDTH=1024 "${cq4[@]}" --ntok 40 --step "${g4[@]}"
# The steps above turn V too (the default), the CPU turning each V head
# before rounding and back after. With V as it comes (TS5_KV_TURN_V=0) the
# same 1%. Turned on the GPU only, the two land 6% apart.
"${q4[@]}" TS5_KV_TURN_V=0 "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${q4[@]}" TS5_KV_TURN_V=0 TS5_ATTN_PF=0 "${cq4[@]}" --ntok 300 "${shapes[1]}" "${shapes[3]}"
"${q4[@]}" TS5_KV_TURN_V=0 "${cq4[@]}" --ntok 60 --step "${shapes[@]}" "${h64[@]}"
"${q4[@]}" TS5_KV_TURN_V=0 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 160 --step "${shapes[0]}" "${shapes[1]}" "${shapes[3]}"
"${q4[@]}" TS5_KV_TURN_V=0 LP_NATIVE_VECTOR_WIDTH=1024 "${cq4[@]}" "${g4[@]}"
"${q4[@]}" TS5_KV_TURN_V=0 LP_NATIVE_VECTOR_WIDTH=1024 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 40 --step "${g4[@]}"
! "${q4[@]}" TS5_CPU_KV_TURN_V=0 "${cq4[@]}" --ntok 300 "${shapes[4]}" > /dev/null || unexpected $LINENO
echo "== 8-bit K with 4-bit V (TS5_KV=k8v4), against the CPU rounding the same way (TS5_CPU_KV=k8v4)"
# Every attention kernel's _k8v4 build (K words as the 8-bit cache's, V
# words as the 4-bit one's), turned and not, V turned (the default) and not. Against a CPU
# rounding both in 8 bits, the GPU lands 4% off, not under 1%.
k8v4=(env TS5_KV=k8v4 TS5_CPU_KV=k8v4)
"${k8v4[@]}" "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${k8v4[@]}" TS5_ATTN_PF=0 "${cq4[@]}" --ntok 300 "${shapes[1]}" "${shapes[3]}"
"${k8v4[@]}" TS5_ATTN_PF128=0 TS5_ATTN_PF_SLIM=0 "${cq4[@]}" --ntok 300 "${shapes[1]}" "${shapes[3]}"
"${k8v4[@]}" "${cq4[@]}" --ntok 60 --step "${shapes[@]}" "${h64[@]}"
"${k8v4[@]}" TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 160 --step "${shapes[0]}" "${shapes[1]}" "${shapes[3]}"
"${k8v4[@]}" TS5_ATTN_WIDE8=0 TS5_ATTN_SPLITS=7 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 100 --step "${shapes[3]}"
"${k8v4[@]}" TS5_ATTN_DEC64=0 "${cq4[@]}" --ntok 60 --step "${h64[@]}"
"${k8v4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 "${cq4[@]}" "${g4[@]}"
"${k8v4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 TS5_ATTN_PF_SLIM=0 "${cq4[@]}" "${g4[@]}"
"${k8v4[@]}" LP_NATIVE_VECTOR_WIDTH=1024 TS5_ATTN_DEC=1 "${cq4[@]}" --ntok 40 --step "${g4[@]}"
"${k8v4[@]}" TS5_KV_TURN=0 "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${k8v4[@]}" TS5_KV_TURN_V=0 "${cq4[@]}" --ntok 300 "${shapes[@]}"
"${k8v4[@]}" TS5_KV_TURN_V=0 "${cq4[@]}" --ntok 60 --step "${shapes[@]}" "${h64[@]}"
"${k8v4[@]}" TS5_KV_TURN_V=0 LP_NATIVE_VECTOR_WIDTH=1024 "${cq4[@]}" --ntok 40 --step "${g4[@]}"
! TS5_KV=k8v4 TS5_CPU_KV=q8 "${cq4[@]}" --ntok 300 "${shapes[4]}" > /dev/null || unexpected $LINENO
TS5_KV=k8v4 TS5_PROFILE=1 "$ts5" bench -m "${shapes[1]}" --backend vk --pp 16 --tg 8 > "$work/bench-k8v4.txt" 2>&1
grep -q "qkv_post_k8v4tv " "$work/bench-k8v4.txt"
grep -q "attn_pf128_k8v4 " "$work/bench-k8v4.txt"
grep -q "attn_dec[0-9]*_k8v4 " "$work/bench-k8v4.txt"
grep -q "KV cache .*(k8v4, Q, K and V turned" "$work/bench-k8v4.txt"
# The profile names the 4-bit builds.
TS5_KV=q4 TS5_PROFILE=1 "$ts5" bench -m "${shapes[1]}" --backend vk --pp 16 --tg 8 > "$work/bench-q4.txt"
grep -q "qkv_post_q4tv " "$work/bench-q4.txt"
grep -q "attn_unturn " "$work/bench-q4.txt"
grep -q "attn_pf128_q4 " "$work/bench-q4.txt"
grep -q "attn_dec[0-9]*_q4 " "$work/bench-q4.txt"
TS5_KV=q4 TS5_KV_TURN_V=0 TS5_PROFILE=1 "$ts5" bench -m "${shapes[1]}" --backend vk --pp 16 --tg 8 > "$work/bench-q4t.txt" 2>&1
grep -q "qkv_post_q4t " "$work/bench-q4t.txt"
! grep -q "attn_unturn" "$work/bench-q4t.txt" || unexpected $LINENO
grep -q "KV cache .*(q4, Q and K turned" "$work/bench-q4t.txt"

echo "== attn_pf256s and attn_pf512s (the default; TS5_ATTN_PF_SLIM=0 the wider builds): the same logits as attn_pf256 and attn_pf512, byte for byte"
# The builds with fewer registers (tiles of 16 keys; no tile loaded ahead)
# score the keys in the same order, so 600 tokens (three prompt chunks) come
# out identical with the f16, the 8-bit and the 4-bit cache, and with the keys split.
slim_toks=$(python3 -c "print(' '.join(str(t + 3 * (i // 16)) for i, t in enumerate([1, 450, 4996, 17354, 1701, 432, 35, 279, 3974, 13, 2000, 77, 8, 1234, 999, 42] * 38) if i < 600))")
for m in "${shapes[3]}" "$work/gemma4-kmix.gguf"; do
    for env in TS5_KV=f16 TS5_KV=q8 TS5_KV=q4 TS5_KV=k8v4 TS5_ATTN_SPLITS=3; do
        for s in 0 1; do
            env LP_NATIVE_VECTOR_WIDTH=1024 $env TS5_ATTN_PF_SLIM=$s TS5_PROFILE=1 "$ts5" logits -m "$m" \
                -o "$work/slim-$s.bin" $slim_toks 2> "$work/slim-$s.err"
        done
        grep -q "attn_pf[25][51][62]s" "$work/slim-1.err"
        if grep -q "attn_pf[25][51][62]s" "$work/slim-0.err"; then echo "slim build with TS5_ATTN_PF_SLIM=0"; exit 1; fi
        cmp "$work/slim-0.bin" "$work/slim-1.bin"
        echo "same logits: $(basename "$m") $env"
    done
    LP_NATIVE_VECTOR_WIDTH=1024 TS5_PROFILE=1 "$ts5" logits -m "$m" -o "$work/slim-d.bin" $slim_toks 2> "$work/slim-d.err"
    grep -q "attn_pf[25][51][62]s" "$work/slim-d.err" || { echo "FAIL: the slim builds are not the default"; exit 1; }
done

echo "== rmsnorm_early and rmsnorm2_early (the norms with their loads up front, the default; TS5_NORM_EARLY=0 the old kernels): the same logits, byte for byte"
# The same sums in the same order: a prompt at once (and on Gemma 4 one
# token a call) comes out identical, also with 16 values a thread
# (TS5_NORM_EARLY=16, the build for rows of up to 8,192, here with most of its
# slots clamped). Rows of 256, 512 and 896, and 5,376 (Gemma 3 27B's), past
# the old kernels' 4,096, so their loops for the rest run (3 passes for half
# the threads, 2 for the others) against the early builds' 11 values a
# thread, the last clamped for half of them. Gemma's norms add into the
# residual, two at once (rmsnorm2_early), and Gemma 4's times its layer scale.
g3x="$work/gemma3-5376.gguf"
[ -s "$g3x" ] || python3 "$here/make_test_model.py" --arch gemma3 --quant Q8_0 --layers 1 --embd 5376 --heads 2 \
    --kv-heads 1 --head-dim 256 --ff 256 --window 64 --vocab "$work/ggml-vocab-llama-spm.gguf" "$g3x" > /dev/null
for m in "$work/llama-Q8_0.gguf" "${shapes[1]}" "$g3" "$g3x" "$work/gemma4-kmix.gguf"; do
    ne_toks=$(seq 5 11 600)
    [ "$m" = "$g3x" ] && ne_toks=$(seq 5 13 520)  # its 32,000 x 5,376 output, on the CPU's Vulkan
    steps=("")
    case "$(basename "$m")" in gemma4*) steps=("" --step) ;; esac
    for step in "${steps[@]}"; do
        TS5_NORM_EARLY=0 TS5_PROFILE=1 "$ts5" logits -m "$m" -o "$work/ne-0.bin" $step $ne_toks 2> "$work/ne-0.err"
        if grep -q "rmsnorm2*_early " "$work/ne-0.err"; then echo "FAIL: an early norm ran with TS5_NORM_EARLY=0"; exit 1; fi
        for e in "" TS5_NORM_EARLY=16; do
            env $e TS5_PROFILE=1 "$ts5" logits -m "$m" -o "$work/ne-1.bin" $step $ne_toks 2> "$work/ne-1.err"
            cmp "$work/ne-0.bin" "$work/ne-1.bin"
            grep -q "rmsnorm_early " "$work/ne-1.err" || { echo "FAIL: rmsnorm_early didn't run ($e)"; exit 1; }
            case "$(basename "$m")" in gemma*) grep -q "rmsnorm2_early " "$work/ne-1.err" ||
                { echo "FAIL: rmsnorm2_early didn't run ($e)"; exit 1; } ;; esac
        done
        echo "same logits: $(basename "$m") ${step:-at once}"
    done
done
rm -f "$work/ne-0.bin" "$work/ne-1.bin"

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
# 128 (7 and 5 a KV head), 64 and 256, Q4_K, and Gemma 4's 512 with its 8-token
# window; batches of 6 and 8, one slice over two 128-key tiles
# (TS5_ATTN_SPLITS=1), the 8- and 4-bit caches; Q8_0 and Q5_0 in rows of 896
# values. 140 tokens: more than one 128-key slice for attn_dec. (The build
# before this rule failed it on the heads of 256 and with TS5_ATTN_SPLITS=3,
# which now splits these lengths as the default does.)
vt=$(seq 3 7 975)
for m in "${shapes[1]}" "${shapes[2]}" "${shapes[3]}" "$work/qwen2-Q4_K_M.gguf" "$work/gemma4-kmix.gguf" \
    "${shapes[0]}" "$q50" "${rows[0]}"; do
    for e in "TS5_ATTN_SPLITS=0:6 8" "TS5_ATTN_SPLITS=1:8" "TS5_KV=q8:6" "TS5_KV=q4:6"; do
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
! grep -q "draft model \(stopped\|failed\)" "$work/pad-on.err" || unexpected $LINENO
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" -p "$nr" -n 100 --temp 0 > "$work/padr-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-pad.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padr-on.txt" 2> "$work/padr-on.err"
cmp "$work/padr-off.txt" "$work/padr-on.txt"
grep -q "draft model .* ready" "$work/padr-on.err"  # wider drafter accepted too
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/padr-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padr-on.err" || unexpected $LINENO
echo "== a target whose token list itself is padded (and the reverse), as Qwen 7B pads against the 0.5B"
mk_k qwen2-padlist qwen2 qwen2 --pad-tokens 128
TS5_DRAFT=0 "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" -p "$nr" -n 100 --temp 0 > "$work/padl-off.txt" 2> /dev/null
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-kmix.gguf" "$ts5" run --backend cpu -m "$work/qwen2-padlist.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padl-on.txt" 2> "$work/padl-on.err"
cmp "$work/padl-off.txt" "$work/padl-on.txt"
grep -q "draft model .* ready" "$work/padl-on.err"  # the drafter's shorter token list is accepted
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/padl-on.err"
! grep -q "draft model \(stopped\|failed\)" "$work/padl-on.err" || unexpected $LINENO
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-padlist.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/padlr-on.txt" 2> "$work/padlr-on.err"
cmp "$work/padr-off.txt" "$work/padlr-on.txt"
grep -q "draft model .* ready" "$work/padlr-on.err"  # a longer one too
! grep -q "draft model \(stopped\|failed\)" "$work/padlr-on.err" || unexpected $LINENO
echo "== a drafter whose control tokens are named differently drafts (DeepSeek's Qwens, Gemma 4); one text token differing is refused"
mk_k qwen2-ctl qwen2 qwen2 --rename-controls
TS5_DRAFT=7 TS5_DRAFT_MODEL="$work/qwen2-ctl.gguf" "$ts5" run --backend cpu -m "$work/qwen2-kmix.gguf" \
    -p "$nr" -n 100 --temp 0 > "$work/ctl-on.txt" 2> "$work/ctl-on.err"
cmp "$work/padr-off.txt" "$work/ctl-on.txt"
grep -q "draft model: [1-9][0-9]* control or placeholder tokens named differently" "$work/ctl-on.err"
grep -q "draft model .* ready" "$work/ctl-on.err"
grep -q "helper [0-9]*/[1-9][0-9]* accepted" "$work/ctl-on.err"
! grep -q "draft model \(stopped\|failed\|not used\)" "$work/ctl-on.err" || unexpected $LINENO
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
! grep -q "named differently\|draft model \(stopped\|failed\|not used\)" "$work/sp-on.err" || unexpected $LINENO
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
! grep -q "draft model \(stopped\|failed\)" "$work/padb-on.err" || unexpected $LINENO
echo "== ts5 check (GPU vs CPU in one command, as run on the console), then again on the saved answers"
rm -f "$work"/qwen3-Q4_0.gguf*.ts5ref
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
! "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep -q perplexity || unexpected $LINENO  # test tokens: no next-token measure
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q8 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
# The turned cache's answers (.q8tv: Q, K and V turned) are saved apart from
# the plain one's (.q8, TS5_KV_TURN=0), which round other values.
[ -s "$work/qwen3-Q4_0.gguf.q8tv.ts5ref" ] && [ ! -e "$work/qwen3-Q4_0.gguf.q8.ts5ref" ]
TS5_KV=q8 TS5_KV_TURN=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q8 TS5_KV_TURN=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
python3 "$here/q8_refcmp.py" "$work/qwen3-Q4_0.gguf" | grep -c "vs 16-bit cache, CPU" | grep -qx 2
# TS5_CHECK_REF_KV=f16: the GPU's 8-bit cache against the 16-bit answers
# saved above, with no CPU run.
TS5_KV=q8 TS5_CHECK_REF_KV=f16 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null |
    grep "cpu answers saved earlier: PASS$"
# The 4-bit cache: its own answers (.q4tv), then against the 16-bit ones (its
# cost, which no bar judges here: on these test models about as much as the
# CPU's own, 7 to 10% of the largest logit).
TS5_KV=q4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
[ -s "$work/qwen3-Q4_0.gguf.q4tv.ts5ref" ] && [ ! -e "$work/qwen3-Q4_0.gguf.q4.ts5ref" ]
python3 "$here/q8_refcmp.py" "$work/qwen3-Q4_0.gguf" | grep -q "4-bit turned, V too vs 16-bit cache, CPU, 16 positions"
{ TS5_KV=q4 TS5_CHECK_REF_KV=f16 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2>&1 > /dev/null || true; } |
    grep -q "check against the answers with a cache in f16 (the GPU keeps q4)"
# With V as it comes (TS5_KV_TURN_V=0): answers of their own (.q4t, .q8t).
TS5_KV=q4 TS5_KV_TURN_V=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=q4 TS5_KV_TURN_V=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
TS5_KV=q8 TS5_KV_TURN_V=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
[ -s "$work/qwen3-Q4_0.gguf.q4t.ts5ref" ] && [ -s "$work/qwen3-Q4_0.gguf.q8t.ts5ref" ]
python3 "$here/q8_refcmp.py" "$work/qwen3-Q4_0.gguf" | grep -c "[48]-bit turned vs 16-bit cache, CPU, 16 positions" | grep -qx 2
# 8-bit K with 4-bit V: answers of their own (.k8v4tv, .k8v4t).
TS5_KV=k8v4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_KV=k8v4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
TS5_KV=k8v4 TS5_KV_TURN_V=0 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
[ -s "$work/qwen3-Q4_0.gguf.k8v4t.ts5ref" ] && [ -s "$work/qwen3-Q4_0.gguf.k8v4tv.ts5ref" ]
python3 "$here/q8_refcmp.py" "$work/qwen3-Q4_0.gguf" | grep -q "8-bit K 4-bit V turned, V too vs 16-bit cache, CPU, 16 positions"
{ TS5_KV=k8v4 TS5_CHECK_REF_KV=f16 "$ts5" check -m "$work/qwen3-Q4_0.gguf" 2>&1 > /dev/null || true; } |
    grep -q "check against the answers with a cache in f16 (the GPU keeps k8v4)"
# What each half of the cache costs, in the CPU's answers (saved apart):
# only K in 4 bits, and only V, turned and back. Turning V and back without
# rounding changes nothing (within 1e-5 of the largest logit). Only the
# saved answers count here, not the GPU's pass or fail against them. These
# keep V's default turn off (TS5_KV_TURN_V=0), so TS5_CPU_KV_TURN_V alone
# decides V's, and the names say so (.q4t-...).
export TS5_KV_TURN_V=0
TS5_CPU_KV_V=f16 TS5_CHECK_REF_KV=q4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" --fresh > /dev/null 2>&1 || true
TS5_CPU_KV_K=f16 TS5_CPU_KV_TURN_V=1 TS5_CHECK_REF_KV=q4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" --fresh \
    > /dev/null 2>&1 || true
python3 "$here/q8_refcmp.py" "$work/qwen3-Q4_0.gguf" 16 "$work/qwen3-Q4_0.gguf.q4t-Vf16.ts5ref" \
    "$work/qwen3-Q4_0.gguf.q4t-Kf16-Vt.ts5ref" | grep -c "q4t-[KV]f16.* vs 16-bit cache, CPU, 16 positions" | grep -qx 2
for t in 0 1; do
    TS5_CPU_KV_V=f32 TS5_CPU_KV_TURN_V=$t TS5_CHECK_REF_KV=q4 "$ts5" check -m "$work/qwen3-Q4_0.gguf" --fresh \
        > /dev/null 2>&1 || true
done
python3 - "$work/qwen3-Q4_0.gguf" <<'PY'
import struct, sys
import numpy as np
def load(p):
    with open(p, 'rb') as f:
        h = struct.unpack('<8s4I2Q', f.read(40))
        return np.fromfile(f, dtype=np.float32).reshape(h[2], h[3])
a, b = load(sys.argv[1] + '.q4t-Vf32.ts5ref'), load(sys.argv[1] + '.q4t-Vf32-Vt.ts5ref')
d = float(np.abs(a - b).max() / np.abs(a).max())
print(f"V turned and back, unrounded: {d:.2g} of the largest logit")
assert d < 1e-5, d
PY
unset TS5_KV_TURN_V
# On a text's tokens (-p): answers saved apart from the test tokens', then used again.
txt=$(head -c 3000 "$engine/DESIGN.md")
"$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "$txt" 2> /dev/null |
    grep ", over the bar at 0, on the text's next tokens perplexity [0-9.e+]* on the GPU and [0-9.e+]* on the CPU, KL [0-9.e-]* on average .*, cpu [0-9]* ms: PASS$"
"$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "$txt" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
ls "$work"/qwen3-Q4_0.gguf.t*.ts5ref > /dev/null
! "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 -p "hi" 2>&1 | grep -q PASS || unexpected $LINENO
# Saved answers of an older version are kept beside the new file, not written over.
python3 -c 'import struct, sys; f = open(sys.argv[1], "r+b"); f.seek(8); f.write(struct.pack("<I", 1))' \
    "$work/qwen3-Q4_0.gguf.ts5ref"
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2>&1 > /dev/null |
    grep "kept the earlier saved answers (another version or model file) as $work/qwen3-Q4_0.gguf.ts5ref.v1-[0-9]*$"
ls "$work"/qwen3-Q4_0.gguf.ts5ref.v1-* > /dev/null
"$ts5" check -m "$work/qwen3-Q4_0.gguf" 2> /dev/null | grep "cpu answers saved earlier: PASS$"
rm -f "$work"/qwen3-Q4_0.gguf.ts5ref.v1-*
# A check whose logits and CPU reference don't fit in the RAM left beside the
# system's reserve is refused before it takes any, with how many tokens would.
{ "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 100000 2>&1 > /dev/null || true; } |
    grep "not enough memory for a 100000-token check: the two sides' logits take about [0-9.]* GiB of RAM, [0-9.]* GiB is free and [0-9.]* GiB is kept for the system (TS5_SYSTEM_RESERVE_GIB); about [0-9]* tokens would fit"
echo "== a mixture-of-experts file is refused as one, also under an architecture name the engine reads (Mixtral's is llama)"
moe="$work/llama-experts.gguf"
[ -s "$moe" ] || python3 "$here/make_test_model.py" --arch llama --quant F32 --layers 1 --experts 8 \
    --vocab "$work/ggml-vocab-qwen2.gguf" "$moe" > /dev/null
{ "$ts5" run -m "$moe" -p hi -n 1 2>&1 > /dev/null || true; } |
    grep "mixture-of-experts models are not supported yet (this one has 8 experts)"
! "$ts5" run -m "$moe" -p hi -n 1 > /dev/null 2>&1 || unexpected $LINENO
echo "== -c auto waits while memory just freed is still coming back (TS5_MEMINFO: a scripted /proc/meminfo)"
# 12 MiB more every quarter second, here for 3 s, then flat. The wait goes on
# while 4 s add 16 MiB or more.
python3 "$here/meminfo_curve.py" "$work/meminfo" 12 3 10 &
mc=$!
sleep 0.2
TS5_MEMINFO="$work/meminfo" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "waited [4-7]\.[0-9]* s for memory just freed to come back"
wait $mc
# As the kernel shows it: steps of 60 MiB every 1.25 s, with nothing between
# (a rule that looks at 1 s or less stops in a gap), for 5 s.
python3 "$here/meminfo_curve.py" "$work/meminfo" 12 5 12 1.25 &
mc=$!
sleep 0.2
TS5_MEMINFO="$work/meminfo" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "waited [6-9]\.[0-9]* s for memory just freed to come back"
wait $mc
# The PS5's own curve right after a full build (+572 MiB over 22 s): the
# wait stops near 22 s, when 4 s add less than 16 MiB.
python3 "$here/meminfo_curve.py" "$work/meminfo" 0 ck 28 &
mc=$!
sleep 0.2
TS5_MEMINFO="$work/meminfo" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "waited 2[0-4]\.[0-9]* s for memory just freed to come back"
wait $mc
# And after a server stopped (+98 MiB over 13 s, steps 1 to 3 s apart): the
# wait goes on past 9 s (a 2 s window stopped near 7.5 s, 0.03 GiB short)
# and stops in the tail (the last 16 MiB come over 4 s).
python3 "$here/meminfo_curve.py" "$work/meminfo" 0 serve 20 &
mc=$!
sleep 0.2
TS5_MEMINFO="$work/meminfo" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep -E "waited (9|1[0-5])\.[0-9]* s for memory just freed to come back"
wait $mc
# Nothing coming back: the wait ends at its first look (1.5 s), no line.
python3 "$here/meminfo_curve.py" "$work/meminfo" 0 0 4 &
mc=$!
sleep 0.2
! TS5_MEMINFO="$work/meminfo" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep -q "waited" || unexpected $LINENO
wait $mc
echo "== memory: a load that would leave the system less than its reserve is refused, with the numbers"
# (The refused runs exit 1, which pipefail would take for the grep's.)
{ TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null || true; } |
    grep "not enough memory: the weights and buffers take [0-9.]* GiB and the smallest context [0-9.]* GiB, [0-9.]* GiB is free"
{ TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 2048 -p hi -n 1 2>&1 > /dev/null || true; } |
    grep "and a context of 2048 tokens [0-9.]* GiB"
! TS5_SYSTEM_RESERVE_GIB=100000 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 2048 -p hi -n 1 > /dev/null 2>&1 || unexpected $LINENO
"$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "GiB of it kept free for the system"
echo "== -c auto with TS5_KV=q8: an f16 cache, as the test model's whole context fits in f16"
TS5_KV=q8 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "KV cache [0-9.]* GiB (f16, "
echo "== -c auto with TS5_KV unset: 8 bits where f16 holds under 24,576 tokens and less than the model's context,"
echo "   then 8-bit K and 4-bit V where 8 bits still hold less and that holds an eighth more"
# 2.002 GiB leaves this model 2,048 tokens of f16 (tests' arithmetic, as the
# 4.5 and 3 GiB below: a change in what a load takes moves it), 3,072 in 8
# bits and its whole 4,096 in 8-bit K and 4-bit V.
TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2> "$work/kvauto.err" > /dev/null
grep "8-bit KV cache: [0-9]* tokens, where f16 fits 2048 (TS5_KV=f16 keeps f16)" "$work/kvauto.err"
grep "8-bit K and 4-bit V KV cache: [0-9]* tokens, where 8 bits fit [0-9]* (TS5_KV=q8 keeps 8 bits)" "$work/kvauto.err"
grep "KV cache [0-9.]* GiB (k8v4, Q, K and V turned, " "$work/kvauto.err"
TS5_KV=f16 TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "KV cache [0-9.]* GiB (f16, 2048 tokens)"
# Asked for, 8 bits stay 8 bits; without the V turn the automatic choice
# stops at 8 bits too (4-bit V unturned failed the check on the PS5).
TS5_KV=q8 TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2> "$work/kvq8.err" > /dev/null
grep "KV cache [0-9.]* GiB (q8, Q, K and V turned, " "$work/kvq8.err"
! grep "4-bit V KV cache:" "$work/kvq8.err" || unexpected $LINENO
TS5_KV_TURN_V=0 TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2> "$work/kvnov.err" > /dev/null
grep "KV cache [0-9.]* GiB (q8, Q and K turned, " "$work/kvnov.err"
! grep "4-bit V KV cache:" "$work/kvnov.err" || unexpected $LINENO
# Only for an eighth more tokens or better: at 2.005 GiB, with YaRN past the
# 4,096-token window and a 10,240-token cap, 8 bits hold 9,216 and stay.
TS5_YARN=1 TS5_CTX_MAX=10240 TS5_GPU_MEMORY_GIB=2.005 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 \
    2> "$work/kvmargin.err" > /dev/null
grep "KV cache [0-9.]* GiB (q8, Q, K and V turned, 9216 tokens)" "$work/kvmargin.err"
! grep "4-bit V KV cache:" "$work/kvmargin.err" || unexpected $LINENO
# TS5_KV=q4 (asked for, never automatic): 4 bits, more tokens than 8.
TS5_KV=q4 TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2> "$work/kvq4.err" > /dev/null
q8_tok=$(sed -n 's/.*KV cache [0-9.]* GiB (q8, Q, K and V turned, \([0-9]*\) tokens).*/\1/p' "$work/kvq8.err")
q4_tok=$(sed -n 's/.*KV cache [0-9.]* GiB (q4, Q, K and V turned, \([0-9]*\) tokens).*/\1/p' "$work/kvq4.err")
[ -n "$q4_tok" ] && [ "$q4_tok" -gt "$q8_tok" ] || { echo "FAIL: 4-bit context $q4_tok, 8-bit $q8_tok"; exit 1; }
# TS5_KV=k8v4 (asked for): more than 8 bits, at most as many as 4, and as
# many as the automatic choice.
TS5_KV=k8v4 TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2> "$work/kvk8v4.err" > /dev/null
k8v4_tok=$(sed -n 's/.*KV cache [0-9.]* GiB (k8v4, Q, K and V turned, \([0-9]*\) tokens).*/\1/p' "$work/kvk8v4.err")
auto_tok=$(sed -n 's/.*KV cache [0-9.]* GiB (k8v4, Q, K and V turned, \([0-9]*\) tokens).*/\1/p' "$work/kvauto.err")
[ -n "$k8v4_tok" ] && [ "$k8v4_tok" -gt "$q8_tok" ] && [ "$k8v4_tok" -le "$q4_tok" ] && [ "$k8v4_tok" = "$auto_tok" ] ||
    { echo "FAIL: 8-bit K 4-bit V context $k8v4_tok (automatic $auto_tok), 8-bit $q8_tok, 4-bit $q4_tok"; exit 1; }
TS5_GPU_MEMORY_GIB=2.002 "$ts5" run -m "$work/qwen3-F32.gguf" -c 2048 -p hi -n 1 2>&1 > /dev/null |
    grep "KV cache [0-9.]* GiB (f16, 2048 tokens)"
echo "== -c auto with TS5_KV unset, past 24,576 tokens of f16: on llama, qwen3 and qwen35 a smaller cache"
echo "   where f16 falls an eighth or more short of the model's own window; never on a Gemma"
ql="$work/qwen3-long-F32.gguf"
[ -s "$ql" ] || python3 "$here/make_test_model.py" --arch qwen3 --quant F32 --ctx 65536 \
    --vocab "$work/ggml-vocab-qwen2.gguf" "$ql" > /dev/null
g3l="$work/gemma3-long-F32.gguf"
[ -s "$g3l" ] || python3 "$here/make_test_model.py" --arch gemma3 --quant F32 --layers 6 --embd 512 --heads 2 \
    --kv-heads 1 --window 64 --rope-linear 8 --ctx 65536 --vocab "$work/ggml-vocab-llama-spm.gguf" "$g3l" > /dev/null
g4l="$work/gemma4-long-F32.gguf"
[ -s "$g4l" ] || python3 "$here/make_test_model.py" --arch gemma4 --quant F32 --layers 3 --global-hd 512 --ctx 65536 \
    --vocab "$work/ggml-vocab-qwen2.gguf" "$g4l" > /dev/null
# The f16 tokens a run's cache line says, or the "where f16 fits" line's.
f16_of() { sed -n -e 's/.*where f16 fits \([0-9]*\) .*/\1/p' -e 's/.*KV cache [0-9.]* GiB (f16, \([0-9]*\) tokens).*/\1/p' "$1" | head -1; }
# 2.028 GiB: f16 about 28,700 of its 65,536 tokens, 8 bits about 54,000,
# 8-bit K and 4-bit V the whole window (Llama 3.1 8B on the PS5: 54,272,
# 101,376, 131,072).
TS5_GPU_MEMORY_GIB=2.028 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvlong.err" > /dev/null
n=$(f16_of "$work/kvlong.err")
[ -n "$n" ] && [ "$n" -ge 24576 ] && [ $((n + n / 8)) -le 65536 ] || { echo "FAIL: f16 '$n' tokens at 2.028 GiB"; exit 1; }
grep "8-bit KV cache: [0-9]* tokens, where f16 fits $n (TS5_KV=f16 keeps f16)" "$work/kvlong.err"
grep "8-bit K and 4-bit V KV cache: 65536 tokens, where 8 bits fit [0-9]* (TS5_KV=q8 keeps 8 bits)" "$work/kvlong.err"
grep "KV cache [0-9.]* GiB (k8v4, Q, K and V turned, 65536 tokens)" "$work/kvlong.err"
# 2.04: f16 40,960, 8 bits the whole window, and 8 bits stay (Qwen3.5 9B:
# 175,104 in f16, 262,144 in 8 bits).
TS5_GPU_MEMORY_GIB=2.04 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvlong8.err" > /dev/null
grep "KV cache [0-9.]* GiB (q8, Q, K and V turned, 65536 tokens)" "$work/kvlong8.err"
! grep "4-bit V KV cache:" "$work/kvlong8.err" || unexpected $LINENO
# 2.06: f16 62,464, less than an eighth short of the window: f16 stays, and
# with YaRN too (it doesn't stretch a context within the window).
for yarn in 0 1; do
    TS5_YARN=$yarn TS5_GPU_MEMORY_GIB=2.06 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvlongf.err" > /dev/null
    n=$(f16_of "$work/kvlongf.err")
    [ -n "$n" ] && [ "$n" -lt 65536 ] && [ $((n + n / 8)) -gt 65536 ] ||
        { echo "FAIL: f16 '$n' tokens at 2.06 GiB, TS5_YARN=$yarn: not just short of the window"; exit 1; }
    ! grep "8-bit KV cache:" "$work/kvlongf.err" || unexpected $LINENO
done
echo "f16 keeps $n of 65536 tokens, with YaRN and without"
# With YaRN, f16 that holds the model's own window stays f16 past it (Coder
# 7B on the PS5: 114,688 tokens), and f16 well short of it takes 8 bits and
# then 8-bit K and 4-bit V past it.
TS5_YARN=1 TS5_GPU_MEMORY_GIB=2.07 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvyarnf.err" > /dev/null
grep "YaRN: positions stretched" "$work/kvyarnf.err"
grep "KV cache [0-9.]* GiB (f16, " "$work/kvyarnf.err"
TS5_YARN=1 TS5_GPU_MEMORY_GIB=2.04 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvyarn8.err" > /dev/null
grep "8-bit KV cache: [0-9]* tokens, where f16 fits 40960 (TS5_KV=f16 keeps f16)" "$work/kvyarn8.err"
grep "KV cache [0-9.]* GiB (k8v4, Q, K and V turned, " "$work/kvyarn8.err"
grep "YaRN: positions stretched" "$work/kvyarn8.err"
# The Gemmas keep f16 under 24,576 tokens and past it (Gemma 3 and 4 12B
# fail the check in 8 bits): Gemma 3 at 19,456 and 29,696, Gemma 4 (V from K,
# heads of 512) at 15,360.
for run in "$g3l 2.021" "$g3l 2.03" "$g4l 2.03"; do
    set -- $run
    TS5_GPU_MEMORY_GIB=$2 "$ts5" run -m "$1" -c auto -p hi -n 1 2> "$work/kvgemma.err" > /dev/null
    grep "KV cache [0-9.]* GiB (f16, " "$work/kvgemma.err"
    ! grep "8-bit KV cache:" "$work/kvgemma.err" || unexpected $LINENO
done
# Not even 1,024 tokens: refused, with no cache changes on the way.
{ TS5_GPU_MEMORY_GIB=1.5 "$ts5" run -m "$ql" -c auto -p hi -n 1 2> "$work/kvnone.err" > /dev/null || true; }
grep "not enough GPU memory left for a context of 1024 tokens" "$work/kvnone.err"
! grep "KV cache:" "$work/kvnone.err" || unexpected $LINENO
echo "== YaRN (TS5_YARN): Vulkan vs CPU stretched 4x, 300 tokens and one per call; ts5 check; -c auto past the window"
TS5_YARN=4 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --ntok 300 \
    "$work/qwen3-Q4_0.gguf" "$work/qwen2-Q8_0.gguf"
TS5_YARN=4 python3 "$here/compare.py" --ts5 "$ts5" --ref cpu --tol 0.002 --backend vk --step "$work/qwen3-Q4_0.gguf"
# It changes the logits (so the passes above test it), and only with TS5_YARN.
"$ts5" logits -m "$work/qwen3-Q4_0.gguf" -o "$work/yarn0.bin" 1 2 3 4 5 6 7 8 2> /dev/null
TS5_YARN=2 "$ts5" logits -m "$work/qwen3-Q4_0.gguf" -o "$work/yarn2.bin" 1 2 3 4 5 6 7 8 2> /dev/null
if cmp -s "$work/yarn0.bin" "$work/yarn2.bin"; then echo "TS5_YARN=2 changed nothing"; exit 1; fi
TS5_YARN=0 "$ts5" logits -m "$work/qwen3-Q4_0.gguf" -o "$work/yarn2.bin" 1 2 3 4 5 6 7 8 2> /dev/null
cmp "$work/yarn0.bin" "$work/yarn2.bin"
# The check runs its CPU reference stretched alike, and keeps no saved answers for it.
TS5_YARN=2 "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_YARN=2 "$ts5" check -m "$work/qwen3-Q4_0.gguf" -n 64 2> /dev/null | grep ", cpu [0-9]* ms: PASS$"
TS5_YARN=1 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "YaRN: positions stretched 4.00x over the 4096-token window, for 16384 tokens of context"
TS5_YARN=1 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 6144 -p hi -n 1 2>&1 > /dev/null |
    grep "YaRN: positions stretched 1.50x over the 4096-token window, for 6144 tokens of context"
if TS5_YARN=1 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -c 4096 -p hi -n 1 2>&1 > /dev/null | grep YaRN; then
    echo "YaRN on within the model's own window"; exit 1
fi
TS5_YARN=1 "$ts5" run -m "$work/llama-Q8_0.gguf" -c auto -p hi -n 1 2>&1 > /dev/null | grep "context 4096$"
echo "== -c auto with a draft model: what is left after the context, or room kept for it (TS5_HELPER_ROOM=1)"
TS5_GPU_MEMORY_GIB=4.5 TS5_DRAFT_MODEL="$work/qwen3-Q4_0.gguf" "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 \
    2>&1 > /dev/null | grep "draft model qwen3-Q4_0.gguf ready"
TS5_GPU_MEMORY_GIB=4.5 TS5_HELPER_ROOM=1 TS5_DRAFT_MODEL="$work/qwen3-Q4_0.gguf" "$ts5" run -m "$work/qwen3-F32.gguf" \
    -c auto -p hi -n 1 2>&1 > /dev/null | grep "draft model qwen3-Q4_0.gguf ready"
# Where the model doesn't fit beside that room, it loads alone, before any
# weight goes up, with the context it has with no helper.
TS5_GPU_MEMORY_GIB=3 TS5_HELPER_ROOM=1 TS5_DRAFT_MODEL="$work/qwen3-Q4_0.gguf" "$ts5" run -m "$work/qwen3-F32.gguf" \
    -c auto -p hi -n 1 2> "$work/alone.err" > /dev/null
grep "qwen3-F32.gguf doesn't fit with room for the draft model (not enough .*); loading it alone" "$work/alone.err"
grep "draft model not used: \(no room for it beside qwen3-F32.gguf\|not enough GPU memory left for it\)" "$work/alone.err"
grep "backend ready in [0-9.]* s, context 4096$" "$work/alone.err"
TS5_GPU_MEMORY_GIB=3 "$ts5" run -m "$work/qwen3-F32.gguf" -c auto -p hi -n 1 2>&1 > /dev/null |
    grep "backend ready in [0-9.]* s, context 4096$"
echo "== a model's own tuned settings (ts5-tune --models): where nothing set them, for that file only"
tune=$(mktemp -d)
mkdir -p "$tune/tokenstation/tune"
printf '# a test\nsize=%s\nTS5_DRAFT=2\nTS5_DRAFT_MODEL=helper.gguf\nRADV_PERFTEST=nocswave32\n' \
    "$(stat -c %s "$work/qwen3-Q4_0.gguf")" > "$tune/tokenstation/tune/qwen3-Q4_0.gguf.env"
XDG_CONFIG_HOME="$tune" "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null |
    grep -x "ts5: settings for qwen3-Q4_0.gguf from $tune/tokenstation/tune/qwen3-Q4_0.gguf.env: TS5_DRAFT=2 TS5_DRAFT_MODEL=$work/helper.gguf"
XDG_CONFIG_HOME="$tune" TS5_DRAFT=4 "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null |
    grep -x "ts5: settings for qwen3-Q4_0.gguf from .*: TS5_DRAFT_MODEL=$work/helper.gguf"
! XDG_CONFIG_HOME="$tune" TS5_MODEL_TUNE= "$ts5" run -m "$work/qwen3-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null | grep -q "settings for" || unexpected $LINENO
! XDG_CONFIG_HOME="$tune" "$ts5" run -m "$work/qwen2-Q4_0.gguf" -p hi -n 1 2>&1 > /dev/null | grep -q "settings for" || unexpected $LINENO
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
        "$work/llama-F32.gguf" "$work/qwen3-Q4_K_M.gguf" "$work/qwen35-F32.gguf"
    echo "== Gemma 3 vs llama.cpp: 600 tokens across the window, with its linear RoPE scale"
    python3 "$here/compare.py" --ts5 "$ts5" --ref "$ref" --backend vk --ntok 600 "$g3"
    echo "== Gemma 3 12B's heads vs llama.cpp: generation after a 1,100-token prompt"
    "$ts5" logits -m "$g3w" -o "$work/tail-vk.bin" --tail 8 $tail_toks 2> /dev/null
    TS5_REF_TAIL=8 "$ref" "$g3w" "$work/tail-ref.bin" $tail_toks 2> /dev/null
    python3 - "$work/tail-vk.bin" "$work/tail-ref.bin" <<'PY'
import sys
import numpy as np
a = np.fromfile(sys.argv[1], np.float32).reshape(-1, 32000)
b = np.fromfile(sys.argv[2], np.float32).reshape(-1, 32000)
d = float((np.abs(a - b).max(axis=1) / (b.max(axis=1) - b.min(axis=1))).max())
if a.shape != b.shape or d > 0.002:
    sys.exit(f"FAIL: generation rows {len(a)} and {len(b)}, ts5 vs llama.cpp {d:.5f} of the logit range")
print(f"8 rows, ts5 vs llama.cpp {d:.6f} of the logit range")
PY
    rm -f "$work/tail-vk.bin" "$work/tail-ref.bin"
    echo "== YaRN vs llama.cpp's (rope scaling yarn over the test models' 4096-token window)"
    for f in 1.5 4; do
        TS5_YARN=$f TS5_REF_YARN=$f:4096 python3 "$here/compare.py" --ts5 "$ts5" --ref "$ref" --backend vk \
            "$work/qwen2-F32.gguf" "$work/qwen3-F32.gguf"
    done
    echo "== tokenizer vs llama-tokenize"
    python3 "$here/tokenizer_test.py" --ts5 "$ts5" --ref "$bin/llama-tokenize" "$work/qwen2-Q4_0.gguf" \
        "$work/llama-Q4_0.gguf" "$work/llama3-Q8_0.gguf" "$work/qwen35-Q4_K_M.gguf"
fi
echo "all tests passed"
