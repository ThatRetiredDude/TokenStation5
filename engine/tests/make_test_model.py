#!/usr/bin/env python3
"""Writes a tiny random-weight GGUF for correctness tests.

    make_test_model.py --arch qwen2 --vocab ggml-vocab-qwen2.gguf out.gguf

The tokenizer is copied from a vocab-only GGUF (llama.cpp ships them under
models/), so llama.cpp can load the file and act as the reference. Weights are
F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1 or a Q4_K/Q6_K mix like llama.cpp's
Q4_K_M (--quant); llama-quantize makes the other K-quant variants. Dimensions are multiples of 256 so K-quants apply to every
matrix.
"""
import argparse

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quants


def copy_tokenizer(reader, writer, pad_tokens=0, rename_controls=False, rename_token=-1, rename_to="[RENAMED]",
                   pad_name="[PAD", pad_type=5, plain_spaces=False):
    tf = reader.fields.get("tokenizer.ggml.token_type")
    types = [tf.parts[i].tolist()[0] for i in tf.data] if tf else []
    for field in reader.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            etype = field.types[1]
            if etype == GGUFValueType.STRING:
                vals = [bytes(field.parts[i]).decode("utf-8", errors="surrogateescape") for i in field.data]
            else:
                vals = [field.parts[i].tolist()[0] for i in field.data]
            if field.name == "tokenizer.ggml.tokens":
                # As DeepSeek's distilled Qwens and Gemma 4 do: the chat's
                # control tokens under other names, the text tokens as they were.
                if rename_controls:
                    vals = [f"<|ctl{i}|>" if i < len(types) and types[i] in (3, 4) else v for i, v in enumerate(vals)]
                if rename_token >= 0:
                    vals[rename_token] = rename_to
                if plain_spaces:
                    vals = [v.replace("\u2581", " ") for v in vals]
            # Qwen 2.5 7B GGUFs pad the token list itself with placeholders;
            # the parallel scores and types arrays grow with it.
            if field.name == "tokenizer.ggml.tokens":
                close = ">" if pad_name.startswith("<") else "]"
                vals += [f"{pad_name}{len(vals) + i}{close}" for i in range(pad_tokens)]
            elif field.name == "tokenizer.ggml.scores":
                vals += [0.0] * pad_tokens
            elif field.name == "tokenizer.ggml.token_type":
                vals += [pad_type] * pad_tokens  # UNUSED, unless told otherwise
            writer.add_array(field.name, vals)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            val = field.parts[field.data[0]].tolist()[0]
            writer.add_key_value(field.name, val, vtype)


def quantize_q4_k(x):
    """Q4_K: per 256 values d, dmin (f16), 6-bit scales and mins for eight
    32-value sub-blocks, then 4-bit values; x ~ d * sc * q - dmin * m."""
    b = x.reshape(-1, 8, 32).astype(np.float64)
    lo = np.minimum(b.min(axis=2), 0.0)
    scale = (b.max(axis=2) - lo) / 15.0
    d = np.maximum(scale.max(axis=1), 1e-12) / 63.0
    dmin = np.maximum((-lo).max(axis=1), 1e-12) / 63.0
    d16, dmin16 = d.astype(np.float16), dmin.astype(np.float16)
    d, dmin = d16.astype(np.float64), dmin16.astype(np.float64)
    sc = np.clip(np.round(scale / d[:, None]), 0, 63).astype(np.uint8)
    mn = np.clip(np.round(-lo / dmin[:, None]), 0, 63).astype(np.uint8)
    step = np.maximum(d[:, None] * sc, 1e-30)[:, :, None]
    q = np.clip(np.round((b + (dmin[:, None] * mn)[:, :, None]) / step), 0, 15).astype(np.uint8)
    s = np.zeros((b.shape[0], 12), np.uint8)
    s[:, 0:4] = (sc[:, :4] & 63) | ((sc[:, 4:] >> 4) << 6)
    s[:, 4:8] = (mn[:, :4] & 63) | ((mn[:, 4:] >> 4) << 6)
    s[:, 8:12] = (sc[:, 4:] & 15) | ((mn[:, 4:] & 15) << 4)
    qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(-1, 128)
    out = np.concatenate([d16.view(np.uint8).reshape(-1, 2), dmin16.view(np.uint8).reshape(-1, 2), s, qs], axis=1)
    return out.reshape(x.shape[0], -1)


def quantize_q5_k(x):
    """Q5_K: as Q4_K (d, dmin, 6-bit scales and mins per 32), then 32 bytes
    of fifth bits (bit j of byte l: sub-block j, value l) and the 4-bit low
    parts; x ~ d * sc * q - dmin * m with q < 32."""
    b = x.reshape(-1, 8, 32).astype(np.float64)
    lo = np.minimum(b.min(axis=2), 0.0)
    scale = (b.max(axis=2) - lo) / 31.0
    d = np.maximum(scale.max(axis=1), 1e-12) / 63.0
    dmin = np.maximum((-lo).max(axis=1), 1e-12) / 63.0
    d16, dmin16 = d.astype(np.float16), dmin.astype(np.float16)
    d, dmin = d16.astype(np.float64), dmin16.astype(np.float64)
    sc = np.clip(np.round(scale / d[:, None]), 0, 63).astype(np.uint8)
    mn = np.clip(np.round(-lo / dmin[:, None]), 0, 63).astype(np.uint8)
    step = np.maximum(d[:, None] * sc, 1e-30)[:, :, None]
    q = np.clip(np.round((b + (dmin[:, None] * mn)[:, :, None]) / step), 0, 31).astype(np.uint8)
    s = np.zeros((b.shape[0], 12), np.uint8)
    s[:, 0:4] = (sc[:, :4] & 63) | ((sc[:, 4:] >> 4) << 6)
    s[:, 4:8] = (mn[:, :4] & 63) | ((mn[:, 4:] >> 4) << 6)
    s[:, 8:12] = (sc[:, 4:] & 15) | ((mn[:, 4:] & 15) << 4)
    qs = ((q[:, 0::2, :] & 15) | ((q[:, 1::2, :] & 15) << 4)).reshape(-1, 128)
    qh = np.zeros((b.shape[0], 32), np.uint8)
    for j in range(8):
        qh |= ((q[:, j, :] >> 4) << j).astype(np.uint8)
    out = np.concatenate([d16.view(np.uint8).reshape(-1, 2), dmin16.view(np.uint8).reshape(-1, 2), s, qh, qs], axis=1)
    return out.reshape(x.shape[0], -1)


def quantize_q6_k(x):
    """Q6_K: per 256 values 6-bit values (low nibbles, then high 2 bits),
    an int8 scale per 16 and d (f16); x ~ d * sc * (q - 32)."""
    b = x.reshape(-1, 16, 16).astype(np.float64)
    scale = np.abs(b).max(axis=2) / 31.0
    d = np.maximum(scale.max(axis=1), 1e-12) / 127.0
    d16 = d.astype(np.float16)
    d = d16.astype(np.float64)
    sc = np.clip(np.round(scale / d[:, None]), -128, 127).astype(np.int8)
    step = np.where(sc == 0, 1.0, d[:, None] * sc)[:, :, None]
    q = (np.clip(np.round(b / step), -32, 31) + 32).astype(np.uint8).reshape(-1, 2, 4, 32)  # [blk][half][i][l]
    ql = np.zeros((q.shape[0], 2, 64), np.uint8)
    ql[:, :, 0:32] = (q[:, :, 0] & 15) | ((q[:, :, 2] & 15) << 4)
    ql[:, :, 32:64] = (q[:, :, 1] & 15) | ((q[:, :, 3] & 15) << 4)
    qh = (q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4) | ((q[:, :, 3] >> 4) << 6)
    out = np.concatenate([ql.reshape(-1, 128), qh.reshape(-1, 64), sc.view(np.uint8), d16.view(np.uint8).reshape(-1, 2)],
                         axis=1)
    return out.reshape(x.shape[0], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen2", choices=["llama", "qwen2", "qwen3", "gemma4", "qwen35"])
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--global-hd", type=int, default=0,
                    help="gemma4: the full-attention layers' head size (every 3rd layer; default twice the others'), "
                         "on one K/V head without a V projection, as the 12B has it")
    ap.add_argument("--window", type=int, default=8, help="gemma4: the sliding layers' window")
    ap.add_argument("--rope-base", type=float, default=0, help="gemma4: the full layers' RoPE base (0: the default)")
    ap.add_argument("--prop-rope", type=float, default=0,
                    help="gemma4: proportional RoPE, rotating this fraction of the full layers' pairs (rope_freqs "
                         "1, the rest 1e30) in place of random rope_freqs")
    ap.add_argument("--qk-gain", type=float, default=1.0,
                    help="gemma4: scale on the q and k norms' weights, for sharp attention as trained models have")
    ap.add_argument("--ssm-k-heads", type=int, default=2, help="qwen35: the linear-attention layers' key heads")
    ap.add_argument("--ssm-v-heads", type=int, default=4, help="qwen35: their value heads (a multiple of the keys')")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="qwen35: scale on the linear layers' decay (A), smaller for a longer memory")
    ap.add_argument("--experts", type=int, default=0,
                    help="write <arch>.expert_count (the weights stay dense): a mixture-of-experts file's header")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pad-vocab", type=int, default=0,
                    help="extra embedding and output rows past the token list, as released models pad")
    ap.add_argument("--pad-tokens", type=int, default=0,
                    help="placeholder entries added to the token list itself (with rows to match), as Qwen 7B pads")
    ap.add_argument("--rename-controls", action="store_true",
                    help="name the control tokens differently (a drafter with another chat format)")
    ap.add_argument("--rename-token", type=int, default=-1, help="name this one text token differently")
    ap.add_argument("--rename-to", default="[RENAMED]", help="the renamed token's new name")
    ap.add_argument("--plain-spaces", action="store_true",
                    help="spell SentencePiece's U+2581 as a space in every token, as Gemma 4's files do")
    ap.add_argument("--pad-name", default="[PAD", help="the padding tokens' name prefix ([PAD or <unused, as Gemma)")
    ap.add_argument("--pad-type", type=int, default=5, help="the padding tokens' type (5 unused; Gemma 4 types its placeholders 1)")
    ap.add_argument("--boost-token", type=int, default=-1,
                    help="make this output row huge, so greedy sampling soon picks it: for guard tests")
    ap.add_argument("--quant", default="F32", choices=["F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K_M", "Q5_K_M"],
                    help="type for the 2-D weights; Q4_K_M: Q4_K, with Q6_K for attn_v, ffn_down and output, "
                         "as llama-quantize's mix; Q5_K_M: the same with Q5_K")
    ap.add_argument("out")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    reader = GGUFReader(a.vocab)
    n_vocab = len(reader.fields["tokenizer.ggml.tokens"].data)
    hd = a.embd // a.heads

    w = GGUFWriter(a.out, a.arch)
    w.add_name(f"ts5-test-{a.arch}")
    w.add_context_length(4096)
    w.add_embedding_length(a.embd)
    # Qwen3.5's files count its multi-token-prediction layer, whose weights
    # they leave out.
    w.add_block_count(a.layers + (1 if a.arch == "qwen35" else 0))
    w.add_feed_forward_length(a.ff)
    if a.experts:
        w.add_key_value(f"{a.arch}.expert_count", a.experts, GGUFValueType.UINT32)
    w.add_head_count(a.heads)
    if a.arch != "gemma4":
        w.add_head_count_kv(a.kv_heads)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0 if a.arch != "llama" else 10000.0)
    if a.arch in ("qwen3", "qwen35"):
        w.add_key_length(hd)
        w.add_value_length(hd)
    # Qwen3.5 as the 9B's file has it: every 4th layer full attention (its
    # query projection gives each head's query and an output gate, a quarter
    # of each head rotates, M-RoPE sections), the rest Gated DeltaNet linear
    # attention with fewer key heads than value heads and heads as wide as the
    # full layers'.
    q35 = a.arch == "qwen35"
    full = [(i + 1) % 4 == 0 for i in range(a.layers)]
    nk, nv = a.ssm_k_heads, a.ssm_v_heads
    conv_dim = 2 * nk * hd + nv * hd
    if q35:
        w.add_rope_dimension_count(hd // 4)
        pairs = hd // 8  # the rotated quarter's pairs, split over M-RoPE's axes as the 9B's [11, 11, 10]
        s = -(-pairs // 3)
        w.add_array("qwen35.rope.dimension_sections", [s, s, pairs - 2 * s, 0])
        w.add_key_value("qwen35.full_attention_interval", 4, GGUFValueType.UINT32)
        w.add_key_value("qwen35.nextn_predict_layers", 1, GGUFValueType.UINT32)
        w.add_array("qwen35.attention.recurrent_layers", [not f for f in full] + [False])
        w.add_key_value("qwen35.ssm.conv_kernel", 4, GGUFValueType.UINT32)
        w.add_key_value("qwen35.ssm.inner_size", nv * hd, GGUFValueType.UINT32)
        w.add_key_value("qwen35.ssm.state_size", hd, GGUFValueType.UINT32)
        w.add_key_value("qwen35.ssm.time_step_rank", nv, GGUFValueType.UINT32)
        w.add_key_value("qwen35.ssm.group_count", nk, GGUFValueType.UINT32)
    # Gemma 4 as the 12B's file has it: every 3rd layer here (6th there) is a
    # full-attention layer with wider heads on one K/V head and no V
    # projection; the rest slide; V is normalized; each layer's output is
    # scaled; the logits are soft-capped.
    g4 = a.arch == "gemma4"
    ghd = a.global_hd or 2 * hd
    swa = [(i % 3) != 2 for i in range(a.layers)]
    if g4:
        w.add_array("gemma4.attention.sliding_window_pattern", swa)
        w.add_array("gemma4.attention.head_count_kv", [a.kv_heads if s else 1 for s in swa])
        w.add_key_length(ghd)
        w.add_value_length(ghd)
        w.add_key_length_swa(hd)
        w.add_value_length_swa(hd)
        w.add_rope_dimension_count(ghd)
        w.add_key_value("gemma4.rope.dimension_count_swa", hd, GGUFValueType.UINT32)
        w.add_key_value("gemma4.rope.freq_base_swa", 10000.0, GGUFValueType.FLOAT32)
        w.add_sliding_window(a.window)
        if a.rope_base:
            w.add_rope_freq_base(a.rope_base)
        w.add_final_logit_softcapping(30.0)
        w.add_key_value("gemma4.attention.shared_kv_layers", 0, GGUFValueType.UINT32)
        w.add_key_value("gemma4.embedding_length_per_layer_input", 0, GGUFValueType.UINT32)
    w.add_file_type(0)
    copy_tokenizer(reader, w, a.pad_tokens, a.rename_controls, a.rename_token, a.rename_to, a.pad_name, a.pad_type,
                   a.plain_spaces)

    kmix = a.quant in ("Q4_K_M", "Q5_K_M")
    qtype = getattr(GGMLQuantizationType, a.quant[:4]) if kmix else getattr(GGMLQuantizationType, a.quant)

    def t(name, *shape, scale=None, boost_row=-1):
        # numpy shape is (ne1, ne0): rows are outputs, columns are inputs.
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        data = (rng.standard_normal(shape) * s).astype(np.float32)
        if boost_row >= 0:
            data[boost_row, :] = 100.0  # its logit dwarfs the rest whenever sum(h) > 0
        if kmix:
            if shape[-1] % 256:
                w.add_tensor(name, quants.quantize(data, GGMLQuantizationType.Q8_0),
                             raw_dtype=GGMLQuantizationType.Q8_0)
            elif any(k in name for k in ("attn_v", "ffn_down", "output")) and "attn_output" not in name:
                w.add_tensor(name, quantize_q6_k(data), raw_dtype=GGMLQuantizationType.Q6_K)
            elif qtype == GGMLQuantizationType.Q5_K:
                w.add_tensor(name, quantize_q5_k(data), raw_dtype=GGMLQuantizationType.Q5_K)
            else:
                w.add_tensor(name, quantize_q4_k(data), raw_dtype=GGMLQuantizationType.Q4_K)
        elif qtype == GGMLQuantizationType.F32:
            w.add_tensor(name, data)
        elif qtype == GGMLQuantizationType.F16:
            w.add_tensor(name, data.astype(np.float16))
        else:
            w.add_tensor(name, quants.quantize(data, qtype), raw_dtype=qtype)

    def norm(name, n):
        w.add_tensor(name, (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32))

    def f32(name, data):
        w.add_tensor(name, np.asarray(data, np.float32))

    t("token_embd.weight", n_vocab + a.pad_tokens + a.pad_vocab, a.embd, scale=1.0)
    if g4:
        if a.prop_rope:
            f = np.full(ghd // 2, 1e30, np.float32)
            f[: int(a.prop_rope * (ghd // 2))] = 1.0
        else:
            f = (1.0 + rng.random(ghd // 2)).astype(np.float32)
        w.add_tensor("rope_freqs.weight", f)
    for i in range(a.layers):
        p = f"blk.{i}."
        lhd, lkv = (hd, a.kv_heads) if not g4 or swa[i] else (ghd, 1)
        norm(p + "attn_norm.weight", a.embd)
        if q35:
            # The norm before the FFN goes by post_attention_norm here.
            norm(p + "post_attention_norm.weight", a.embd)
            if full[i]:
                t(p + "attn_q.weight", 2 * a.heads * hd, a.embd)
                t(p + "attn_k.weight", a.kv_heads * hd, a.embd)
                t(p + "attn_v.weight", a.kv_heads * hd, a.embd)
                t(p + "attn_output.weight", a.embd, a.heads * hd)
                norm(p + "attn_q_norm.weight", hd)
                norm(p + "attn_k_norm.weight", hd)
            else:
                t(p + "attn_qkv.weight", conv_dim, a.embd)
                t(p + "attn_gate.weight", nv * hd, a.embd)
                f32(p + "ssm_conv1d.weight", rng.standard_normal((conv_dim, 4)) * 0.5)
                f32(p + "ssm_dt.bias", rng.standard_normal(nv) * 0.5)
                # -exp(A_log), as the converter stores it: decays of exp(softplus(.) * a).
                f32(p + "ssm_a", -a.decay * rng.uniform(0.2, 2.0, nv))
                t(p + "ssm_beta.weight", nv, a.embd)
                t(p + "ssm_alpha.weight", nv, a.embd)
                norm(p + "ssm_norm.weight", hd)
                t(p + "ssm_out.weight", a.embd, nv * hd)
            t(p + "ffn_gate.weight", a.ff, a.embd)
            t(p + "ffn_up.weight", a.ff, a.embd)
            t(p + "ffn_down.weight", a.embd, a.ff)
            continue
        t(p + "attn_q.weight", a.heads * lhd, a.embd)
        t(p + "attn_k.weight", lkv * lhd, a.embd)
        if not g4 or swa[i]:
            t(p + "attn_v.weight", lkv * lhd, a.embd)
        t(p + "attn_output.weight", a.embd, a.heads * lhd)
        if g4:
            for n in ("q", "k"):
                w.add_tensor(p + f"attn_{n}_norm.weight", (a.qk_gain * (1.0 + 0.1 * rng.standard_normal(lhd))).astype(np.float32))
            norm(p + "post_attention_norm.weight", a.embd)
            norm(p + "post_ffw_norm.weight", a.embd)
            w.add_tensor(p + "layer_output_scale.weight", np.array([0.9 + 0.2 * rng.random()], np.float32))
        if a.arch == "qwen2":
            for n, dim in (("q", a.heads * hd), ("k", a.kv_heads * hd), ("v", a.kv_heads * hd)):
                w.add_tensor(p + f"attn_{n}.bias", (0.1 * rng.standard_normal(dim)).astype(np.float32))
        if a.arch == "qwen3":
            norm(p + "attn_q_norm.weight", hd)
            norm(p + "attn_k_norm.weight", hd)
        norm(p + "ffn_norm.weight", a.embd)
        t(p + "ffn_gate.weight", a.ff, a.embd)
        t(p + "ffn_up.weight", a.ff, a.embd)
        t(p + "ffn_down.weight", a.embd, a.ff)
    norm("output_norm.weight", a.embd)
    t("output.weight", n_vocab + a.pad_tokens + a.pad_vocab, a.embd, boost_row=a.boost_token)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {a.arch}, vocab {n_vocab}, {a.layers} layers")


if __name__ == "__main__":
    main()
