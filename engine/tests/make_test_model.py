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


def copy_tokenizer(reader, writer, pad_tokens=0):
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
            # Qwen 2.5 7B GGUFs pad the token list itself with placeholders;
            # the parallel scores and types arrays grow with it.
            if field.name == "tokenizer.ggml.tokens":
                vals += [f"[PAD{len(vals) + i}]" for i in range(pad_tokens)]
            elif field.name == "tokenizer.ggml.scores":
                vals += [0.0] * pad_tokens
            elif field.name == "tokenizer.ggml.token_type":
                vals += [5] * pad_tokens  # UNUSED
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
    ap.add_argument("--arch", default="qwen2", choices=["llama", "qwen2", "qwen3"])
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--embd", type=int, default=256)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pad-vocab", type=int, default=0,
                    help="extra embedding and output rows past the token list, as released models pad")
    ap.add_argument("--pad-tokens", type=int, default=0,
                    help="placeholder entries added to the token list itself (with rows to match), as Qwen 7B pads")
    ap.add_argument("--quant", default="F32", choices=["F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q4_K_M"],
                    help="type for the 2-D weights; Q4_K_M: Q4_K, with Q6_K for attn_v, ffn_down and output, "
                         "as llama-quantize's mix")
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
    w.add_block_count(a.layers)
    w.add_feed_forward_length(a.ff)
    w.add_head_count(a.heads)
    w.add_head_count_kv(a.kv_heads)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(1000000.0 if a.arch != "llama" else 10000.0)
    if a.arch == "qwen3":
        w.add_key_length(hd)
        w.add_value_length(hd)
    w.add_file_type(0)
    copy_tokenizer(reader, w, a.pad_tokens)

    kmix = a.quant == "Q4_K_M"
    qtype = GGMLQuantizationType.Q4_K if kmix else getattr(GGMLQuantizationType, a.quant)

    def t(name, *shape, scale=None):
        # numpy shape is (ne1, ne0): rows are outputs, columns are inputs.
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        data = (rng.standard_normal(shape) * s).astype(np.float32)
        if kmix:
            if shape[-1] % 256:
                w.add_tensor(name, quants.quantize(data, GGMLQuantizationType.Q8_0),
                             raw_dtype=GGMLQuantizationType.Q8_0)
            elif any(k in name for k in ("attn_v", "ffn_down", "output")) and "attn_output" not in name:
                w.add_tensor(name, quantize_q6_k(data), raw_dtype=GGMLQuantizationType.Q6_K)
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

    t("token_embd.weight", n_vocab + a.pad_tokens + a.pad_vocab, a.embd, scale=1.0)
    for i in range(a.layers):
        p = f"blk.{i}."
        norm(p + "attn_norm.weight", a.embd)
        t(p + "attn_q.weight", a.heads * hd, a.embd)
        t(p + "attn_k.weight", a.kv_heads * hd, a.embd)
        t(p + "attn_v.weight", a.kv_heads * hd, a.embd)
        t(p + "attn_output.weight", a.embd, a.heads * hd)
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
    t("output.weight", n_vocab + a.pad_tokens + a.pad_vocab, a.embd)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {a.arch}, vocab {n_vocab}, {a.layers} layers")


if __name__ == "__main__":
    main()
