#!/usr/bin/env python3
"""Writes a tiny random-weight GGUF for correctness tests.

    make_test_model.py --arch qwen2 --vocab ggml-vocab-qwen2.gguf out.gguf

The tokenizer is copied from a vocab-only GGUF (llama.cpp ships them under
models/), so llama.cpp can load the file and act as the reference. Weights are
F32, F16, Q8_0 or Q4_0 (--quant); llama-quantize makes the K-quant variants. Dimensions
are multiples of 256 so K-quants apply to every matrix.
"""
import argparse

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFValueType, GGUFWriter, quants


def copy_tokenizer(reader, writer):
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
            writer.add_array(field.name, vals)
        elif vtype == GGUFValueType.STRING:
            writer.add_string(field.name, bytes(field.parts[field.data[0]]).decode("utf-8"))
        else:
            val = field.parts[field.data[0]].tolist()[0]
            writer.add_key_value(field.name, val, vtype)


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
    ap.add_argument("--quant", default="F32", choices=["F32", "F16", "Q8_0", "Q4_0"],
                    help="type for the 2-D weights (K-quants need llama-quantize)")
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
    copy_tokenizer(reader, w)

    qtype = getattr(GGMLQuantizationType, a.quant)

    def t(name, *shape, scale=None):
        # numpy shape is (ne1, ne0): rows are outputs, columns are inputs.
        s = scale if scale is not None else 1.0 / np.sqrt(shape[-1])
        data = (rng.standard_normal(shape) * s).astype(np.float32)
        if qtype == GGMLQuantizationType.F32:
            w.add_tensor(name, data)
        elif qtype == GGMLQuantizationType.F16:
            w.add_tensor(name, data.astype(np.float16))
        else:
            w.add_tensor(name, quants.quantize(data, qtype), raw_dtype=qtype)

    def norm(name, n):
        w.add_tensor(name, (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32))

    t("token_embd.weight", n_vocab, a.embd, scale=1.0)
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
    t("output.weight", n_vocab, a.embd)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {a.arch}, vocab {n_vocab}, {a.layers} layers")


if __name__ == "__main__":
    main()
