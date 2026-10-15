#!/usr/bin/env python3
"""Writes ps5/catalog.h: the files TokenStation5 can download on the console,
each pinned to a Hugging Face commit with its size and SHA-256, and for a
model the header fields the title's memory estimate reads.

    scripts/catalog.py              # look everything up, write ps5/catalog.h
    scripts/catalog.py --dry-run    # print the file instead
    scripts/catalog.py --gguf FILE  # print what a local GGUF's header gives

Run it on a computer that can reach huggingface.co (HF_TOKEN for gated
repos). The models are the rows of models.tsv with a `console` column, which
says what each is good for on the Models page. The speech models and voices
are listed below, from the URLs scripts/ts5-ps5.sh fetches them from. A
model's file is the first .gguf in its repo whose name has file_match, as
scripts/fetch-models.sh picks it, at the repo's newest commit. Its header is
read with ranged requests (it sits in the first few MiB), and its weight bytes
are counted as the engine counts them (engine/src/gguf.cpp, model.cpp), so a
row's MEMORY NEEDED before the download is what the model shows after it.
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = "https://huggingface.co"

# Speech to text (whisper.cpp's conversions) and the voice (kokoro.cpp's
# conversions of Kokoro-82M and its voices): key, repo, path in the repo,
# name on the PS5, name on screen.
SPEECH = [
    ("speech-base-en", "ggerganov/whisper.cpp", "ggml-base.en.bin", "ggml-base.en.bin", "Base (English)"),
    ("speech-small-en", "ggerganov/whisper.cpp", "ggml-small.en.bin", "ggml-small.en.bin", "Small (English)"),
]
VOICE = [
    ("kokoro", "simonfxr/kokoro.cpp-GGUF", "kokoro-82m-f16.gguf", "kokoro-82m-f16.gguf", "Kokoro"),
    ("voice-af-heart", "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-af_heart.gguf",
     "kokoro-voice-af_heart.gguf", "Heart"),
    ("voice-am-michael", "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-am_michael.gguf",
     "kokoro-voice-am_michael.gguf", "Michael"),
    ("voice-bf-emma", "simonfxr/kokoro.cpp-GGUF", "voices/kokoro-voice-bf_emma.gguf",
     "kokoro-voice-bf_emma.gguf", "Emma"),
]

# GGML types the engine has blocks for: values per block, bytes per block.
# Any other type counts 0 bytes, as in gguf.cpp.
BLOCKS = {0: (1, 4), 1: (1, 2), 30: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),
          8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210)}


def curl(args, out=None):
    cmd = ["curl", "-fsSL", "--retry", "3"]
    if os.environ.get("HF_TOKEN"):
        cmd += ["-H", "Authorization: Bearer " + os.environ["HF_TOKEN"]]
    cmd += args
    if out:
        cmd += ["-o", out]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.exit("catalog: curl %s failed: %s" % (" ".join(args), r.stderr.decode(errors="replace").strip()))
    return r.stdout


def api(path):
    return json.loads(curl([HF + "/api/" + path]))


def commit_of(repo):
    sha = api("models/" + repo).get("sha", "")
    if len(sha) != 40:
        sys.exit("catalog: %s: no commit in the API's answer" % repo)
    return sha


def entries(repo, commit, folder=""):
    return api("models/%s/tree/%s%s" % (repo, commit, "/" + folder if folder else ""))


def pinned(repo, commit, path):
    """The file's size and SHA-256 at the commit (its LFS object)."""
    folder = os.path.dirname(path)
    for e in entries(repo, commit, folder):
        if e.get("path") == path:
            lfs = e.get("lfs") or {}
            if not lfs.get("oid") or not lfs.get("size"):
                sys.exit("catalog: %s/%s isn't an LFS file, so it has no SHA-256 in the API" % (repo, path))
            return int(lfs["size"]), lfs["oid"]
    sys.exit("catalog: %s has no %s at %s" % (repo, path, commit))


class Header:
    """A GGUF header read from bytes that may stop short: `need` says how
    many more to fetch."""

    class Short(Exception):
        pass

    def __init__(self, data):
        self.d, self.p = data, 0

    def take(self, n):
        if self.p + n > len(self.d):
            raise Header.Short()
        b = self.d[self.p:self.p + n]
        self.p += n
        return b

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        return self.take(self.u64()).decode("utf-8", errors="replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.u32(), self.u64()
            size = {0: 1, 1: 1, 7: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}.get(et)
            if size:
                self.take(size * n)  # numbers the catalog doesn't need (token scores, types)
                return n
            return [self.value(et) for _ in range(n)]
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def parse(self):
        if self.take(4) != b"GGUF":
            sys.exit("catalog: not a GGUF file")
        version, n_tensors, n_kv = self.u32(), self.u64(), self.u64()
        if version < 2:
            sys.exit("catalog: GGUF v1")
        kv = {}
        for _ in range(n_kv):
            k = self.string()
            kv[k] = self.value(self.u32())
        tensors = {}
        for _ in range(n_tensors):
            name = self.string()
            ne = [self.u64() for _ in range(self.u32())]
            t, off = self.u32(), self.u64()
            n = 1
            for d in ne:
                n *= d
            vals, size = BLOCKS.get(t, (0, 0))
            tensors[name] = (ne, off + (n // vals * size if vals else 0))
        return kv, tensors


def hparams(kv, tensors):
    """What model.cpp reads, and data_bytes()."""
    a = kv.get("general.architecture", "") + "."
    n_embd = kv.get(a + "embedding_length", 0)
    n_head = kv.get(a + "attention.head_count", 0)
    h = {
        "name": kv.get("general.name", ""),
        "n_vocab": tensors["token_embd.weight"][0][1] if "token_embd.weight" in tensors else 0,
        "n_embd": n_embd,
        "n_layer": kv.get(a + "block_count", 0),
        "n_ff": kv.get(a + "feed_forward_length", 0),
        "n_head": n_head,
        "n_head_kv": kv.get(a + "attention.head_count_kv", n_head),
        "head_dim": kv.get(a + "attention.key_length", n_embd // n_head if n_head else 0),
        "ctx_train": kv.get(a + "context_length", 4096),
        "weight_bytes": max((end for _, end in tensors.values()), default=0),
    }
    for k, v in h.items():
        if k != "name" and not isinstance(v, int):
            sys.exit("catalog: %s is %r, not a number" % (k, v))
    return h


def remote_header(repo, commit, path):
    url = "%s/%s/resolve/%s/%s" % (HF, repo, commit, path)
    size = 16 << 20
    with tempfile.TemporaryDirectory() as d:
        part = os.path.join(d, "head")
        while True:
            curl(["-r", "0-%d" % (size - 1), url], out=part)
            with open(part, "rb") as f:
                data = f.read()
            try:
                return hparams(*Header(data).parse())
            except Header.Short:
                if size >= 256 << 20:
                    sys.exit("catalog: %s's header is longer than 256 MiB" % path)
                size *= 2


def pick(repo, commit, match):
    for e in entries(repo, commit):
        p = e.get("path", "")
        low = p.lower()
        if low.endswith(".gguf") and "mmproj" not in low and "-of-0" not in low and match.lower() in low:
            return p
    sys.exit("catalog: no %s .gguf in %s" % (match, repo))


def same_as(file, match):
    """What a file on the PS5 has in its name when it is this model, in any
    precision and from any uploader: lower case, without the quantization,
    an uploader's "Org_" prefix, "Meta-" or "-Instruct" (as
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" gives "llama-3.1-8b")."""
    s = file[:-5].lower() if file.lower().endswith(".gguf") else file.lower()
    q = "-" + match.lower()
    if q in s:
        s = s[:s.rindex(q)]
    head = s.split("-", 1)[0]
    if "_" in head:
        s = s[s.index("_") + 1:]
    if s.startswith("meta-"):
        s = s[5:]
    if s.endswith("-instruct"):
        s = s[:-9]
    return s


def c_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def models():
    rows = []
    with open(os.path.join(ROOT, "models.tsv")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) >= 8 and cols[7].strip() and cols[7].strip() != "-":
                rows.append(cols)
    return rows


def write(out):
    lines = []
    for cols in models():
        key, repo, match, good_for = cols[0], cols[1], cols[2], cols[7].strip()
        commit = commit_of(repo)
        path = pick(repo, commit, match)
        size, sha = pinned(repo, commit, path)
        h = remote_header(repo, commit, path)
        print("catalog: %s: %s at %s, %d bytes, %s" % (key, path, commit[:12], size, h["name"]), file=sys.stderr)
        same = same_as(os.path.basename(path), match)
        lines.append("    {CatalogEntry::Model, %s, %s, %s, %s,\n     %s, %s,\n     %s,\n     %s, %du, %s,\n"
                     "     %d, %d, %d, %d, %d, %d, %d, %d, %du},"
                     % (c_str(key), c_str(h["name"] or stem), c_str(good_for), c_str(same), c_str(repo),
                        c_str(path), c_str(os.path.basename(path)), c_str(commit), size, c_str(sha),
                        h["n_vocab"], h["n_embd"], h["n_layer"], h["n_ff"], h["n_head"], h["n_head_kv"],
                        h["head_dim"], h["ctx_train"], h["weight_bytes"]))
    commits = {}
    for kind, rows in (("Speech", SPEECH), ("Voice", VOICE)):
        for key, repo, path, file, name in rows:
            if repo not in commits:
                commits[repo] = commit_of(repo)
            commit = commits[repo]
            size, sha = pinned(repo, commit, path)
            print("catalog: %s: %s at %s, %d bytes" % (key, path, commit[:12], size), file=sys.stderr)
            lines.append("    {CatalogEntry::%s, %s, %s, \"\", %s,\n     %s, %s,\n     %s,\n     %s, %du, %s},"
                         % (kind, c_str(key), c_str(name), c_str(file.lower()), c_str(repo), c_str(path),
                            c_str(file), c_str(commit), size, c_str(sha)))
    text = HEAD + "\n".join(lines) + "\n" + TAIL
    if out == "-":
        sys.stdout.write(text)
    else:
        with open(out, "w") as f:
            f.write(text)
        print("catalog: wrote %s" % os.path.relpath(out, ROOT), file=sys.stderr)


HEAD = """// What TokenStation5 can download on the console (downloads.h): the models
// on the Models page, and the speech model and voices for Settings. Each is
// pinned to a commit on Hugging Face with its size and SHA-256, and a model
// carries the header fields its memory estimate reads (library.h), so a row
// can say how much memory the model needs before it's on the PS5.
// Written by scripts/catalog.py from models.tsv; run it again to change it.
#pragma once
#include <cstdint>

namespace app {

struct CatalogEntry {
    enum Kind { Model, Speech, Voice } kind;
    const char *key;       // its row in models.tsv, or its line in scripts/catalog.py
    const char *name;      // on screen; a model's general.name
    const char *good_for;  // a model's, on screen
    const char *same;      // lower case: a file on the PS5 whose name has it is this one already
    const char *repo;      // on huggingface.co
    const char *path;      // the file in the repo
    const char *file;      // its name on the PS5, in the models folder
    const char *commit;    // the repo's commit it is pinned to
    uint64_t bytes;
    const char *sha256;    // hex
    // A model's header fields (model.h HParams) and its weight bytes.
    int n_vocab = 0, n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0, n_head_kv = 0, head_dim = 0, ctx_train = 0;
    uint64_t weight_bytes = 0;
};

inline const CatalogEntry kCatalog[] = {
"""

TAIL = """};

}  // namespace app
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="print the file instead of writing it")
    ap.add_argument("--gguf", metavar="FILE", help="print what a local GGUF's header gives, and stop")
    a = ap.parse_args()
    if a.gguf:
        with open(a.gguf, "rb") as f:
            data = f.read(256 << 20)
        print(json.dumps(hparams(*Header(data).parse()), indent=1))
        return
    write("-" if a.dry_run else os.path.join(ROOT, "ps5", "catalog.h"))


if __name__ == "__main__":
    main()
