#!/usr/bin/env python3
# Packs the English (US) pronunciation lexicon the PS5 title's G2P reads
# (ps5/tts/g2p.cpp) into one sorted, binary-searchable file:
#
#   1. misaki's us_gold (Apache-2.0), as it is, in misaki's US phoneme symbols;
#   2. CMUdict (BSD-style), mapped from ARPAbet to the same symbols, for the
#      words gold lacks;
#   3. misaki's us_silver only with --with-silver. It is off by default: where
#      gold and silver came from is still an open question for the owner.
#
# It downloads pinned revisions, checks their sha256, and writes the same
# bytes every time from the same inputs. Python 3, standard library only.
#
#   python3 scripts/pack-g2p-lexicon.py                 # writes ps5/tts/lexicon.bin
#   python3 scripts/pack-g2p-lexicon.py --with-silver   # gold + silver + CMUdict
#   python3 scripts/pack-g2p-lexicon.py --compare       # CMUdict mapping vs gold, no output
#
# Downloads are kept in --cache (default ~/.cache/ts5-g2p), not in the repo.
#
# Format, little-endian:
#   "TS5G", u32 version (1),
#   u32 symbol count, then each symbol as u8 byte length + UTF-8; phoneme
#       code k (1..count) is symbol k-1,
#   u32 tag count, then each tag as u8 length + ASCII; tag byte 0xC0+i is tag i,
#   u32 entry count n, u32 record offsets[n + 1] from the start of the records,
#   records, sorted by key bytes: key (ASCII), 0, source (0 gold, 1 silver,
#       2 CMUdict), the default reading's phoneme codes, then for each other
#       reading its tag byte and phoneme codes. A reading with no codes means
#       "spell the letters" (misaki's None, used for acronyms read as nouns).
#
# Homographs keep every reading misaki's gold gives, keyed by its tags (NOUN,
# VERB, ADJ, VBD, ..., and None: the reading before a pause); the G2P picks
# one from the words around it and falls back to DEFAULT.
import argparse, hashlib, json, os, struct, sys, urllib.request

RAW = "https://raw.githubusercontent.com"
MISAKI = "hexgrad/misaki/fba1236595f2d2bf21d414ba6e57d25256afada3"  # misaki 0.9.4
CMUDICT = "cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690"
SOURCES = {
    "us_gold.json": (f"{RAW}/{MISAKI}/misaki/data/us_gold.json",
                     "dc414872a49a28ae6c141463d502fd945f3b2fde040484fdc47d00cc4612686f"),
    "us_silver.json": (f"{RAW}/{MISAKI}/misaki/data/us_silver.json",
                       "de8f67be911bb6c659187b4a65fd966b6a30e56350e0f790d763210b053ac475"),
    "cmudict.dict": (f"{RAW}/{CMUDICT}/cmudict.dict",
                     "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22"),
}

# misaki's US symbols (its US_VOCAB), plus ɐ, the weak "a" its G2P writes.
SYMBOLS = list("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔɐ")
CODE = {s: i + 1 for i, s in enumerate(SYMBOLS)}
TAGS = ["DEFAULT", "None", "NOUN", "VERB", "ADJ", "ADV", "DT", "VBD", "VBN", "VBP"]
GOLD, SILVER, CMU = 0, 1, 2

# Kokoro-82M v1.0's 114 input symbols (config.json "vocab"). Every symbol the
# lexicon holds must be one of them, or Kokoro would drop it.
KOKORO_VOCAB = set(';:,.!?—…"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ')


def fetch(name, cache):
    url, sha = SOURCES[name]
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as r:
            data = r.read()
        with open(path + ".part", "wb") as f:
            f.write(data)
        os.replace(path + ".part", path)
    data = open(path, "rb").read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        sys.exit(f"{path}: sha256 {got}, expected {sha} (delete it to download again)")
    return data


# ARPAbet to misaki. Vowels carry a stress digit; misaki puts its stress mark
# right before the stressed vowel.
VOWELS = {"AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "W", "AY": "I", "EH": "ɛ", "ER": "ɜɹ",
          "EY": "A", "IH": "ɪ", "IY": "i", "OW": "O", "OY": "Y", "UH": "ʊ", "UW": "u"}
CONSONANTS = {"B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h", "JH": "ʤ", "K": "k",
              "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t",
              "TH": "θ", "V": "v", "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ"}


SIBILANTS = ("T", "D", "S", "Z", "SH", "ZH", "CH", "JH")


def from_arpabet(phones):
    """CMUdict phones ("HH AH0 L OW1") as misaki writes them ("həlˈO").

    The rules for t and the unstressed vowels follow what gold writes for
    the 33k words both have; scripts/pack-g2p-lexicon.py --compare measures it."""
    ph = [(p[:-1], p[-1]) if p[-1].isdigit() else (p, None) for p in phones]
    n = len(ph)
    if n > 2 and ph[-1] == ("IY", "2"):  # gold leaves a final -y unstressed
        ph[-1] = ("IY", "0")
    name = lambda i: ph[i][0] if 0 <= i < n else None
    vowel = lambda i: 0 <= i < n and ph[i][1] is not None
    # Only consonants from i to the end: i is in the last syllable.
    last_syllable = lambda i: all(not vowel(j) for j in range(i, n))
    out = []
    for i, (p, s) in enumerate(ph):
        if p == "":
            continue
        if s is None:
            sym = CONSONANTS[p]
            if p == "T" and i > 0:
                prev = i - 1 if name(i - 1) != "R" or i < 2 else i - 2
                if vowel(prev) and name(i + 1) == "AH" and ph[i + 1][1] == "0" and name(i + 2) == "N" and \
                        last_syllable(i + 3) and prev == i - 1:
                    # A t before a syllabic n is a glottal stop: "button" bˈʌʔn.
                    out.append("ʔn")
                    ph[i + 1] = ph[i + 2] = ("", None)  # consumed
                    continue
                # A t between a vowel (or vowel + r) and an unstressed vowel
                # is the American flap: "water" wˈɔɾəɹ.
                if vowel(prev) and vowel(i + 1) and ph[i + 1][1] == "0":
                    sym = "ɾ"
            out.append(sym)
            continue
        mark = {"1": "ˈ", "2": "ˌ"}.get(s, "")
        if p == "AH":
            v = "ʌ" if s != "0" else "ə"
            if s == "0" and i > 0 and not vowel(i - 1) and last_syllable(i + 2):
                # Syllabic l after a consonant, and syllabic n after t, d, s
                # or z: "pixel" pˈɪksᵊl, "student" stˈudᵊnt.
                if name(i + 1) == "L" and name(i - 1) != "R" or name(i + 1) == "N" and name(i - 1) in ("T", "D", "S", "Z"):
                    v = "ᵊ"
        elif p == "ER":
            v = "ɜɹ" if s != "0" else "əɹ"
        elif p == "EH" and s == "0" and i == 0 and not vowel(i + 2):
            v = "ə" if name(i + 1) == "M" else "ɪ"  # "embark", "expect"
        elif p == "IH" and s == "0":
            # Unstressed ih: the reduced ᵻ of -ed and -es, ɪ before ng, at the
            # start and in a last syllable ("-ic", "-ish", "-ive"), and ə in
            # "-ness", "-less", "-ate" and before a consonant and a vowel.
            if i == n - 2 and name(i + 1) in ("D", "Z") and name(i - 1) in SIBILANTS:
                v = "ᵻ"
            elif name(i + 1) == "NG":
                v = "ɪ"
            elif last_syllable(i + 1):
                v = "ə" if i == n - 2 and name(i + 1) in ("S", "T") and name(i - 1) in ("N", "L") else "ɪ"
            elif i == 0:
                v = "ə" if vowel(i + 2) else "ɪ"
            elif vowel(i + 2) or name(i - 1) == "D":
                v = "ə"
            else:
                v = "ɪ"
        else:
            v = VOWELS[p]
        out.append(mark + v)
    return "".join(out)


def read_cmudict(data):
    words = {}
    for line in data.decode("utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        if "(" in word or word in words:  # alternates: keep the first reading
            continue
        words[word] = phones
    return words


def gold_has(lex, w):
    """True if misaki's lexicon (with its grown capitalized forms) covers w."""
    return w in lex or w.capitalize() in lex or (len(w) > 1 and w.lower() in lex and w == w.lower().capitalize())


def encode(ps):
    return bytes(CODE[c] for c in ps)


def record(key, source, value):
    out = key.encode("ascii") + b"\0" + bytes([source])
    if isinstance(value, str):
        return out + encode(value)
    out += encode(value["DEFAULT"] or "")
    for tag in sorted((t for t in value if t != "DEFAULT"), key=TAGS.index):
        out += bytes([0xC0 + TAGS.index(tag)]) + encode(value[tag] or "")
    return out


def compare(gold, cmu):
    same = total = 0
    diffs = []
    for w, v in gold.items():
        if isinstance(v, str) and w in cmu:
            total += 1
            ps = from_arpabet(cmu[w])
            if ps == v:
                same += 1
            elif len(diffs) < 60:
                diffs.append(f"  {w}: gold {v}, cmudict {ps}")
    print(f"CMUdict mapped exactly as gold: {same} of {total} ({100 * same / total:.1f}%)")
    print("\n".join(diffs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "ps5", "tts", "lexicon.bin"))
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "ts5-g2p"))
    ap.add_argument("--with-silver", action="store_true", help="also pack misaki's us_silver (off by default)")
    ap.add_argument("--compare", action="store_true", help="report how well the CMUdict mapping matches gold")
    args = ap.parse_args()

    gold = json.loads(fetch("us_gold.json", args.cache))
    cmu = read_cmudict(fetch("cmudict.dict", args.cache))
    if args.compare:
        return compare(gold, cmu)
    entries = {k: (GOLD, v) for k, v in gold.items()}
    if args.with_silver:
        silver = json.loads(fetch("us_silver.json", args.cache))
        for k, v in silver.items():
            if not gold_has(gold, k):
                entries[k] = (SILVER, v)
    added = 0
    for w, phones in cmu.items():
        if not gold_has(entries, w) and w.isascii():
            entries[w] = (CMU, from_arpabet(phones))
            added += 1

    for k, (_, v) in entries.items():
        for ps in [v] if isinstance(v, str) else v.values():
            bad = set(ps or "") - set(SYMBOLS)
            if bad or not set(ps or "") <= KOKORO_VOCAB:
                sys.exit(f"{k}: {ps!r} has symbols outside misaki's US set or Kokoro's vocab: {bad}")
    if not set(SYMBOLS + ["T"]) <= KOKORO_VOCAB:
        sys.exit("a symbol is missing from Kokoro's vocab")

    keys = sorted(entries, key=lambda k: k.encode("ascii"))
    records = [record(k, *entries[k]) for k in keys]
    head = b"TS5G" + struct.pack("<I", 1)
    head += struct.pack("<I", len(SYMBOLS)) + b"".join(bytes([len(s.encode())]) + s.encode() for s in SYMBOLS)
    head += struct.pack("<I", len(TAGS)) + b"".join(bytes([len(t)]) + t.encode() for t in TAGS)
    offsets, pos = [], 0
    for r in records:
        offsets.append(pos)
        pos += len(r)
    offsets.append(pos)
    blob = head + struct.pack("<I", len(keys)) + struct.pack(f"<{len(offsets)}I", *offsets) + b"".join(records)
    with open(args.out, "wb") as f:
        f.write(blob)
    n_gold = sum(1 for s, _ in entries.values() if s == GOLD)
    n_silver = sum(1 for s, _ in entries.values() if s == SILVER)
    print(f"{args.out}: {len(keys)} entries (gold {n_gold}, silver {n_silver}, CMUdict {added}), "
          f"{len(blob)} bytes, sha256 {hashlib.sha256(blob).hexdigest()}")


main()
