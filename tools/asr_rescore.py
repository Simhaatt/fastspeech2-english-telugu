"""Rescore saved ASR transcripts with a script-safe normaliser.

Whisper's BasicTextNormalizer drops Unicode combining marks, which in Telugu are the vowel signs and the
virama, so it would compare consonant skeletons only. This keeps every letter and mark and removes only
punctuation and symbols. CER is reported with and without spaces (word-splitting differences such as
'న్యూరో ఫార్మకో...' vs one word should not count as letter errors).
"""
import json
import re
import sys
import unicodedata

import jiwer


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


def score(refs, hyps):
    r = [norm(x) for x in refs]
    h = [norm(x) or "<empty>" for x in hyps]
    return {"cer": jiwer.cer(r, h), "cer_nospace": jiwer.cer([x.replace(" ", "") for x in r],
                                                            [x.replace(" ", "") for x in h]),
            "wer": jiwer.wer(r, h), "n": len(r)}


d = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("real_vocoded", "synthesized"):
    v = d["val"][key]
    s = score(v["ref"], v["hyp"])
    short = sum(len(norm(h)) < 0.8 * len(norm(r)) for r, h in zip(v["ref"], v["hyp"]))
    print(f"{key:14s} CER {s['cer']:.3f}  CER(no spaces) {s['cer_nospace']:.3f}  WER {s['wer']:.3f}  "
          f"n {s['n']}  hyps <80% of ref length: {short}")
t = d["tests"]
s = score([x["ref"] for x in t], [x["hyp"] for x in t])
print(f"{'test':14s} CER {s['cer']:.3f}  CER(no spaces) {s['cer_nospace']:.3f}  WER {s['wer']:.3f}")
for x in t:
    s = score([x["ref"]], [x["hyp"]])
    print(f"   {x['id']:14s} CER {s['cer']:.3f}  CER(no spaces) {s['cer_nospace']:.3f}  WER {s['wer']:.3f}")
