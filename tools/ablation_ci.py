"""Paired bootstrap for ablation_eval.py output: is a CER difference between two conditions larger than the
noise from having only N clips? Resamples clips with replacement (same clips for both conditions) and
reports the 95% interval of CER(condition) - CER(pred), with CER pooled over the resampled clips.

    python ablation_ci.py <ablation json> [B]
"""
import json
import re
import sys
import unicodedata

import jiwer
import numpy as np

d = json.load(open(sys.argv[1], encoding="utf-8"))
B = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
if "whisper-base.en" in d["asr"]:
    from transformers import WhisperProcessor
    norm = WhisperProcessor.from_pretrained(d["asr"]).tokenizer.normalize
else:
    def norm(t):
        t = unicodedata.normalize("NFC", t).lower()
        t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
        return re.sub(r"\s+", " ", t).strip()


def edits(cond):
    """Per clip: (character edits, reference characters)."""
    v = d["per_clip"][cond]
    out = []
    for r, h in zip(v["ref"], v["hyp"]):
        r, h = norm(r), norm(h) or "<empty>"
        m = jiwer.process_characters(r, h)
        out.append((m.substitutions + m.deletions + m.insertions, len(r)))
    return np.array(out, dtype=float)


rng = np.random.default_rng(0)
base = edits("pred")
idx = rng.integers(0, len(base), size=(B, len(base)))
print(f"{sys.argv[1]}  ({len(base)} clips, {B} resamples)")
for cond in d["per_clip"]:
    e = edits(cond)
    cer = e[:, 0].sum() / e[:, 1].sum()
    diff = e[idx, 0].sum(1) / e[idx, 1].sum(1) - base[idx, 0].sum(1) / base[idx, 1].sum(1)
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print(f"  {cond:13s} CER {100 * cer:5.1f}   minus pred: {100 * (cer - base[:, 0].sum() / base[:, 1].sum()):+5.1f}"
          f"  [95% {100 * lo:+5.1f}, {100 * hi:+5.1f}]")
# per-clip means: MCD (dB) and F0 RMSE (cents), paired against pred
for key in ("mcd", "f0_rmse"):
    b = np.array(d["per_clip"]["pred"][key], dtype=float)
    for cond in ("real_dur", "real_pe", "real_all"):
        x = np.array(d["per_clip"][cond][key], dtype=float) - b
        boot = np.nanmean(x[idx], 1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {key:8s} {cond:9s} minus pred: {np.nanmean(x):+7.2f}  [95% {lo:+7.2f}, {hi:+7.2f}]")
