"""Checks that the training data and released checkpoints are what the report says.

1. For every clip: phone durations sum to the mel length, and pitch/energy/duration have one value per phone.
2. Re-computes the validation loss of the released fp16 checkpoint on the full validation set with the training
   code's own evaluate() (fp32, CPU), to compare with the logged best value.
3. English: the speaker embedding changes the voice (per-speaker real F0 vs predicted F0 for two speakers).

    python sanity_checks.py <project> <lang> <config> <ckpt> <preprocessed dir> <logged best val>
"""
import json
import os
import sys

import numpy as np
import torch

proj, lang, cfg, ckpt, pre_dir, logged = sys.argv[1:7]
sys.path.insert(0, os.path.join(proj, "core"))
os.chdir(proj)
from config_utils import load_config  # noqa: E402
from dataset import Dataset  # noqa: E402
from infer_core import load_model  # noqa: E402
from model import FastSpeech2Loss  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
import train_core  # noqa: E402

pre, mc, tc = load_config(cfg)
pre["path"]["preprocessed_path"] = pre_dir

# 1. durations vs mel length, one value per phone
bad_len, bad_count, n = [], [], 0
for split in ("train.txt", "val.txt"):
    for line in open(os.path.join(pre_dir, split), encoding="utf-8"):
        base, spk, phones, _ = line.rstrip("\n").split("|")
        nph = len(phones.strip("{}").split())
        f = lambda kind: os.path.join(pre_dir, kind, f"{spk}-{kind}-{base}.npy")
        d = np.load(f("duration"))
        mel_frames = np.load(f("mel"), mmap_mode="r").shape[0]
        n += 1
        if d.sum() != mel_frames:
            bad_len.append((base, int(d.sum()), mel_frames))
        if not (len(d) == len(np.load(f("pitch"))) == len(np.load(f("energy"))) == nph):
            bad_count.append(base)
print(f"[{lang}] 1. clips {n}: duration sum != mel frames in {len(bad_len)}, per-phone count mismatch in "
      f"{len(bad_count)}", bad_len[:3], bad_count[:3])

# 2. validation loss of the released checkpoint, computed by the training code
model = load_model((pre, mc, tc), ckpt, torch.device("cpu"))
ds = Dataset("val.txt", pre, tc, sort=False, drop_last=False)
loader = DataLoader(ds, batch_size=tc["optimizer"]["batch_size"] * 4, shuffle=False, collate_fn=ds.collate_fn)
val = train_core.evaluate(model, FastSpeech2Loss(pre, mc), loader, torch.device("cpu"), False)
names = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
print(f"[{lang}] 2. validation loss of {os.path.basename(ckpt)}: "
      + " ".join(f"{k} {v:.4f}" for k, v in zip(names, val)) + f" | logged best total {float(logged):.4f}")

# 3. speaker embedding (English only)
if lang == "English":
    stats = json.load(open(os.path.join(pre_dir, "stats.json")))
    p_mean, p_std = stats["pitch"][2:]
    for spk in ("en_s073", "en_s034"):
        vals = []
        for line in open(os.path.join(pre_dir, "train.txt"), encoding="utf-8"):
            base, s, _, _ = line.split("|")
            if s == spk:
                p = np.load(os.path.join(pre_dir, "pitch", f"{s}-pitch-{base}.npy"))
                d = np.load(os.path.join(pre_dir, "duration", f"{s}-duration-{base}.npy"))
                vals.append(p[d > 0])
        real = np.concatenate(vals) * p_std + p_mean
        sub = "" if spk == "en_s073" else os.path.join("other_speakers", spk)
        jd = os.path.join(proj, "Results", "English", sub, "variance_adaptor_plots", "json")
        pred = np.concatenate([json.load(open(os.path.join(jd, f"{t}.json")))["pitch_hz"]
                               for t in ("E1", "E2a", "E2b", "E3", "E4")])
        print(f"[{lang}] 3. {spk}: real training F0 median {np.median(real):.0f} Hz | predicted on test "
              f"sentences median {np.median(pred):.0f} Hz")
