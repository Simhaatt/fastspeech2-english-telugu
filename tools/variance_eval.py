"""Released models vs ground truth on validation sentences (same phones, free-running prediction).

Per utterance: within-utterance spread (SD) of predicted vs real phone-level pitch and energy,
phone-level correlations, log-duration correlation, total-length ratio, and two mel over-smoothing
measures: global variance (per-bin variance over time) and spectral flux (mean |frame-to-frame change|),
each as predicted / real. Prints means and medians over N random validation clips.

    python variance_eval.py <project> <lang> <ckpt> <config> <preprocessed dir or zip::prefix> [N]
"""
import io
import json
import os
import random
import sys
import zipfile

import numpy as np
import torch

proj, lang, ckpt, cfg, data = sys.argv[1:6]
N = int(sys.argv[6]) if len(sys.argv) > 6 else 60
sys.path.insert(0, os.path.join(proj, "core"))
os.chdir(proj)
from config_utils import load_config  # noqa: E402
from infer_core import load_model  # noqa: E402
from text import text_to_sequence  # noqa: E402

if "::" in data:
    zpath, prefix = data.split("::")
    zf = zipfile.ZipFile(zpath)
    read = lambda rel: zf.read(prefix + "/" + rel)
else:
    read = lambda rel: open(os.path.join(data, rel), "rb").read()
npy = lambda rel: np.load(io.BytesIO(read(rel)))

configs = load_config(cfg)
configs[0]["path"]["preprocessed_path"] = os.path.dirname(os.path.abspath(ckpt))
model = load_model(configs, ckpt, torch.device("cpu"))
spk_map = json.loads(read("speakers.json"))
lines = read("val.txt").decode("utf-8").strip().split("\n")
random.Random(0).shuffle(lines)

rows = []
with torch.no_grad():
    for line in lines[:N]:
        base, spk, phones, _ = line.split("|")
        seq = np.array(text_to_sequence(phones, []))
        out = model(torch.LongTensor([spk_map[spk]]), torch.from_numpy(seq).long()[None],
                    torch.LongTensor([len(seq)]), len(seq))
        p_pred, e_pred = out[2][0].numpy(), out[3][0].numpy()
        d_pred = out[5][0].numpy().astype(float)
        mel_pred = out[1][0, : out[9][0]].numpy()
        p_gt = npy(f"pitch/{spk}-pitch-{base}.npy")
        e_gt = npy(f"energy/{spk}-energy-{base}.npy")
        d_gt = npy(f"duration/{spk}-duration-{base}.npy").astype(float)
        mel_gt = npy(f"mel/{spk}-mel-{base}.npy").astype(np.float32)
        if mel_gt.shape[0] == 80 and mel_gt.shape[1] != 80:
            mel_gt = mel_gt.T
        k = d_gt > 0
        corr = lambda a, b: np.corrcoef(a, b)[0, 1]
        gv = lambda m: m.var(0)
        flux = lambda m: np.abs(np.diff(m, axis=0)).mean()
        rows.append({
            "pitch_sd": p_pred[k].std() / p_gt[k].std(),
            "energy_sd": e_pred[k].std() / e_gt[k].std(),
            "pitch_r": corr(p_pred[k], p_gt[k]),
            "energy_r": corr(e_pred[k], e_gt[k]),
            "logdur_r": corr(np.log(d_pred + 1), np.log(d_gt + 1)),
            "length": d_pred.sum() / d_gt.sum(),
            "gv_all": (gv(mel_pred) / gv(mel_gt)).mean(),
            "gv_0_20": (gv(mel_pred)[:20] / gv(mel_gt)[:20]).mean(),
            "gv_40_80": (gv(mel_pred)[40:] / gv(mel_gt)[40:]).mean(),
            "flux": flux(mel_pred) / flux(mel_gt),
        })

print(f"== {lang} | {os.path.relpath(ckpt, proj)} | {len(rows)} validation clips")
for key in rows[0]:
    v = np.array([r[key] for r in rows])
    print(f"   {key:10s} mean {v.mean():6.3f}  median {np.median(v):6.3f}  [p10 {np.percentile(v, 10):6.3f}, p90 {np.percentile(v, 90):6.3f}]")
