"""Shared helpers for dataset selection / subsampling (both languages)."""
import io
import json
import os

import numpy as np
import soundfile as sf


def decode(audio_bytes, target_sr=None):
    """Decode audio bytes -> mono float32 (optionally resampled)."""
    wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if target_sr and sr != target_sr:
        import soxr

        wav = soxr.resample(wav, sr, target_sr, quality="HQ")
        sr = target_sr
    return wav, sr


def quality(wav, sr):
    """Cheap quality descriptors used to rank utterances.

    snr_db: ratio of loud (90th pct) to quiet (10th pct) 25 ms frame energy -
            a rough noise-floor estimate that works without a VAD model.
    clip:   fraction of samples at |x| >= 0.999.
    lead/trail: seconds of near-silence at the edges."""
    n = int(0.025 * sr)
    hop = int(0.010 * sr)
    if len(wav) < n * 4:
        return {"snr_db": 0.0, "clip_frac": 0.0, "rms_db": -99.0, "lead_s": 0.0, "trail_s": 0.0}
    frames = np.lib.stride_tricks.sliding_window_view(wav, n)[::hop]
    e = (frames ** 2).mean(axis=1) + 1e-10
    lo, hi = np.percentile(e, 10), np.percentile(e, 90)
    thr = lo * 10 ** (0.4 * np.log10(hi / lo))  # 40% of the way up in log-energy
    voiced = np.where(e > thr)[0]
    lead = voiced[0] * hop / sr if len(voiced) else 0.0
    trail = (len(e) - 1 - voiced[-1]) * hop / sr if len(voiced) else 0.0
    return {
        "snr_db": float(10 * np.log10(hi / lo)),
        "clip_frac": float(np.mean(np.abs(wav) >= 0.999)),
        "rms_db": float(10 * np.log10(np.mean(wav ** 2) + 1e-10)),
        "lead_s": float(lead),
        "trail_s": float(trail),
    }


def write_utt(raw_dir, speaker, uid, wav, sr, text):
    d = os.path.join(raw_dir, speaker)
    os.makedirs(d, exist_ok=True)
    peak = np.abs(wav).max() + 1e-9
    if peak > 0.99:
        wav = wav / peak * 0.99
    sf.write(os.path.join(d, uid + ".wav"), wav, sr, subtype="PCM_16")
    with open(os.path.join(d, uid + ".lab"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def pick_split(df, train_h, val_h, seed=1234, group=None):
    """Draw val then train from a ranked/filtered pool, matching durations.

    If `group` (speaker column) is given, val is drawn proportionally from
    every speaker so the two sets share speakers but never utterances."""
    rng = np.random.RandomState(seed)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    total_h = train_h + val_h
    if group is None:
        df["cum"] = df["dur"].cumsum() / 3600
        pool = df[df["cum"] <= total_h].copy()
        pool["split"] = "train"
        pool_h = pool["dur"].sum() / 3600
        val_target = min(val_h, pool_h * val_h / total_h)  # keep the ratio if the pool is short
        vcum = pool["dur"].cumsum() / 3600
        pool.loc[vcum <= val_target, "split"] = "val"
        return pool.drop(columns=["cum"])
    parts = []
    avail_h = df["dur"].sum() / 3600
    keep_frac = min(1.0, total_h / avail_h)
    for g, d in df.groupby(group):
        d = d.copy()
        d["cum"] = d["dur"].cumsum()
        d = d[d["cum"] <= d["dur"].sum() * keep_frac + 1e-6].copy()
        d["split"] = "train"
        vtarget = d["dur"].sum() * val_h / total_h
        vc = d["dur"].cumsum()
        d.loc[vc - d["dur"] < vtarget, "split"] = "val"   # include the clip that crosses the target
        if (d["split"] == "train").sum() == 0 and len(d) > 1:
            d.iloc[-1, d.columns.get_loc("split")] = "train"
        parts.append(d.drop(columns=["cum"]))
    return __import__("pandas").concat(parts, ignore_index=True)


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
