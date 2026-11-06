"""Where does the error come from? Swap predicted prosody for the real values, one part at a time.

For N random validation clips the released model makes four mels from the same phones:
  pred      everything predicted (the normal system)
  real_dur  real durations, predicted pitch and energy
  real_pe   predicted durations, real pitch and energy
  real_all  real durations, pitch and energy (teacher-forced)
plus the real mel ("real_vocoded"). All five go through the same HiFi-GAN and the same Whisper model.

Measures against the real recording, per condition:
  MCD     mel-cepstral distortion (dB) from the 80-band log-mel, coefficients 1-24, DTW-aligned.
          Computed on the model's mel, so the vocoder plays no part.
  F0 RMSE in cents over frames voiced in both, and V/UV error (% of frames), from WORLD (dio+stonemask)
          F0 of the vocoded audio, with the vocoded real mel as the reference so the vocoder cancels out.
  CER/WER from Whisper (English normaliser for English; letters and marks kept for Telugu).

    python tools/ablation_eval.py <project> <en|te> <ckpt> <config> <preprocessed dir> <asr model> <out json>
                            [N] [sample wav dir]

With the default seed and N=30 these are the same clips as asr_eval.py, so 'pred' and 'real_vocoded' repeat it.
"""
import json
import os
import random
import re
import sys
import unicodedata

import jiwer
import librosa
import numpy as np
import pyworld
import scipy.fft
import soxr
import torch
from scipy.io import wavfile

proj, lang, ckpt, cfg, data, asr_name, out_json = sys.argv[1:8]
N = int(sys.argv[8]) if len(sys.argv) > 8 else 30
wav_dir = sys.argv[9] if len(sys.argv) > 9 else None
sys.path.insert(0, os.path.join(proj, "core"))
os.chdir(proj)
from config_utils import load_config  # noqa: E402
from infer_core import load_model  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor, logging  # noqa: E402

torch.set_num_threads(os.cpu_count())
logging.set_verbosity_error()
npy = lambda rel: np.load(os.path.join(data, rel))
configs = load_config(cfg)
pre, mcfg, _ = configs
pre["path"]["preprocessed_path"] = os.path.dirname(os.path.abspath(ckpt))
dev = torch.device("cpu")
model = load_model(configs, ckpt, dev)
vocoder = get_vocoder(mcfg, dev)
sr = pre["preprocessing"]["audio"]["sampling_rate"]
hop = pre["preprocessing"]["stft"]["hop_length"]
spk_map = json.load(open(os.path.join(data, "speakers.json")))

proc = WhisperProcessor.from_pretrained(asr_name)
asr = WhisperForConditionalGeneration.from_pretrained(asr_name).eval()
if lang == "te":
    asr.generation_config.forced_decoder_ids = proc.get_decoder_prompt_ids(language="telugu", task="transcribe")


def te_norm(t):
    # Whisper's basic normaliser drops combining marks (Telugu vowel signs); keep them, drop only P/S
    t = unicodedata.normalize("NFC", t).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return re.sub(r"\s+", " ", t).strip()


norm = proc.tokenizer.normalize if lang == "en" else te_norm


@torch.no_grad()
def transcribe(wav_int16):
    x = soxr.resample(wav_int16.astype(np.float32) / 32768.0, sr, 16000)
    feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features
    ids = asr.generate(feats, max_new_tokens=440)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def to_wav(mel_Tx80):
    m = torch.from_numpy(np.ascontiguousarray(mel_Tx80.T))[None]
    return vocoder_infer(m, vocoder, mcfg, pre, lengths=[mel_Tx80.shape[0] * hop])[0]


def mfcc(mel_Tx80):
    # the 80 log-mel bins are a log spectrum on a warped (mel) axis; its cepstrum is DCT-II / (2 * 80)
    return scipy.fft.dct(mel_Tx80, type=2, axis=1)[:, 1:25] / (2 * mel_Tx80.shape[1])


def align(a, b):
    """DTW path (pairs of frame indices) between two mels. Used for every condition, also when the lengths
    match, so that conditions with real durations are not scored on a stricter alignment."""
    _, wp = librosa.sequence.dtw(mfcc(a).T, mfcc(b).T, metric="euclidean")
    wp = wp[::-1]
    return wp[:, 0], wp[:, 1]


def mcd(a, b, ia, ib):
    d = mfcc(a)[ia] - mfcc(b)[ib]
    return float(np.mean(10 / np.log(10) * np.sqrt(2 * (d ** 2).sum(1))))


def f0(wav_int16):
    x = wav_int16.astype(np.float64) / 32768.0
    f, t = pyworld.dio(x, sr, frame_period=hop / sr * 1000)
    return pyworld.stonemask(x, f, t, sr)


def f0_err(fa, fb, ia, ib):
    n = min(len(fa), len(fb))
    k = (ia < n) & (ib < n)
    a, b = fa[ia[k]], fb[ib[k]]
    both = (a > 0) & (b > 0)
    rmse = float(np.sqrt(np.mean((1200 * np.log2(a[both] / b[both])) ** 2))) if both.any() else np.nan
    return rmse, float(np.mean((a > 0) != (b > 0)))


CONDS = ("pred", "real_dur", "real_pe", "real_all")
lines = open(os.path.join(data, "val.txt"), encoding="utf-8").read().strip().split("\n")
random.Random(1).shuffle(lines)
res = {c: {"ref": [], "hyp": [], "mcd": [], "f0_rmse": [], "vuv": [], "len": []}
       for c in ("real_vocoded",) + CONDS}
with torch.no_grad():
    for k, line in enumerate(lines[:N]):
        base, spk, phones, raw = line.split("|")
        mel_gt = npy(f"mel/{spk}-mel-{base}.npy").astype(np.float32)
        if mel_gt.shape[0] == 80 and mel_gt.shape[1] != 80:
            mel_gt = mel_gt.T
        d_gt = npy(f"duration/{spk}-duration-{base}.npy")
        p_gt = torch.from_numpy(npy(f"pitch/{spk}-pitch-{base}.npy").astype(np.float32))[None]
        e_gt = torch.from_numpy(npy(f"energy/{spk}-energy-{base}.npy").astype(np.float32))[None]
        seq = torch.from_numpy(np.array(text_to_sequence(phones, []))).long()[None]
        T = int(d_gt.sum())
        mels = {"real_vocoded": mel_gt}
        for c in CONDS:
            kw = {}
            if c in ("real_dur", "real_all"):
                kw.update(mel_lens=torch.LongTensor([T]), max_mel_len=T,
                          d_targets=torch.from_numpy(d_gt.astype(np.int64))[None])
            if c in ("real_pe", "real_all"):
                kw.update(p_targets=p_gt, e_targets=e_gt)
            out = model(torch.LongTensor([spk_map[spk]]), seq, torch.LongTensor([seq.shape[1]]), seq.shape[1], **kw)
            mels[c] = out[1][0, : out[9][0]].numpy()
        wavs = {c: to_wav(m) for c, m in mels.items()}
        f0_ref = f0(wavs["real_vocoded"])
        for c, m in mels.items():
            r = res[c]
            r["ref"].append(raw)
            r["hyp"].append(transcribe(wavs[c]))
            r["len"].append(len(m) / len(mel_gt))
            if c != "real_vocoded":
                ia, ib = align(m, mel_gt)
                r["mcd"].append(mcd(m, mel_gt, ia, ib))
                rm, vuv = f0_err(f0(wavs[c]), f0_ref, ia, ib)
                r["f0_rmse"].append(rm)
                r["vuv"].append(vuv)
            if wav_dir and k < 3:
                os.makedirs(wav_dir, exist_ok=True)
                wavfile.write(os.path.join(wav_dir, f"{k + 1}_{base}_{c}.wav"), sr, wavs[c])
        print(f"{k + 1}/{N}", flush=True)


def asr_score(refs, hyps):
    r = [norm(t) for t in refs]
    h = [norm(t) or "<empty>" for t in hyps]
    keep = [i for i, t in enumerate(r) if t.strip()]
    r, h = [r[i] for i in keep], [h[i] for i in keep]
    return {"cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h)}


summary = {}
for c, r in res.items():
    s = asr_score(r["ref"], r["hyp"])
    for key in ("mcd", "f0_rmse", "vuv", "len"):
        if r[key]:
            s[key] = float(np.nanmean(r[key]))
    summary[c] = s
json.dump({"asr": asr_name, "n": N, "summary": summary, "per_clip": res}, open(out_json, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"== {lang} | {N} validation clips")
print(f"{'condition':13s} {'CER':>6s} {'WER':>6s} {'MCD dB':>7s} {'F0 cents':>9s} {'V/UV %':>7s} {'length':>7s}")
for c, s in summary.items():
    print(f"{c:13s} {100 * s['cer']:6.1f} {100 * s['wer']:6.1f} {s.get('mcd', np.nan):7.2f} "
          f"{s.get('f0_rmse', np.nan):9.1f} {100 * s.get('vuv', np.nan):7.1f} {s['len']:7.3f}")
