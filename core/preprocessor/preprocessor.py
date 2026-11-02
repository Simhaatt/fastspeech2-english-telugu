"""Feature extraction for FastSpeech 2 (mel, phone durations, pitch, energy).

Based on ming024/FastSpeech2 preprocessor/preprocessor.py. Changes:
  * reads MFA 2.x/3.x TextGrids: silences are empty intervals there, which
    the reference code silently dropped (`tgt` skips empty intervals by
    default) so phone durations no longer summed to the mel length. Empty
    intervals are now kept and labelled "sp"; "spn" (unknown word) utterances
    are skipped.
  * the phone tier is found by name suffix ("phones" or "<spk> - phones").
  * train/val split comes from a split file written by the data-selection
    step (speaker-stratified, no utterance overlap) instead of a random
    shuffle, so the split is reproducible and documented.
  * parallel extraction (multiprocessing), per-language F0 search range,
    and pyworld harvest/dio choice.
  * writes vocab.txt (word counts of the training transcripts), used by the
    Telugu front-end to repair words split by a stray space.
"""
import json
import os
from collections import Counter
from multiprocessing import Pool

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from tqdm import tqdm

import audio as Audio

SIL = {"sil", "sp", "spn", ""}


def _phone_tier(tg):
    for t in tg.tiers:
        if t.name == "phones" or t.name.endswith("phones"):
            return t
    raise KeyError("no phones tier")


class Preprocessor:
    def __init__(self, config):
        self.config = config
        pp = config["preprocessing"]
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.split_file = config["path"].get("split_path")
        self.sampling_rate = pp["audio"]["sampling_rate"]
        self.hop_length = pp["stft"]["hop_length"]
        self.pitch_phoneme_averaging = pp["pitch"]["feature"] == "phoneme_level"
        self.energy_phoneme_averaging = pp["energy"]["feature"] == "phoneme_level"
        self.pitch_normalization = pp["pitch"]["normalization"]
        self.energy_normalization = pp["energy"]["normalization"]
        self.f0_floor = pp["pitch"].get("f0_floor", 60.0)
        self.f0_ceil = pp["pitch"].get("f0_ceil", 500.0)
        self.f0_method = pp["pitch"].get("method", "dio")
        self.max_frames = config.get("max_seq_len_frames", 1000)
        self.n_jobs = pp.get("n_jobs", os.cpu_count())
        self.stft_args = (
            pp["stft"]["filter_length"], pp["stft"]["hop_length"],
            pp["stft"]["win_length"], pp["mel"]["n_mel_channels"],
            pp["audio"]["sampling_rate"], pp["mel"]["mel_fmin"], pp["mel"]["mel_fmax"],
        )
        self.STFT = None

    # ------------------------------------------------------------------ main
    def build_from_path(self):
        for d in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)

        jobs = []
        speakers = sorted(
            s for s in os.listdir(self.in_dir) if os.path.isdir(os.path.join(self.in_dir, s))
        )
        for spk in speakers:
            for wav_name in sorted(os.listdir(os.path.join(self.in_dir, spk))):
                if wav_name.endswith(".wav"):
                    base = wav_name[:-4]
                    tg = os.path.join(self.out_dir, "TextGrid", spk, base + ".TextGrid")
                    if os.path.exists(tg):
                        jobs.append((spk, base))
        print(f"{len(jobs)} utterances with alignments, {len(speakers)} speakers")

        with Pool(self.n_jobs) as pool:
            results = list(tqdm(pool.imap(self._job, jobs, chunksize=8), total=len(jobs)))

        out, pitch_all, energy_all, n_frames, skipped = [], [], [], 0, Counter()
        for r in results:
            if isinstance(r, str):
                skipped[r] += 1
                continue
            info, p, e, n = r
            out.append(info)
            pitch_all.append(p)
            energy_all.append(e)
            n_frames += n
        print("skipped:", dict(skipped))
        used_speakers = sorted({o.split("|")[1] for o in out})
        speaker_map = {s: i for i, s in enumerate(used_speakers)}

        pitch_all = np.concatenate(pitch_all)
        energy_all = np.concatenate(energy_all)
        pm, ps = (pitch_all.mean(), pitch_all.std()) if self.pitch_normalization else (0.0, 1.0)
        em, es = (energy_all.mean(), energy_all.std()) if self.energy_normalization else (0.0, 1.0)
        pmin, pmax = self.normalize(os.path.join(self.out_dir, "pitch"), pm, ps)
        emin, emax = self.normalize(os.path.join(self.out_dir, "energy"), em, es)

        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speaker_map, f, indent=1)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump({"pitch": [float(pmin), float(pmax), float(pm), float(ps)],
                       "energy": [float(emin), float(emax), float(em), float(es)]}, f)

        hours = n_frames * self.hop_length / self.sampling_rate / 3600
        print(f"Total (silence-trimmed) speech: {hours:.2f} h")

        # split
        split = {}
        if self.split_file and os.path.exists(self.split_file):
            with open(self.split_file) as f:
                for line in f:
                    k, v = line.strip().split("\t")
                    split[k] = v
        train, val = [], []
        for m in out:
            (val if split.get(m.split("|")[0]) == "val" else train).append(m)
        if not split:
            rng = np.random.RandomState(1234)
            idx = rng.permutation(len(out))
            nv = max(1, len(out) // 6)
            val = [out[i] for i in idx[:nv]]
            train = [out[i] for i in idx[nv:]]
        for name, rows in (("train.txt", train), ("val.txt", val)):
            with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
        summary = self._summary(train, val)
        with open(os.path.join(self.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
        print(json.dumps(summary, indent=1))

        vocab = Counter()
        for m in train:
            vocab.update(m.split("|")[3].split())
        with open(os.path.join(self.out_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            for w, c in vocab.most_common():
                f.write(f"{w} {c}\n")
        return out

    def _summary(self, train, val):
        def hours(rows):
            tot = 0
            for m in rows:
                spk, base = m.split("|")[1], m.split("|")[0]
                d = np.load(os.path.join(self.out_dir, "duration", f"{spk}-duration-{base}.npy"))
                tot += d.sum()
            return round(tot * self.hop_length / self.sampling_rate / 3600, 3)

        return {
            "train_utts": len(train), "val_utts": len(val),
            "train_hours_trimmed": hours(train), "val_hours_trimmed": hours(val),
            "train_speakers": len({m.split("|")[1] for m in train}),
            "val_speakers": len({m.split("|")[1] for m in val}),
            "overlap_utts": len({m.split("|")[0] for m in train} & {m.split("|")[0] for m in val}),
        }

    def _job(self, args):
        if self.STFT is None:
            import torch

            torch.set_num_threads(1)
            self.STFT = Audio.stft.TacotronSTFT(*self.stft_args)
        try:
            return self.process_utterance(*args)
        except Exception as e:  # keep going, report reason
            return "error:" + type(e).__name__

    # -------------------------------------------------------------- per utt
    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, f"{basename}.wav")
        text_path = os.path.join(self.in_dir, speaker, f"{basename}.lab")
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, f"{basename}.TextGrid")

        textgrid = tgt.io.read_textgrid(tg_path, include_empty_intervals=True)
        phone, duration, start, end = self.get_alignment(_phone_tier(textgrid))
        if start >= end or len(phone) == 0:
            return "empty_alignment"
        if "spn" in phone:
            return "unknown_word_spn"
        if sum(duration) > self.max_frames:
            return "too_long"
        text = "{" + " ".join(phone) + "}"

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[int(self.sampling_rate * start): int(self.sampling_rate * end)].astype(np.float32)
        with open(text_path, encoding="utf-8") as f:
            raw_text = f.readline().strip("\n").replace("|", " ")

        frame_period = self.hop_length / self.sampling_rate * 1000
        x = wav.astype(np.float64)
        if self.f0_method == "harvest":
            pitch, t = pw.harvest(x, self.sampling_rate, f0_floor=self.f0_floor,
                                  f0_ceil=self.f0_ceil, frame_period=frame_period)
        else:
            pitch, t = pw.dio(x, self.sampling_rate, f0_floor=self.f0_floor,
                              f0_ceil=self.f0_ceil, frame_period=frame_period)
            pitch = pw.stonemask(x, pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return "no_voiced_frames"

        mel, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel = mel[:, : sum(duration)]
        energy = energy[: sum(duration)]
        if mel.shape[1] < sum(duration):
            # rounding at the very end of the file: shave the last phone
            diff = sum(duration) - mel.shape[1]
            if duration[-1] <= diff:
                return "length_mismatch"
            duration[-1] -= diff
            pitch = pitch[: sum(duration)]

        if self.pitch_phoneme_averaging:
            nz = np.where(pitch != 0)[0]
            fn = interp1d(nz, pitch[nz], fill_value=(pitch[nz[0]], pitch[nz[-1]]), bounds_error=False)
            pitch = fn(np.arange(0, len(pitch)))
            pitch = self._phone_avg(pitch, duration)
        if self.energy_phoneme_averaging:
            energy = self._phone_avg(energy, duration)

        tag = f"{speaker}-{{}}-{basename}.npy"
        np.save(os.path.join(self.out_dir, "duration", tag.format("duration")), np.array(duration))
        np.save(os.path.join(self.out_dir, "pitch", tag.format("pitch")), pitch)
        np.save(os.path.join(self.out_dir, "energy", tag.format("energy")), energy)
        np.save(os.path.join(self.out_dir, "mel", tag.format("mel")), mel.T)
        return ("|".join([basename, speaker, text, raw_text]),
                self.remove_outlier(pitch), self.remove_outlier(energy), mel.shape[1])

    @staticmethod
    def _phone_avg(x, duration):
        out, pos = np.zeros(len(duration)), 0
        for i, d in enumerate(duration):
            out[i] = np.mean(x[pos: pos + d]) if d > 0 else 0.0
            pos += d
        return out

    def get_alignment(self, tier):
        phones, durations = [], []
        start_time = end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text.strip()
            if p == "":
                p = "sp"
            if p == "sil":
                p = "sp"
            if not phones and p in SIL:
                continue  # trim leading silence
            if not phones:
                start_time = s
            phones.append(p)
            if p not in SIL:
                end_time = e
                end_idx = len(phones)
            durations.append(int(np.round(e * self.sampling_rate / self.hop_length)
                                 - np.round(s * self.sampling_rate / self.hop_length)))
        return phones[:end_idx], durations[:end_idx], start_time, end_time

    @staticmethod
    def remove_outlier(values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lo, hi = p25 - 1.5 * (p75 - p25), p25 + 1.5 * (p75 - p25) + (p75 - p25)
        return values[np.logical_and(values > lo, values < hi)]

    @staticmethod
    def normalize(in_dir, mean, std):
        vmax, vmin = np.finfo(np.float64).min, np.finfo(np.float64).max
        for fn in os.listdir(in_dir):
            p = os.path.join(in_dir, fn)
            v = (np.load(p) - mean) / std
            np.save(p, v)
            vmax, vmin = max(vmax, v.max()), min(vmin, v.min())
        return vmin, vmax
