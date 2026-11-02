"""Inference: text -> phones -> FastSpeech 2 -> HiFi-GAN -> wav, plus the
variance-adaptor predictions (duration, F0, energy) saved as JSON and plotted.

Replaces ming024 synthesize.py for this assignment. Additions:
  * language-specific front-ends (text/frontend_en.py, text/frontend_te.py)
  * reads test sentences from data/test_sentences.tsv by language
  * de-normalises predicted pitch (Hz) and energy using stats.json
  * per-sentence plots in variance_adaptor_plots/{duration,pitch,energy}
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from config_utils import ROOT, load_config
from model import FastSpeech2
from text import text_to_sequence
from utils.model import get_vocoder, vocoder_infer
from plots import plot_variance


def find_meta(pre, ckpt, name):
    """stats.json / speakers.json / vocab.txt: from the preprocessed data if it is
    there, else next to the checkpoint (the released repo ships them there)."""
    for d in (pre["path"]["preprocessed_path"], os.path.dirname(os.path.abspath(ckpt))):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in the preprocessed data or next to {ckpt}")


def load_model(configs, ckpt, device):
    pre, model_cfg, _ = configs
    model = FastSpeech2(pre, model_cfg).to(device)
    sd = torch.load(ckpt, map_location=device)
    sd = sd.get("model", sd)
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    return model


def build_frontend(pre, ckpt, repair=True):
    lang = pre["preprocessing"]["text"]["language"]
    if lang == "en":
        from text.frontend_en import EnglishFrontend

        return EnglishFrontend(pre["path"]["lexicon_path"], repair=repair)
    from text.frontend_te import TeluguFrontend

    try:
        vocab = find_meta(pre, ckpt, "vocab.txt")
    except FileNotFoundError:
        vocab = None  # split-word repair then has nothing to check against
    return TeluguFrontend(vocab, repair=repair)


def word_spans(phones, trace):
    """Map each word of the front-end trace to [start, end) phone indices."""
    spans, i = [], 0
    for t in trace:
        word, ph = t[0], t[-1].split()
        while i < len(phones) and phones[i] == "sp":
            i += 1
        spans.append((word, i, i + len(ph)))
        i += len(ph)
    return spans


@torch.no_grad()
def synthesize(model, vocoder, configs, text, speaker_id, frontend, controls=(1.0, 1.0, 1.0), stats=None):
    pre, model_cfg, _ = configs
    phones, info = frontend(text)
    seq = np.array(text_to_sequence("{" + " ".join(phones) + "}", []))
    dev = next(model.parameters()).device
    speakers = torch.LongTensor([speaker_id]).to(dev)
    texts = torch.from_numpy(seq).long().unsqueeze(0).to(dev)
    src_lens = torch.LongTensor([len(seq)]).to(dev)
    out = model(speakers, texts, src_lens, len(seq),
                p_control=controls[0], e_control=controls[1], d_control=controls[2])
    mel_post = out[1][0, : out[9][0]].transpose(0, 1)
    hop = pre["preprocessing"]["stft"]["hop_length"]
    wav = vocoder_infer(mel_post.unsqueeze(0), vocoder, model_cfg, pre,
                        lengths=[out[9][0].item() * hop])[0]
    if stats is None:
        with open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) as f:
            stats = json.load(f)
    p_mean, p_std = stats["pitch"][2:]
    e_mean, e_std = stats["energy"][2:]
    dur = out[5][0, : len(seq)].cpu().numpy().astype(int)
    pitch_n = out[2][0, : len(seq)].cpu().numpy()
    energy_n = out[3][0, : len(seq)].cpu().numpy()
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    pred = {
        "text": text,
        "normalised": info["normalised"],
        "repairs": info["repairs"],
        "words": [list(t) for t in info["trace"]],
        "word_spans": word_spans(phones, info["trace"]),
        "phones": phones,
        "duration_frames": dur.tolist(),
        "duration_ms": (dur * hop / sr * 1000).round(1).tolist(),
        "pitch_norm": pitch_n.round(4).tolist(),
        "pitch_hz": (pitch_n * p_std + p_mean).round(2).tolist(),
        "energy_norm": energy_n.round(4).tolist(),
        "energy": (energy_n * e_std + e_mean).round(4).tolist(),
        "total_frames": int(out[9][0].item()),
        "total_sec": round(out[9][0].item() * hop / sr, 3),
        "speaker_id": int(speaker_id),
        "pitch_mean_hz": round(p_mean, 2),
    }
    return wav, sr, pred, mel_post.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", required=True, help="Results/<Language>")
    ap.add_argument("--text", default=None, help="synthesize one sentence instead of the test set")
    ap.add_argument("--only", nargs="*", default=None, help="subset of test-sentence ids, e.g. E3")
    ap.add_argument("--speaker", default=None, help="speaker name (multi-speaker models)")
    ap.add_argument("--no_repair", action="store_true", help="disable split-word repair")
    ap.add_argument("--suffix", default="", help="added to output file names")
    ap.add_argument("--pitch_control", type=float, default=1.0)
    ap.add_argument("--energy_control", type=float, default=1.0)
    ap.add_argument("--duration_control", type=float, default=1.0)
    args = ap.parse_args()

    configs = load_config(args.config)
    pre = configs[0]
    # released repo: no preprocessed data; the model's pitch/energy bins and the metadata
    # are then built from the stats.json etc. stored next to the checkpoint
    if not os.path.exists(os.path.join(pre["path"]["preprocessed_path"], "stats.json")):
        pre["path"]["preprocessed_path"] = os.path.dirname(os.path.abspath(args.ckpt))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(configs, args.ckpt, device)
    vocoder = get_vocoder(configs[1], device)
    frontend = build_frontend(pre, args.ckpt, repair=not args.no_repair)
    with open(find_meta(pre, args.ckpt, "speakers.json")) as f:
        spk_map = json.load(f)
    with open(find_meta(pre, args.ckpt, "stats.json")) as f:
        stats = json.load(f)
    spk_name = args.speaker or configs[1].get("inference_speaker") or next(iter(spk_map))
    spk_id = spk_map.get(spk_name, 0)
    lang = pre["preprocessing"]["text"]["language"]

    if args.text:
        items = [("custom", args.text)]
    else:
        with open(os.path.join(ROOT, "data", "test_sentences.tsv"), encoding="utf-8") as f:
            items = [(r["id"], r["text"]) for r in csv.DictReader(f, delimiter="\t") if r["language"] == lang]
        if args.only:
            items = [it for it in items if it[0] in args.only]

    audio_dir = os.path.join(args.out_dir, "test_generated_audios")
    plot_root = os.path.join(args.out_dir, "variance_adaptor_plots")
    os.makedirs(audio_dir, exist_ok=True)
    for sub in ("duration", "pitch", "energy", "overview", "json"):
        os.makedirs(os.path.join(plot_root, sub), exist_ok=True)
    controls = (args.pitch_control, args.energy_control, args.duration_control)
    for sid, text in items:
        wav, sr, pred, mel = synthesize(model, vocoder, configs, text, spk_id, frontend, controls, stats)
        pred["speaker"] = spk_name
        name = sid + args.suffix
        wavfile.write(os.path.join(audio_dir, name + ".wav"), sr, wav)
        with open(os.path.join(plot_root, "json", name + ".json"), "w", encoding="utf-8") as f:
            json.dump(pred, f, ensure_ascii=False, indent=1)
        try:
            plot_variance(pred, mel, plot_root, name, lang, sr, pre["preprocessing"]["stft"]["hop_length"])
        except Exception as e:  # never lose the audio because of a plotting problem
            print(f"  plot failed for {name}: {e!r}")
            import matplotlib.pyplot as plt

            plt.close("all")
        print(f"{name}: {pred['total_sec']} s, {len(pred['phones'])} phones, repairs={pred['repairs']}")


if __name__ == "__main__":
    main()
