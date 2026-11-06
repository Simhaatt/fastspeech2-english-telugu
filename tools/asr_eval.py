"""Intelligibility by ASR (Whisper), with a ceiling to separate ASR errors from TTS errors.

For N random validation clips: (a) the real mel through HiFi-GAN ("real, vocoded") and (b) the released
FastSpeech 2 model's mel from the same phones through HiFi-GAN ("synthesized"). Both go through the same
Whisper model; CER/WER against the transcript. Also transcribes the test-sentence WAVs.

    python tools/asr_eval.py <project> <en|te> <ckpt> <config> <preprocessed dir or zip::prefix> <asr model>
                       <test wav dir> <test refs tsv> <out json> [N]
"""
import csv
import io
import json
import os
import random
import sys
import zipfile

import jiwer
import numpy as np
import soxr
import torch

(proj, lang, ckpt, cfg, data, asr_name, test_dir, test_refs, out_json) = sys.argv[1:10]
N = int(sys.argv[10]) if len(sys.argv) > 10 else 30
sys.path.insert(0, os.path.join(proj, "core"))
os.chdir(proj)
from config_utils import load_config  # noqa: E402
from infer_core import load_model  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402
from scipy.io import wavfile  # noqa: E402

torch.set_num_threads(os.cpu_count())
if "::" in data:
    zpath, prefix = data.split("::")
    zf = zipfile.ZipFile(zpath)
    read = lambda rel: zf.read(prefix + "/" + rel)
else:
    read = lambda rel: open(os.path.join(data, rel), "rb").read()

configs = load_config(cfg)
pre, mcfg, _ = configs
pre["path"]["preprocessed_path"] = os.path.dirname(os.path.abspath(ckpt))
dev = torch.device("cpu")
model = load_model(configs, ckpt, dev)
vocoder = get_vocoder(mcfg, dev)
sr = pre["preprocessing"]["audio"]["sampling_rate"]
hop = pre["preprocessing"]["stft"]["hop_length"]
spk_map = json.loads(read("speakers.json"))

proc = WhisperProcessor.from_pretrained(asr_name)
asr = WhisperForConditionalGeneration.from_pretrained(asr_name).eval()
gen_kw = {}
if lang == "te":
    gen_kw["forced_decoder_ids"] = proc.get_decoder_prompt_ids(language="telugu", task="transcribe")
    asr.generation_config.forced_decoder_ids = gen_kw.pop("forced_decoder_ids")
norm = proc.tokenizer.normalize if lang == "en" else proc.tokenizer.basic_normalize


def to_wav(mel_80xT):
    return vocoder_infer(mel_80xT[None], vocoder, mcfg, pre, lengths=[mel_80xT.shape[1] * hop])[0]


@torch.no_grad()
def transcribe(wav_int16, rate):
    x = soxr.resample(wav_int16.astype(np.float32) / 32768.0, rate, 16000)
    feats = proc(x, sampling_rate=16000, return_tensors="pt").input_features
    # Whisper's limit is 448 tokens including the prompt; Telugu needs many tokens per word, and 200 cut
    # long sentences short
    ids = asr.generate(feats, max_new_tokens=440)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def score(refs, hyps):
    r = [norm(t) for t in refs]
    h = [norm(t) for t in hyps]
    keep = [i for i, t in enumerate(r) if t.strip()]
    r, h = [r[i] for i in keep], [h[i] if h[i].strip() else "<empty>" for i in keep]
    return {"cer": jiwer.cer(r, h), "wer": jiwer.wer(r, h), "n": len(r)}


lines = read("val.txt").decode("utf-8").strip().split("\n")
random.Random(1).shuffle(lines)
res = {"real_vocoded": {"ref": [], "hyp": []}, "synthesized": {"ref": [], "hyp": []}}
with torch.no_grad():
    for k, line in enumerate(lines[:N]):
        base, spk, phones, raw = line.split("|")
        mel_gt = np.load(io.BytesIO(read(f"mel/{spk}-mel-{base}.npy"))).astype(np.float32)
        if mel_gt.shape[1] == 80:
            mel_gt = mel_gt.T
        seq = np.array(text_to_sequence(phones, []))
        out = model(torch.LongTensor([spk_map[spk]]), torch.from_numpy(seq).long()[None],
                    torch.LongTensor([len(seq)]), len(seq))
        mel_pred = out[1][0, : out[9][0]].transpose(0, 1)
        for key, mel in (("real_vocoded", torch.from_numpy(mel_gt)), ("synthesized", mel_pred)):
            res[key]["ref"].append(raw)
            res[key]["hyp"].append(transcribe(to_wav(mel), sr))
        print(f"{k + 1}/{N}", flush=True)

tests = []
with open(test_refs, encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        rate, w = wavfile.read(os.path.join(test_dir, r["id"] + ".wav"))
        hyp = transcribe(w, rate)
        s = score([r["ref"]], [hyp])
        tests.append({"id": r["id"], "ref": r["ref"], "hyp": hyp, "cer": s["cer"], "wer": s["wer"]})

summary = {key: score(v["ref"], v["hyp"]) for key, v in res.items()}
summary["test_sentences"] = score([t["ref"] for t in tests], [t["hyp"] for t in tests])
json.dump({"asr": asr_name, "summary": summary, "tests": tests, "val": res}, open(out_json, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps(summary, indent=1))
for t in tests:
    print(f"{t['id']:14s} CER {t['cer']:.3f} WER {t['wer']:.3f}\n   ref: {t['ref']}\n   hyp: {t['hyp']}")
