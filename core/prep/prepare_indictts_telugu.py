"""Select ~5 h train + ~1 h val of single-speaker Telugu from IndicTTS
(SPRINGLab/IndicTTS_Telugu on Hugging Face) and write an MFA-ready corpus.

Selection strategy (documented in the report):
  1. keep one voice (gender chosen after the audit) -> single-speaker model
  2. drop clips shorter than --min_dur or longer than --max_dur seconds
     (FastSpeech 2 memory grows with length; very long clips also align worse)
  3. drop clips whose transcript has Latin letters or no Telugu at all
  4. drop clips with clipping or a low estimated SNR, and exact duplicate
     transcripts (so the same sentence can't land in both train and val)
  5. shuffle with a fixed seed and take val first (1 h), then train (5 h)
Outputs: raw_data/<speaker>/<id>.wav (22.05 kHz mono) + .lab, split.tsv,
selection.csv, lexicon.txt (rule-based G2P for every corpus word).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prep.common import decode, dump_json, pick_split, quality, write_utt  # noqa: E402
from text.frontend_te import normalise, tokenize, word_to_phones  # noqa: E402

TEL = re.compile(r"[ఀ-౿]")


def label_names(pq_file):
    meta = pq.read_schema(pq_file).metadata or {}
    try:
        info = json.loads(meta[b"huggingface"])
        return info["info"]["features"]["gender"]["names"]
    except Exception:
        return None


def iter_rows(pq_file, columns=None):
    """Yield (row index, row dict), reading one row group (100 clips) at a time.
    Each file holds 1-2 GB of audio, too much to load whole on a machine with
    4 GB of RAM; arrow's freed buffers are handed back after every group."""
    pf = pq.ParquetFile(pq_file)
    i = 0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g, columns=columns, use_threads=False)
        for k in range(0, t.num_rows, 10):
            for r in t.slice(k, 10).to_pylist():
                yield i, r
                i += 1
        del t
        pa.default_memory_pool().release_unused()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--out", required=True, help="corpus root (raw_data parent)")
    ap.add_argument("--gender", required=True)
    ap.add_argument("--train_h", type=float, default=5.0)
    ap.add_argument("--val_h", type=float, default=1.0)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=11.0)
    ap.add_argument("--min_snr", type=float, default=25.0)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.hf_dir, "**", "*.parquet"), recursive=True))
    names = label_names(files[0])
    print("gender labels:", names)
    rows = []
    for f in files:
        for i, r in iter_rows(f, ["audio", "text", "gender"]):
            g = r["gender"]
            g = names[g] if names is not None and isinstance(g, (int, np.integer)) else str(g)
            if g != args.gender:
                continue
            wav, sr = decode(r["audio"]["bytes"])
            q = quality(wav, sr)
            rows.append({"uid": f"te_{os.path.basename(f)[:-8]}_{i:05d}", "file": f, "row": i,
                         "text": r["text"], "dur": len(wav) / sr, "orig_sr": sr, **q})
    df = pd.DataFrame(rows)
    n0, h0 = len(df), df.dur.sum() / 3600
    log = {"available_utts": n0, "available_hours": round(h0, 3)}

    df["text_norm"] = df.text.map(normalise)
    keep = (df.dur >= args.min_dur) & (df.dur <= args.max_dur)
    log["after_duration_filter"] = int(keep.sum())
    keep &= ~df.text.str.contains(r"[A-Za-z]") & df.text.str.contains(TEL)
    log["after_text_filter"] = int(keep.sum())
    keep &= (df.clip_frac < 1e-3) & (df.snr_db >= args.min_snr)
    log["after_quality_filter"] = int(keep.sum())
    keep &= ~df.text_norm.duplicated(keep="first")
    log["after_dedup"] = int(keep.sum())
    pool = df[keep].copy()
    log["pool_hours"] = round(pool.dur.sum() / 3600, 3)
    sel = pick_split(pool, args.train_h, args.val_h)
    for s in ("train", "val"):
        d = sel[sel.split == s]
        log[f"{s}_utts"] = int(len(d))
        log[f"{s}_hours"] = round(d.dur.sum() / 3600, 3)
        log[f"{s}_dur_mean_s"] = round(d.dur.mean(), 2)
    log["text_overlap_train_val"] = len(set(sel[sel.split == "train"].text_norm) & set(sel[sel.split == "val"].text_norm))
    print(json.dumps(log, indent=1))

    raw = os.path.join(args.out, "raw_data")
    spk = f"te_{args.gender}"
    bar = tqdm(total=len(sel))
    for f, d in sel.groupby("file"):
        want = {r.row: r for r in d.itertuples()}
        for i, row in iter_rows(f, ["audio"]):
            if i not in want:
                continue
            r = want[i]
            wav, sr = decode(row["audio"]["bytes"], args.sr)
            words = [w for w in tokenize(r.text_norm) if TEL.search(w)]
            write_utt(raw, spk, r.uid, wav, sr, " ".join(words))
            bar.update()
    bar.close()
    with open(os.path.join(args.out, "split.tsv"), "w") as f:
        for _, r in sel.iterrows():
            f.write(f"{r.uid}\t{r.split}\n")
    sel.drop(columns=["file"]).to_csv(os.path.join(args.out, "selection.csv"), index=False)
    df.drop(columns=["file", "text"]).to_csv(os.path.join(args.out, "all_candidates.csv"), index=False)

    vocab = set()
    for t in sel.text_norm:
        vocab.update(w for w in tokenize(t) if TEL.search(w))
    with open(os.path.join(args.out, "lexicon.txt"), "w", encoding="utf-8") as f:
        for w in sorted(vocab):
            ph = word_to_phones(w)
            if ph:
                f.write(w + "\t" + " ".join(ph) + "\n")
    log["lexicon_words"] = len(vocab)
    log["speaker"] = spk
    dump_json(log, os.path.join(args.out, "selection_log.json"))


if __name__ == "__main__":
    main()
