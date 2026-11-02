"""Select ~5 h train + ~1 h val of Indian-accented English from Svarah
(ai4bharat/Svarah, gated on Hugging Face) and write an MFA-ready corpus.

Svarah is an ASR benchmark (117 speakers, read + spontaneous speech), not a
TTS corpus, so selection is about cleanliness:
  1. 1 s <= duration <= --max_dur s
  2. transcript is plain English (no digits left after normalisation, no
     tags such as [noise] / <unk>, at least 3 words)
  3. estimated SNR >= --min_snr dB, no clipping, at most --max_edge_sil s of
     leading/trailing silence
  4. speakers with less than --min_spk_min minutes of clean audio are dropped
     (a speaker embedding needs enough examples to be learned)
  5. the val set is drawn *per speaker* (~1/6 of each speaker's selected
     audio): every val speaker is also seen in training (the model has one
     embedding per speaker), but no utterance is in both sets.
Outputs: raw_data/<speaker>/<id>.wav (22.05 kHz mono) + .lab, split.tsv,
selection.csv, dictionary.txt (LibriSpeech lexicon + g2p_en for OOV words).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prep.common import decode, dump_json, pick_split, quality, write_utt  # noqa: E402
from text.frontend_en import normalise, read_lexicon  # noqa: E402

SPK_COLS = ["speaker_id", "speaker", "spk_id", "client_id"]
META_COLS = ["gender", "age-group", "primary_language", "native_place_state",
             "native_place_district", "highest_qualification", "job_category", "occupation_domain"]


def assign_speakers(df):
    """The HF release of Svarah has no speaker-id column (it is only in the
    original meta_speaker_stats.csv). Recover speakers in this order:
      1. an explicit speaker column, if present
      2. the file-name prefix before the first '_' (AI4Bharat naming), if it
         gives a plausible number of speakers (Svarah has 117)
      3. the tuple of speaker metadata (gender, age group, L1, state, district,
         qualification, job, domain) - near-unique per speaker
    Returns (series, method)."""
    for c in SPK_COLS:
        if c in df.columns and df[c].notna().all():
            return df[c].astype(str), c
    pre = df["path"].fillna("").map(lambda p: os.path.basename(p).split("_")[0] if "_" in os.path.basename(p) else "")
    if (pre != "").all() and 60 <= pre.nunique() <= 250:
        return pre, "filename_prefix"
    cols = [c for c in META_COLS if c in df.columns]
    tup = df[cols].astype(str).agg("|".join, axis=1)
    codes = {t: f"s{i:03d}" for i, t in enumerate(sorted(tup.unique()))}
    return tup.map(codes), "metadata_tuple:" + ",".join(cols)


def audio_col(t):
    for c in t.columns:
        v = t[c].iloc[0]
        if isinstance(v, dict) and "bytes" in v:
            return c
    raise KeyError("no audio column")


def clean_text(t):
    t = re.sub(r"\[[^\]]*\]|<[^>]*>|\([^)]*\)", " ", str(t))
    t = normalise(t)
    t = t.replace("-", " ")
    t = re.sub(r"[^A-Za-z' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon", required=True, help="librispeech-lexicon.txt")
    ap.add_argument("--train_h", type=float, default=5.0)
    ap.add_argument("--val_h", type=float, default=1.0)
    ap.add_argument("--min_dur", type=float, default=1.0)
    ap.add_argument("--max_dur", type=float, default=11.0)
    ap.add_argument("--min_snr", type=float, default=20.0)
    ap.add_argument("--max_edge_sil", type=float, default=1.5)
    ap.add_argument("--min_spk_min", type=float, default=1.0)
    ap.add_argument("--sr", type=int, default=22050)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.hf_dir, "**", "*.parquet"), recursive=True))
    rows = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        acol = audio_col(t)
        tcol = "text" if "text" in t.columns else [c for c in t.columns if "text" in c or "transcript" in c][0]
        meta_cols = [c for c in t.columns if c not in (acol, tcol)]
        for i, r in t.iterrows():
            wav, sr = decode(r[acol]["bytes"])
            q = quality(wav, sr)
            rows.append({"uid": f"{os.path.basename(f)[:-8]}_{i:05d}", "file": f, "row": i, "acol": acol,
                         "path": r[acol].get("path"), "text": r[tcol], "dur": len(wav) / sr,
                         "orig_sr": sr, **q, **{c: r[c] for c in meta_cols if np.isscalar(r[c])}})
        del t
    df = pd.DataFrame(rows)
    df["speaker"], spk_method = assign_speakers(df)
    print("speaker ids from:", spk_method, "->", df.speaker.nunique(), "speakers")
    df["uid"] = "en_" + df["speaker"].str.replace(r"[^A-Za-z0-9]", "", regex=True) + "_" + df["uid"]
    log = {"speaker_id_method": spk_method, "available_utts": len(df), "available_hours": round(df.dur.sum() / 3600, 3),
           "available_speakers": int(df.speaker.nunique())}

    df["text_clean"] = df.text.map(clean_text)
    keep = (df.dur >= args.min_dur) & (df.dur <= args.max_dur)
    log["after_duration_filter"] = int(keep.sum())
    keep &= df.text_clean.str.split().str.len() >= 3
    keep &= ~df.text.astype(str).str.contains(r"\[|<|\*|#")
    log["after_text_filter"] = int(keep.sum())
    keep &= (df.snr_db >= args.min_snr) & (df.clip_frac < 1e-3)
    keep &= (df.lead_s <= args.max_edge_sil) & (df.trail_s <= args.max_edge_sil)
    log["after_quality_filter"] = int(keep.sum())
    pool = df[keep].copy()
    spk_min = pool.groupby("speaker").dur.sum() / 60
    good = spk_min[spk_min >= args.min_spk_min].index
    pool = pool[pool.speaker.isin(good)]
    log["after_speaker_filter"] = int(len(pool))
    log["pool_hours"] = round(pool.dur.sum() / 3600, 3)
    log["pool_speakers"] = int(pool.speaker.nunique())
    # prefer the cleanest clips when the pool is larger than the budget
    pool = pool.sort_values("snr_db", ascending=False)
    if pool.dur.sum() / 3600 > (args.train_h + args.val_h) * 1.0:
        pool = pool[pool.dur.cumsum() / 3600 <= (args.train_h + args.val_h) * 1.15]
    sel = pick_split(pool, args.train_h, args.val_h, group="speaker")
    for s in ("train", "val"):
        d = sel[sel.split == s]
        log[f"{s}_utts"], log[f"{s}_hours"] = int(len(d)), round(d.dur.sum() / 3600, 3)
        log[f"{s}_speakers"] = int(d.speaker.nunique())
    log["val_speakers_unseen_in_train"] = len(set(sel[sel.split == "val"].speaker) - set(sel[sel.split == "train"].speaker))
    print(json.dumps(log, indent=1))

    raw = os.path.join(args.out, "raw_data")
    tables = {}
    for _, r in tqdm(sel.sort_values(["file", "row"]).iterrows(), total=len(sel)):
        if r.file not in tables:
            tables = {r.file: pq.read_table(r.file).to_pandas()}
        wav, sr = decode(tables[r.file].iloc[r.row][r.acol]["bytes"], args.sr)
        spk = "en_" + re.sub(r"[^A-Za-z0-9]", "", r.speaker)
        write_utt(raw, spk, r.uid, wav, sr, r.text_clean)
    with open(os.path.join(args.out, "split.tsv"), "w") as f:
        for _, r in sel.iterrows():
            f.write(f"{r.uid}\t{r.split}\n")
    sel.drop(columns=["file"]).to_csv(os.path.join(args.out, "selection.csv"), index=False)
    df.drop(columns=["file"]).to_csv(os.path.join(args.out, "all_candidates.csv"), index=False)

    # MFA dictionary: every pronunciation from LibriSpeech + g2p_en for OOVs
    lex = read_lexicon(args.lexicon)
    from g2p_en import G2p

    g2p = G2p()
    words = sorted({w for t in sel.text_clean for w in t.split()})
    oov = 0
    with open(os.path.join(args.out, "dictionary.txt"), "w") as f:
        for w in words:
            prons = lex.get(w)
            if not prons:
                oov += 1
                prons = [g2p.cmu[w][0]] if w in g2p.cmu else [g2p.predict(w)]
            for p in prons:
                f.write(w + "\t" + " ".join(p) + "\n")
    log["dictionary_words"], log["oov_words_g2p"] = len(words), oov
    dump_json(log, os.path.join(args.out, "selection_log.json"))


if __name__ == "__main__":
    main()
