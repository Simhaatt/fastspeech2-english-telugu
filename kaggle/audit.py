# Data audit: run on Kaggle (CPU, internet on). Prints dataset structure and
# per-speaker / per-gender hour budgets so we can choose the subsets.
import os, io, glob, json, subprocess, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf
from huggingface_hub import snapshot_download

T = "/kaggle/temp/hf"
os.makedirs(T, exist_ok=True)
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("no HF_TOKEN secret:", repr(e)[:200])

def audinfo(b):
    try:
        i = sf.info(io.BytesIO(b)); return i.samplerate, i.frames / i.samplerate, i.channels
    except Exception:
        return None, None, None

def audit(repo, token=None, max_files=None):
    t0 = time.time()
    p = snapshot_download(repo_id=repo, repo_type="dataset", local_dir=f"{T}/{repo.split('/')[1]}", token=token)
    files = sorted(glob.glob(p + "/**/*", recursive=True))
    print(repo, "files:", [(f.replace(p, ""), round(os.path.getsize(f) / 1e6, 1)) for f in files if os.path.isfile(f)][:30])
    pqs = [f for f in files if f.endswith(".parquet")]
    rows = []
    for f in pqs:
        tb = pq.read_table(f)
        print(f.replace(p, ""), tb.schema)
        df = tb.to_pandas()
        acol = [c for c in df.columns if isinstance(df[c].iloc[0], dict) and "bytes" in df[c].iloc[0]]
        for c in acol:
            info = df[c].map(lambda d: audinfo(d["bytes"]))
            df["sr"] = info.map(lambda x: x[0]); df["dur"] = info.map(lambda x: x[1]); df["ch"] = info.map(lambda x: x[2])
            df["path"] = df[c].map(lambda d: d.get("path"))
            df = df.drop(columns=[c])
        df["file"] = os.path.basename(f)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    print(repo, "rows", len(df), "cols", list(df.columns), "time", round(time.time() - t0), "s")
    print(df.head(3).T)
    print("sample rates:", df["sr"].value_counts().to_dict(), "channels:", df["ch"].value_counts().to_dict())
    print("total hours:", round(df["dur"].sum() / 3600, 2))
    print("duration pct:", np.percentile(df["dur"].dropna(), [1, 5, 25, 50, 75, 95, 99]).round(1))
    return df

os.makedirs("/kaggle/working/audit", exist_ok=True)
te = audit("SPRINGLab/IndicTTS_Telugu")
for g, d in te.groupby("gender"):
    ok = d[(d.dur >= 1) & (d.dur <= 15)]
    ok20 = d[(d.dur >= 1) & (d.dur <= 20)]
    print(f"TELUGU gender={g}: n={len(d)} hours={d.dur.sum()/3600:.2f} | 1-15s: n={len(ok)} h={ok.dur.sum()/3600:.2f} | 1-20s: h={ok20.dur.sum()/3600:.2f} | median={d.dur.median():.1f}s")
te.drop(columns=[c for c in te.columns if c == "text"]).to_csv("/kaggle/working/audit/telugu_meta.csv", index=False)
te[["text"]].to_csv("/kaggle/working/audit/telugu_text.csv", index=False)

try:
    sv = audit("ai4bharat/Svarah", token=tok)
    sv.to_csv("/kaggle/working/audit/svarah_meta.csv", index=False)
    for c in sv.columns:
        if sv[c].dtype == object and sv[c].nunique() < 200:
            print("SVARAH col", c, "nunique", sv[c].nunique(), sv[c].value_counts().head(8).to_dict())
except Exception as e:
    print("SVARAH FAILED:", repr(e)[:500])
print("AUDIT DONE")
