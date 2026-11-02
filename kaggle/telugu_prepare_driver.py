# Telugu data preparation (Kaggle CPU, Linux or native Windows): audit -> pick voice ->
# select -> MFA train/align -> features -> TE_OUT (default /kaggle/working/fs2_prepared/Telugu, fp16 mels)
import glob, json, os, re, shutil, subprocess, sys, time

T0 = time.time()
PROJ = os.environ.get("TE_PROJ", "/kaggle/tmp/project")
os.chdir(PROJ)
sys.path.insert(0, "core")
WINDOWS = os.name == "nt"


def run(*args):
    """Run a project script with this same interpreter (not whatever 'python' is on PATH)."""
    cmd = [sys.executable, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if not WINDOWS:  # Kaggle image: add what it lacks (the Windows env is built beforehand)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyworld", "tgt", "soxr", "inflect", "unidecode"],
                   check=True)
from huggingface_hub import snapshot_download
import numpy as np, pandas as pd
from prep.common import decode, quality
from prep.prepare_indictts_telugu import iter_rows, label_names

HF = os.environ.get("TE_HF_DIR", "/kaggle/tmp/hf/IndicTTS_Telugu")
JOBS = int(os.environ.get("TE_JOBS", "4"))
MFA_JOBS = int(os.environ.get("TE_MFA_JOBS", JOBS))
if not os.environ.get("TE_HF_DIR"):
    snapshot_download("SPRINGLab/IndicTTS_Telugu", repo_type="dataset", local_dir=HF)
files = sorted(glob.glob(os.path.join(HF, "**", "*.parquet"), recursive=True))
names = label_names(files[0])


def audit_rows(files, names):
    rows = []
    for f in files:
        for _, r in iter_rows(f, ["audio", "gender"]):
            w, sr = decode(r["audio"]["bytes"])
            g = names[r["gender"]] if names else str(r["gender"])
            rows.append({"gender": g, "dur": len(w) / sr, "sr": sr, **quality(w, sr)})
    return rows


AUDIT_CACHE = "work/Telugu/telugu_audit.csv"   # a rerun skips the audit pass
if os.path.exists(AUDIT_CACHE):
    A = pd.read_csv(AUDIT_CACHE)
elif WINDOWS:  # no fork on Windows; the decoding peak (~1 GB) is freed when the rows are dropped
    A = pd.DataFrame(audit_rows(files, names))
else:
    # in a child process, so the ~1 GB peak of decoding is returned before MFA starts
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("fork")) as ex:
        A = pd.DataFrame(ex.submit(audit_rows, files, names).result())
if not os.path.exists(AUDIT_CACHE):
    os.makedirs(os.path.dirname(AUDIT_CACHE), exist_ok=True)
    A.to_csv(AUDIT_CACHE, index=False)


def summ(d):
    ok = d[(d.dur >= 1) & (d.dur <= 11)]
    return pd.Series({"utts": len(d), "hours": d.dur.sum() / 3600, "median_s": d.dur.median(),
                      "hours_1to11s": ok.dur.sum() / 3600,
                      "hours_1to11s_snr25": ok[ok.snr_db >= 25].dur.sum() / 3600,
                      "median_snr_db": d.snr_db.median(), "sample_rate": d.sr.mode()[0]})


AUDIT = A.groupby("gender").apply(summ).round(2)
print("AUDIT\n" + AUDIT.T.to_string(), flush=True)
# choose the voice with the most clean hours in the usable length range
GENDER = AUDIT["hours_1to11s_snr25"].idxmax()
print("CHOSEN VOICE:", GENDER, flush=True)

cfg = re.sub(r'gender: "\w+"', f'gender: "{GENDER}"', open("Telugu/config.yaml", encoding="utf-8").read())
f0 = (60.0, 350.0) if GENDER == "male" else (100.0, 500.0)
cfg = re.sub(r"f0_floor: [\d.]+", f"f0_floor: {f0[0]}", cfg)
cfg = re.sub(r"f0_ceil: [\d.]+", f"f0_ceil: {f0[1]}", cfg)
open("Telugu/config.yaml", "w", encoding="utf-8").write(cfg)

run("Telugu/preprocess.py", "--stage", "select", "--hf_dir", HF, "--gender", GENDER)
if not WINDOWS:
    subprocess.run(["bash", "core/prep/mfa.sh", "setup"], check=True)
run("Telugu/preprocess.py", "--stage", "align", "--jobs", str(MFA_JOBS))
run("Telugu/preprocess.py", "--stage", "features", "--jobs", str(JOBS))

OUT = os.environ.get("TE_OUT", "/kaggle/working/fs2_prepared/Telugu")
W = "work/Telugu"
os.makedirs(f"{OUT}/corpus", exist_ok=True)
shutil.copytree(f"{W}/preprocessed", f"{OUT}/preprocessed")
shutil.copytree(f"{W}/mfa", f"{OUT}/mfa")
for f in ["selection_log.json", "selection.csv", "split.tsv", "lexicon.txt", "all_candidates.csv"]:
    shutil.copy(f"{W}/corpus/{f}", f"{OUT}/corpus/")
A.to_csv(f"{OUT}/corpus/telugu_audit.csv", index=False)
open(f"{OUT}/corpus/audit_summary.txt", "w").write(AUDIT.T.to_string())
shutil.copy("Telugu/config.yaml", f"{OUT}/config_used.yaml")
mx = 0.0
for f in glob.glob(f"{OUT}/preprocessed/mel/*.npy"):
    a = np.load(f)
    b = a.astype(np.float16)
    mx = max(mx, float(np.abs(a - b.astype(np.float32)).max()))
    np.save(f, b)
print("mel -> float16, max abs error", mx)
print("SELECTION", open(f"{W}/corpus/selection_log.json", encoding="utf-8").read())
print("SUMMARY", open(f"{W}/preprocessed/summary.json").read())
print("STATS", open(f"{W}/preprocessed/stats.json").read())
size = sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(OUT) for x in fs)
print(f"{OUT}: {size / 1e6:.0f} MB")
print(f"TELUGU PREP DONE in {(time.time() - T0) / 3600:.2f} h")
