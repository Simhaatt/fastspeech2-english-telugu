"""Builds the self-contained Kaggle notebooks (code bundle embedded as base64).

    python kaggle/build_notebooks.py   ->  kaggle/FastSpeech2_English.ipynb, kaggle/FastSpeech2_Telugu.ipynb
"""
import base64
import hashlib
import io
import os
import tarfile
import time

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["core", "English", "Telugu", "data", "requirements.txt"]
EXCLUDE_DIRS = {"__pycache__", "checkpoints", "lexicon"}
EXCLUDE_EXT = {".zip", ".tar", ".pth"}


def bundle():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for top in INCLUDE:
            p = os.path.join(ROOT, top)
            if os.path.isfile(p):
                tar.add(p, arcname=top)
                continue
            for d, dirs, files in os.walk(p):
                dirs[:] = [x for x in dirs if x not in EXCLUDE_DIRS]
                for f in files:
                    if os.path.splitext(f)[1] in EXCLUDE_EXT:
                        continue
                    fp = os.path.join(d, f)
                    tar.add(fp, arcname=os.path.relpath(fp, ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


COMMON_SETUP = r'''
import os, sys, time, json, subprocess, shutil, glob
T0 = time.time()
PROJ = "/kaggle/working/project"
# scratch space for big intermediate files (NOT saved as notebook output)
for TMP in ("/kaggle/tmp", "/tmp/fs2"):
    try:
        os.makedirs(TMP, exist_ok=True)
        break
    except OSError:
        continue
WORK = f"{TMP}/work"
os.makedirs(WORK, exist_ok=True)

def sh(cmd, check=True):
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r.returncode

def hours():
    return (time.time() - T0) / 3600

# ---- unpack the project code (embedded below) ----
import base64, io, tarfile
os.makedirs(PROJ, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE_B64)), mode="r:gz").extractall(PROJ)
# a stale notebook re-runs old code; this line in the log tells which build ran
print("code build:", CODE_BUILD, "| energy dtype fix:",
      ".from_numpy(energies).float()" in open(f"{PROJ}/core/utils/tools.py").read())
if not os.path.islink(f"{PROJ}/work"):
    os.symlink(WORK, f"{PROJ}/work")
os.chdir(PROJ)

# ---- assets from the reference implementation: LibriSpeech lexicon + HiFi-GAN ----
if not os.path.exists(f"{TMP}/ming024"):
    sh(f"git clone -q --depth 1 https://github.com/ming024/FastSpeech2.git {TMP}/ming024")
os.makedirs("core/lexicon", exist_ok=True)
shutil.copy(f"{TMP}/ming024/lexicon/librispeech-lexicon.txt", "core/lexicon/")
shutil.copy(f"{TMP}/ming024/hifigan/generator_universal.pth.tar.zip", "core/hifigan/")

# ---- python deps (torch, librosa, numpy come with the Kaggle image) ----
sh("pip install -q pyworld tgt g2p_en==2.1.0 soxr inflect unidecode 2>&1 | tail -2", check=False)
import nltk
for r in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]:
    nltk.download(r, quiet=True)
import torch
print("torch", torch.__version__, "| GPUs:", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(f"setup done at {hours():.2f} h")
'''

MFA_SETUP = r'''
# Montreal Forced Aligner 3.4.3 in its own micromamba env (~5 min). MFA's
# PostgreSQL refuses to run as root, so mfa.sh runs it as user "mfa".
sh("bash core/prep/mfa.sh setup")
print(f"MFA ready at {hours():.2f} h")
'''

TRAIN_HELPERS = r'''
def launch(cmd, log, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    f = open(log, "w")
    print(f"[GPU{gpu}] {cmd}\n   log -> {log}")
    return subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env)

def watch(procs, every_min=15):
    """procs: {name: (Popen, log)} - print the tail of each log until all finish."""
    while True:
        alive = {n: p for n, (p, l) in procs.items() if p.poll() is None}
        for n, (p, l) in procs.items():
            lines = [x for x in open(l).read().splitlines() if x.startswith(("step", "VAL", "  new best", "finished", "Traceback", "RuntimeError"))]
            print(f"--- {n} ({'running' if p.poll() is None else 'exit ' + str(p.returncode)}) @ {hours():.2f} h:", *lines[-3:], sep="\n   ")
        if not alive:
            break
        time.sleep(every_min * 60)
    for n, (p, l) in procs.items():
        if p.returncode != 0:
            print(open(l).read()[-3000:])
            raise RuntimeError(f"{n} failed")
'''



def prep_save(lang, extra_dirs=""):
    return code(r"""
# ---- package everything training needs into ONE file for a Kaggle dataset ----
OUT = "/kaggle/working/prepared"
os.makedirs(f"{OUT}/reports", exist_ok=True)
sh(f"cd {WORK} && tar -cf {OUT}/LANG.tar LANG/corpus LANG/preprocessed EXTRA")
for f in ["corpus/selection_log.json", "corpus/selection.csv", "preprocessed/summary.json",
          "preprocessed/stats.json", "preprocessed/speakers.json"]:
    if os.path.exists(f"work/LANG/{f}"):
        shutil.copy(f"work/LANG/{f}", f"{OUT}/reports/")
shutil.copy("LANG/config.yaml", f"{OUT}/config_used.yaml")
os.chdir("/kaggle/working"); os.unlink(f"{PROJ}/work"); shutil.rmtree(PROJ)
sh(f"ls -lh {OUT} {OUT}/reports")
print(f"PREP DONE in {hours():.2f} h  ->  now: Output tab -> 'New Dataset' (keep it private)")
""".replace("LANG", lang).replace("EXTRA", extra_dirs))


FIND_DATA = r'''
def find_prepared(lang, roots, dest, tmp):
    """Locate <lang>/preprocessed/stats.json in anything attached as input, whatever the dataset
    is called and however deep it sits: an extracted folder (Kaggle unzips uploaded zips), or
    a .zip / .tar that was left packed. Puts the data at `dest`; returns where it came from."""
    import tarfile, zipfile
    def find(pattern):
        return sorted({p for r in roots for p in glob.glob(os.path.join(r, "**", pattern), recursive=True)})
    shutil.rmtree(dest, ignore_errors=True)
    key = f"{lang}/preprocessed/stats.json"
    dirs = [p[: -len("/preprocessed/stats.json")] for p in find(key)]
    if dirs:
        if len(dirs) > 1:
            print("several copies found, using the first:", *dirs, sep="\n  ")
        os.makedirs(dest)
        for sub in os.listdir(dirs[0]):   # link, read-only input is fine
            os.symlink(os.path.join(dirs[0], sub), os.path.join(dest, sub))
        return dirs[0]
    for arc in find("*.zip") + find("*.tar"):
        packed = zipfile.ZipFile(arc) if arc.endswith(".zip") else tarfile.open(arc)
        names = packed.namelist() if arc.endswith(".zip") else packed.getnames()
        hit = [n for n in names if n.endswith(key)]
        if not hit:
            continue
        prefix = hit[0][: -len("preprocessed/stats.json")]          # e.g. "fs2_prepared/Telugu/"
        unpack = os.path.join(tmp, "unpack_" + lang)
        shutil.rmtree(unpack, ignore_errors=True)
        members = [n for n in names if n.startswith(prefix)]
        if arc.endswith(".zip"):
            packed.extractall(unpack, members)
        else:
            packed.extractall(unpack, [packed.getmember(n) for n in members])
        shutil.move(os.path.join(unpack, prefix), dest)
        return arc
    print("Nothing with", key, "under", roots, "- what is attached:")
    for r in roots:
        for d, subs, files in os.walk(r):
            if d.count(os.sep) - r.count(os.sep) < 3:
                print("  ", d, f"({len(files)} files)")
    raise SystemExit(f"Prepared {lang} data not found: Add Input -> your private prepared-data dataset")
'''


def train_load(lang):
    return code(FIND_DATA + r"""
# ---- load the prepared data (attach it via Add Input first; the name does not matter) ----
src = find_prepared("__LANG__", [DATA_DIR] if DATA_DIR else ["/kaggle/input"], f"{WORK}/__LANG__", TMP)
print("prepared __LANG__ data from:", src)
if os.path.exists(f"{WORK}/__LANG__/config_used.yaml"):
    shutil.copy(f"{WORK}/__LANG__/config_used.yaml", "__LANG__/config.yaml")   # same settings as the prep run
print(open("work/__LANG__/preprocessed/summary.json").read())
print(f"data ready at {hours():.2f} h")
""".replace("__LANG__", lang))


def english_prep_nb(b64):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# FastSpeech 2 — English · step 1/2: data preparation (CPU)

**Settings:** Accelerator **None** (CPU is enough; saves GPU quota), Internet **on**,
**Add Input → your uploaded Svarah dataset** (the parquet files). Without it the notebook downloads from
Hugging Face, which needs Add-ons → Secrets → `HF_TOKEN`.

**Run:** *Save Version → Save & Run All (Commit)* (~1–1.5 h). Then open the version → **Output** →
**New Dataset**, name it `fs2-english-prepared`, keep it **private**. Step 2 trains from that dataset.

Does: download Svarah → select ~5 h train / ~1 h val → MFA alignment → mel / duration / pitch / energy features.
"""),
        code("CODE_B64 = '" + b64 + "'\nCODE_BUILD = '" + build_id(b64) + "'"),
        code(COMMON_SETUP),
        code(MFA_SETUP),
        md("## 1 · Download Svarah and select ~5 h train / ~1 h val"),
        code(r'''
# uses your uploaded Kaggle dataset if attached (Add Input), otherwise downloads from Hugging Face
up = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/test-*.parquet", recursive=True)})
if up:
    HF = up[0]
    print("using uploaded Svarah parquet files in", HF)
else:
    from kaggle_secrets import UserSecretsClient
    from huggingface_hub import snapshot_download
    HF = f"{TMP}/hf/Svarah"
    snapshot_download("ai4bharat/Svarah", repo_type="dataset", local_dir=HF,
                      token=UserSecretsClient().get_secret("HF_TOKEN"))
sh(f"du -sh {HF}; find {HF} -name '*.parquet' | head")
sh(f"python English/preprocess.py --stage select --hf_dir {HF}")
print(open("work/English/corpus/selection_log.json").read())
'''),
        md("## 2 · Forced alignment (MFA `english_us_arpa`, LibriSpeech lexicon + g2p_en for OOV words)"),
        code(r'''
sh("python English/preprocess.py --stage align --jobs 4")
n_tg = len(glob.glob("work/English/preprocessed/TextGrid/*/*.TextGrid"))
n_wav = len(glob.glob("work/English/corpus/raw_data/*/*.wav"))
print(f"aligned {n_tg}/{n_wav} utterances at {hours():.2f} h")
'''),
        md("## 3 · Features (mel, durations, phone-level F0 and energy)"),
        code(r'''
sh("python English/preprocess.py --stage features --jobs 4")
print(open("work/English/preprocessed/summary.json").read())
'''),
        prep_save("English"),
    ]
    return nb


def english_train_nb(b64):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# FastSpeech 2 — English · step 2/2: training + inference (GPU)

**Settings:** Accelerator **GPU T4 ×2**, Internet **on**, **Add Input → your private prepared-data dataset** (any name: the notebook finds `English/` and `Telugu/` inside whatever is attached).

**Run:** *Save Version → Save & Run All (Commit)*. Training stops on a time budget inside Kaggle's 12 h limit.
Then open the version → **Output** and download `English_results.zip`.

Two models train at the same time, one per T4:
* **scratch** — random init, the main submission (`English/checkpoints/`)
* **finetune** — initialised from the public LJSpeech FastSpeech 2 checkpoint (ablation, `English/checkpoints/finetune/`)
"""),
        code("""
# ---- knobs ----
DATA_DIR        = None     # None: search everything attached under /kaggle/input (any dataset name)
TIME_BUDGET_H   = 11.3     # Kaggle hard limit is 12 h; leaves room for inference + packaging
RUN_FINETUNE    = True     # ablation run on the second GPU
FINETUNE_STEPS  = 30000
SCRATCH_STEPS   = 60000    # upper bound; the time budget may stop it first
"""),
        code("CODE_B64 = '" + b64 + "'\nCODE_BUILD = '" + build_id(b64) + "'"),
        code(COMMON_SETUP),
        train_load("English"),
        md("## 4 · Training (scratch on GPU0, LJSpeech fine-tune on GPU1)"),
        code(TRAIN_HELPERS),
        code(r'''
budget = TIME_BUDGET_H - hours() - 0.6
print(f"training budget: {budget:.2f} h")
procs = {}
procs["scratch"] = (launch(f"python English/train.py --total_step {SCRATCH_STEPS} --max_hours {budget:.2f}",
                           "/kaggle/working/train_scratch.log", 0), "/kaggle/working/train_scratch.log")
if RUN_FINETUNE:
    lj = None
    try:
        # ming024's Drive folder (1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F) holds LJSpeech_900000.zip next to the
        # AISHELL3 and LibriTTS checkpoints: fetch only the LJSpeech zip by its file id and unpack it
        sh("pip install -q gdown", check=False)
        pre = f"{TMP}/pretrained"
        os.makedirs(pre, exist_ok=True)
        z = f"{pre}/LJSpeech_900000.zip"
        if not os.path.exists(z):
            sh(f"gdown 1r3fYhnblBJ8hDKDSUDtidJ-BN-xAM9pe -O {z}", check=False)
        if os.path.exists(z):
            import zipfile
            zipfile.ZipFile(z).extractall(f"{pre}/LJSpeech")
        c = sorted(glob.glob(f"{pre}/LJSpeech/**/*.pth.tar", recursive=True))
        print("checkpoints in the LJSpeech zip:", c)
        lj = c[-1] if c else None
    except Exception as e:
        print("pretrained download failed:", repr(e))
    if lj:
        print("LJSpeech checkpoint:", lj)
        cfg = open("English/config.yaml").read().replace("lr_scale: 1.0", "lr_scale: 0.5").replace("warm_up_step: 4000", "warm_up_step: 1000")
        open("English/config_finetune.yaml", "w").write(cfg)
        procs["finetune"] = (launch(f"python English/train.py --config English/config_finetune.yaml --tag finetune --init_from {lj} "
                                    f"--total_step {FINETUNE_STEPS} --max_hours {budget:.2f}",
                                    "/kaggle/working/train_finetune.log", 1), "/kaggle/working/train_finetune.log")
    else:
        print("!! LJSpeech checkpoint not available - skipping the fine-tune ablation")
watch(procs)
print(f"training done at {hours():.2f} h")
'''),
        md("## 5 · Inference on the test sentences + variance-adaptor plots"),
        code(r'''
import pandas as pd
sel = pd.read_csv("work/English/corpus/selection.csv")
tr = sel[sel.split == "train"]
stats = tr.groupby("speaker").agg(hours=("dur", lambda x: x.sum() / 3600), snr=("snr_db", "median"))
gcol = [c for c in sel.columns if c.lower() == "gender"]
if gcol:
    stats["gender"] = tr.groupby("speaker")[gcol[0]].first()
stats["score"] = stats.hours.rank() + stats.snr.rank()
stats = stats.sort_values("score", ascending=False)
print(stats.head(10))
spk_ids = json.load(open("work/English/preprocessed/speakers.json"))
key = lambda s: "en_" + "".join(ch for ch in str(s) if ch.isalnum())
main_spk = key(stats.index[0])
extra = []
if gcol:
    for g in stats.gender.dropna().unique():
        s = key(stats[stats.gender == g].index[0])
        if s != main_spk:
            extra.append(s)
print("main inference speaker:", main_spk, "| extra:", extra)
json.dump({"main": main_spk, "extra": extra, "table": stats.head(10).reset_index().to_dict("records")},
          open("work/English/inference_speakers.json", "w"), indent=1, default=str)

R = "Results/English"
sh(f"python English/inference.py --speaker {main_spk} --out_dir {R}")
sh(f"python English/inference.py --speaker {main_spk} --out_dir {R} --no_repair --suffix _no_repair --only E3")
for s in extra:
    sh(f"python English/inference.py --speaker {s} --out_dir {R}/other_speakers/{s}")
if os.path.exists("English/checkpoints/finetune/best.pth.tar"):
    sh(f"python English/inference.py --config English/config_finetune.yaml --ckpt English/checkpoints/finetune/best.pth.tar --speaker {main_spk} --out_dir {R}/finetune_ablation")
'''),
        md("## 6 · Loss plots, release checkpoints, packaging"),
        code(r'''
runs = "scratch:work/English/logs" + (" finetune:work/English/logs/finetune" if os.path.exists("work/English/logs/finetune/losses.csv") else "")
ck = "scratch:English/checkpoints" + (" finetune:English/checkpoints/finetune" if os.path.exists("English/checkpoints/finetune/best.pth.tar") else "")
sh(f"python core/export.py --lang English --runs {runs} --ckpt_dirs {ck}")
shutil.copy("work/English/inference_speakers.json", "Results/English/logs/")
for l in glob.glob("/kaggle/working/train_*.log"):
    shutil.copy(l, "Results/English/logs/")
os.unlink(f"{PROJ}/work")      # don't save the big intermediate features
# deliverable zip: code + configs + fp16 checkpoints + results (full-precision ckpts stay in the output dir)
sh("cd /kaggle/working && zip -qr English_results.zip project/Results/English project/English "
   "-x '*/last.pth.tar' '*/best.pth.tar' '*/[0-9]*.pth.tar' && ls -lh English_results.zip")
print(f"ALL DONE in {hours():.2f} h")
'''),
    ]
    return nb


def telugu_prep_nb(b64):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# FastSpeech 2 — Telugu · step 1/2: audit + data preparation (CPU)

**Settings:** Accelerator **None**, Internet **on**, **Add Input → your uploaded IndicTTS Telugu dataset**
(the 4 parquet files). Without it the notebook downloads them from Hugging Face.

The audit (printed first) already picked the **male** voice: 6.22 h of clips in 1–11 s vs 1.89 h for the
female voice. Set `TELUGU_GENDER = None` only to re-run the audit on its own.

**Run:** *Save Version → Save & Run All (Commit)* (~1.5–2 h; trains a Telugu MFA aligner). The training
notebook can then use this notebook's output directly (Add Input → Your Work → this notebook), or make it a
**private** dataset from the version's Output tab.
"""),
        code("""
# ---- knobs ----
TELUGU_GENDER = "male"     # None -> audit only and stop
"""),
        code("CODE_B64 = '" + b64 + "'\nCODE_BUILD = '" + build_id(b64) + "'"),
        code(COMMON_SETUP),
        md("## 0 · Download IndicTTS Telugu and audit both voices"),
        code(r'''
from huggingface_hub import snapshot_download
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, "core")
from prep.common import decode, quality
from prep.prepare_indictts_telugu import iter_rows, label_names
up =sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train-0000*-of-00004.parquet", recursive=True)})
if up:
    HF = up[0]
    print("using uploaded IndicTTS Telugu parquet files in", HF)
else:
    HF = f"{TMP}/hf/IndicTTS_Telugu"
    snapshot_download("SPRINGLab/IndicTTS_Telugu", repo_type="dataset", local_dir=HF)
files = sorted(glob.glob(f"{HF}/**/*.parquet", recursive=True))
names = label_names(files[0]); print("gender labels:", names)
rows = []
for f in files:
    for i, r in iter_rows(f, ["audio", "text", "gender"]):
        w, sr = decode(r["audio"]["bytes"])
        g = names[r["gender"]] if names else r["gender"]
        rows.append({"gender": g, "dur": len(w) / sr, "sr": sr, "chars": len(r["text"]), **quality(w, sr)})
A = pd.DataFrame(rows)
A.to_csv("/kaggle/working/telugu_audit.csv", index=False)
def summ(d):
    ok = d[(d.dur >= 1) & (d.dur <= 11)]
    return pd.Series({"utts": len(d), "hours": d.dur.sum()/3600, "median_s": d.dur.median(), "p95_s": d.dur.quantile(.95),
                      "hours_1to11s": ok.dur.sum()/3600, "hours_1to11s_snr25": ok[ok.snr_db >= 25].dur.sum()/3600,
                      "median_snr_db": d.snr_db.median(), "clipped_utts": int((d.clip_frac > 1e-3).sum()),
                      "sample_rate": d.sr.mode()[0]})
AUDIT = A.groupby("gender").apply(summ).round(2)
print(AUDIT.T.to_string())
if TELUGU_GENDER is None:
    raise SystemExit("Audit done - set TELUGU_GENDER and re-run (Save & Run All).")
'''),
        code(MFA_SETUP),
        md("## 1 · Select ~5 h train / ~1 h val of one voice"),
        code(r'''
sh(f"python Telugu/preprocess.py --stage select --hf_dir {HF} --gender {TELUGU_GENDER}")
print(open("work/Telugu/corpus/selection_log.json").read())
'''),
        md("## 2 · Train a Telugu MFA acoustic model on this corpus and align it (rule-based G2P lexicon)"),
        code(r'''
sh("python Telugu/preprocess.py --stage align --jobs 4")
n_tg = len(glob.glob("work/Telugu/preprocessed/TextGrid/*/*.TextGrid"))
n_wav = len(glob.glob("work/Telugu/corpus/raw_data/*/*.wav"))
print(f"aligned {n_tg}/{n_wav} utterances at {hours():.2f} h")
'''),
        md("## 3 · Features"),
        code(r'''
# F0 search range from the chosen voice
import re
cfg = re.sub(r'gender: "\w+"', f'gender: "{TELUGU_GENDER}"', open("Telugu/config.yaml").read())
f0 = (60.0, 350.0) if TELUGU_GENDER == "male" else (100.0, 500.0)
cfg = re.sub(r"f0_floor: [\d.]+", f"f0_floor: {f0[0]}", cfg)
cfg = re.sub(r"f0_ceil: [\d.]+", f"f0_ceil: {f0[1]}", cfg)
open("Telugu/config.yaml", "w").write(cfg)
sh("python Telugu/preprocess.py --stage features --jobs 4")
print(open("work/Telugu/preprocessed/summary.json").read())
'''),
        code("""shutil.copy("/kaggle/working/telugu_audit.csv", f"{WORK}/Telugu/corpus/")
open(f"{WORK}/Telugu/corpus/audit_summary.txt", "w").write(AUDIT.T.to_string())"""),
        prep_save("Telugu", "Telugu/mfa"),
    ]
    return nb


def telugu_train_nb(b64):
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# FastSpeech 2 — Telugu · step 2/2: training + inference (GPU)

**Settings:** Accelerator **GPU T4 ×2** (one GPU is used), Internet **on**,
**Add Input → your private prepared-data dataset** (any name: the notebook finds `English/` and `Telugu/` inside whatever is attached).

**Run:** *Save Version → Save & Run All (Commit)*. Then the version → **Output** → download `Telugu_results.zip`.
"""),
        code("""
# ---- knobs ----
DATA_DIR      = None       # None: search everything attached under /kaggle/input (any dataset name)
TIME_BUDGET_H = 11.3
TOTAL_STEPS   = 60000      # upper bound; the time budget may stop it first
"""),
        code("CODE_B64 = '" + b64 + "'\nCODE_BUILD = '" + build_id(b64) + "'"),
        code(COMMON_SETUP),
        train_load("Telugu"),
        md("## 4 · Training"),
        code(TRAIN_HELPERS),
        code(r'''
budget = TIME_BUDGET_H - hours() - 0.5
print(f"training budget: {budget:.2f} h")
procs = {"telugu": (launch(f"python Telugu/train.py --total_step {TOTAL_STEPS} --max_hours {budget:.2f}",
                          "/kaggle/working/train_telugu.log", 0), "/kaggle/working/train_telugu.log")}
watch(procs)
print(f"training done at {hours():.2f} h")
'''),
        md("## 5 · Inference + variance-adaptor plots"),
        code(r'''
R = "Results/Telugu"
sh(f"python Telugu/inference.py --out_dir {R}")
sh(f"python Telugu/inference.py --out_dir {R} --no_repair --suffix _no_repair --only T3")
'''),
        md("## 6 · Loss plots, release checkpoint, packaging"),
        code(r'''
sh("python core/export.py --lang Telugu --runs scratch:work/Telugu/logs --ckpt_dirs scratch:Telugu/checkpoints")
for f in ["telugu_audit.csv", "audit_summary.txt"]:
    if os.path.exists(f"work/Telugu/corpus/{f}"):
        shutil.copy(f"work/Telugu/corpus/{f}", "Results/Telugu/logs/")
os.makedirs("Telugu/mfa_model", exist_ok=True)   # the Telugu MFA acoustic model trained in step 1
for z in glob.glob("work/Telugu/mfa/*.zip"):
    shutil.copy(z, "Telugu/mfa_model/")
shutil.copy("/kaggle/working/train_telugu.log", "Results/Telugu/logs/")
os.unlink(f"{PROJ}/work")
sh("cd /kaggle/working && zip -qr Telugu_results.zip project/Results/Telugu project/Telugu "
   "-x '*/last.pth.tar' '*/best.pth.tar' '*/[0-9]*.pth.tar' && ls -lh Telugu_results.zip")
print(f"ALL DONE in {hours():.2f} h")
'''),
    ]
    return nb


def build_id(b64):
    return time.strftime("%Y-%m-%d %H:%M") + " " + hashlib.sha1(b64.encode()).hexdigest()[:8]


if __name__ == "__main__":
    b64 = bundle()
    print("bundle base64 size:", len(b64))
    for name, fn in (("FS2_English_1_prepare.ipynb", english_prep_nb), ("FS2_English_2_train.ipynb", english_train_nb),
                     ("FS2_Telugu_1_prepare.ipynb", telugu_prep_nb), ("FS2_Telugu_2_train.ipynb", telugu_train_nb)):
        nb = fn(b64)
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbf.write(nb, os.path.join(ROOT, "kaggle", name))
        print("wrote", name)
