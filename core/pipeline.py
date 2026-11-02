"""Shared entry points used by <Language>/{preprocess,train,inference}.py."""
import argparse
import os
import subprocess
import sys

import yaml

CORE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CORE)
if CORE not in sys.path:
    sys.path.insert(0, CORE)


def _run(cmd, env=None):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _mfa(args, jobs):
    """Montreal Forced Aligner 3.x. Linux (Kaggle): core/prep/mfa.sh, which runs MFA as a
    non-root user. Windows: the mfa executable of the conda env in $MFA_ENV, called directly
    (no root problem there); the env's DLL/tool folders are put on PATH first."""
    cmd, corpus, dictionary, model, out = args
    if os.name != "nt":
        _run(["bash", os.path.join(CORE, "prep", "mfa.sh"), cmd, corpus, dictionary, model, out, str(jobs)])
        return
    prefix = os.environ["MFA_ENV"]
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([prefix, os.path.join(prefix, "Library", "bin"), os.path.join(prefix, "Scripts"),
                                   os.path.join(prefix, "Library", "usr", "bin"), env.get("PATH", "")])
    mfa = os.path.join(prefix, "Scripts", "mfa.exe")
    os.makedirs(out, exist_ok=True)
    common = ["--clean", "--overwrite", "-j", str(jobs), "--output_format", "long_textgrid"]
    if cmd == "train":
        os.makedirs(os.path.dirname(model), exist_ok=True)
        _run([mfa, "train", *common, "--single_speaker", "--output_directory", out, corpus, dictionary, model], env)
    else:
        if not model.endswith(".zip"):
            _run([mfa, "model", "download", "acoustic", model], env)
        beams = os.path.join(os.path.dirname(out), "mfa_align.yaml")
        with open(beams, "w") as f:
            f.write("beam: 100\nretry_beam: 400\n")  # accented, partly spontaneous speech
        _run([mfa, "align", *common, "-c", beams, corpus, dictionary, model, out], env)


def preprocess_main(lang_dir):
    from config_utils import load_config

    ap = argparse.ArgumentParser(description="dataset selection -> MFA alignment -> features")
    ap.add_argument("--config", default=os.path.join(lang_dir, "config.yaml"))
    ap.add_argument("--stage", default="all", choices=["select", "align", "features", "all"])
    ap.add_argument("--hf_dir", default=None, help="local snapshot of the HF dataset (select stage)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--gender", default=None, help="Telugu: override selection.gender")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)["preprocess"]
    pre, _, _ = load_config(args.config)
    lang = pre["preprocessing"]["text"]["language"]
    corpus = pre["path"]["corpus_path"]
    sel = raw_cfg["selection"]
    stages = ["select", "align", "features"] if args.stage == "all" else [args.stage]

    if "select" in stages:
        assert args.hf_dir, "--hf_dir is required for the select stage"
        if lang == "en":
            _run([sys.executable, os.path.join(CORE, "prep", "prepare_svarah_english.py"),
                  "--hf_dir", args.hf_dir, "--out", corpus, "--lexicon", pre["path"]["lexicon_path"],
                  "--train_h", str(sel["train_hours"]), "--val_h", str(sel["val_hours"]),
                  "--min_dur", str(sel["min_dur"]), "--max_dur", str(sel["max_dur"]),
                  "--min_snr", str(sel["min_snr_db"]), "--max_edge_sil", str(sel["max_edge_silence_s"]),
                  "--min_spk_min", str(sel["min_speaker_minutes"])])
        else:
            _run([sys.executable, os.path.join(CORE, "prep", "prepare_indictts_telugu.py"),
                  "--hf_dir", args.hf_dir, "--out", corpus, "--gender", args.gender or sel["gender"],
                  "--train_h", str(sel["train_hours"]), "--val_h", str(sel["val_hours"]),
                  "--min_dur", str(sel["min_dur"]), "--max_dur", str(sel["max_dur"]),
                  "--min_snr", str(sel["min_snr_db"])])

    if "align" in stages:
        tg = os.path.join(pre["path"]["preprocessed_path"], "TextGrid")
        if lang == "en":
            _mfa(["align", pre["path"]["raw_path"], os.path.join(ROOT, raw_cfg["mfa"]["dictionary"]),
                  raw_cfg["mfa"]["acoustic_model"], tg], args.jobs)
        else:
            _mfa(["train", pre["path"]["raw_path"], pre["path"]["lexicon_path"],
                  os.path.join(ROOT, raw_cfg["mfa"]["model_out"]), tg], args.jobs)

    if "features" in stages:
        from preprocessor.preprocessor import Preprocessor

        pre["preprocessing"]["n_jobs"] = args.jobs
        Preprocessor(pre).build_from_path()


def train_main(lang_dir):
    sys.argv[1:1] = [] if "--config" in sys.argv else ["--config", os.path.join(lang_dir, "config.yaml")]
    import train_core

    train_core.main()


def inference_main(lang_dir):
    lang = os.path.basename(lang_dir)
    if "--config" not in sys.argv:
        sys.argv += ["--config", os.path.join(lang_dir, "config.yaml")]
    if "--ckpt" not in sys.argv:
        ck = os.path.join(lang_dir, "checkpoints", "best.pth.tar")
        if not os.path.exists(ck):  # a fresh clone has only the released fp16 checkpoint
            ck = os.path.join(lang_dir, "checkpoints", "best_model_fp16.pth.tar")
        sys.argv += ["--ckpt", ck]
    if "--out_dir" not in sys.argv:
        sys.argv += ["--out_dir", os.path.join(ROOT, "Results", lang)]
    import infer_core

    infer_core.main()
