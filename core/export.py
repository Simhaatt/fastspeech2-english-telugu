"""Post-training packaging for one language:
  * loss plots -> Results/<Lang>/training_loss.png, validation_loss.png
  * fp16, model-only release checkpoint (fits under GitHub's 100 MB limit)
  * copies of the CSV logs and data-selection reports into Results/<Lang>/logs

    python core/export.py --lang English --runs scratch:work/English/logs \
                          finetune:work/English/logs/finetune
"""
import argparse
import glob
import json
import os
import shutil

import torch

from config_utils import ROOT
from plots import plot_losses, plot_train_vs_val


def slim(src, dst):
    ck = torch.load(src, map_location="cpu")
    sd = {k: (v.half() if v.is_floating_point() and "bins" not in k else v) for k, v in ck["model"].items()}
    torch.save({"model": sd, "step": ck.get("step"), "best_val": ck.get("best_val")}, dst)
    return os.path.getsize(dst) / 1e6, ck.get("step"), ck.get("best_val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="label:log_dir (first = main run)")
    ap.add_argument("--ckpt_dirs", nargs="*", default=None, help="label:ckpt_dir, same order as --runs")
    args = ap.parse_args()

    res = os.path.join(ROOT, "Results", args.lang)
    logs_out = os.path.join(res, "logs")
    os.makedirs(logs_out, exist_ok=True)
    runs = [r.split(":", 1) for r in args.runs]  # split once: Windows paths contain a drive colon
    runs = [(label, os.path.join(ROOT, path, "losses.csv")) for label, path in runs]
    runs = [(l, p) for l, p in runs if os.path.exists(p)]
    main_label, main_csv = runs[0]
    plot_losses([runs[0]], os.path.join(res, "training_loss.png"), "train",
                f"{args.lang} FastSpeech 2 - training losses ({main_label})", ema=0.0)
    plot_losses([runs[0]], os.path.join(res, "validation_loss.png"), "val",
                f"{args.lang} FastSpeech 2 - validation losses ({main_label}, full val set every 1k steps)")
    plot_train_vs_val(main_csv, os.path.join(res, "train_vs_val.png"),
                      f"{args.lang} - train vs validation ({main_label})")
    if len(runs) > 1:
        plot_losses(runs, os.path.join(res, "compare_runs_validation.png"), "val",
                    f"{args.lang} - validation losses: " + " vs ".join(l for l, _ in runs))
        plot_losses(runs, os.path.join(res, "compare_runs_training.png"), "train",
                    f"{args.lang} - training losses: " + " vs ".join(l for l, _ in runs))
        for l, p in runs[1:]:
            plot_train_vs_val(p, os.path.join(res, f"train_vs_val_{l}.png"), f"{args.lang} - train vs validation ({l})")
    for l, p in runs:
        shutil.copy(p, os.path.join(logs_out, f"losses_{l}.csv"))
        info = os.path.join(os.path.dirname(p), "run_info.json")
        if os.path.exists(info):
            shutil.copy(info, os.path.join(logs_out, f"run_info_{l}.json"))

    summary = {}
    for spec in args.ckpt_dirs or []:
        l, d = spec.split(":", 1)
        d = os.path.join(ROOT, d)
        src = os.path.join(d, "best.pth.tar")
        if os.path.exists(src):
            mb, step, best = slim(src, os.path.join(d, "best_model_fp16.pth.tar"))
            summary[l] = {"release_ckpt": os.path.relpath(os.path.join(d, "best_model_fp16.pth.tar"), ROOT),
                          "size_mb": round(mb, 1), "best_step": step, "best_val_total": best}
            # inference needs these next to the checkpoint when the preprocessed data is absent
            for f in ("stats.json", "speakers.json", "vocab.txt"):
                p = os.path.join(ROOT, "work", args.lang, "preprocessed", f)
                if os.path.exists(p):
                    shutil.copy(p, d)
    pre = os.path.join(ROOT, "work", args.lang)
    for f in ["corpus/selection_log.json", "corpus/selection.csv", "corpus/split.tsv",
              "preprocessed/summary.json", "preprocessed/stats.json", "preprocessed/speakers.json",
              "preprocessed/vocab.txt"]:
        p = os.path.join(pre, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(logs_out, os.path.basename(p)))
    with open(os.path.join(logs_out, "checkpoints.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
