"""Training loop for FastSpeech 2.

Based on ming024/FastSpeech2 train.py + evaluate.py. Changes:
  * one YAML per language (sections preprocess / model / train)
  * mixed precision (torch.cuda.amp) for ~2x speed on T4 GPUs
  * full validation pass every `val_step`, losses written to CSV
    (train losses are averaged over each logging window) -> loss plots
  * keeps `last.pth.tar` + `best.pth.tar` (lowest val total loss) and
    periodic checkpoints; wall-clock budget (`max_hours`) for Kaggle sessions
  * `--init_from` for fine-tuning from another checkpoint: copies every
    tensor whose shape matches, copies the overlapping rows of the phone
    embedding, and leaves new tensors (e.g. speaker table) at their init
  * `--resume` continues from last.pth.tar (model + optimiser + step)
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_utils import load_config
from dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from utils.tools import to_device

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def init_from_checkpoint(model, path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:  # the 2021 ming024 release may hold non-tensor objects; it is a trusted source
        print("weights_only load failed, loading the full pickle:", repr(e)[:200])
        sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    own = model.state_dict()
    copied, partial, skipped = [], [], []
    for k, v in sd.items():
        if k not in own:
            skipped.append(k)
            continue
        if "pitch_bins" in k or "energy_bins" in k:
            skipped.append(k)  # bins depend on the new data statistics
            continue
        if own[k].shape == v.shape:
            own[k] = v.float()
            copied.append(k)
        elif k.endswith("src_word_emb.weight"):
            n = min(own[k].shape[0], v.shape[0])
            own[k][:n] = v[:n].float()
            partial.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    print(f"init_from {path}: copied {len(copied)}, partial {partial}, skipped {skipped}")


def evaluate(model, loss_fn, loader, device, amp):
    model.eval()
    sums, n = np.zeros(6), 0
    with torch.no_grad():
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(*(batch[2:]))
                losses = loss_fn(batch, [o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out])
                bs = len(batch[0])
                sums += np.array([l.item() for l in losses]) * bs
                n += bs
    model.train()
    return sums / max(n, 1)


def save(path, model, optimizer, step, extra=None):
    extra = {k: float(v) for k, v in (extra or {}).items()}
    torch.save({"model": model.state_dict(), "optimizer": optimizer._optimizer.state_dict(),
                "step": int(step), **extra}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from last.pth.tar")
    ap.add_argument("--init_from", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--tag", default=None, help="run name (sub-folder for ckpt/logs)")
    ap.add_argument("--total_step", type=int, default=None)
    ap.add_argument("--max_hours", type=float, default=None)
    args = ap.parse_args()

    preprocess_config, model_config, train_config = configs = load_config(args.config, args.tag)
    tc = train_config
    if args.total_step:
        tc["step"]["total_step"] = args.total_step
    max_hours = args.max_hours or tc["step"].get("max_hours", 1e9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(tc.get("amp", True)) and device.type == "cuda"
    torch.manual_seed(tc.get("seed", 1234))
    np.random.seed(tc.get("seed", 1234))

    ckpt_dir, log_dir = tc["path"]["ckpt_path"], tc["path"]["log_path"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_set = Dataset("train.txt", preprocess_config, tc, sort=True, drop_last=True)
    val_set = Dataset("val.txt", preprocess_config, tc, sort=False, drop_last=False)
    bs = tc["optimizer"]["batch_size"]
    group = 4
    train_loader = DataLoader(train_set, batch_size=bs * group, shuffle=True,
                              collate_fn=train_set.collate_fn, num_workers=tc.get("num_workers", 2),
                              persistent_workers=tc.get("num_workers", 2) > 0)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, collate_fn=val_set.collate_fn)
    print(f"train {len(train_set)} utts, val {len(val_set)} utts, device {device}, amp {amp}")

    model = FastSpeech2(preprocess_config, model_config).to(device)
    step = 0
    last = os.path.join(ckpt_dir, "last.pth.tar")
    ck = None
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        step = ck["step"]
        print("resumed from step", step)
    elif args.init_from:
        init_from_checkpoint(model, args.init_from)
        model.to(device)
    optimizer = ScheduledOptim(model, tc, model_config, step)
    if ck is not None:
        optimizer.load_state_dict(ck["optimizer"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"FastSpeech2 parameters: {n_param/1e6:.2f} M")
    with open(os.path.join(log_dir, "run_info.json"), "w") as f:
        json.dump({"params": n_param, "train_utts": len(train_set), "val_utts": len(val_set),
                   "init_from": args.init_from, "amp": amp, "config": args.config}, f, indent=1)

    loss_csv = os.path.join(log_dir, "losses.csv")
    new_file = not os.path.exists(loss_csv) or step == 0
    fcsv = open(loss_csv, "w" if new_file else "a", newline="")
    writer = csv.writer(fcsv)
    if new_file:
        writer.writerow(["step", "split"] + LOSS_NAMES + ["lr", "elapsed_min"])
    best = ck.get("best_val", math.inf) if ck else math.inf

    total_step = tc["step"]["total_step"]
    log_step, val_step, save_step = tc["step"]["log_step"], tc["step"]["val_step"], tc["step"]["save_step"]
    grad_clip = tc["optimizer"]["grad_clip_thresh"]
    t0 = time.time()
    window, wn = np.zeros(6), 0
    model.train()
    done = False
    while not done:
        for batchs in train_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    out = model(*(batch[2:]))
                out = [o.float() if torch.is_tensor(o) and o.is_floating_point() else o for o in out]
                losses = loss_fn(batch, out)
                total = losses[0]
                if not torch.isfinite(total):
                    print(f"step {step}: non-finite loss, batch skipped")
                    optimizer.zero_grad()
                    continue
                scaler.scale(total).backward()
                scaler.unscale_(optimizer._optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer._update_learning_rate()
                scaler.step(optimizer._optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1
                window += np.array([l.item() for l in losses])
                wn += 1
                elapsed = (time.time() - t0) / 60
                lr = optimizer._optimizer.param_groups[0]["lr"]

                if step % log_step == 0:
                    avg = window / wn
                    writer.writerow([step, "train"] + [f"{x:.5f}" for x in avg] + [f"{lr:.3e}", f"{elapsed:.1f}"])
                    fcsv.flush()
                    print(f"step {step} | " + " ".join(f"{n} {v:.4f}" for n, v in zip(LOSS_NAMES, avg))
                          + f" | lr {lr:.2e} | {elapsed:.1f} min", flush=True)
                    window, wn = np.zeros(6), 0

                if step % val_step == 0:
                    v = evaluate(model, loss_fn, val_loader, device, amp)
                    writer.writerow([step, "val"] + [f"{x:.5f}" for x in v] + [f"{lr:.3e}", f"{elapsed:.1f}"])
                    fcsv.flush()
                    print(f"VAL step {step} | " + " ".join(f"{n} {x:.4f}" for n, x in zip(LOSS_NAMES, v)), flush=True)
                    if v[0] < best:
                        best = v[0]
                        save(os.path.join(ckpt_dir, "best.pth.tar"), model, optimizer, step, {"best_val": best})
                        print(f"  new best val {best:.4f} at step {step}")
                    save(last, model, optimizer, step, {"best_val": best})

                if step % save_step == 0:
                    save(os.path.join(ckpt_dir, f"{step}.pth.tar"), model, optimizer, step, {"best_val": best})

                if step >= total_step or elapsed / 60 > max_hours:
                    done = True
                    break
            if done:
                break
    v = evaluate(model, loss_fn, val_loader, device, amp)
    writer.writerow([step, "val"] + [f"{x:.5f}" for x in v] + [f"{lr:.3e}", f"{(time.time()-t0)/60:.1f}"])
    if v[0] < best:
        best = v[0]
        save(os.path.join(ckpt_dir, "best.pth.tar"), model, optimizer, step, {"best_val": best})
    save(last, model, optimizer, step, {"best_val": best})
    fcsv.close()
    print(f"finished at step {step}, best val {best:.4f}")


if __name__ == "__main__":
    main()
