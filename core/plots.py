"""Plots: variance-adaptor predictions per test sentence and loss curves."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated categorical slots (light mode) + text inks
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "lines.linewidth": 2,
})


def _word_label(word, lang, phones=None):
    """Matplotlib cannot shape Telugu script (conjuncts and vowel signs come
    out in the wrong order), so Telugu words are labelled with their
    romanised phone string; the JSON next to each plot keeps the original."""
    if lang == "te":
        return "".join(p for p in (phones or []) if p != "sp") or word, None
    return word, None


def _shade_words(ax, pred, starts, ends, lang, y_text=None, shade=True):
    for k, (word, a, b) in enumerate(pred["word_spans"]):
        if a >= len(starts) or b <= a:
            continue
        x0, x1 = starts[a], ends[b - 1]
        if shade and k % 2 == 0:
            ax.axvspan(x0, x1, color="#f1f0ec", zorder=0, lw=0)
        lab, fp = _word_label(word, lang, pred["phones"][a:b])
        yt = ax.get_ylim()[1] if y_text is None else y_text
        ax.text((x0 + x1) / 2, yt, lab, ha="center", va="bottom", fontsize=7.5, color=INK2,
                fontproperties=fp, rotation=0 if (x1 - x0) > 0.25 else 30, clip_on=False)


def plot_variance(pred, mel, root, name, lang, sr, hop):
    ph = pred["phones"]
    dur_f = np.array(pred["duration_frames"])
    dur_s = dur_f * hop / sr
    starts = np.concatenate([[0], np.cumsum(dur_s)[:-1]])
    ends = starts + dur_s
    mids = (starts + ends) / 2
    title = f'{name}  ·  {pred["total_sec"]:.2f} s  ·  {len(ph)} phones'

    # ---- duration: one bar per phone, grouped by word
    fig, ax = plt.subplots(figsize=(max(8, 0.22 * len(ph)), 3.6))
    x = np.arange(len(ph))
    ax.bar(x, np.array(pred["duration_ms"]), color=C_BLUE, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ph, rotation=90, fontsize=7)
    ax.set_ylabel("predicted duration (ms)")
    ax.set_xlim(-0.6, len(ph) - 0.4)
    ax.set_title("Duration predictor  ·  " + title, loc="left", pad=22)
    idx_s = np.arange(len(ph)) - 0.45
    idx_e = np.arange(len(ph)) + 0.45
    _shade_words(ax, pred, idx_s, idx_e, lang)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "duration", name + ".png"), dpi=150)
    plt.close(fig)

    # ---- pitch: phoneme-level F0 (Hz) as steps over time
    def step_plot(values, ylabel, color, sub, ref=None):
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * pred["total_sec"] + 2), 3.4))
        v = np.array(values)
        keep = dur_f > 0
        ax.hlines(v[keep], starts[keep], ends[keep], color=color, lw=2.2)
        ax.plot(mids[keep], v[keep], "o", ms=3.2, color=color)
        if ref is not None:
            ax.axhline(ref, ls="--", lw=1, color=INK2)
            # stats.json mean over all training speakers, not the chosen speaker's own mean
            ax.text(pred["total_sec"], ref, " training-set mean", va="center", fontsize=7, color=INK2)
        ax.set_xlim(0, pred["total_sec"])
        lo, hi = v[keep].min(), v[keep].max()
        pad = 0.12 * (hi - lo + 1e-6)
        ax.set_ylim(lo - pad, hi + 2.2 * pad)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{sub} predictor  ·  " + title, loc="left", pad=22)
        _shade_words(ax, pred, starts, ends, lang)
        for i in np.where(keep)[0]:
            if ph[i] != "sp":
                ax.text(mids[i], lo - pad * 0.9, ph[i], ha="center", va="bottom", fontsize=5.5,
                        color=INK2, rotation=90)
        fig.tight_layout()
        return fig

    p_mean = pred.get("pitch_mean_hz")
    fig = step_plot(pred["pitch_hz"], "predicted F0 (Hz)", C_ORANGE, "Pitch", p_mean)
    fig.savefig(os.path.join(root, "pitch", name + ".png"), dpi=150)
    plt.close(fig)
    fig = step_plot(pred["energy"], "predicted energy (frame L2 norm)", C_AQUA, "Energy")
    fig.savefig(os.path.join(root, "energy", name + ".png"), dpi=150)
    plt.close(fig)

    # ---- overview: mel + F0 + energy on a shared time axis
    t = np.arange(mel.shape[1]) * hop / sr
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1, 1]})
    axs[0].imshow(mel, origin="lower", aspect="auto", cmap="magma",
                  extent=[0, t[-1] if len(t) else 1, 0, mel.shape[0]])
    axs[0].grid(False)
    axs[0].set_ylabel("mel bin")
    axs[0].set_title(f"Generated mel-spectrogram and variance-adaptor outputs  ·  {title}", loc="left", pad=20)
    # labels only: the shading bands are drawn above the image and would hide every other word
    _shade_words(axs[0], pred, starts, ends, lang, y_text=mel.shape[0], shade=False)
    f_frame = np.repeat(pred["pitch_hz"], dur_f)
    e_frame = np.repeat(pred["energy"], dur_f)
    tt = np.arange(len(f_frame)) * hop / sr
    axs[1].plot(tt, f_frame, color=C_ORANGE)
    axs[1].set_ylabel("F0 (Hz)")
    axs[2].plot(tt, e_frame, color=C_AQUA)
    axs[2].set_ylabel("energy")
    axs[2].set_xlabel("time (s)")
    for b in ends:
        for a in axs[1:]:
            a.axvline(b, color=GRID, lw=0.5, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "overview", name + ".png"), dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ losses
def read_losses(path):
    rows = {"train": [], "val": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["split"]].append({k: float(v) for k, v in r.items() if k != "split"})
    return rows


NAMES = [("total", "Total"), ("mel", "Mel L1 (before PostNet)"), ("mel_postnet", "Mel L1 (after PostNet)"),
         ("duration", "Log-duration MSE"), ("pitch", "Pitch MSE (normalised)"), ("energy", "Energy MSE (normalised)")]


def plot_losses(runs, out_png, split, title, ema=0.0):
    """runs: list of (label, csv_path). One small multiple per loss term."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 6.2))
    colors = [C_BLUE, C_ORANGE, C_VIOLET]
    for ri, (label, path) in enumerate(runs):
        r = read_losses(path)[split]
        if not r:
            continue
        steps = np.array([x["step"] for x in r])
        for ax, (k, nm) in zip(axs.flat, NAMES):
            y = np.array([x[k] for x in r])
            if ema and len(y) > 3:
                s, out = y[0], []
                for v in y:
                    s = ema * s + (1 - ema) * v
                    out.append(s)
                y = np.array(out)
            ax.plot(steps, y, color=colors[ri % 3], label=label, lw=1.8 if split == "train" else 2,
                    marker=None if split == "train" else "o", ms=3)
            ax.set_title(nm, loc="left")
            ax.set_xlabel("step")
    for ax in axs.flat:
        ax.set_yscale("log")
    if len(runs) > 1:
        axs[0, 0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_train_vs_val(path, out_png, title):
    r = read_losses(path)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k, nm in zip(axs, ["total", "mel_postnet", "duration"], ["Total", "Mel L1 (PostNet)", "Log-duration MSE"]):
        for split, c in (("train", C_BLUE), ("val", C_ORANGE)):
            rr = r[split]
            if rr:
                ax.plot([x["step"] for x in rr], [x[k] for x in rr], color=c, label=split,
                        marker="o" if split == "val" else None, ms=3)
        ax.set_title(nm, loc="left")
        ax.set_xlabel("step")
        ax.set_yscale("log")
    axs[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
