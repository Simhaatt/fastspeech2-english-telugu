"""One config.yaml per language -> the (preprocess, model, train) dicts that
the reference code expects. Relative paths are resolved against the project
root, so scripts work from any working directory."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(p):
    if p is None or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(ROOT, p))


def load_config(path, tag=None):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre, model, train = cfg["preprocess"], cfg["model"], cfg["train"]
    for sec in (pre["path"], train["path"]):
        for k, v in list(sec.items()):
            sec[k] = _abs(v)
    if tag:
        for k in ("ckpt_path", "log_path", "result_path"):
            if k in train["path"]:
                train["path"][k] = os.path.join(train["path"][k], tag)
    pre["max_seq_len_frames"] = model["max_seq_len"]
    model["vocoder"]["path"] = _abs(model["vocoder"].get("path", "core/hifigan"))
    return pre, model, train
