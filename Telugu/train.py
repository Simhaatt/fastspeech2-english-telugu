"""Train FastSpeech 2.

    python Telugu/train.py                      # from scratch
    python Telugu/train.py --resume             # continue from checkpoints/last.pth.tar
    python Telugu/train.py --init_from X.pth.tar --tag finetune   # fine-tune
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from pipeline import train_main  # noqa: E402

if __name__ == "__main__":
    train_main(HERE)
