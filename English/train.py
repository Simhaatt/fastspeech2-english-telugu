"""Train FastSpeech 2.

    python English/train.py                      # from scratch
    python English/train.py --resume             # continue from checkpoints/last.pth.tar
    python English/train.py --init_from X.pth.tar --tag finetune   # fine-tune
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from pipeline import train_main  # noqa: E402

if __name__ == "__main__":
    train_main(HERE)
