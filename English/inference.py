"""Synthesize the test sentences (or --text "...") and plot the
variance-adaptor predictions.

    python English/inference.py                         # test set -> Results/English
    python English/inference.py --text "..." --out_dir out/
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from pipeline import inference_main  # noqa: E402

if __name__ == "__main__":
    inference_main(HERE)
