"""Data selection/subsampling, MFA alignment and feature extraction.

    python Telugu/preprocess.py --stage all --hf_dir /path/to/hf_snapshot
    stages: select | align | features | all
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "core"))
from pipeline import preprocess_main  # noqa: E402

if __name__ == "__main__":
    preprocess_main(HERE)
