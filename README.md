# FastSpeech 2: Indian-accented English (Svarah) and Telugu (IndicTTS)

Two FastSpeech 2 text-to-speech models, trained separately from scratch:

| | English | Telugu |
|---|---|---|
| Data | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), 99 speakers | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu), one male voice |
| Train / validation (selected) | 3,465 / 742 clips, 4.99 h / 1.07 h (no shared clips) | 2,629 / 526 clips, 5.00 h / 1.00 h (no shared sentences) |
| Train / validation (after silence trimming) | 4.64 h / 1.00 h | 4.14 h / 0.82 h |
| Model | multi-speaker FastSpeech 2, 35.2 M parameters | single-speaker FastSpeech 2, same size |
| Aligner | MFA 3.4.3, pretrained `english_us_arpa` | MFA 3.4.3, Telugu model trained on this corpus |
| Vocoder | pretrained universal HiFi-GAN (not trained) | same |

The code is based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2). All changes,
including the Telugu support, are listed in [Report/Technical_Report.pdf](Report/Technical_Report.pdf)
(section 11) and in the docstrings of the changed files. Section 2 of the report explains how the
system works for readers new to speech synthesis.

## Results

| | English | Telugu |
|---|---|---|
| Steps trained / best step | 60,000 / 14,000 | 60,000 / 12,000 |
| Best validation loss (total) | 2.354 | 2.048 |
| ASR character error, real speech through the vocoder | 6.4 % | 3.4 % |
| ASR character error, synthesised (same 30 sentences) | 10.0 % | 4.7 % |
| Predicted vs real log-duration correlation | 0.61 | 0.85 |

Telugu (one speaker, read speech) comes out close to real speech. English (99 speakers, about 3 minutes
each, partly spontaneous) is understandable but sounds mumbled: its spectrograms change only 63 % as fast
between frames as real speech. The fine-tuned-from-LJSpeech ablation reached 2.369, no better than training
from scratch. Giving the models the real pitch and energy instead of their predictions brings the
spectrograms 9 % (English) and 20 % (Telugu) closer to real speech; real durations change almost nothing.
With all prosody taken from the recording, English still stays further from real speech than Telugu, so the
mumbling comes from the decoder, not from the timing. Details and plots: [the report](Report/Technical_Report.pdf).

## Repository layout

```
English/  train.py  preprocess.py  inference.py  config.yaml  checkpoints/
Telugu/   train.py  preprocess.py  inference.py  config.yaml  checkpoints/  mfa_model/
core/            model, training, inference, text front-ends, data preparation (ming024 code + changes)
data/            test_sentences.tsv (the 10 test sentences, text as given)
kaggle/          notebook builder + the notebooks used for training on Kaggle
tools/           evaluation: speech-recognition check, predicted-vs-real prosody, ablation, sanity checks
Results/<Lang>/  training_loss.png  validation_loss.png  train_vs_val.png  test_generated_audios/
                 (English also: compare_runs_{training,validation}.png, train_vs_val_finetune.png)
                 variance_adaptor_plots/{duration,pitch,energy,overview,json}/
                 ablation_samples/  (3 validation clips with real vs predicted prosody)
                 logs/  (loss CSVs, training logs, data selection, split, asr_check.json, variance_eval*.txt,
                        ablation.json, sanity_checks.txt)
Report/          Technical_Report.pdf (+ LaTeX source)
requirements.txt
```

`Telugu/mfa_model/telugu_acoustic.zip` is the Montreal Forced Aligner acoustic model trained on the selected
Telugu corpus (MFA has no pretrained Telugu model).

`checkpoints/best_model_fp16.pth.tar` is the checkpoint with the lowest validation loss, stored in fp16
(about 70 MB, under GitHub's 100 MB limit). It is converted back to fp32 when loaded. The same folder holds
`stats.json` (pitch/energy normalisation), `speakers.json` and `vocab.txt`, which inference needs.

`Results/English/` also has `other_speakers/en_s034/` (the same sentences in a female voice) and
`finetune_ablation/` (the model fine-tuned from the LJSpeech checkpoint), with their own audio and plots.

## Setup

Python 3.10 to 3.13.

```bash
pip install -r requirements.txt
python -c "import nltk; [nltk.download(r) for r in ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict']]"
# HiFi-GAN universal vocoder weights from the reference repository
curl -L -o core/hifigan/generator_universal.pth.tar.zip \
  https://github.com/ming024/FastSpeech2/raw/master/hifigan/generator_universal.pth.tar.zip
```

Data preparation also needs Montreal Forced Aligner 3.4.3. `bash core/prep/mfa.sh setup` installs it in its
own micromamba environment (`/opt/mfa_env`) and creates a user `mfa`, because MFA's PostgreSQL server refuses
to run as root. Run it as root (or with sudo).

## Inference with the released checkpoints

```bash
# the 5 English test sentences -> Results/English/ (speaker en_s073, male; the list is in speakers.json)
python English/inference.py --ckpt English/checkpoints/best_model_fp16.pth.tar --speaker en_s073
# the fine-tuned-from-LJSpeech ablation
python English/inference.py --config English/config_finetune.yaml \
  --ckpt English/checkpoints/finetune/best_model_fp16.pth.tar --speaker en_s073 --out_dir Results/English/finetune_ablation
# the same sentences in a female voice
python English/inference.py --speaker en_s034 --out_dir Results/English/other_speakers/en_s034
# E3 without the split-word repair (E3_no_repair.wav)
python English/inference.py --speaker en_s073 --no_repair --suffix _no_repair --only E3
# the 5 Telugu test sentences -> Results/Telugu/
python Telugu/inference.py --ckpt Telugu/checkpoints/best_model_fp16.pth.tar
python Telugu/inference.py --no_repair --suffix _no_repair --only T3
# any sentence
python Telugu/inference.py --ckpt Telugu/checkpoints/best_model_fp16.pth.tar --text "..." --out_dir out/
```

Options: `--only E3` (subset of test ids), `--no_repair --suffix _no_repair` (switch off the split-word repair),
`--pitch_control`, `--energy_control`, `--duration_control`. Each sentence gives a WAV, a JSON with the
predicted durations (frames and ms), F0 (Hz) and energy per phone, and duration, pitch, energy and overview
plots. A GPU is used when available; CPU works.

## Training from scratch

```bash
# 1. select + subsample the data, align with MFA, extract features   (run as root for MFA)
python English/preprocess.py --stage all --hf_dir /path/to/svarah_parquet_folder
python Telugu/preprocess.py  --stage all --hf_dir /path/to/indictts_telugu_parquet_folder --gender male
# 2. train (stops at total_step or after max_hours; keeps best.pth.tar and last.pth.tar)
python English/train.py
python Telugu/train.py
python English/train.py --resume                          # continue from last.pth.tar
# 3. loss plots + fp16 release checkpoints
python core/export.py --lang English --runs scratch:work/English/logs finetune:work/English/logs/finetune   --ckpt_dirs scratch:English/checkpoints finetune:English/checkpoints/finetune
python core/export.py --lang Telugu  --runs scratch:work/Telugu/logs  --ckpt_dirs scratch:Telugu/checkpoints
```

All settings (data selection, features, model, optimiser, steps) are in `<Lang>/config.yaml`. Paths in
it are relative to the repository root; intermediate files go to `work/<Lang>/`.

English fine-tuning ablation from the public LJSpeech FastSpeech 2 checkpoint
(`LJSpeech_900000.pth.tar` from the ming024 release):

```bash
python English/train.py --config English/config_finetune.yaml --tag finetune \
  --init_from /path/to/LJSpeech_900000.pth.tar --total_step 30000
```
`config_finetune.yaml` is `config.yaml` with `lr_scale: 0.5` and `warm_up_step: 1000`.

## Evaluation

The scripts in `tools/` produce the numbers in sections 8.1, 9 and 12 of the report. They need the
prepared features (`work/<Lang>/preprocessed/`) and, for the speech-recognition check, `pip install
transformers jiwer` (the Whisper models download from Hugging Face on first use). Each script's header
lists its arguments; for example:

```bash
# all clips line up, checkpoint reproduces the logged validation loss, speaker table works, LR schedule
python tools/sanity_checks.py . English English/config.yaml English/checkpoints/best_model_fp16.pth.tar \
  work/English/preprocessed 2.3540
# predicted vs real pitch/energy/duration and mel smoothing on 80 validation clips
python tools/variance_eval.py . Telugu Telugu/checkpoints/best_model_fp16.pth.tar Telugu/config.yaml \
  work/Telugu/preprocessed 80
# Whisper on real (vocoded) vs synthesised validation clips and on the test sentences
python tools/asr_eval.py . te Telugu/checkpoints/best_model_fp16.pth.tar Telugu/config.yaml \
  work/Telugu/preprocessed vasista22/whisper-telugu-base Results/Telugu/test_generated_audios \
  tools/asr_refs_te.tsv asr_te.json 30
python tools/asr_rescore.py asr_te.json      # script-safe scoring for Telugu
# real vs predicted duration / pitch+energy: Whisper CER, MCD and F0 error on the same 30 clips
python tools/ablation_eval.py . te Telugu/checkpoints/best_model_fp16.pth.tar Telugu/config.yaml   work/Telugu/preprocessed vasista22/whisper-telugu-base ablation_te.json 30 ablation_samples_te
python tools/ablation_ci.py ablation_te.json  # paired bootstrap intervals for the differences
```

## How the models were actually trained (Kaggle)

1. Data preparation ran on CPU before Kaggle, with the same `preprocess.py` code. Telugu ran natively on
   Windows with `kaggle/telugu_prepare_driver.py` (MFA 3.4.3 from conda-forge). On Linux `core/pipeline.py`
   calls MFA through `core/prep/mfa.sh`; on Windows it calls the `mfa` executable of the conda environment
   named by the `MFA_ENV` environment variable, which must be set. The prepared features of both languages
   were uploaded as one private Kaggle dataset. `kaggle/FS2_*_1_prepare.ipynb` do the same preparation in a
   Kaggle CPU session; the released models did not use them.
2. `kaggle/FS2_English_2_train.ipynb` and `kaggle/FS2_Telugu_2_train.ipynb` (GPU T4 x2) train, run
   inference on the test sentences, plot, and package the results. English trains the scratch model and the
   LJSpeech fine-tune at the same time, one per GPU.
3. `python kaggle/build_notebooks.py` rebuilds the notebooks; the project code is embedded in them.

## Credits

FastSpeech 2 implementation, HiFi-GAN weights and the LibriSpeech lexicon in `core/lexicon/`:
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT). Data: Svarah (AI4Bharat), IndicTTS
(IIT Madras, via SPRINGLab on Hugging Face). Aligner: Montreal Forced Aligner. Speech-recognition check:
`openai/whisper-base.en` and `vasista22/whisper-telugu-base` (Hugging Face).
