#!/usr/bin/env bash
# Montreal Forced Aligner (MFA 3.x) helper for Kaggle / Colab / any Linux box.
#   bash core/prep/mfa.sh setup
#   bash core/prep/mfa.sh align  <corpus_dir> <dictionary> <acoustic_model> <out_textgrid_dir> [jobs]
#   bash core/prep/mfa.sh train  <corpus_dir> <dictionary> <out_model.zip>  <out_textgrid_dir> [jobs]
# MFA 3 runs a PostgreSQL server, which refuses to start as root (the default
# user on Kaggle), so every MFA call is made as an unprivileged user "mfa".
set -euo pipefail
MFA_ENV=${MFA_ENV:-/opt/mfa_env}
MM=${MM:-/opt/micromamba/bin/micromamba}
cmd=$1; shift

run_as_mfa() {
  if [ "$(id -u)" = "0" ]; then
    su mfa -s /bin/bash -c "export PATH=$MFA_ENV/bin:\$PATH MFA_ROOT_DIR=/home/mfa/Documents/MFA; $*"
  else
    PATH=$MFA_ENV/bin:$PATH bash -c "$*"
  fi
}

case "$cmd" in
  setup)
    if [ ! -x "$MFA_ENV/bin/mfa" ]; then
      mkdir -p /opt/micromamba/bin
      curl -sSL -o $MM https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
      chmod +x $MM
      # 3.4.3 is the version this pipeline was tested with
      $MM create -y -q -p "$MFA_ENV" -c conda-forge "montreal-forced-aligner=3.4.3" python=3.11 >/dev/null
    fi
    id -u mfa >/dev/null 2>&1 || useradd -m -s /bin/bash mfa
    run_as_mfa "mfa version"
    ;;
  align)
    corpus=$1; dict=$2; am=$3; out=$4; jobs=${5:-4}
    chmod -R a+rwX "$(dirname "$corpus")" "$(dirname "$out")" 2>/dev/null || true
    mkdir -p "$out"; chmod a+rwX "$out"
    if [[ "$am" != *.zip ]]; then
      if ! run_as_mfa "mfa model download acoustic $am"; then
        # fallback: fetch the release asset directly (GitHub API rate limits)
        url="https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-${am}-v3.0.0/${am}.zip"
        echo "model download via API failed, trying $url"
        curl -fsSL -o /tmp/${am}.zip "$url" && chmod a+r /tmp/${am}.zip && am=/tmp/${am}.zip
      fi
    fi
    # wider beams than the default (10/40): Svarah is accented, spontaneous speech
    printf "beam: 100\nretry_beam: 400\n" > /tmp/mfa_align.yaml; chmod a+r /tmp/mfa_align.yaml
    run_as_mfa "cd /tmp && mfa align --clean --overwrite -j $jobs -c /tmp/mfa_align.yaml --output_format long_textgrid '$corpus' '$dict' '$am' '$out'"
    ;;
  train)
    corpus=$1; dict=$2; model=$3; out=$4; jobs=${5:-4}
    chmod -R a+rwX "$(dirname "$corpus")" 2>/dev/null || true
    mkdir -p "$out" "$(dirname "$model")"; chmod a+rwX "$out" "$(dirname "$model")"
    # --single_speaker: one voice, so let MFA split it across all jobs
    run_as_mfa "cd /tmp && mfa train --clean --overwrite --single_speaker -j $jobs --output_format long_textgrid --output_directory '$out' '$corpus' '$dict' '$model'"
    ;;
  *) echo "unknown command $cmd"; exit 1;;
esac
