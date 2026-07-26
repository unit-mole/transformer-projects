#!/usr/bin/env bash
set -euo pipefail

: "${HF_SPACE_REPO:?Set HF_SPACE_REPO, for example anmol-unitmole/06-multimodal-visual-question-answering-transformer}"
: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face write token}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_REPO}" "$TMP/space"
find "$TMP/space" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$ROOT/space/." "$TMP/space/"
cd "$TMP/space"
git add .
git commit -m "Deploy Project 06 static VQA Space" || true
git push
