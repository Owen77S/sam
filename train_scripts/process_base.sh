#!/usr/bin/env bash
# Prépare ../data/base : supprime les dossiers mask/ puis génère les annotations
# RLE + bbox (segmentation_bbox/*.json) de chaque sous-dossier avec masks_to_rle.py.
set -eu

SCRIPTS="$(dirname "$0")/python_scripts"
BASE="$(dirname "$0")/../data/base"

find "$BASE" -mindepth 2 -type d -name mask -prune -exec rm -rf {} +

for dir in "$BASE"/*/; do
  python3 "$SCRIPTS/masks_to_rle.py" "$dir"
done
