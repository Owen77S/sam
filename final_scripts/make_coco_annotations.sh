#!/usr/bin/env bash
# Génère les annotations COCO (annotation_<classe>.json) dans fold_X/all de
# ../data/base avec python_scripts/make_coco_annotations.py.
set -eu

DIR="$(dirname "$0")"
BASE="$DIR/../data/base"

for dir in "$BASE"/fold_*/all; do
  python3 "$DIR/python_scripts/make_coco_annotations.py" "$dir"
done
