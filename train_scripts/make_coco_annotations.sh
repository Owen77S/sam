#!/usr/bin/env bash
# Génère les annotations COCO (annotation_<classe>.json) dans fold_X/train et
# fold_X/test de ../data/base avec make_coco_annotations.py.
set -eu

DIR="$(dirname "$0")"
BASE="$DIR/../data/base"

for dir in "$BASE"/fold_*/train "$BASE"/fold_*/test; do
  python3 "$DIR/make_coco_annotations.py" "$dir"
done
