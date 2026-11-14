#!/usr/bin/env bash
# Prépare data/base à partir des folds décompressés :
#   1. supprime les dossiers mask/ (seuls les masques couleur servent)
#   2. génère les annotations RLE + bbox (segmentation_bbox/*.json) avec masks_to_rle.py
#   3. génère les annotations COCO (annotation_<classe>.json) de fold_X/train et
#      fold_X/test avec make_coco_annotations.py
# Relançable : les sorties sont régénérées à l'identique.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${DATA_BASE:-$HERE/../../data/base}"

find "$BASE" -mindepth 2 -type d -name mask -prune -exec rm -rf {} +

for dir in "$BASE"/*/; do
  python3 "$HERE/masks_to_rle.py" "$dir"
done

for dir in "$BASE"/fold_*/train "$BASE"/fold_*/test; do
  python3 "$HERE/make_coco_annotations.py" "$dir"
done
