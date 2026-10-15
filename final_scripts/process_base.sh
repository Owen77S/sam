#!/usr/bin/env bash
# Prépare ../data/base : supprime les dossiers mask/, fusionne fold_X/train et
# fold_X/test en fold_X/all, puis génère les annotations RLE + bbox
# (segmentation_bbox/*.json) avec masks_to_rle.py.
set -eu

SCRIPTS="$(dirname "$0")/python_scripts"
BASE="$(dirname "$0")/../data/base"

find "$BASE" -mindepth 2 -type d -name mask -prune -exec rm -rf {} +

# mv -n n'écrase rien : en cas de nom en double, le fichier reste dans train/
# ou test/, rmdir échoue et le script s'arrête.
for fold in "$BASE"/fold_*/; do
  for split in train test; do
    [ -d "$fold$split" ] || continue
    for sub in "$fold$split"/*/; do
      name="$(basename "$sub")"
      mkdir -p "${fold}all/$name"
      mv -n "$sub"* "${fold}all/$name/"
      rmdir "$sub"
    done
    rmdir "$fold$split"
  done
done

for fold in "$BASE"/fold_*/; do
  python3 "$SCRIPTS/masks_to_rle.py" "${fold}all"
done
