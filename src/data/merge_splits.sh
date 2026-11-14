#!/usr/bin/env bash
# Fusionne fold_N/train et fold_N/test en fold_N/all, pour l'entraînement final
# sur toutes les images, puis génère les annotations COCO de all/.
# Les fichiers sont des liens durs : rien n'est copié, et train/ et test/ restent
# intacts (l'éval par fold continue de fonctionner).
#
# Usage : merge_splits.sh <N>
# Prérequis : prepare.sh a été lancé. Relançable : all/ est reconstruit.
# Un nom de fichier présent à la fois dans train/ et test/ fait échouer le script.
set -eu

[ $# -eq 1 ] || { echo "Usage : $0 <numéro de fold>" >&2; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${DATA_BASE:-$HERE/../../data/base}"
fold="$BASE/fold_$1"

for split in train test; do
  [ -d "$fold/$split" ] || { echo "$fold/$split est introuvable" >&2; exit 1; }
done

# On construit dans all.tmp puis on renomme : all/ n'existe que complet.
rm -rf "$fold/all.tmp"
for split in train test; do
  for sub in "$fold/$split"/*/; do
    name="$(basename "$sub")"
    mkdir -p "$fold/all.tmp/$name"
    # ln (et non cp -al, qui écraserait en silence) refuse un nom déjà présent
    find "$sub" -maxdepth 1 -type f -exec ln -t "$fold/all.tmp/$name" {} + \
      || { echo "ERREUR : liaison impossible dans $name/ (nom en double entre train/ et test/ ?)" >&2; exit 1; }
  done
done

python3 "$HERE/make_coco_annotations.py" "$fold/all.tmp"

rm -rf "$fold/all"
mv "$fold/all.tmp" "$fold/all"
echo "$fold/all : $(find "$fold/all/img" -type f | wc -l) images"
