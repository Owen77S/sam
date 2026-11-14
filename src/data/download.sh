#!/usr/bin/env bash
# Télécharge fold_N.zip (Zenodo, record 14478210) dans data/base/, vérifie leur md5
# et les décompresse en fold_N/. Relançable : reprend les téléchargements
# interrompus et saute les fichiers et dossiers déjà valides.
#
# Usage : download.sh [N…]   (par défaut : les 5 folds ; ex. « download.sh 1 3 »)
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${DATA_BASE:-$ROOT/data/base}"
URL="https://zenodo.org/api/records/14478210/files"
FOLDS="${*:-1 2 3 4 5}"

# md5 publiés par Zenodo (https://zenodo.org/api/records/14478210)
declare -A MD5=(
  [fold_1.zip]=3bef0c0229de65a2cb618b18f97ac949
  [fold_2.zip]=8aa65b0d2008c287ca80fc4b1a219722
  [fold_3.zip]=c53045ee5dd0c3c243d95253bfd4d0ef
  [fold_4.zip]=af7dd83a88af4f9034db0d6b11f4a4ff
  [fold_5.zip]=8c998cdddb40861b788dd84986c08252
)

for i in $FOLDS; do
  if [ -z "${MD5[fold_$i.zip]:-}" ]; then
    echo "fold inconnu : $i (valeurs possibles : 1 à 5)" >&2
    exit 1
  fi
done

mkdir -p "$DEST"
cd "$DEST"

status=0
for i in $FOLDS; do
  f="fold_$i.zip"
  sum="${MD5[$f]}  $f"
  if md5sum -c --status <<<"$sum" 2>/dev/null; then
    echo "$f: déjà présent, md5 OK"
    continue
  fi
  curl -fL -C - --retry 5 -o "$f" "$URL/$f/content"
  md5sum -c <<<"$sum" || status=1
done

# Décompression : chaque zip contient fold_X/{train,test}/. On extrait dans un
# dossier temporaire puis on déplace, pour que fold_X/ n'existe que complet.
for i in $FOLDS; do
  d="fold_$i"
  if [ -d "$d" ]; then
    echo "$d/: déjà décompressé"
    continue
  fi
  md5sum -c --status <<<"${MD5[$d.zip]}  $d.zip" 2>/dev/null || continue
  echo "$d.zip: décompression"
  rm -rf "$d.tmp"
  if python3 -m zipfile -e "$d.zip" "$d.tmp" && [ -d "$d.tmp/$d" ]; then
    mv "$d.tmp/$d" "$d" && rm -rf "$d.tmp"
  else
    echo "$d.zip: échec de la décompression" >&2
    status=1
  fi
done
exit $status
