#!/usr/bin/env bash
# Télécharge fold_1.zip (Zenodo, record 14478210) dans ../data/base/, vérifie son
# md5 et le décompresse en fold_1/. Relançable : reprend un téléchargement
# interrompu et saute le fichier et le dossier déjà valides.
set -u

DEST="$(dirname "$0")/../data/base"
URL="https://zenodo.org/api/records/14478210/files"
F=fold_1
# md5 publié par Zenodo (https://zenodo.org/api/records/14478210)
SUM="3bef0c0229de65a2cb618b18f97ac949  $F.zip"

mkdir -p "$DEST"
cd "$DEST"

if md5sum -c --status <<<"$SUM" 2>/dev/null; then
  echo "$F.zip: déjà présent, md5 OK"
else
  curl -fL -C - --retry 5 -o "$F.zip" "$URL/$F.zip/content"
  md5sum -c <<<"$SUM" || exit 1
fi

# Décompression : le zip contient fold_1/{train,test}/. On extrait dans un
# dossier temporaire puis on déplace, pour que fold_1/ n'existe que complet.
if [ -d "$F" ]; then
  echo "$F/: déjà décompressé"
  exit 0
fi
echo "$F.zip: décompression"
rm -rf "$F.tmp"
if python3 -m zipfile -e "$F.zip" "$F.tmp" && [ -d "$F.tmp/$F" ]; then
  mv "$F.tmp/$F" "$F" && rm -rf "$F.tmp"
else
  echo "$F.zip: échec de la décompression" >&2
  exit 1
fi
