#!/usr/bin/env bash
# Teste le pipeline de données (src/data/) sur tests/fixtures/sample, sans Docker ni GPU.
# Reconstruit un faux fold dans un dossier temporaire, lance prepare.sh puis merge_splits.sh
# et compare les sorties aux références versionnées.
# Prérequis : python3 avec numpy et Pillow.
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/.."
SAMPLE="$HERE/fixtures/sample"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export DATA_BASE="$TMP/base"
fold="$DATA_BASE/fold_1"

fail() { echo "ÉCHEC : $*" >&2; exit 1; }

# Faux fold : sample.png en train, la même image sous un autre nom en test
mkdir -p "$fold"/{train,test}/{img,mask_color}
cp "$SAMPLE/img/sample.png" "$fold/train/img/sample.png"
cp "$SAMPLE/mask_color/sample.tiff" "$fold/train/mask_color/sample.tiff"
cp "$SAMPLE/img/sample.png" "$fold/test/img/sample_b.png"
cp "$SAMPLE/mask_color/sample.tiff" "$fold/test/mask_color/sample_b.tiff"

bash "$ROOT/src/data/prepare.sh" >/dev/null
cmp "$fold/train/segmentation_bbox/sample.json" "$SAMPLE/segmentation_bbox/sample.json" \
  || fail "RLE + bbox différents de la référence"
for c in LM FC LC VV; do
  cmp "$fold/train/annotation_$c.json" "$SAMPLE/annotation_$c.json" \
    || fail "annotation_$c.json différent de la référence"
done
echo "prepare.sh      OK : RLE et annotations COCO identiques aux références"

bash "$ROOT/src/data/merge_splits.sh" 1 >/dev/null
[ "$(find "$fold/all/img" -type f | wc -l)" -eq 2 ] || fail "all/img devrait contenir 2 images"
[ "$(stat -c %h "$fold/all/img/sample.png")" -eq 2 ] || fail "all/ n'utilise pas de liens durs"
[ "$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['images']))" \
     "$fold/all/annotation_LM.json")" -eq 2 ] || fail "annotation_LM.json de all/ devrait lister 2 images"
[ -f "$fold/train/img/sample.png" ] && [ -f "$fold/test/img/sample_b.png" ] \
  || fail "merge_splits.sh a modifié train/ ou test/"
echo "merge_splits.sh OK : 2 images en liens durs, train/ et test/ intacts"

# Un nom présent dans train/ et test/ doit faire échouer la fusion
cp "$SAMPLE/img/sample.png" "$fold/test/img/sample.png"
if bash "$ROOT/src/data/merge_splits.sh" 1 >/dev/null 2>&1; then
  fail "merge_splits.sh aurait dû refuser un nom en double"
fi
echo "merge_splits.sh OK : un nom en double est refusé"
