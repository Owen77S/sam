#!/usr/bin/env bash
# Enchaîne les étapes de préparation des données. S'arrête à la première erreur.
set -euo pipefail

DIR="$(dirname "$0")"

bash "$DIR/download_fold.sh"
bash "$DIR/process_base.sh"
bash "$DIR/make_coco_annotations.sh"
