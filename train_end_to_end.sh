#!/usr/bin/env bash
# Chaîne complète sur une machine avec Docker + GPU :
#   1. téléchargement et préparation des données (train_scripts/)
#   2. pour chaque classe du fold : éval du SAM3 d'origine, puis finetuning (qui s'évalue sur test/)
#   3. rédaction de resultat.md (SAM3 d'origine vs finetuné)
#
# Usage : bash train_end_to_end.sh
# Prérequis : docker compose, nvidia-container-toolkit, curl, python3, et .env contenant HF_TOKEN.
# Relançable : les étapes déjà terminées (fichier DONE dans leur dossier de logs) sont sautées.
set -euo pipefail

cd "$(dirname "$0")"

export FOLD=1
CLASSES="LM FC LC VV"
COMPOSE=(docker compose -f docker-compose-train.yml)
TRAIN=(python sam3/train/train.py --use-cluster 0 --num-gpus 1 -c)

# 1. Données : le téléchargement tourne sur l'hôte (l'image n'a pas curl),
#    la préparation dans le conteneur (numpy, Pillow)
bash train_scripts/download_folds.sh
export CLASS=LM  # requis par docker-compose-train.yml, sans effet sur la préparation
"${COMPOSE[@]}" build
"${COMPOSE[@]}" run --rm train bash -c \
  "bash /app/train_scripts/process_base.sh && bash /app/train_scripts/make_coco_annotations.sh"

# 2. Évals et entraînements
for CLASS in $CLASSES; do
  export CLASS

  base_dir="logs/base_fold_${FOLD}_${CLASS}"
  if [ ! -f "$base_dir/DONE" ]; then
    echo "=== Éval du SAM3 d'origine : fold $FOLD, $CLASS"
    rm -rf "$base_dir"
    "${COMPOSE[@]}" run --rm -e LOG_DIR="/app/$base_dir" train \
      "${TRAIN[@]}" configs/finetune/eval_base.yaml
    touch "$base_dir/DONE"
  fi

  ft_dir="logs/fold_${FOLD}_${CLASS}"
  if [ ! -f "$ft_dir/DONE" ]; then
    echo "=== Finetuning + éval : fold $FOLD, $CLASS"
    # pas de rm : un entraînement interrompu reprend depuis son dernier checkpoint
    "${COMPOSE[@]}" run --rm train "${TRAIN[@]}" configs/finetune/docker_config.yaml
    touch "$ft_dir/DONE"
  fi
done

# 3. Rapport
python3 - "$FOLD" $CLASSES <<'EOF'
import glob, json, sys
from datetime import date

fold, classes = sys.argv[1], sys.argv[2:]
NAMES = {"LM": "Lumen", "FC": "Fibrous cap", "LC": "Lipid core", "VV": "Vasa vasorum"}
METRICS = [
    ("AP", "AP (IoU 0,50:0,95)"),
    ("AP_50", "AP@50"),
    ("AP_75", "AP@75"),
    ("AR_maxDets@100", "AR@100"),
    ("AP_small", "AP petits"),
    ("AP_medium", "AP moyens"),
    ("AP_large", "AP grands"),
]

def last_val_stats(log_dir):
    """Métriques COCO segm de la dernière éval écrite dans val_stats.json."""
    path = glob.glob(f"{log_dir}/logs/**/val_stats.json", recursive=True)[0]
    with open(path) as f:
        stats = json.loads(f.read().splitlines()[-1])
    return {name: v for k, v in stats.items() for name, _ in METRICS
            if k.endswith(f"coco_eval_segm_{name}")}

def fmt(v):
    # COCO renvoie -1 quand aucune structure de cette taille n'existe dans test/
    return "—" if v is None or v < 0 else f"{v:.3f}"

def gain(b, f):
    return "—" if b is None or f is None or b < 0 or f < 0 else f"{f - b:+.3f}"

results = {c: (last_val_stats(f"logs/base_fold_{fold}_{c}"),
               last_val_stats(f"logs/fold_{fold}_{c}")) for c in classes}

out = [
    f"# Résultats — SAM3 d'origine vs finetuné (fold {fold})",
    "",
    f"Généré le {date.today().isoformat()} par `train_end_to_end.sh`. "
    f"Évaluation sur `data/base/fold_{fold}/test/`, segmentation (masques), métriques COCO entre 0 et 1, "
    "plus haut = meilleur.",
    "",
    "## Synthèse",
    "",
    "| Classe | AP d'origine | AP finetuné | Gain |",
    "|---|---|---|---|",
]
for c, (b, f) in results.items():
    out.append(f"| {NAMES[c]} ({c}) | {fmt(b.get('AP'))} | {fmt(f.get('AP'))} | {gain(b.get('AP'), f.get('AP'))} |")

for c, (b, f) in results.items():
    out += ["", f"## {NAMES[c]} ({c})", "", "| Métrique | SAM3 d'origine | Finetuné | Gain |", "|---|---|---|---|"]
    for key, label in METRICS:
        out.append(f"| {label} | {fmt(b.get(key))} | {fmt(f.get(key))} | {gain(b.get(key), f.get(key))} |")

out += [
    "",
    "## Lire les métriques",
    "",
    "- **AP** : le chiffre principal. Précision moyenne en exigeant un recouvrement (IoU) "
    "masque prédit / masque annoté de 0,50 à 0,95.",
    "- **AP@50** : un recouvrement de 50 % suffit — le modèle trouve-t-il la bonne structure ?",
    "- **AP@75** : il faut 75 % de recouvrement — le contour est-il précis ?",
    "- **AR@100** : rappel, part des structures annotées retrouvées (jusqu'à 100 prédictions par image).",
    "- **AP petits / moyens / grands** : AP selon la taille de la structure "
    "(< 32², 32²–96², > 96² pixels). « — » : aucune structure de cette taille dans test/.",
    "",
]
with open("resultat.md", "w") as fh:
    fh.write("\n".join(out))
print("resultat.md écrit")
EOF
