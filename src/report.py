"""Rédige reports/resultat_fold_<fold>.md : SAM3 d'origine vs finetuné, métriques COCO segm.

Usage : python src/report.py <fold> <classe>…   (depuis la racine du dépôt ; `make report`)

Lit logs/base_fold_<fold>_<classe>/ (éval d'origine) et logs/fold_<fold>_<classe>/ (finetuné).
"""
import glob, json, sys
from datetime import date
from pathlib import Path

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
    f"Généré le {date.today().isoformat()} par `make report`. "
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
out_path = Path("reports") / f"resultat_fold_{fold}.md"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("\n".join(out), encoding="utf-8")
print(f"{out_path} écrit")
