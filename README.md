# Finetuning de SAM3 — segmentation de structures par classe

Finetune [SAM3](https://github.com/facebookresearch/sam3) (Meta) sur le jeu
[Zenodo 14478210](https://zenodo.org/records/14478210) (5 folds) et compare le modèle finetuné
au SAM3 d'origine. Un modèle par classe, le nom de la classe servant de prompt textuel :

| Code | Classe | Prompt |
|---|---|---|
| LM | Lumen | `lumen` |
| FC | Fibrous cap | `fibrous cap` |
| LC | Lipid core | `lipid core` |
| VV | Vasa vasorum | `vasa vasorum` |

## Démarrage rapide

Prérequis : Docker, [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/),
`curl`, `python3`, `make`, et un fichier `.env` (`cp .env.example .env`, puis renseigner `HF_TOKEN`).

```bash
make all                      # build, données du fold 1, 4 évals + 4 finetunings, rapport
cat reports/resultat_fold_1.md
```

`make help` liste toutes les cibles. Les variables se surchargent sur la ligne de commande :
`make train FOLD=2 CLASSES="LC VV"`.

| Cible | Rôle |
|---|---|
| `build` | construit l'image GPU (`docker/Dockerfile`) |
| `data` | télécharge les folds `FOLDS` (défaut : `FOLD`) puis génère RLE et annotations COCO |
| `data-final` | fusionne train + test de `FOLD` en `fold_N/all` (liens durs) |
| `eval-base` | évalue le SAM3 d'origine sur `test/`, par classe |
| `train` | finetune puis évalue sur `test/`, par classe |
| `train-final` | entraînement final sans validation sur `fold_N/all` — `CLASS=LC` obligatoire |
| `report` | écrit `reports/resultat_fold_N.md` (d'origine vs finetuné) |
| `infer` | `make infer IMAGE=… [WEIGHTS=…] [PROMPT=lumen] [OUT=…]` |
| `shell` | shell interactif dans le conteneur |
| `check` | teste le pipeline de données sur `tests/fixtures/` (sans Docker ni GPU) |

**Reprise.** Chaque étape terminée laisse un fichier `DONE` dans son dossier de `logs/` et n'est pas
relancée. L'éval d'origine repart de zéro ; un finetuning interrompu reprend à son dernier
checkpoint. Pour refaire une étape, supprimer son `DONE`.

**Un seul GPU** : le Makefile ne lance jamais deux jobs en parallèle (`.NOTPARALLEL`).

## Organisation

```
Makefile  compose.yaml  docker/Dockerfile  requirements.txt  .env.example
configs/            configs Hydra du projet (finetune, finetune_final, eval_base)
src/data/           download.sh, prepare.sh, merge_splits.sh, masks_to_rle.py, make_coco_annotations.py
src/infer.py        inférence (SAM3 d'origine sans --weights)
src/report.py       rapport d'origine vs finetuné
tests/              fixtures + check_data.sh (make check)
tools/              setup_pod.sh (RunPod sans Docker), capture_slices.py
sam3/               copie du dépôt de Meta, avec quelques écarts (voir plus bas)
data/  logs/        ignorés par git : données et sorties d'entraînement
reports/            rapports générés, suivis par git
```

## Données

`make data` produit, pour chaque `data/base/fold_N/{train,test}/` :

```
img/*.png                       images
mask_color/*.tiff               masques couleur (fournis) ; les mask/ bruts sont supprimés
segmentation_bbox/*.json        RLE non compressé + bbox par structure   (masks_to_rle.py)
annotation_{LM,FC,LC,VV}.json   COCO, un fichier par classe              (make_coco_annotations.py)
```

Chaque fichier COCO liste toutes les images ; celles sans la classe servent d'exemples négatifs.
`make data-final` ajoute `fold_N/all/` : mêmes fichiers en liens durs, donc `train/` et `test/`
restent intacts. Un nom présent dans les deux splits fait échouer la fusion.

## Configs Hydra

Les configs vivent dans `configs/` et non dans `sam3/`. `train.py` charge ses configs avec
`initialize_config_module("sam3.train")` ; le lien `sam3/sam3/train/configs/project -> ../../../../configs`
les rend visibles sous le nom `configs/project/<nom>.yaml` sans toucher au code de Meta.
`DATASET`, `CLASS` et `LOG_DIR` sont lus dans l'environnement (fixés par le Makefile).

## RunPod sans Docker

```bash
bash tools/setup_pod.sh && source ~/.bashrc    # à refaire après chaque redémarrage du pod
cd /app/sam3 && python sam3/train/train.py -c configs/project/finetune.yaml --use-cluster 0 --num-gpus 1
```

Le script lie `/app` vers le dépôt (les configs utilisent des chemins en `/app/...`) et place
le cache Hugging Face sur `/workspace`. Les données (`data/`) sont à envoyer séparément.

## Essai rapide sur `small_fold`

Petit jeu (20 images en train, 10 en test) pour tester la chaîne sans télécharger un fold. Il vit
dans `data/small_fold/` (ignoré par git) et n'a pas d'annotations au départ :

```bash
python3 src/data/masks_to_rle.py data/small_fold
python3 src/data/make_coco_annotations.py data/small_fold/train
python3 src/data/make_coco_annotations.py data/small_fold/test
docker compose run --rm -w /app/sam3 -e DATASET=small_fold -e CLASS=LC -e LOG_DIR=/app/logs/small_fold_LC \
  sam3 python sam3/train/train.py --use-cluster 0 --num-gpus 1 -c configs/project/finetune.yaml
```

Éviter VV (5 annotations en train, 0 en test : métriques à −1). `make train` vise `base/fold_N`
et ne convient pas ici.

## Écarts de `sam3/` avec le dépôt de Meta

- `sam3/sam3/model/vitdet.py` : `Mlp.forward` n'utilise `addmm_act` qu'à l'inférence
  (il détache les poids, inutilisable avec les gradients activés).
- `sam3/sam3/train/data/` : ajouté après coup (omis du premier import par une règle `.gitignore`).
- `sam3/sam3/train/configs/project` : lien symbolique vers `configs/` (voir ci-dessus).
- `sam3/sam3/train/configs/finetune/` : supprimé ; nos configs sont dans `configs/`, et son
  `config.yaml` doublait `roboflow_v100/roboflow_v100_full_ft_100_images.yaml`.

## Outils

`tools/capture_slices.py` capture une zone de l'écran puis appuie sur → jusqu'à la dernière image
(à lancer avec le Python de Windows ; voir l'en-tête du fichier). Les captures sont à ranger dans
`data/slicer/`.
