# CLAUDE.md

Finetuning de SAM3 (Meta) sur 4 classes (LM, FC, LC, VV) du jeu Zenodo 14478210, avec comparaison
au modèle d'origine. `README.md` décrit l'organisation, les cibles `make` et le format des données ;
ce fichier ne couvre que ce qu'il faut savoir avant de modifier ou de lancer quelque chose.

## Avant de lancer quoi que ce soit
- `make check` est la seule vérification fonctionnelle sans GPU ni Docker (pipeline de données sur
  `tests/fixtures/`). Le relancer après tout changement dans `src/data/`.
- `make -n <cible>` affiche les commandes sans les exécuter.
- `train`, `eval-base`, `train-final` et `infer` exigent Docker et un GPU ; la machine de dev n'en a
  souvent pas (vérifier `nvidia-smi`). Ne pas lancer ces cibles, ni `make data` / `make all`
  (téléchargement de plusieurs Go, heures de GPU), sans l'accord de l'utilisateur.
- Valider une config Hydra sans GPU : la composer avec `hydra-core` (`initialize_config_dir` sur
  `sam3/sam3/train`, `config_name="configs/project/<nom>"`) et comparer `OmegaConf.to_yaml(cfg)`
  avant/après, sans `resolve=True` (les résolveurs `times`/`add`/`string` viennent de sam3 et exigent torch).

## Pièges
- **Ne pas mettre nos fichiers dans `sam3/`** (copie du dépôt de Meta). Nos configs sont dans `configs/`,
  vues par Hydra via le lien `sam3/sam3/train/configs/project` ; on les référence en
  `configs/project/<nom>.yaml`. Tout écart nécessaire avec l'amont se note dans la section
  « Écarts de `sam3/` » du README.
- Les chemins des configs sont ceux du conteneur (`/app/...`, le dépôt monté ; `tools/setup_pod.sh`
  crée le même lien sur RunPod). `DATASET`, `CLASS` et `LOG_DIR` sont lus dans l'environnement.
- `make train` et `make eval-base` visent `data/base/fold_N` en dur ; `data/small` se teste avec
  `make smoke` (`configs/smoke.yaml` : 10 époques, `supercategory` fixée à `small`).
- La reprise repose sur des fichiers `DONE` dans `logs/` : supprimer le `DONE` pour refaire une étape.
  Le conteneur écrit `logs/` en root.
- `merge_splits.sh` fabrique `fold_N/all` avec `ln` (liens durs) : ne pas passer à `cp -al`, qui écrase
  en silence un nom en double.
- `data/` (≈ 12 Go), `logs/` et `.env` (`HF_TOKEN`) sont ignorés par git : ne jamais les committer, ne pas
  lister `data/` récursivement, ne pas afficher `.env`.

## Conventions
- Français pour les commentaires, messages, docs et commits (voir `git log`).
