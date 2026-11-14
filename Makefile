# Point d'entrée du projet : `make help`.
#
# Variables (toutes surchargeables : make train FOLD=2 CLASSES="LC VV") :
#   FOLD     fold utilisé (1 à 5)                       défaut : 1
#   CLASSES  classes traitées par eval-base/train/report  défaut : LM FC LC VV
#   FOLDS    folds téléchargés par `make data`          défaut : $(FOLD)
#   CLASS    classe unique, requise par train-final

FOLD    ?= 1
CLASSES ?= LM FC LC VV
FOLDS   ?= $(FOLD)

RUN   := docker compose run --rm
TRAIN := python sam3/train/train.py --use-cluster 0 --num-gpus 1 -c

BASE_DONE := $(foreach c,$(CLASSES),logs/base_fold_$(FOLD)_$(c)/DONE)
FT_DONE   := $(foreach c,$(CLASSES),logs/fold_$(FOLD)_$(c)/DONE)
# éval d'origine puis finetuning, classe par classe
ALL_DONE  := $(foreach c,$(CLASSES),logs/base_fold_$(FOLD)_$(c)/DONE logs/fold_$(FOLD)_$(c)/DONE)

.DEFAULT_GOAL := help
# un seul GPU : jamais deux jobs en parallèle, même avec make -j
.NOTPARALLEL:
.PHONY: help build shell data data-final eval-base train train-final report all infer check

help: ## Liste les cibles
	@awk -F':.*## ' '/^[a-z-]+:.*## /{printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.env:
	@echo "ERREUR : .env est absent. Copier .env.example en .env et y mettre HF_TOKEN." >&2; exit 1

build: .env ## Construit l'image Docker (GPU)
	docker compose build

shell: .env ## Shell interactif dans le conteneur
	$(RUN) sam3 bash

data: .env ## Télécharge les folds (FOLDS=…) puis prépare RLE + annotations COCO
	bash src/data/download.sh $(FOLDS)
	$(RUN) sam3 bash src/data/prepare.sh

data-final: .env ## Fusionne train+test du fold en fold_N/all (liens durs)
	$(RUN) sam3 bash src/data/merge_splits.sh $(FOLD)

eval-base: $(BASE_DONE) ## Évalue le SAM3 d'origine sur test/ (une passe par classe)

train: $(FT_DONE) ## Finetune puis évalue sur test/ (une passe par classe)

# Reprise : une étape terminée laisse un fichier DONE et n'est pas relancée.
# L'éval d'origine repart de zéro ; un finetuning interrompu reprend à son dernier checkpoint.
$(BASE_DONE): logs/base_fold_$(FOLD)_%/DONE: | .env
	rm -rf $(@D)
	$(RUN) -w /app/sam3 -e DATASET=base/fold_$(FOLD) -e CLASS=$* \
	  -e LOG_DIR=/app/logs/base_fold_$(FOLD)_$* sam3 $(TRAIN) configs/project/eval_base.yaml
	touch $@

$(FT_DONE): logs/fold_$(FOLD)_%/DONE: | .env
	$(RUN) -w /app/sam3 -e DATASET=base/fold_$(FOLD) -e CLASS=$* \
	  -e LOG_DIR=/app/logs/fold_$(FOLD)_$* sam3 $(TRAIN) configs/project/finetune.yaml
	touch $@

train-final: .env ## Entraînement final sans validation, sur fold_N/all (CLASS=LM|FC|LC|VV)
	@test -n "$(CLASS)" || { echo "ERREUR : préciser CLASS=LM, FC, LC ou VV" >&2; exit 1; }
	$(RUN) -w /app/sam3 -e DATASET=base/fold_$(FOLD)/all -e CLASS=$(CLASS) \
	  -e LOG_DIR=/app/logs/final_$(CLASS) sam3 $(TRAIN) configs/project/finetune_final.yaml

report: reports/resultat_fold_$(FOLD).md ## Écrit le rapport SAM3 d'origine vs finetuné

reports/resultat_fold_$(FOLD).md: $(ALL_DONE) src/report.py
	python3 src/report.py $(FOLD) $(CLASSES)

all: build data report ## Chaîne complète : build, données, évals, finetunings, rapport

infer: .env ## Inférence : IMAGE=… [WEIGHTS=…] [PROMPT=lumen] [OUT=…] (chemins relatifs au dépôt)
	@test -n "$(IMAGE)" || { echo "ERREUR : préciser IMAGE=chemin/image.png" >&2; exit 1; }
	$(RUN) sam3 python src/infer.py $(IMAGE) $(if $(WEIGHTS),--weights $(WEIGHTS)) \
	  $(if $(PROMPT),--prompt "$(PROMPT)") $(if $(OUT),--out $(OUT))

check: ## Teste le pipeline de données sur tests/fixtures (sans Docker ni GPU)
	bash tests/check_data.sh
