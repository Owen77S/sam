#!/bin/bash
# Prépare un pod RunPod pour le finetuning de sam3.
# À relancer après chaque (re)démarrage : le disque du conteneur est vidé, seul /workspace persiste.
#
# Usage : bash setup_pod.sh   (depuis le dépôt cloné sur le disque local, ex. /root/sam)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# torch : on garde celui du template s'il est assez récent et voit le GPU, sinon on l'installe
if python -c "import torch, sys; v = tuple(int(x) for x in torch.__version__.split('.')[:2]); sys.exit(0 if v >= (2, 7) and torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "torch déjà présent : $(python -c 'import torch; print(torch.__version__)')"
else
    pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
fi

pip install -r requirements_vps.txt

# La config utilise des chemins en /app/... : on fait pointer /app sur le dépôt
if [ "$REPO_DIR" != /app ]; then
    if [ -e /app ] && [ ! -L /app ]; then
        echo "ERREUR : /app existe déjà et n'est pas un lien, impossible de le faire pointer sur $REPO_DIR" >&2
        exit 1
    fi
    ln -sfn "$REPO_DIR" /app
fi

# Variables d'environnement pour les prochains shells
add_to_bashrc() {
    grep -qxF "$1" ~/.bashrc 2>/dev/null || echo "$1" >> ~/.bashrc
}
# sam3 est importé depuis le dépôt local, pour que Hydra trouve configs/finetune/
add_to_bashrc "export PYTHONPATH=/app/sam3"
# cache Hugging Face sur le volume persistant : le checkpoint n'est téléchargé qu'une fois
if [ -d /workspace ]; then
    add_to_bashrc "export HF_HOME=/workspace/hf"
fi

# Vérifications
python -c "import torch; assert torch.cuda.is_available(), 'GPU non visible par torch'; print('GPU :', torch.cuda.get_device_name(0))"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ATTENTION : HF_TOKEN n'est pas défini (variables d'environnement du pod dans la console RunPod)"
fi
if [ ! -d data/small_fold ]; then
    echo "ATTENTION : data/small_fold est absent (data/ n'est pas dans git, à envoyer à part)"
fi

echo
echo "Pod prêt. Pour lancer le finetuning :"
echo "  source ~/.bashrc"
echo "  cd /app/sam3 && python sam3/train/train.py -c configs/finetune/docker_config.yaml --use-cluster 0 --num-gpus 1"
