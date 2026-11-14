FROM python:3.12-slim

# PYTHONPATH : sam3 est importé depuis le dépôt local monté dans /app/sam3 (et non depuis GitHub),
# pour que Hydra trouve nos configs (lien sam3/sam3/train/configs/project -> configs/)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/sam3

WORKDIR /app

# Index CUDA 12.8 de PyTorch (versions du README de sam3) ; les wheels embarquent les bibliothèques CUDA,
# seul le pilote NVIDIA de l'hôte est nécessaire
RUN pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["bash"]
