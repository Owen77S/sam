FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# git : nécessaire pour installer sam3 depuis GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Index CPU de PyTorch (pas de GPU sur cette machine) ; PyPI reste l'index principal
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# sam3 code en dur device="cuda" à la construction du modèle (position_encoding.py, decoder.py) ;
# on le force sur cpu puisqu'il n'y a pas de GPU
RUN SAM3_DIR=$(python -c "import importlib.util as u; print(u.find_spec('sam3').submodule_search_locations[0])") \
    && sed -i 's/device="cuda")$/device="cpu")/' "$SAM3_DIR/model/position_encoding.py" \
    && sed -i '/feat_size, feat_size, device="cuda"/s/"cuda"/"cpu"/' "$SAM3_DIR/model/decoder.py" \
    && ! grep -n 'device="cuda"' "$SAM3_DIR/model/position_encoding.py" "$SAM3_DIR/model/decoder.py"

# addmm_act force fc1 en bfloat16 et renvoie du bfloat16, alors que fc2 reste en float32 sur CPU
# (pas d'autocast) -> "mat1 and mat2 must have the same dtype" ; on recaste vers le dtype d'entrée.
# pin_memory() exige un accélérateur : on l'enlève (inutile sur cpu).
RUN SAM3_DIR=$(python -c "import importlib.util as u; print(u.find_spec('sam3').submodule_search_locations[0])") \
    && sed -i 's/^    self = self.to(torch.bfloat16)/    orig_dtype = mat1.dtype\n&/; s/return y.view(mat1.shape\[:-1\] + (y.shape\[-1\],))/&.to(orig_dtype)/' "$SAM3_DIR/perflib/fused.py" \
    && sed -i 's/scale.pin_memory().to(/scale.to(/' "$SAM3_DIR/model/geometry_encoders.py" \
    && grep -c 'orig_dtype' "$SAM3_DIR/perflib/fused.py" | grep -qx 3 \
    && ! grep -n 'scale.pin_memory()' "$SAM3_DIR/model/geometry_encoders.py"

CMD ["bash"]
