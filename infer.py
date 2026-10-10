"""Inférence avec le SAM3 finetuné, ou avec le SAM3 d'origine sans --weights.

Usage : python infer.py <image> [--weights <poids.pt>] [--prompt lumen] [--out sortie.jpg]

<poids.pt> : checkpoint écrit par l'entraînement (checkpoints/checkpoint.pt) ou
ses seuls poids (clé "model"). Sans --weights, les poids d'origine sont
téléchargés depuis Hugging Face. Le prompt doit être le nom de catégorie utilisé
au finetuning : lumen, fibrous cap, lipid core ou vasa vasorum.
"""
import argparse

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("image")
parser.add_argument("--weights")
parser.add_argument("--prompt", default="lumen")
parser.add_argument("--out", default="img_segmented.jpg")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint_path n'est pas utilisable ici : build_sam3_image_model ne garde que les clés
# préfixées "detector." (format Hugging Face), absentes d'un checkpoint d'entraînement
model = build_sam3_image_model(device=device, load_from_HF=args.weights is None)
if args.weights is not None:
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict.get("model", state_dict))
processor = Sam3Processor(model, device=device)

image = Image.open(args.image).convert("RGB")
state = processor.set_image(image)
output = processor.set_text_prompt(state=state, prompt=args.prompt)
masks, scores = output["masks"], output["scores"]

result = np.array(image, dtype=np.float32)
rng = np.random.default_rng(0)
for mask in masks:
    m = mask.squeeze().cpu().numpy().astype(bool)
    result[m] = 0.5 * result[m] + 0.5 * rng.integers(0, 255, 3)
Image.fromarray(result.astype(np.uint8)).save(args.out)
print(f"{len(masks)} masque(s), scores : {[round(float(s), 3) for s in scores]} -> {args.out}")
