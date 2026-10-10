"""Convertit les masques couleur (mask_color/*.tiff) en polygones + bbox.

Usage : python masks_to_polygons.py <dossier_racine> [--preview]

Pour chaque dossier contenant img/ et mask_color/, écrit
segmentation_bbox/<nom>.json (et segmentation_preview/<nom>.png avec --preview).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

BACKGROUND = (128, 128, 128)
CLASSES = {
    "LM": (228, 30, 199),   # magenta
    "FC": (123, 171, 226),  # bleu clair
    "LC": (125, 227, 127),  # vert
    "VV": (208, 2, 27),     # rouge
}


def find_mask(mask_dir, stem):
    for ext in (".tiff", ".tif"):
        path = mask_dir / (stem + ext)
        if path.exists():
            return path
    return None


def extract_objects(mask):
    """mask : tableau (H, W, 3) RGB -> liste d'objets {label, segmentation, bbox}."""
    objects = []
    for label, color in CLASSES.items():
        binary = (mask == color).all(axis=-1).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if len(c) >= 3]
        if not contours:
            continue
        objects.append({
            "label": label,
            "segmentation": [c.reshape(-1).tolist() for c in contours],
            "bbox": list(cv2.boundingRect(np.concatenate(contours))),
        })
    return objects


def warn_unknown_colors(mask, name):
    known = set(CLASSES.values()) | {BACKGROUND}
    for color in np.unique(mask.reshape(-1, 3), axis=0):
        if tuple(color.tolist()) not in known:
            print(f"  ATTENTION {name} : couleur inconnue {tuple(color.tolist())} ignorée")


def save_preview(image, objects, path):
    overlay = image.copy()
    for obj in objects:
        polygons = [np.array(p, dtype=np.int32).reshape(-1, 2) for p in obj["segmentation"]]
        cv2.fillPoly(overlay, polygons, CLASSES[obj["label"]])
    preview = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
    for obj in objects:
        x, y, w, h = obj["bbox"]
        color = CLASSES[obj["label"]]
        cv2.rectangle(preview, (x, y), (x + w - 1, y + h - 1), color, 2)
        cv2.putText(preview, obj["label"], (x, max(y - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    Image.fromarray(preview).save(path)


def process_folder(folder, preview):
    out_dir = folder / "segmentation_bbox"
    out_dir.mkdir(exist_ok=True)
    if preview:
        preview_dir = folder / "segmentation_preview"
        preview_dir.mkdir(exist_ok=True)

    for img_path in sorted((folder / "img").glob("*.png")):
        mask_path = find_mask(folder / "mask_color", img_path.stem)
        if mask_path is None:
            print(f"  ATTENTION {img_path.name} : pas de masque, image ignorée")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        if mask.size != image.size:
            print(f"  ATTENTION {img_path.name} : masque {mask.size} redimensionné en {image.size}")
            mask = mask.resize(image.size, Image.NEAREST)
        mask = np.array(mask)

        warn_unknown_colors(mask, mask_path.name)
        objects = extract_objects(mask)
        annotation = {
            "image": img_path.name,
            "width": image.width,
            "height": image.height,
            "objects": objects,
        }
        (out_dir / (img_path.stem + ".json")).write_text(json.dumps(annotation))
        if preview:
            save_preview(np.array(image), objects, preview_dir / img_path.name)
        print(f"  {img_path.name} : {', '.join(o['label'] for o in objects) or 'aucun objet'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dossier racine")
    parser.add_argument("--preview", action="store_true",
                        help="enregistre aussi une image de contrôle par image")
    args = parser.parse_args()

    folders = [d for d in [args.root, *sorted(args.root.rglob("*"))]
               if (d / "img").is_dir() and (d / "mask_color").is_dir()]
    if not folders:
        parser.error(f"aucun dossier contenant img/ et mask_color/ sous {args.root}")
    for folder in folders:
        print(folder)
        process_folder(folder, args.preview)


if __name__ == "__main__":
    main()
