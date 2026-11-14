"""Convertit les masques couleur (mask_color/*.tiff) en RLE non compressé + bbox.

Usage : python masks_to_rle.py <dossier_racine> [--preview]

Pour chaque dossier contenant img/ et mask_color/, écrit
segmentation_bbox/<nom>.json (et segmentation_preview/<nom>.png avec --preview).

RLE : {"size": [hauteur, largeur], "counts": [...]}, pixels parcourus colonne
par colonne (convention COCO), la première séquence compte toujours des 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

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


def rle_encode(binary):
    """binary : tableau (H, W) booléen -> {"size": [H, W], "counts": [...]}."""
    flat = binary.flatten(order="F")
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0] + counts
    return {"size": list(binary.shape), "counts": counts}


def rle_decode(rle):
    """Inverse de rle_encode -> tableau (H, W) booléen."""
    values = np.arange(len(rle["counts"])) % 2 == 1
    return np.repeat(values, rle["counts"]).reshape(rle["size"], order="F")


def extract_objects(mask):
    """mask : tableau (H, W, 3) RGB -> liste d'objets {label, segmentation, bbox}."""
    objects = []
    for label, color in CLASSES.items():
        binary = (mask == color).all(axis=-1)
        if not binary.any():
            continue
        ys, xs = np.nonzero(binary)
        x, y = int(xs.min()), int(ys.min())
        objects.append({
            "label": label,
            "segmentation": rle_encode(binary),
            "bbox": [x, y, int(xs.max()) - x + 1, int(ys.max()) - y + 1],
        })
    return objects


def warn_unknown_colors(mask, name):
    known = set(CLASSES.values()) | {BACKGROUND}
    for color in np.unique(mask.reshape(-1, 3), axis=0):
        if tuple(color.tolist()) not in known:
            print(f"  ATTENTION {name} : couleur inconnue {tuple(color.tolist())} ignorée")


def save_preview(image, objects, path):
    preview = image.astype(np.float32)
    for obj in objects:
        binary = rle_decode(obj["segmentation"])
        preview[binary] = 0.5 * preview[binary] + 0.5 * np.array(CLASSES[obj["label"]])
    preview = Image.fromarray(preview.astype(np.uint8))
    draw = ImageDraw.Draw(preview)
    for obj in objects:
        x, y, w, h = obj["bbox"]
        color = CLASSES[obj["label"]]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        draw.text((x, max(y - 12, 0)), obj["label"], fill=color)
    preview.save(path)


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
