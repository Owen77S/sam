"""Assemble les segmentation_bbox/*.json en fichiers d'annotation COCO, un par classe.

Usage : python make_coco_annotations.py <dossier>

<dossier> contient img/ (images .png) et segmentation_bbox/ (sorties de
rle/masks_to_rle.py). Écrit <dossier>/annotation_<classe>.json pour chaque
classe ; chaque fichier liste toutes les images (celles sans la classe servent
d'exemples négatifs).
"""
import argparse
import json
from pathlib import Path

CLASSES = {  # label -> (category_id, name = prompt textuel SAM3)
    "LM": (1, "lumen"),
    "FC": (2, "fibrous cap"),
    "LC": (3, "lipid core"),
    "VV": (4, "vasa vasorum"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="dossier contenant img/ et segmentation_bbox/")
    args = parser.parse_args()

    img_dir = args.root / "img"
    json_dir = args.root / "segmentation_bbox"
    if not img_dir.is_dir() or not json_dir.is_dir():
        parser.error(f"{args.root} doit contenir img/ et segmentation_bbox/")

    images = []
    annotations = {label: [] for label in CLASSES}
    for img_path in sorted(img_dir.glob("*.png")):
        json_path = json_dir / (img_path.stem + ".json")
        if not json_path.exists():
            print(f"ATTENTION {img_path.name} : pas de {json_path.name}, image ignorée")
            continue
        data = json.loads(json_path.read_text())

        image_id = len(images) + 1
        images.append({
            "id": image_id,
            "file_name": f"img/{img_path.name}",
            "width": data["width"],
            "height": data["height"],
        })
        for obj in data["objects"]:
            anns = annotations[obj["label"]]
            anns.append({
                "id": len(anns) + 1,
                "image_id": image_id,
                "category_id": CLASSES[obj["label"]][0],
                "bbox": obj["bbox"],
                "area": sum(obj["segmentation"]["counts"][1::2]),
                "iscrowd": 0,
                "segmentation": obj["segmentation"],
            })

    for label, (category_id, name) in CLASSES.items():
        coco = {
            "info": {},
            "images": images,
            "categories": [{"id": category_id, "name": name}],
            "annotations": annotations[label],
        }
        out_path = args.root / f"annotation_{label}.json"
        out_path.write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
        print(f"{out_path} : {len(images)} images, {len(annotations[label])} annotations")


if __name__ == "__main__":
    main()
