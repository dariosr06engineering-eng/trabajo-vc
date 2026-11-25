"""Mide cuánto se parecen las máscaras convertidas a YOLO a las máscaras originales.

    python src/check_conversion.py --data ../data/fruits_part1/train --yolo ../datasets/fruits_yolo

Para cada fruta, vuelve a pintar su polígono de YOLO como máscara (con polygon2mask,
la misma función que usa Ultralytics al entrenar) y calcula su IoU con la máscara
original del annotations.json. Da los resultados para las frutas que se evalúan
(iscrowd = 0 en el original), separados por número de trozos y por si tienen agujeros.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from ultralytics.data.utils import polygon2mask


def pieces_and_holes(mask):
    """Número de trozos (8-conectividad) y si la máscara tiene algún agujero."""
    n, _ = cv2.connectedComponents(mask, connectivity=8)
    padded = np.pad(mask, 1)
    nb, _ = cv2.connectedComponents((padded == 0).astype(np.uint8), connectivity=4)
    return n - 1, nb - 1 > 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True, help="carpeta del train con annotations.json")
    p.add_argument("--yolo", type=Path, required=True, help="dataset convertido por prepare_dataset.py")
    args = p.parse_args()

    coco = json.loads((args.data / "annotations.json").read_text(encoding="utf-8"))
    anns_of = defaultdict(list)
    for a in coco["annotations"]:          # mismo orden que usa convert_coco
        anns_of[a["image_id"]].append(a)

    ious, by_pieces, by_holes = [], defaultdict(list), defaultdict(list)
    mismatched = 0
    for im in coco["images"]:
        h, w = im["height"], im["width"]
        stem = Path(im["file_name"]).stem
        label = next(args.yolo.glob(f"labels/*/{stem}.txt"))
        rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        anns = anns_of[im["id"]]
        if len(rows) != len(anns):
            mismatched += 1
            continue
        for a, row in zip(anns, rows):
            if a["iscrowd"]:
                continue                    # no se evalúan
            gt = mask_utils.decode(a["segmentation"])
            poly = (np.array(row[1:]).reshape(-1, 2) * [w, h]).reshape(-1)
            pred = polygon2mask((h, w), [poly])
            inter = np.logical_and(gt, pred).sum()
            union = np.logical_or(gt, pred).sum()
            iou = inter / union
            ious.append(iou)
            n, hole = pieces_and_holes(gt)
            by_pieces[min(n, 5)].append(iou)
            by_holes[hole].append(iou)

    ious = np.array(ious)
    print(f"imágenes con distinto número de filas que de anotaciones: {mismatched}")
    print(f"frutas evaluadas comparadas: {len(ious)}")
    print(f"IoU medio {ious.mean():.3f}, mediana {np.median(ious):.3f}")
    for t in (0.5, 0.75, 0.9, 0.95):
        print(f"  IoU >= {t:.2f}: {100 * (ious >= t).mean():.1f} %")
    print("por número de trozos:")
    for k in sorted(by_pieces):
        v = np.array(by_pieces[k])
        print(f"  {k if k < 5 else '5+'} trozo(s): {len(v):5d} frutas, IoU medio {v.mean():.3f}, "
              f"IoU >= 0.75: {100 * (v >= 0.75).mean():.1f} %")
    for k in (False, True):
        v = np.array(by_holes[k])
        print(f"  {'con' if k else 'sin'} agujeros: {len(v):5d} frutas, IoU medio {v.mean():.3f}")


if __name__ == "__main__":
    main()
