"""Análisis de errores de unas predicciones sobre las imágenes de validación (solo CPU).

    python tools/analisis_errores.py --pred predictions.json

Muestra:
  - qué porcentaje de frutas se encuentra (IoU >= 0.5) y con máscara precisa (IoU >= 0.75), por tamaño y por número
    de trozos de la fruta;
  - si las máscaras predichas salen más grandes o más pequeñas que las reales (área predicha / área real);
  - cómo cambia la AP si se erosionan o dilatan las máscaras 1 píxel.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
import evaluate as ev  # noqa: E402

SIZE_EDGES = [20, 50, 100, 200, 400, 1024, 10 ** 9]
SIZE_NAMES = ["20-49", "50-99", "100-199", "200-399", "400-1023", ">=1024"]


def rle(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pred", type=Path, required=True, help="predicciones de main.py detect")
    p.add_argument("--gt", type=Path, default=ROOT / "data" / "fruits_part1" / "train" / "annotations.json")
    p.add_argument("--val-ids", type=Path, default=ROOT / "datasets" / "fruits_yolo" / "val_ids.txt")
    args = p.parse_args()

    gt = ev.load_gt(args.gt)
    val_ids = [int(x) for x in args.val_ids.read_text().split()]

    def score(preds, name):
        preds, _ = ev.validate(preds, gt)
        m = ev.evaluate(gt, preds, val_ids)
        print(f"{name:20s} AP {m['AP']:.3f} | AP50 {m['AP50']:.3f} | AP75 {m['AP75']:.3f} | small {m['AP_small']:.3f}")

    preds = [p for p in json.loads(args.pred.read_text()) if p["image_id"] in set(val_ids)]
    dets_of = defaultdict(list)
    for d in preds:
        dets_of[(d["image_id"], d["category_id"])].append(d)

    # Mejor IoU de cada fruta evaluada con alguna detección de su especie
    by_size, by_pieces, area_ratio = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in val_ids:
        for a in gt.imgToAnns[i]:
            if a["iscrowd"]:
                continue
            dets = dets_of[(i, a["category_id"])]
            best, ratio = 0.0, None
            if dets:
                ious = mask_utils.iou([d["segmentation"] for d in dets], [a["segmentation"]], [0])[:, 0]
                k = int(np.argmax(ious))
                best = float(ious[k])
                ratio = mask_utils.area(dets[k]["segmentation"]) / a["area"]
            pieces = cv2.connectedComponents(mask_utils.decode(a["segmentation"]), connectivity=8)[0] - 1
            s = next(n for e, n in zip(SIZE_EDGES[1:], SIZE_NAMES) if a["area"] < e)
            by_size[s].append(best)
            by_pieces[min(pieces, 5)].append(best)
            if best >= 0.5:
                area_ratio[s].append(ratio)

    print(f"{'área (px)':>10s} {'frutas':>7s} {'encontradas':>12s} {'IoU>=0.75':>10s} {'IoU medio*':>11s} {'pred/real*':>11s}")
    for s in SIZE_NAMES:
        v, r = np.array(by_size[s]), np.array(area_ratio[s])
        print(f"{s:>10s} {len(v):7d} {100 * (v >= 0.5).mean():11.1f}% {100 * (v >= 0.75).mean():9.1f}% "
              f"{v[v >= 0.5].mean():11.3f} {np.median(r):11.2f}")
    print("(* solo frutas encontradas; 'pred/real' = mediana del área predicha / área real)\n")
    print(f"{'trozos':>7s} {'frutas':>7s} {'encontradas':>12s} {'IoU>=0.75':>10s} {'IoU medio*':>11s}")
    for k in sorted(by_pieces):
        v = np.array(by_pieces[k])
        print(f"{k if k < 5 else '5+':>7} {len(v):7d} {100 * (v >= 0.5).mean():11.1f}% "
              f"{100 * (v >= 0.75).mean():9.1f}% {v[v >= 0.5].mean():11.3f}")

    print()
    score(preds, "tal cual")
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for name, op in (("erosión 1 px", cv2.erode), ("dilatación 1 px", cv2.dilate)):
        out = []
        for d in preds:
            m = op(mask_utils.decode(d["segmentation"]), kernel)
            if m.any():
                out.append(dict(d, segmentation=rle(m)))
        score(out, name)


if __name__ == "__main__":
    main()
