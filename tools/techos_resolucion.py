"""Techos de AP en validación: qué AP sacaría un detector perfecto (encuentra todas las frutas y acierta la especie)
si sus máscaras tuvieran alguna limitación. Solo CPU.

    python tools/techos_resolucion.py

  - Etiquetas YOLO: si el modelo reprodujera exactamente nuestros polígonos (un único polígono por fruta).
  - Resolución 1/s: si la máscara solo pudiera representarse a 1/s de la resolución de la imagen y luego se ampliara
    con interpolación bilineal, como hace YOLO-seg (genera las máscaras a 1/4 de la resolución de su entrada).
    Es una aproximación: se simula con la media de cada bloque de s x s píxeles; una red puede hacerlo algo mejor.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from ultralytics.data.utils import polygon2mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter_kit"))
import evaluate as ev  # noqa: E402


def rle(m):
    r = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def at_stride(m, s):
    h, w = m.shape
    low = cv2.resize(m.astype(np.float32), (w // s, h // s), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


def main():
    gt = ev.load_gt(ROOT / "data" / "fruits_part1" / "train" / "annotations.json")
    val_ids = [int(x) for x in (ROOT / "datasets" / "fruits_yolo" / "val_ids.txt").read_text().split()]
    images = {im["id"]: im for im in gt.dataset["images"]}
    anns_of = defaultdict(list)
    for a in gt.dataset["annotations"]:
        anns_of[a["image_id"]].append(a)

    real, labels = [], []
    for i in val_ids:
        im = images[i]
        h, w = im["height"], im["width"]
        lbl = ROOT / "datasets" / "fruits_yolo" / "labels" / "val" / f"{Path(im['file_name']).stem}.txt"
        lines = lbl.read_text().splitlines()
        assert len(lines) == len(anns_of[i]), "prepare_dataset.py escribe una línea por anotación, en el mismo orden"
        for a, line in zip(anns_of[i], lines):
            if a["iscrowd"]:
                continue
            v = list(map(float, line.split()))
            poly = (np.array(v[1:]).reshape(-1, 2) * [w, h]).reshape(-1)
            labels.append((i, a["category_id"], polygon2mask((h, w), [poly]).astype(bool)))
            real.append((i, a["category_id"], mask_utils.decode(a["segmentation"]).astype(bool)))

    def score(items, name, s=None):
        preds = []
        for i, c, m in items:
            m = at_stride(m, s) if s else m
            if m.any():
                preds.append({"image_id": i, "category_id": c, "score": 1.0, "segmentation": rle(m)})
        preds, _ = ev.validate(preds, gt)
        r = ev.evaluate(gt, preds, val_ids)
        print(f"{name:40s} AP {r['AP']:.3f} | AP50 {r['AP50']:.3f} | AP75 {r['AP75']:.3f} | small {r['AP_small']:.3f}")

    score(real, "máscaras reales (debe dar 1.0)")
    score(labels, "etiquetas YOLO")
    for s in (4, 2):
        score(real, f"máscaras reales a 1/{s} de resolución", s)
        score(labels, f"etiquetas YOLO a 1/{s} de resolución", s)


if __name__ == "__main__":
    main()
