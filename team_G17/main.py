"""Proyecto final de Visión por Computador, grupo G17.

    python main.py detect --input <input folder> --output <predictions file> --device cuda

Parte 1: detecta las frutas de cada imagen de <input folder>/images (la lista está
en <input folder>/images.json) y escribe en <predictions file> un JSON con el
formato de resultados de COCO: una entrada por fruta con su máscara (RLE), su
especie y su confianza.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from detector import FruitDetector  # noqa: E402

MAX_DETS = 300


def to_rle(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")      # bytes -> str, para poder escribirlo en JSON
    return rle


def progress(k, n, name):
    print(f"\rDetectando frutas {k}/{n}: {name}".ljust(60), end="\n" if k == n else "", flush=True)


def detect(args):
    images = json.loads((args.input / "images.json").read_text())["images"]
    detector = FruitDetector(device=args.device, species_vote=args.species_vote)
    results = []
    for k, im in enumerate(images, 1):
        progress(k, len(images), im["file_name"])
        dets = detector.detect(args.input / "images" / im["file_name"])
        dets = sorted(dets, key=lambda d: -d[2])[:MAX_DETS]
        for mask, category_id, score in dets:
            results.append({"image_id": im["id"], "category_id": category_id, "score": score,
                            "segmentation": to_rle(mask)})
    args.output.write_text(json.dumps(results))
    print(f"{len(results)} detecciones escritas en {args.output}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="task", required=True)
    d = sub.add_parser("detect")
    d.add_argument("--input", type=Path, required=True)
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    d.add_argument("--species-vote", action="store_true", help="una sola especie por imagen (votación)")
    args = p.parse_args()
    if args.task == "detect":
        detect(args)


if __name__ == "__main__":
    main()
