"""Prepara el dataset de la Parte 1 para entrenar YOLO de segmentación.

    python src/prepare_dataset.py --data ../data/fruits_part1/train --out ../datasets/fruits_yolo

Pasos:
  1. Agrupa las imágenes por árbol: dos imágenes son del mismo árbol si comparten
     algún fruit_id (la misma fruta física vista en las dos).
  2. Separa train y validación: --val-per-tree imágenes de cada árbol van a
     validación (por defecto 10 de 50, es decir, 80/20), elegidas al azar con
     una semilla fija para que la separación sea siempre la misma.
  3. Convierte las anotaciones COCO (máscaras RLE) al formato de YOLO-seg: un
     polígono por fruta. Hace lo mismo que convert_coco de Ultralytics (contornos
     exteriores de la máscara y, si la fruta está partida en trozos, los une con
     merge_multi_segment de Ultralytics), con dos diferencias:
       - Usa también los fragmentos con iscrowd = 1 (menos de 20 píxeles), que
         convert_coco se salta. Si no, sus píxeles contarían como fondo.
       - Escribe cada punto del contorno desplazado medio píxel (x + 0.5). Al
         entrenar, Ultralytics vuelve a pintar el polígono truncando las
         coordenadas a entero; sin el desplazamiento, valores como 99.99996 se
         truncan a 99 y la máscara se mueve un píxel, lo que en frutas de 10x10
         píxeles baja el IoU de 0.995 a 0.75 (medido con check_conversion.py).
  4. Copia las imágenes y escribe data.yaml.

Resultado en --out:
    images/train/  images/val/     las imágenes
    labels/train/  labels/val/     un .txt por imagen: <clase> x1 y1 x2 y2 ... (normalizado)
    data.yaml                      el fichero que recibe model.train(data=...)
    train_ids.txt  val_ids.txt     los image_id de cada parte (val_ids.txt sirve para evaluate.py --images-file)
"""

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from ultralytics.data.converter import merge_multi_segment


def group_by_tree(annotations, image_ids):
    """Agrupa las imágenes que comparten algún fruit_id (union-find). Devuelve una lista de listas de image_id."""
    parent = {i: i for i in image_ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    images_of_fruit = defaultdict(set)
    for a in annotations:
        images_of_fruit[a["fruit_id"]].add(a["image_id"])
    for imgs in images_of_fruit.values():
        first, *rest = imgs
        for i in rest:
            parent[find(i)] = find(first)

    trees = defaultdict(list)
    for i in image_ids:
        trees[find(i)].append(i)
    return sorted((sorted(t) for t in trees.values()), key=lambda t: t[0])


def split_train_val(trees, val_per_tree, seed):
    rng = random.Random(seed)
    train, val = [], []
    for tree in trees:
        chosen = set(rng.sample(tree, val_per_tree))
        val += [i for i in tree if i in chosen]
        train += [i for i in tree if i not in chosen]
    return sorted(train), sorted(val)


def annotation_to_polygon(ann, h, w):
    """Polígono YOLO (lista x1 y1 x2 y2 ... normalizada) de una anotación con la máscara en RLE."""
    mask = mask_utils.decode(ann["segmentation"])
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    polygons = [c.reshape(-1).tolist() for c in contours if len(c) >= 3]   # YOLO necesita al menos 3 puntos
    if not polygons:
        # Fragmento diminuto (1-2 píxeles en línea): se usa el rectángulo de sus píxeles
        x, y, bw, bh = ann["bbox"]
        pts = np.array([[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]], dtype=np.float64)
    elif len(polygons) > 1:
        pts = np.concatenate(merge_multi_segment(polygons), axis=0).astype(np.float64)
    else:
        pts = np.array(polygons[0], dtype=np.float64).reshape(-1, 2)
    pts = (pts + 0.5) / [w, h]       # +0.5: ver el punto 3 de la explicación del principio
    return np.clip(pts, 0.0, 1.0).reshape(-1).tolist()


def write_labels(coco, ids, labels_dir, include_crowd):
    """Un .txt por imagen con una línea por fruta: <clase> x1 y1 x2 y2 ..."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    anns_of = defaultdict(list)
    for a in coco["annotations"]:
        anns_of[a["image_id"]].append(a)
    images = {im["id"]: im for im in coco["images"]}
    n = 0
    for i in ids:
        im = images[i]
        lines = []
        for a in anns_of[i]:
            if a["iscrowd"] and not include_crowd:
                continue
            poly = annotation_to_polygon(a, im["height"], im["width"])
            lines.append(" ".join([str(a["category_id"] - 1)] + [f"{v:.6f}" for v in poly]))
        (labels_dir / f"{Path(im['file_name']).stem}.txt").write_text("\n".join(lines) + "\n")
        n += len(lines)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True, help="carpeta del train: images/ y annotations.json")
    p.add_argument("--out", type=Path, required=True, help="carpeta de salida del dataset en formato YOLO")
    p.add_argument("--val-per-tree", type=int, default=10, help="imágenes de cada árbol para validación (10)")
    p.add_argument("--seed", type=int, default=0, help="semilla de la separación train/val (0)")
    p.add_argument("--no-crowd", action="store_true", help="no usar los fragmentos iscrowd para entrenar")
    p.add_argument("--overwrite", action="store_true", help="borrar --out si ya existe")
    args = p.parse_args()

    coco = json.loads((args.data / "annotations.json").read_text(encoding="utf-8"))
    image_ids = [im["id"] for im in coco["images"]]
    trees = group_by_tree(coco["annotations"], image_ids)
    train_ids, val_ids = split_train_val(trees, args.val_per_tree, args.seed)
    print(f"{len(trees)} árboles ({', '.join(str(len(t)) for t in trees)} imágenes); "
          f"train {len(train_ids)}, val {len(val_ids)}")

    if args.out.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out} ya existe: usa --overwrite para rehacerlo")
        shutil.rmtree(args.out)

    for name, ids in (("train", train_ids), ("val", val_ids)):
        n = write_labels(coco, ids, args.out / "labels" / name, include_crowd=not args.no_crowd)
        print(f"{name}: {n} frutas convertidas")

    file_of = {im["id"]: im["file_name"] for im in coco["images"]}
    for name, ids in (("train", train_ids), ("val", val_ids)):
        dst = args.out / "images" / name
        dst.mkdir(parents=True, exist_ok=True)
        for i in ids:
            shutil.copy2(args.data / "images" / file_of[i], dst / file_of[i])
        (args.out / f"{name}_ids.txt").write_text("\n".join(map(str, ids)) + "\n")

    # Las clases de YOLO empiezan en 0: category_id 1..4 -> 0..3
    names = "\n".join(f"  {c['id'] - 1}: {c['name']}" for c in sorted(coco["categories"], key=lambda c: c["id"]))
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"Dataset YOLO en {args.out.resolve()}")


if __name__ == "__main__":
    main()
