"""Entrena YOLO de segmentación con el dataset preparado por prepare_dataset.py.

    python src/train.py --data ../datasets/fruits_yolo/data.yaml --name y26s_1280

Parte de unos pesos preentrenados en COCO (fine-tuning) y guarda el entrenamiento
en ../runs/<name>/. El mejor modelo (según la validación) queda en
../runs/<name>/weights/best.pt; con --save-to se copia además a otra ruta, por
ejemplo weights/fruits.pt, que es el que carga main.py.

Opciones principales:
    --model       pesos de partida (yolo26s-seg.pt). Si no están, Ultralytics los descarga
    --imgsz       tamaño de entrada (1280: las imágenes son de 1280x960, así no se reducen)
    --epochs      número de épocas (100)
    --batch       imágenes por iteración (4)
    --mask-ratio  reducción de las máscaras al entrenar (Ultralytics usa 4 por defecto)
    --mosaic      probabilidad de usar mosaic (juntar 4 imágenes en una). 1.0 por defecto
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]      # carpeta del repositorio (encima de team_G17/)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True, help="data.yaml del dataset en formato YOLO")
    p.add_argument("--name", required=True, help="nombre del entrenamiento (carpeta dentro de runs/)")
    p.add_argument("--model", default="yolo26s-seg.pt", help="pesos preentrenados de partida")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--mask-ratio", type=int, default=4)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--device", default="0", help="GPU a usar (0) o cpu")
    p.add_argument("--save-to", type=Path, help="copiar aquí el mejor modelo al terminar")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        mask_ratio=args.mask_ratio,
        mosaic=args.mosaic,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
    )
    best = Path(model.trainer.best)
    print(f"Mejor modelo: {best}")
    if args.save_to:
        args.save_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, args.save_to)
        print(f"Copiado a {args.save_to}")


if __name__ == "__main__":
    main()
