"""Detector de frutas de la Parte 1: YOLO de segmentación ajustado con nuestro dataset.

Para cada imagen devuelve una lista de (máscara, category_id, score), el formato
que pide el enunciado.
"""

from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "fruits.pt"

IMGSZ = 1280           # las imágenes son de 1280x960: así no se reducen
CONF = 0.001           # umbral muy bajo: la AP premia devolver muchas detecciones ordenadas por score
MAX_DET = 300          # máximo que tiene en cuenta la evaluación por imagen


class FruitDetector:
    def __init__(self, device="cuda", weights=WEIGHTS, species_vote=False):
        self.device = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.half = self.device != "cpu"           # media precisión en GPU: más rápido y menos memoria
        self.model = YOLO(str(weights), task="segment")
        self.species_vote = species_vote

    def detect(self, image_path):
        """Lista de (mask, category_id, score). mask: array booleano (alto, ancho) del tamaño de la imagen."""
        r = self.model.predict(
            str(image_path), imgsz=IMGSZ, conf=CONF, max_det=MAX_DET, device=self.device, half=self.half,
            retina_masks=True,             # máscaras a la resolución de la imagen, no a 1/4
            verbose=False,
        )[0]
        if r.masks is None or len(r.boxes) == 0:
            return []
        masks = (r.masks.data > 0.5).cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy().astype(float)
        if self.species_vote:
            classes = vote_species(classes, scores)
        return [(m, int(c) + 1, float(s))          # clases de YOLO 0..3 -> category_id 1..4
                for m, c, s in zip(masks, classes, scores) if m.any()]


def vote_species(classes, scores):
    """Una sola especie por imagen: la que más confianza acumula entre todas las detecciones."""
    totals = np.bincount(classes, weights=scores, minlength=4)
    return np.full_like(classes, int(np.argmax(totals)))
