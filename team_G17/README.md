# Grupo G17. Parte 1: detección de frutas

**Miembros:** Darío Sanjuán Romero y Laura Santiago Medina.

```
python main.py detect --input <carpeta> --output <predicciones.json> --device cuda
```

## Método

Segmentación de instancias con **YOLO11s-seg** (Ultralytics), ajustado (*fine-tuning*) con nuestras imágenes a partir de los pesos preentrenados en COCO. Para cada imagen, el modelo da directamente la máscara, la especie y la confianza de cada fruta.

**Por qué YOLO:** es el mismo flujo de trabajo que la práctica 7.5 de la asignatura (cargar un modelo preentrenado, ajustarlo y predecir), pero en su variante de segmentación, que devuelve una máscara por objeto. Además es rápido y ligero, así que cabe con margen en la máquina de evaluación (RTX 2060 de 6 GB, 20 minutos).

**Decisiones principales**, tomadas a partir del análisis de los datos:
- **Resolución nativa.** El 87,5 % de las frutas miden menos de 32×32 píxeles (la mediana, unos 13×13). Si se redujera la imagen, muchas desaparecerían, así que se detecta sobre la imagen completa a su tamaño original (`imgsz=1280`).
- **Entrenamiento con recortes.** Con la imagen completa a 1280 el entrenamiento no cabía en la memoria de la GPU (hay hasta 110 frutas por imagen). Entrenamos con recortes de 640×640 de las imágenes, **sin reducirlas**, de modo que las frutas tienen el mismo tamaño en píxeles al entrenar que al detectar.
- **Conversión propia de las máscaras** (COCO RLE → polígono de YOLO), en `src/prepare_dataset.py`. Hace lo mismo que la de Ultralytics, pero corrige un desplazamiento de un píxel que esta introducía al volver a pintar los polígonos. Así, el IoU medio entre las etiquetas convertidas y las originales pasa de 0,766 a 0,928. Las frutas que una hoja parte en varios trozos se unen en un solo polígono, como exige el formato de YOLO.
- **Los fragmentos `iscrowd` (menos de 20 píxeles) se usan al entrenar** como frutas: la evaluación los ignora, pero si se dejaran fuera, el modelo aprendería que las frutas muy pequeñas son fondo.
- **Al detectar:** umbral de confianza muy bajo (0,001) y hasta 300 detecciones por imagen, porque la AP premia devolver muchas detecciones ordenadas por confianza; máscaras a la resolución de la imagen (`retina_masks=True`); y media precisión en la GPU.
- **Sin internet y escribiendo solo en `/tmp`:** `main.py` desactiva las comprobaciones de conexión y las instalaciones automáticas de Ultralytics, y manda su fichero de configuración a la carpeta temporal.

## Entrenamiento

**Datos:** solo el train de `fruits_part1` del Campus Virtual (200 imágenes). Ningún dataset externo.

**Validación:** las 200 imágenes son de 4 árboles (uno por especie), que identificamos porque las imágenes del mismo árbol comparten `fruit_id`. Separamos 10 imágenes de cada árbol para validación (40) y usamos las otras 160 para entrenar. Como no hay árboles distintos con los que validar, la validación sale algo optimista respecto al test.

**Cómo reproducirlo** (desde esta carpeta):

```
python src/prepare_dataset.py --data <fruits_part1>/train --out <dataset> --crop 640
python src/train.py --data <dataset>/data.yaml --name y11s_crop640 --model yolo11s-seg.pt --imgsz 640 --batch 8 --time 0.16 --workers 2 --save-to weights/fruits.pt
```

- `prepare_dataset.py` separa train y validación (`--val-per-tree 10`, `--seed 0`), convierte las anotaciones a polígonos de YOLO y guarda los recortes (`--crop 640`; sin esa opción guarda las imágenes completas). También escribe `val_ids.txt`, para puntuar solo la validación con `evaluate.py --images-file`.
- `train.py` parte de `yolo11s-seg.pt` (Ultralytics lo descarga la primera vez) y entrena durante `--time` horas (0,16 h ≈ 10 minutos, 34 épocas en una RTX 3060) con `--batch 8` e `--imgsz 640`; el resto de opciones son las de Ultralytics por defecto (aumentos de datos incluidos). Guarda el entrenamiento en `../runs/<name>/` y el mejor modelo en `../runs/<name>/weights/best.pt`, que `--save-to` copia a `weights/fruits.pt`, el modelo que carga `main.py`.
- `check_conversion.py` mide la calidad de la conversión de las máscaras.

## Resultados en validación

Con `evaluate.py` del starter kit, sobre las 40 imágenes de validación:

| AP | AP50 | AP75 | AP naranja / manzana / pera / limón | AP_agnostic | accuracy | AP_small / medium / large |
|---|---|---|---|---|---|---|
| **0,331** | 0,605 | 0,304 | 0,308 / 0,442 / 0,263 / 0,313 | 0,338 | 1,000 | 0,254 / 0,825 / 0,953 |

La especie se acierta siempre; el margen de mejora está en las frutas más pequeñas. En una RTX 3060, `main.py detect` tarda unos 1,1 s por imagen (incluida la carga del modelo) y usa como máximo 1,6 GB de memoria de GPU.

## Recursos externos

- **Ultralytics** 8.4.218: entrenamiento e inferencia de YOLO. https://github.com/ultralytics/ultralytics
- **Pesos preentrenados `yolo11s-seg.pt`** (COCO), de Ultralytics: punto de partida del ajuste. https://docs.ultralytics.com/models/yolo11/
- **PyTorch** y **torchvision**. https://pytorch.org
- **pycocotools**: lectura de las anotaciones y codificación RLE de las máscaras. https://github.com/cocodataset/cocoapi
- **OpenCV** y **NumPy**: contornos y procesado de las máscaras. https://opencv.org · https://numpy.org

No usamos datasets externos.
