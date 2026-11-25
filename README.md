# Proyecto final de Visión por Computador: detectar y contar frutas en árboles

Asignatura: Visión por Computador, curso 2026-27, ETSI Informática (UMA).

**Grupo:** G17. **Miembros:** Darío Sanjuán Romero y Laura Santiago Medina.

> Este README es el documento central del proyecto. Aquí iremos explicando qué hay que hacer, cómo está organizado el repositorio, qué hemos analizado de los datos y qué decisiones técnicas tomamos (y por qué).
>
> **Estado actual:** terminada la fase de estudio. Las decisiones de partida están tomadas (sección 7) y el plan de trabajo está en la sección 6. Empezamos con el código.
>
> Las preguntas que tenemos abiertas (para el profesor o para resolver con experimentos) están en [DUDAS.md](DUDAS.md).

---

## 1. De qué va el proyecto

Los agricultores necesitan saber cuánta fruta tiene cada árbol antes de la cosecha. Contarla a mano es lento, así que la idea es que un robot o vehículo con cámara recorra el huerto, haga fotos de cada árbol y las cuente automáticamente.

Los árboles son **sintéticos** (generados con Blender), así que tenemos etiquetas exactas de cada fruta. Hay cuatro especies:

| `category_id` | Especie |
|---|---|
| 1 | naranja (orange) |
| 2 | manzana (apple) |
| 3 | pera (pear) |
| 4 | limón (lemon) |

El proyecto se hace en parejas y tiene dos partes, cada una con su propia fecha de entrega:

| | Parte 1: detección en imágenes | Parte 2: conteo a partir de una secuencia |
|---|---|---|
| Entrada | una imagen de un árbol | varias imágenes alrededor de un árbol + la pose de la cámara en cada una |
| Salida | para cada fruta visible: **máscara + especie + confianza** | número de frutas de cada especie en el árbol |
| Evaluación | métricas COCO de segmentación (IoU de máscaras) | diferencia entre nuestro conteo y el real |

La **Parte 2 todavía no está publicada** (sale después de la entrega de la Parte 1), así que de momento nos centramos en la Parte 1.

---

## 2. Organización del repositorio

```
trabajo-vc/
├── data/                  material que da el Campus Virtual para hacer el proyecto
│   └── fruits_part1.zip   dataset de la Parte 1 (≈70 MB)
├── info/
│   ├── statement/         enunciado: Introduccion.pdf y Parte 1.pdf
│   └── course content/    contenido de la asignatura
│       ├── TEORIA/        PDFs de teoría (L2–L8)
│       ├── PRACTICAS/     prácticas hechas por nosotros (notebooks + PDFs exportados de Colab)
│       ├── CONTENIDO_TEMAx.md, ALL.md, INDICE.md   resúmenes de cada tema
├── starter_kit/           herramientas oficiales de la Parte 1 (evaluate.py, check_submission.py, visualize.py, example_submission/)
├── team_G17/             la carpeta que se entrega: ver abajo
├── DUDAS.md               preguntas abiertas y cómo se van resolviendo
└── README.md              este documento
```

- `data/`: todo lo que viene del Campus para hacer el proyecto (datos).
- `info/`: el enunciado, nuestras prácticas anteriores y la teoría. Sirve de referencia, no forma parte de la entrega.
- `starter_kit/`: las herramientas oficiales para puntuar y comprobar la entrega. Tampoco se entrega.
- `README.md`: la explicación de todo el proyecto y de las decisiones técnicas. Es nuestro cuaderno de trabajo; la entrega tiene su propio README, más corto.

**Formato de entrega** (según el enunciado): una carpeta `team_<nombre>/` (la nuestra: `team_G17/`, que se sube como `team_G17.zip`) con `README.md`, `requirements.txt` (versiones exactas con `==`), `main.py` y, opcionalmente, `src/` y `weights/`. Tamaño máximo: 500 MB. **No se puede incluir el dataset**, ni entornos virtuales (`.venv/`), ni cachés (`__pycache__/`).

**Cómo vamos a organizar la entrega:**

```
team_G17/
├── README.md          lo que pide el enunciado: miembros, método, entrenamiento, resultados, recursos externos
├── requirements.txt   dependencias con versión exacta
├── main.py            punto de entrada: python main.py detect ...
├── src/               el resto del código: preparar el dataset, entrenar, inferir
│   ├── prepare_dataset.py    separa train/val y convierte las anotaciones a formato YOLO-seg
│   └── check_conversion.py   mide la calidad de esa conversión
└── weights/           el modelo entrenado (.pt)
```

El código de entrenamiento va dentro de `src/` porque el enunciado exige entregarlo. Lo que se genera al trabajar (el dataset descomprimido, el dataset convertido a formato YOLO, las carpetas `runs/` de Ultralytics y el entorno virtual) se queda **fuera** de `team_G17/` y fuera de git, con un `.gitignore`.

---

## 3. Requisitos de la Parte 1 (resumen del enunciado)

**Qué hay que producir:** un único fichero JSON en el *formato de resultados de COCO*, con una entrada por detección:

```json
{"image_id": 1, "category_id": 2, "score": 0.93,
 "segmentation": {"size": [960, 1280], "counts": "..."}}
```

- La máscara va codificada en **RLE** (run-length encoding) con `pycocotools`.
- Se tienen en cuenta como mucho **300 detecciones por imagen**.
- El `score` importa: ordena las detecciones al calcular la AP. Si un método no da confianza, se pone 1.

**Cómo lo ejecutan ellos:**

```
python main.py detect --input <carpeta> --output <predicciones.json> --device cuda
```

La carpeta de entrada tiene `images/` y `images.json` (sin anotaciones). El programa debe terminar con código de salida 0.

**Máquina de evaluación:**

| Recurso | Límite |
|---|---|
| Sistema / Python | Ubuntu 24.04, **Python 3.12** |
| GPU | NVIDIA RTX 2060, **6 GB** |
| CPU / RAM | 4 núcleos, 12 GB |
| Tiempo | **20 minutos para todo el test**, incluida la carga del modelo |
| Internet | **Solo** durante `pip install -r requirements.txt`. Durante la detección no hay, así que los pesos preentrenados tienen que estar en `weights/` |
| Ficheros | No se puede modificar la carpeta de entrada; solo se puede escribir la salida y temporales en `/tmp` |

Si hace falta instalar PyTorch para una versión concreta de CUDA, se añade `--extra-index-url <url>` en `requirements.txt`.

**Métrica principal:** mask AP@[.50:.95]. Es la AP de cada especie con umbrales de IoU 0.50, 0.55, …, 0.95, promediada (4 especies × 10 umbrales = 40 valores). Esto implica que:
- **Las máscaras tienen que ser precisas**: una máscara aproximada solo puntúa en los umbrales bajos.
- **Las cuatro especies pesan lo mismo**, aunque haya más frutas de unas que de otras.
- Una detección con la especie equivocada cuenta como falso positivo, y la fruta real queda como no encontrada.

`evaluate.py` también da AP50, AP75, AP por especie, `AP_agnostic` (sin tener en cuenta la especie), `accuracy` de especie y AP_small / AP_medium / AP_large. Sirven para saber dónde falla el método.

**Normas:** se permite cualquier método (clásico, aprendizaje o mezcla) y se pueden usar modelos preentrenados y librerías públicas, siempre que se citen en el README. Si el método se entrena, hay que incluir el código de entrenamiento y explicar cómo se ejecuta.

**Starter kit:** `evaluate.py`, `check_submission.py`, `visualize.py` y `example_submission/` están en el repositorio de la asignatura `uma_cv_2027`, en la carpeta `Final project/Part 1/`. Los tenemos copiados en `starter_kit/`.

---

## 4. Análisis del dataset de la Parte 1

Datos sacados directamente de `data/fruits_part1.zip` (`train/annotations.json`):

| Dato | Valor |
|---|---|
| Imágenes | 200, todas de 1280×960 |
| Anotaciones totales | 16.002 |
| Fragmentos ignorados (`iscrowd: 1`, menos de 20 píxeles visibles) | 4.926 |
| Frutas que cuentan en la evaluación | 11.076 |
| Por especie | manzana 3.460 · pera 2.863 · limón 2.421 · naranja 2.332 |
| Frutas por imagen | mínimo 10, mediana 57, máximo 110 |
| Frutas físicas distintas (`fruit_id`) | 839 |

**Tamaño de las frutas** (área visible en píxeles, sin contar los fragmentos ignorados):

| Percentil | 10 % | 25 % | 50 % | 75 % | 90 % |
|---|---|---|---|---|---|
| Área (px) | 32 | 61 | 172 | 466 | 1.309 |

Usando los rangos de COCO: **el 87,5 % son *small*** (menos de 32×32 = 1.024 px), el 12,2 % *medium* y solo el 0,3 % *large*. La mediana (172 px) equivale a una fruta de unos **13×13 píxeles** en una imagen de 1280×960.

**Conclusiones del análisis:**

1. **El principal reto son las frutas pequeñas.** Casi todas ocupan muy pocos píxeles y muchas están tapadas en parte por hojas. Si se reduce la imagen (por ejemplo, a 640 px, que es lo típico en YOLO), muchas frutas pasan a medir 2-3 píxeles y es prácticamente imposible segmentarlas bien. Hay que trabajar a alta resolución o por trozos de la imagen.
2. **Cada imagen contiene una sola especie.** Lo hemos comprobado en las 200 imágenes del train, y sabemos que cada árbol es de una sola especie (ver D1 en [DUDAS.md](DUDAS.md)). Esto permite decidir la especie por votación entre todas las detecciones de la imagen. Antes de aplicarlo a todas las detecciones, mediremos en validación cuánto mejora la AP.
3. **Hay 4 árboles, uno por especie, con 50 imágenes cada uno.** Si unimos las imágenes que comparten algún `fruit_id`, salen exactamente 4 grupos de 50 imágenes, cada uno de una sola especie y sin ningún `fruit_id` en común entre grupos. Trabajamos con esto (ver D3 en [DUDAS.md](DUDAS.md)). La consecuencia es importante: **no podemos validar con un árbol que el modelo no haya visto** sin quedarnos sin ejemplos de una especie. La validación tendrá que hacerse con imágenes de los mismos árboles y, por tanto, saldrá **más optimista** que el resultado real en el test (que son árboles distintos). El enunciado ya avisa de que la nota del test suele ser más baja.
4. **Solo cuentan las frutas de 20 píxeles de área o más** (no 20×20). Las de menos de 20 píxeles están anotadas con `iscrowd: 1` y la evaluación las ignora. Aun así, el 71,5 % de las frutas que se evalúan miden menos de 400 px (ver D10 en [DUDAS.md](DUDAS.md)).
5. **La mayoría de las frutas están partidas en trozos.** Por culpa de las hojas y ramas, el 66,8 % de las frutas evaluadas tienen una máscara de 2 o más trozos (el 18,3 %, de 5 o más), y el 19 % tienen algún agujero. Esto importa al convertir las máscaras al formato de YOLO, que usa un único polígono por fruta (ver D6 en [DUDAS.md](DUDAS.md)).

---

## 5. Qué cubre el contenido de la asignatura

Hemos revisado la teoría (L2–L8) y nuestras prácticas para ver qué nos sirve:

| Tema / práctica | Qué aporta al proyecto |
|---|---|
| **7.5 Detección con YOLO** (Ultralytics) | Lo más directo: cargar un modelo preentrenado en COCO, hacer *fine-tuning* con un dataset propio (`train`, `freeze`, `epochs`, `imgsz`), umbral de confianza, mAP e IoU. Pero en la práctica solo se detectan **cajas**, no máscaras. |
| **5.3 SAM** | Genera máscaras **sin clase** (*class-agnostic*) a partir de prompts (puntos o cajas). Podría usarse para refinar máscaras dándole como prompt las cajas de un detector. |
| **6.3 CLIP** | Clasificación *zero-shot* de recortes. Posible alternativa para decidir la especie. |
| **Tema 5** (umbral, Otsu, K-means, HSV) | Enfoque clásico. Útil para entender el problema y como punto de partida, pero lo vemos poco competitivo: las peras son verdes como las hojas y las frutas son diminutas. |
| **Tema 6** (forma y color) | Descriptores (circularidad, color medio…) que podrían ayudar a filtrar falsos positivos. |
| **Tema 8** (deep learning) | Train/validación/test, sobreajuste, *data augmentation*, *transfer learning* y *fine-tuning*. Base teórica para entrenar. |
| **Tema 5, teoría** | Explica qué es la segmentación de instancias y menciona Mask R-CNN, pero **no hay práctica** de segmentación de instancias. |

**Conclusión:** el proyecto se puede hacer, pero **el contenido de la asignatura no basta por sí solo**. Lo más cercano (la práctica 7.5) detecta cajas, y aquí hacen falta máscaras por instancia y además en objetos muy pequeños. Hay que investigar lo que sigue.

---

## 6. Plan de trabajo

El orden está pensado para tener **cuanto antes una versión completa que funcione de principio a fin** (aunque puntúe poco) y mejorarla después midiendo cada cambio.

### Fase 0. Preparar el entorno
1. ✅ Instalar **Python 3.12**, que es la versión con la que se evalúa. Hecho: Python 3.12.10, junto a la 3.14 que ya había. `python` sigue siendo la 3.14; la 3.12 se usa con `py -3.12`.
2. ✅ Crear un entorno virtual e instalar PyTorch con CUDA, Ultralytics y `pycocotools`. Hecho: `.venv` con Python 3.12.10, PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0, Ultralytics 8.4.218, pycocotools 2.0.11 y faster-coco-eval 1.8.1. PyTorch detecta la RTX 3060.
3. ✅ Descomprimir `data/fruits_part1.zip` en `data/` y añadir un `.gitignore` para no subir a git el dataset descomprimido, el convertido, `runs/` ni el entorno virtual.

### Fase 1. Comprobar las herramientas
4. ✅ Ejecutar la `example_submission/` del starter kit sobre el train y puntuarla con `evaluate.py`. No es nuestro método: solo sirve para comprobar que el entorno y la evaluación funcionan, y para tener una nota de referencia mínima. Hecho: AP = 0,0065 (ver sección 8). Para ejecutarla hubo que instalar `scipy` en el entorno.

### Fase 2. Preparar los datos
5. ✅ **Separar train y validación** (D7): 10 imágenes de cada árbol para validación (40 en total) y 160 para entrenar. Los árboles se identifican agrupando las imágenes que comparten `fruit_id`. Los ids de validación se guardan en un fichero (`val_ids.txt`) para usarlo con `evaluate.py --images-file`. Hecho con `src/prepare_dataset.py` (semilla 0): 10 imágenes de validación por especie y ninguna repetida.
6. ✅ **Convertir las anotaciones a formato YOLO-seg** (D6). Incluimos también los fragmentos `iscrowd` como frutas (ver la decisión en la sección 7). Hecho: las 16.002 anotaciones (12.716 en train y 3.286 en validación).
7. ✅ **Medir la calidad de la conversión**: volver a pintar cada polígono como máscara y calcular su IoU con la máscara original. Hecho con `src/check_conversion.py`: `convert_coco` tal cual daba un IoU medio de 0,766 por un problema de redondeo, así que escribimos nuestra propia conversión, que llega a **0,928** (detalle en D6).

### Fase 3. Primera versión completa

> **Estado:** el código está escrito (`src/train.py`, `src/detector.py` y `main.py`), pero todavía no hemos conseguido entrenar: a 1280 px el entrenamiento no cabe en la memoria de la GPU. Las pruebas hechas, la causa y lo que hay que probar están en D12 ([DUDAS.md](DUDAS.md)).
>
> Comandos (desde la raíz del repositorio):
> ```
> .venv\Scripts\python.exe team_G17\src\train.py --data datasets\fruits_yolo\data.yaml --name <nombre> --model yolo11s-seg.pt --batch 2 --epochs 1
> .venv\Scripts\python.exe team_G17\main.py detect --input data\fruits_part1\train --output predictions.json --device cuda
> .venv\Scripts\python.exe starter_kit\evaluate.py --gt data\fruits_part1\train\annotations.json --pred predictions.json --images-file datasets\fruits_yolo\val_ids.txt
> ```
> `main.py` carga el modelo de `team_G17/weights/fruits.pt`: hay que copiarlo ahí desde `runs/<nombre>/weights/best.pt` (o entrenar con `--save-to team_G17\weights\fruits.pt`).

8. Entrenar YOLO-seg con fine-tuning a `imgsz=1280` en la RTX 3060. **Ojo con dos opciones de Ultralytics** que por defecto perjudican a las frutas pequeñas: al entrenar, las máscaras se reducen a 1/4 de resolución (`mask_ratio=4`), y al predecir se devuelven máscaras de baja resolución reescaladas (`retina_masks=False`). Hay que probar `mask_ratio` más bajo y `retina_masks=True`.
9. Escribir `main.py detect`: inferencia → máscaras → RLE → JSON en formato COCO.
10. Puntuar la validación con `evaluate.py` y anotar el resultado en la sección 8.
11. Medir el tiempo por imagen y la memoria de GPU, y pasar `check_submission.py`.

### Fase 4. Mejoras (cada una se mide en validación antes de quedárnosla)
- Votación de la especie por imagen (D1).
- Ajustar el tamaño del modelo, el número de épocas y el umbral de confianza.
- Si la resolución completa no basta para las frutas pequeñas: partir la imagen en trozos solapados (D5).
- Probar a refinar las máscaras con SAM, si el tiempo lo permite.
- Reentrenar con las 200 imágenes para el modelo final, con los ajustes elegidos.

### Parte 2 (cuando se publique)

Lo que ya sabemos: el detector de la Parte 1 es el punto de partida, cada anotación trae un `fruit_id` que identifica la fruta física en todas las imágenes donde aparece, y en la Parte 2 nos darán la pose de la cámara de cada imagen. Todo eso servirá para no contar dos veces la misma fruta vista desde ángulos distintos. El método concreto lo decidiremos cuando tengamos el enunciado.

---

## 7. Decisiones técnicas

Cada decisión indica de qué duda sale (ver [DUDAS.md](DUDAS.md)) y por qué la tomamos. Si un experimento la contradice, se cambia aquí y se explica.

| Decisión | Por qué | Duda |
|---|---|---|
| **Modelo: YOLO de segmentación** (Ultralytics, variantes `-seg`) con fine-tuning desde pesos preentrenados | Es el mismo flujo que la práctica 7.5, pero devuelve una máscara por fruta. Plan B: Mask R-CNN (torchvision). | D4 |
| **Resolución: imagen a tamaño real** (`imgsz=1280`) | Con 640 px la fruta mediana (13×13) se quedaría en unos 6×6. Partir la imagen en trozos solo si esto no basta. | D5 |
| **Conversión de máscaras: propia** (`src/prepare_dataset.py`), basada en la de Ultralytics | `convert_coco` truncaba las coordenadas y desplazaba las máscaras un píxel (IoU medio 0,766). La nuestra hace lo mismo, pero con las coordenadas desplazadas medio píxel, y llega a un IoU medio de 0,928 (0,995 en frutas de un trozo). | D6 |
| **Los fragmentos `iscrowd` se incluyen en el entrenamiento como frutas** | La evaluación ignora las detecciones sobre ellos (ni suman ni restan), así que detectarlos no penaliza. En cambio, si los dejamos fuera, sus píxeles cuentan como fondo y el modelo aprende que una fruta muy pequeña "no es fruta", justo lo contrario de lo que queremos para las de 20-30 px, que sí puntúan. Nuestra conversión los incluye (`convert_coco` se los saltaba). | D6 |
| **Validación: 80/20 por árbol** (10 imágenes de cada árbol) | Las cuatro especies igual de representadas. No se puede separar por árboles (solo hay uno por especie), así que la validación saldrá optimista. | D3, D7 |
| **Especie: una por imagen**, aplicada por votación si mejora la AP en validación | Cada árbol es de una sola especie. | D1 |
| **Entrenamiento en local**, RTX 3060 de 12 GB (plan B: Google Colab) | Sin límites de tiempo ni desconexiones. La GPU de evaluación tiene 6 GB, así que hay que vigilar la memoria de la inferencia. | D8 |
| **Presupuesto de tiempo: ~6 s por imagen**, con margen | Referencia de `check_submission.py`: test de 200 imágenes en 20 minutos, incluida la carga del modelo. | D2 |

## 8. Resultados

Todas las métricas son de `evaluate.py`, el mismo script con el que se puntúa el test.

### Referencia mínima: `example_submission/` del starter kit

Detector por umbral de color (HSV), sin aprendizaje. Al no entrenarse, se puntúa sobre las **200 imágenes del train**. Ejecutado en CPU: 180 s en total, **0,9 s por imagen**.

| Métrica | Valor |
|---|---|
| **AP** (principal) | **0,0065** |
| AP50 / AP75 | 0,0216 / 0,0025 |
| AP naranja / manzana / pera / limón | 0,0034 / 0,0067 / 0,0000 / 0,0160 |
| AP_agnostic | 0,0154 |
| accuracy (especie) | 0,3916 |
| AP_small / AP_medium / AP_large | 0,0026 / 0,0640 / 0,1020 |
| Detecciones / frutas | 15.656 / 11.076 |

**Qué nos dice:**
- La evaluación funciona en nuestra máquina y el formato del JSON es correcto.
- Es la nota que hay que superar con claridad: prácticamente 0.
- Confirma lo que esperábamos del análisis: el umbral de color **no encuentra ninguna pera** (AP 0, porque son verdes como las hojas), acierta la especie solo el 39 % de las veces y falla sobre todo en las frutas pequeñas (AP_small 0,0026 frente a AP_large 0,10).

### Nuestro método

_Pendiente: métricas sobre las 40 imágenes de validación._

## 9. Modelos, datos y librerías externas

El enunciado obliga a citar cada modelo preentrenado, dataset externo o librería no habitual, con su enlace. De momento, previstos:

| Recurso | Para qué | Enlace |
|---|---|---|
| Ultralytics | Entrenar e inferir con YOLO de segmentación | https://github.com/ultralytics/ultralytics |
| Pesos preentrenados de YOLO-seg (COCO) | Punto de partida del fine-tuning. El tamaño concreto del modelo está por decidir | https://docs.ultralytics.com/tasks/segment/ |
| pycocotools | Leer las anotaciones y codificar las máscaras en RLE | https://github.com/cocodataset/cocoapi |

No usamos datasets externos.
