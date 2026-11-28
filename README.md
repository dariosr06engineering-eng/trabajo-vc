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
├── tools/                 scripts de análisis (errores, techos de AP); no forman parte de la entrega
├── team_G17/             la carpeta que se entrega: ver abajo
├── DUDAS.md               preguntas abiertas y cómo se van resolviendo
└── README.md              este documento
```

- `data/`: todo lo que viene del Campus para hacer el proyecto (datos).
- `info/`: el enunciado, nuestras prácticas anteriores y la teoría. Sirve de referencia, no forma parte de la entrega.
- `starter_kit/`: las herramientas oficiales para puntuar y comprobar la entrega. Tampoco se entrega.
- `tools/`: nuestros scripts para analizar los resultados (sección 8). Solo usan la CPU y tampoco se entregan.
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

> **Estado:** ✅ primera versión completa funcionando de principio a fin: **AP = 0,331 en validación** (sección 8).
>
> Comandos (desde la raíz del repositorio):
> ```
> .venv\Scripts\python.exe team_G17\src\prepare_dataset.py --data data\fruits_part1\train --out datasets\fruits_yolo_crop640 --crop 640
> .venv\Scripts\python.exe team_G17\src\train.py --data datasets\fruits_yolo_crop640\data.yaml --name y11s_crop640 --model yolo11s-seg.pt --imgsz 640 --batch 8 --time 0.16 --workers 2 --save-to team_G17\weights\fruits.pt
> cd team_G17
> ..\.venv\Scripts\python.exe main.py detect --input ..\data\fruits_part1\train --output ..\predictions.json --device cuda
> cd ..
> .venv\Scripts\python.exe starter_kit\evaluate.py --gt data\fruits_part1\train\annotations.json --pred predictions.json --images-file datasets\fruits_yolo\val_ids.txt
> ```

8. ✅ Entrenar YOLO-seg con fine-tuning en la RTX 3060. A 1280 px no cabía en memoria (D12), así que **entrenamos con recortes de 640×640 de las imágenes a su resolución original** y detectamos sobre la imagen completa a 1280 (sección 7). Al detectar usamos `retina_masks=True` (máscaras ampliadas a la resolución de la imagen). La resolución interna de las máscaras resultó ser el límite principal: ver la fase 4.
9. ✅ Escribir `main.py detect`: inferencia → máscaras → RLE → JSON en formato COCO.
10. ✅ Puntuar la validación con `evaluate.py` y anotar el resultado en la sección 8.
11. Medir el tiempo por imagen y la memoria de GPU (✅ hecho: 1,14 s por imagen y 1,6 GB de GPU como máximo), y pasar `check_submission.py` (pendiente: falta el `requirements.txt` de la entrega).

### Fase 4. Mejoras (cada una se mide en validación antes de quedárnosla)

**Diagnóstico de la versión 1** (cifras en la sección 8, "Análisis de errores de la versión 1"): el límite actual es la **resolución a la que trabaja YOLO**, no el entrenamiento ni el enunciado.
- YOLO-seg genera las máscaras a **1/4 de la resolución de su entrada**: los prototipos salen de la capa de *stride* 8 ampliada ×2 (`Proto` en `ultralytics/nn/modules/block.py`) y, al detectar, se amplían con interpolación bilineal hasta el tamaño de la imagen (`process_mask_native` en `ultralytics/utils/ops.py`). Una fruta de 13×13 píxeles queda representada en unas 3×3 celdas.
- Simulando un detector perfecto con esa limitación, la AP máxima sería **≈ 0,30**. Nuestro modelo saca 0,331, así que **ya está en ese techo** (por eso la curva de entrenamiento se estanca desde la época 20). Con las máscaras a 1/2 de resolución, el techo sube a **≈ 0,60**.
- La detección tampoco ve bien las frutas diminutas: su capa más fina trabaja a 1/8 de la entrada, y solo encuentra el 13,5 % de las frutas de 20-49 píxeles.
- Lo que **no** limita: la especie (100 % de acierto), el tamaño de las máscaras (ni grandes ni pequeñas) y el formato de polígono único de YOLO (techo 0,81; a 1/2 de resolución su efecto desaparece).

**Plan, por prioridad:**
1. **Experimento ×2: ampliar la imagen al doble.** Preparado, pendiente de ejecutar porque necesita la GPU (detalle abajo).
2. Entrenar más tiempo y, para el modelo final, entrenar con las 200 imágenes (train + validación) con la configuración elegida.
3. `mask_ratio=2` al entrenar. En YOLO11, si las máscaras reales tienen más resolución que los prototipos, la pérdida se calcula ampliando los prototipos (`v8SegmentationLoss` en `ultralytics/utils/loss.py`), así que el modelo aprende bordes más finos. Cuesta más memoria al entrenar.
4. Un modelo más grande (`yolo11m-seg`), vigilando el tiempo de detección.
5. Entrenar sin los fragmentos `iscrowd` (`prepare_dataset.py --no-crowd`), para comprobar si ayudan o molestan.

**Descartado:**
- Votación de la especie: el modelo ya acierta la especie en el 100 % de las detecciones; la AP es la misma.
- Erosionar o dilatar las máscaras 1 píxel: la AP baja de 0,331 a 0,266 y 0,270.
- Mejorar el formato de polígono: no es lo que limita a la resolución a la que trabajamos.
- Refinar las máscaras con SAM: también genera sus máscaras a 1/4 de su entrada (256×256 para 1024×1024), así que no ayudaría con frutas de 13 píxeles.
- Partir la imagen en trozos al detectar (D5): sin ampliar los trozos no cambia la resolución; ampliándolos es lo mismo que el experimento ×2, pero más complicado.

#### Experimento ×2 (preparado, pendiente de ejecutar)

**Idea:** que YOLO vea cada fruta al doble de tamaño. Así las máscaras pasan a 1/2 de la resolución de la imagen original (en vez de 1/4) y las frutas diminutas ocupan celdas enteras en la detección. La entrada y la salida de `main.py` siguen siendo exactamente las del enunciado.

**Cómo:**
- **Entrenamiento:** recortes de 320×320 de las imágenes originales (12 por imagen) que Ultralytics amplía a 640 al cargarlos (`imgsz=640`; `load_image` en `ultralytics/data/base.py` lleva el lado largo de cada imagen a `imgsz`). Cada muestra mide lo mismo que en la versión 1, así que cabe igual en la GPU. El dataset ya está generado: `datasets/fruits_yolo_crop320` (1.920 recortes de train y 480 de validación).
- **Detección:** `main.py detect --imgsz 2560` amplía la imagen de 1280×960 a 2560×1920 (`LetterBox` de Ultralytics amplía con `scaleup=True`), y las máscaras se devuelven al tamaño original. Comprobado en CPU: funciona y las máscaras salen de 960×1280.

**Comandos** (desde la raíz del repositorio):
```
.venv\Scripts\python.exe team_G17\src\train.py --data datasets\fruits_yolo_crop320\data.yaml --name y11s_crop320x2 --model yolo11s-seg.pt --imgsz 640 --batch 8 --time 0.5 --workers 2 --save-to team_G17\weights\fruits_x2.pt
cd team_G17
..\.venv\Scripts\python.exe main.py detect --input ..\data\fruits_part1\train --output ..\predictions_x2.json --device cuda --weights weights\fruits_x2.pt --imgsz 2560
cd ..
.venv\Scripts\python.exe starter_kit\evaluate.py --gt data\fruits_part1\train\annotations.json --pred predictions_x2.json --images-file datasets\fruits_yolo\val_ids.txt
.venv\Scripts\python.exe tools\analisis_errores.py --pred predictions_x2.json
```
Con 12 recortes por imagen, cada época tiene el triple de iteraciones que en la versión 1; con `--time 0.5` (30 minutos) salen unas 20-25 épocas.

**Qué medir y cómo decidir:**
1. La AP en validación frente a 0,331, y en especial la de las frutas pequeñas (`AP_small` y la tabla por tamaños de `tools/analisis_errores.py`).
2. El tiempo por imagen y el pico de memoria de GPU, que `main.py` imprime al terminar. Al detectar se procesa el cuádruple de píxeles, y tiene que caber con margen en la máquina de evaluación (RTX 2060 de 6 GB, 20 minutos para todo el test).
3. Si mejora y cabe, pasa a ser la configuración por defecto: `IMGSZ = 2560` en `src/detector.py` y el nuevo modelo como `weights/fruits.pt`.

### Parte 2 (cuando se publique)

Lo que ya sabemos: el detector de la Parte 1 es el punto de partida, cada anotación trae un `fruit_id` que identifica la fruta física en todas las imágenes donde aparece, y en la Parte 2 nos darán la pose de la cámara de cada imagen. Todo eso servirá para no contar dos veces la misma fruta vista desde ángulos distintos. El método concreto lo decidiremos cuando tengamos el enunciado.

---

## 7. Decisiones técnicas

Cada decisión indica de qué duda sale (ver [DUDAS.md](DUDAS.md)) y por qué la tomamos. Si un experimento la contradice, se cambia aquí y se explica.

| Decisión | Por qué | Duda |
|---|---|---|
| **Modelo: YOLO11s-seg** (Ultralytics) con fine-tuning desde pesos preentrenados en COCO | Es el mismo flujo que la práctica 7.5, pero devuelve una máscara por fruta. Empezamos con YOLO26-seg (la familia de la práctica), pero su entrenamiento calcula muchísimas máscaras por fruta y no cabe en memoria con tantas frutas por imagen (D12); YOLO11-seg no tiene ese problema. | D4, D12 |
| **Resolución nativa: entrenar con recortes de 640×640 y detectar sobre la imagen completa a 1280** | Con la imagen reducida a 640 la fruta mediana (13×13) se quedaría en unos 6×6. Entrenar con la imagen completa a 1280 no cabe en memoria (D12). Con recortes sin reducir, las frutas tienen en el entrenamiento el mismo tamaño en píxeles que al detectar, y cada recorte tiene una cuarta parte de las frutas. YOLO acepta cualquier tamaño de entrada, así que al detectar se usa la imagen entera. | D5, D12 |
| **Conversión de máscaras: propia** (`src/prepare_dataset.py`), basada en la de Ultralytics | `convert_coco` truncaba las coordenadas y desplazaba las máscaras un píxel (IoU medio 0,766). La nuestra hace lo mismo, pero con las coordenadas desplazadas medio píxel, y llega a un IoU medio de 0,928 (0,995 en frutas de un trozo). | D6 |
| **Los fragmentos `iscrowd` se incluyen en el entrenamiento como frutas** | La evaluación ignora las detecciones sobre ellos (ni suman ni restan), así que detectarlos no penaliza. En cambio, si los dejamos fuera, sus píxeles cuentan como fondo y el modelo aprende que una fruta muy pequeña "no es fruta", justo lo contrario de lo que queremos para las de 20-30 px, que sí puntúan. Nuestra conversión los incluye (`convert_coco` se los saltaba). | D6 |
| **Validación: 80/20 por árbol** (10 imágenes de cada árbol) | Las cuatro especies igual de representadas. No se puede separar por árboles (solo hay uno por especie), así que la validación saldrá optimista. | D3, D7 |
| **Especie: sin votación** (la opción `--species-vote` queda disponible) | Cada árbol es de una sola especie, pero el modelo ya acierta la especie en el 100 % de las detecciones de validación: la votación da exactamente la misma AP. | D1 |
| **Entrenamiento en local**, RTX 3060 de 12 GB (plan B: Google Colab) | Sin límites de tiempo ni desconexiones. La GPU de evaluación tiene 6 GB, así que hay que vigilar la memoria de la inferencia. | D8 |
| **Presupuesto de tiempo: ~6 s por imagen**, con margen | Referencia de `check_submission.py`: test de 200 imágenes en 20 minutos, incluida la carga del modelo. Ahora gastamos 1,14 s por imagen en la RTX 3060. | D2 |
| **Inferencia: `conf=0.001`, hasta 300 detecciones, máscaras a resolución completa (`retina_masks=True`), media precisión en GPU** | La AP premia devolver muchas detecciones ordenadas por confianza, y la evaluación tiene en cuenta hasta 300 por imagen. Las máscaras a resolución completa son imprescindibles con frutas tan pequeñas. | D5 |
| **Inferencia con `rect=True`**: la imagen se rellena solo lo mínimo (1280×960) y no hasta un cuadrado (1280×1280) | Ahorra cómputo: a 1280, un 25 % de los píxeles procesados eran relleno, y lo mismo pasaría a 2560 (2560×2560 en vez de 2560×1920). Comprobado en CPU sobre la validación: AP 0,3305 frente a 0,3313 sin `rect`, una diferencia que entra en lo normal (la prueba anterior fue en GPU con media precisión). | D2 |
| **Entrenamiento con 2 *workers*** | En Windows, con los 8 por defecto, algunos procesos fallaban al cargar las DLL de PyTorch (`WinError 1114`) y el entrenamiento se quedaba colgado. | D12 |

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

### Nuestro método: versión 1 (YOLO11s-seg, recortes de 640, 10 minutos de entrenamiento)

Entrenado con las 160 imágenes de train (640 recortes de 640×640) durante 10 minutos (34 épocas) en la RTX 3060. Puntuado con `evaluate.py` sobre las **40 imágenes de validación**, a imagen completa:

| Métrica | Valor |
|---|---|
| **AP** (principal) | **0,331** |
| AP50 / AP75 | 0,605 / 0,304 |
| AP naranja / manzana / pera / limón | 0,308 / 0,442 / 0,263 / 0,313 |
| AP_agnostic | 0,338 |
| accuracy (especie) | **1,000** |
| AP_small / AP_medium / AP_large | **0,254** / 0,825 / 0,953 |
| Detecciones / frutas | 11.846 / 2.315 |

Ejecución completa de `main.py detect` sobre 200 imágenes en la RTX 3060: **229 s (1,14 s por imagen, incluida la carga del modelo)**. Pico de memoria de GPU: 1,6 GB (límite de la máquina de evaluación: 6 GB).

**Qué nos dice:**
- Pasamos de 0,0065 a 0,331: el método funciona.
- **La especie no es un problema**: el 100 % de las detecciones que encuentran una fruta aciertan su especie, y la AP sin tener en cuenta la especie (0,338) es casi igual que la AP. Por eso la votación por imagen no cambia nada.
- **El problema son las frutas pequeñas**: AP 0,25 en las *small*, frente a 0,83 en las *medium* y 0,95 en las *large*. Como el 87,5 % de las frutas son pequeñas, ahí está casi todo el margen de mejora.
- Recordatorio: la validación es de los mismos árboles que el entrenamiento, así que en el test saldrá algo más bajo (D3).

### Análisis de errores de la versión 1

Con `tools/analisis_errores.py` y `tools/techos_resolucion.py` (solo CPU), sobre las 40 imágenes de validación.

**Frutas encontradas, por tamaño** (mejor IoU de cada fruta con alguna detección de su especie):

| Área (píxeles) | Frutas | Encontradas (IoU ≥ 0,5) | Máscara precisa (IoU ≥ 0,75) | IoU medio de las encontradas |
|---|---|---|---|---|
| 20-49 | 422 | 13,5 % | 0,5 % | 0,590 |
| 50-99 | 344 | 37,5 % | 1,7 % | 0,612 |
| 100-199 | 400 | 64,5 % | 9,2 % | 0,639 |
| 200-399 | 425 | 90,4 % | 36,0 % | 0,721 |
| 400-1.023 | 448 | 98,7 % | 77,2 % | 0,823 |
| ≥ 1.024 | 276 | 100 % | 95,7 % | 0,906 |

El área de las máscaras predichas es la correcta en todos los tamaños (mediana del área predicha / área real entre 0,98 y 1,05). Por eso erosionarlas o dilatarlas 1 píxel empeora la AP (0,266 y 0,270).

**Por número de trozos de la fruta** (por las hojas que la tapan): con 1 trozo encuentra el 85,9 % (65,8 % con IoU ≥ 0,75); con 5 o más, el 44,3 % (5,8 %).

**Techos de AP**: la AP que sacaría un detector perfecto (encuentra todas las frutas y acierta la especie) si sus máscaras tuvieran una limitación.

| Limitación | AP | AP50 | AP75 | AP_small |
|---|---|---|---|---|
| Ninguna (máscaras reales) | 1,000 | 1,000 | 1,000 | 1,000 |
| Nuestras etiquetas YOLO (un polígono por fruta) | 0,813 | 0,983 | 0,868 | 0,792 |
| Máscaras a 1/2 de resolución (YOLO con la imagen ×2) | 0,603 | 0,902 | 0,636 | 0,560 |
| Máscaras a 1/4 de resolución (YOLO ahora) | 0,296 | 0,555 | 0,264 | 0,234 |
| *Nuestro modelo, versión 1* | *0,331* | *0,605* | *0,304* | *0,254* |

La reducción de resolución se simula con la media de cada bloque de píxeles y una ampliación bilineal. Es una aproximación: la red puede hacerlo algo mejor, por eso nuestro modelo supera ligeramente su techo de 1/4. Aun así, la diferencia entre 1/4 y 1/2 es muy grande.

**Curva de entrenamiento**: la AP de máscara que calcula Ultralytics en validación (sobre los recortes) pasa de 0,201 en la época 10 a 0,224 en la 20 y a 0,227 en la 34. Se estanca aunque la pérdida siga bajando, como cabe esperar si el límite es la resolución.

## 9. Modelos, datos y librerías externas

El enunciado obliga a citar cada modelo preentrenado, dataset externo o librería no habitual, con su enlace. De momento, previstos:

| Recurso | Para qué | Enlace |
|---|---|---|
| Ultralytics | Entrenar e inferir con YOLO de segmentación | https://github.com/ultralytics/ultralytics |
| Pesos preentrenados `yolo11s-seg.pt` (COCO) | Punto de partida del fine-tuning | https://docs.ultralytics.com/models/yolo11/ |
| pycocotools | Leer las anotaciones y codificar las máscaras en RLE | https://github.com/cocodataset/cocoapi |

No usamos datasets externos.
