# Proyecto final de Visión por Computador: detectar y contar frutas en árboles

Asignatura: Visión por Computador, curso 2026-27, ETSI Informática (UMA).

**Miembros del grupo:** Darío Sanjuán Romero y Laura Santiago Medina.

> Este README es el documento central del proyecto. Aquí iremos explicando qué hay que hacer, cómo está organizado el repositorio, qué hemos analizado de los datos y qué decisiones técnicas tomamos (y por qué).
>
> **Estado actual:** fase de estudio. Todavía no hay código. Lo que aparece en la sección 6 es un **plan propuesto**, todavía sin validar con experimentos.
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
├── DUDAS.md               preguntas abiertas y cómo se van resolviendo
└── README.md              este documento
```

- `data/`: todo lo que viene del Campus para hacer el proyecto (datos).
- `info/`: el enunciado, nuestras prácticas anteriores y la teoría. Sirve de referencia, no forma parte de la entrega.
- `README.md`: la explicación de todo el proyecto y de las decisiones técnicas.

**Formato de entrega** (según el enunciado): una carpeta `team_<nombre>/` con `README.md`, `requirements.txt` (versiones exactas con `==`), `main.py` y, opcionalmente, `src/` y `weights/`. Tamaño máximo: 500 MB. **No se puede incluir el dataset**, ni entornos virtuales (`.venv/`), ni cachés (`__pycache__/`). Cuando empecemos a programar habrá que crear esa carpeta y un `.gitignore`.

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

## 6. Plan propuesto (pendiente de validar)

### Enfoque principal: YOLO de segmentación con fine-tuning

Ultralytics, la librería de la práctica 7.5, también tiene modelos de **segmentación de instancias** (variantes `-seg`), que devuelven caja + clase + máscara para cada objeto. El flujo de trabajo sería el mismo que en la práctica, solo cambiando el tipo de modelo. Por eso lo vemos como la opción más razonable para empezar.

Pasos previstos:
1. Conseguir el starter kit y comprobar que `evaluate.py` funciona con las anotaciones.
2. Convertir las anotaciones de COCO (máscaras en RLE) al formato que pide YOLO-seg (polígonos normalizados). Ojo con las frutas que una hoja parte en varios trozos.
3. Separar train/validación (ver la limitación del apartado 4).
4. Hacer fine-tuning partiendo de pesos preentrenados.
5. Inferencia y escritura del JSON en formato COCO con `pycocotools`.
6. Medir con `evaluate.py` y comprobar la entrega con `check_submission.py`.

### Cosas que hay que investigar

- **Objetos pequeños:** entrenar e inferir a más resolución (`imgsz` alto) y/o dividir la imagen en trozos (*tiling*, por ejemplo con la librería SAHI) y luego juntar las detecciones. Hay que comprobar que cabe en 6 GB de GPU y en 20 minutos.
- **Tiempo de ejecución:** no sabemos cuántas imágenes tiene el test. Habrá que medir cuánto tarda nuestro método por imagen para no pasarnos.
- **Especie por imagen:** cada árbol es de una sola especie, así que la especie se puede decidir por votación en toda la imagen. Hay que medir en validación cuánto mejora la AP.
- **Refinar máscaras con SAM:** probar si mejora la AP en los umbrales altos de IoU (AP75), teniendo en cuenta que SAM es pesado (el `vit_h` de la práctica ocupa mucha memoria).
- **Alternativa:** Mask R-CNN (torchvision), si YOLO-seg no funcionara bien.
- **Entorno:** fijar versiones compatibles con Python 3.12 y CUDA, guardar los pesos en `weights/` y comprobar que nada se descarga durante la ejecución.
- **Dónde entrenar:** Google Colab u otra máquina con GPU.

### Parte 2 (cuando se publique)

Lo que ya sabemos: el detector de la Parte 1 es el punto de partida, cada anotación trae un `fruit_id` que identifica la fruta física en todas las imágenes donde aparece, y en la Parte 2 nos darán la pose de la cámara de cada imagen. Todo eso servirá para no contar dos veces la misma fruta vista desde ángulos distintos. El método concreto lo decidiremos cuando tengamos el enunciado.

---

## 7. Decisiones técnicas

_Se irán añadiendo a medida que hagamos experimentos, cada una con su justificación y los resultados obtenidos._

## 8. Resultados

_Pendiente. Aquí irán las métricas de `evaluate.py` sobre nuestras imágenes de validación._

## 9. Modelos, datos y librerías externas

_Pendiente. El enunciado obliga a citar cada modelo preentrenado, dataset externo o librería no habitual, con su enlace._
