# Dudas del proyecto

Preguntas que tenemos abiertas sobre el proyecto. Cada una indica **por qué importa** y **cómo pensamos resolverla**: preguntando al profesor o comprobándolo nosotros con los datos o con experimentos.

Cuando una duda se resuelva, se marca como resuelta, se apunta la respuesta y, si afecta al método, se refleja en el [README.md](README.md).

**Estados:** 🔴 abierta · 🟢 resuelta

---

## Para preguntar al profesor

### D1. ¿En el test, cada imagen tiene una sola especie? 🟢
- **Qué sabemos:** en las 200 imágenes del train, todas las frutas de una imagen son de la misma especie (comprobado con `annotations.json`). El enunciado no lo dice explícitamente, y el formato de entrada y salida tampoco: la especie va en cada detección porque es el formato estándar de COCO.
- **Por qué importa:** si se cumple, podemos decidir la especie por votación entre todas las detecciones de la imagen, y eso eliminaría casi todos los errores de especie. Si no se cumple y lo usamos, las imágenes con varias especies saldrían mal.
- **Respuesta:** **sí, cada árbol es de una sola especie.** Nos lo ha confirmado un compañero ("cada árbol es solo de una fruta"), y encaja con el enunciado:
  - cada árbol crece "siguiendo los hábitos de su especie";
  - las imágenes se toman alrededor de un único árbol;
  - el fondo son cielos fotografiados en 360° (campos, colinas, nubes), no otros árboles.
- **Consecuencia:** podemos diseñar el método asumiendo una especie por imagen. Como la confirmación no viene por escrito del profesor, primero mediremos en validación si la votación por imagen mejora la AP antes de aplicarla a todas las detecciones.

### D2. ¿Cuántas imágenes tiene el test? 🔴
- **Qué sabemos:** hay 20 minutos para todo el test, incluida la carga del modelo, en una RTX 2060 de 6 GB. El enunciado no da el número de imágenes. El `check_submission.py` del starter kit estima el tiempo suponiendo un test **del mismo tamaño que el train (200 imágenes)**, y avisa si con esa suposición nos pasaríamos de 20 minutos.
- **Por qué importa:** determina cuánto tiempo podemos gastar por imagen (por ejemplo, si podemos permitirnos inferencia por trozos o refinar con SAM).
- **Mientras tanto:** usamos como referencia la del kit: 200 imágenes, es decir, unos **6 segundos por imagen** incluida la carga del modelo, y dejamos margen porque nuestra máquina no es la suya.
- **Cómo resolverla:** confirmar el número con el profesor.

### D3. ¿Las imágenes del train son de 4 árboles (uno por especie)? 🟢
- **Qué sabemos:** si unimos las imágenes que comparten algún `fruit_id`, salen exactamente 4 grupos de 50 imágenes, cada uno de una sola especie y sin frutas en común. Los datos no traen un identificador de árbol.
- **Por qué importa:** si solo hay un árbol por especie, no podemos validar con un árbol que el modelo no haya visto, y nuestra validación será optimista respecto al test (que tiene árboles distintos).
- **Respuesta:** **lo tomamos como 4 árboles, uno por especie.** Los únicos datos que tenemos son los que nos han dado, y en ellos salen 4 grupos. Trabajamos con eso.
- **Consecuencia:** la validación se hará con imágenes de los mismos 4 árboles que el entrenamiento, así que la puntuación de validación será más alta que la del test. Lo tendremos en cuenta al interpretar los resultados (ver D7).

---

## Para resolver nosotros

### D4. ¿Qué modelo de segmentación de instancias usamos? 🟢
- **Respuesta:** **YOLO de segmentación** (variantes `-seg` de Ultralytics) con fine-tuning.
- **Por qué:** es la misma librería y el mismo flujo de trabajo que la práctica 7.5 (cargar un modelo preentrenado, `train`, `predict`), pero devuelve una máscara por objeto, que es lo que pide el proyecto.
- **Plan B:** si no funcionara bien, Mask R-CNN (torchvision).

### D5. ¿Cómo tratamos las frutas pequeñas? 🟢
- **Qué sabemos:** el 87,5 % de las frutas que se evalúan miden menos de 32×32 píxeles, y la mediana es de unos 13×13. YOLO redimensiona la imagen al tamaño de entrada (640 px por defecto), y a esa escala una fruta de 13×13 se queda en unos 6×6.
- **Respuesta:** **primero, la opción sencilla**: entrenar e inferir con la imagen a su tamaño real (`imgsz=1280`), y medir la AP y el tiempo.
- **Si no basta:** partir la imagen en trozos (*tiling*). Los trozos tienen que solaparse para no cortar frutas, y al juntarlos hay que eliminar las detecciones duplicadas. Es más complejo, así que solo lo haremos si la opción sencilla se queda corta.
- **A vigilar:** la memoria de GPU (la de evaluación tiene 6 GB) y el tiempo (ver D2).

### D6. ¿Cómo convertimos las máscaras RLE al formato de YOLO? 🔴
- **El problema:** las anotaciones están en RLE (COCO), y YOLO-seg entrena con un `.txt` por imagen con una línea por objeto: `<clase> x1 y1 x2 y2 ...`, un **único polígono** por objeto, con coordenadas entre 0 y 1 y al menos 3 puntos.
- **Qué hemos visto en nuestras máscaras** (las 11.076 frutas evaluadas):
  | Trozos por fruta | Frutas | % |
  |---|---|---|
  | 1 | 3.677 | 33,2 % |
  | 2 | 2.367 | 21,4 % |
  | 3 | 1.748 | 15,8 % |
  | 4 | 1.261 | 11,4 % |
  | 5 o más | 2.023 | 18,3 % |

  **El 66,8 % de las frutas están partidas en varios trozos** (por hojas o ramas), y el **19 %** tienen algún agujero. No es un caso raro: es lo normal en este dataset.
- **Qué ofrece Ultralytics** (`ultralytics/data/converter.py`, leído del código fuente):
  - `convert_coco(labels_dir, save_dir, use_segments=True, cls91to80=False)` convierte un JSON de COCO a etiquetas YOLO:
    - **Sí admite RLE**: decodifica la máscara y saca sus contornos exteriores con `cv2.findContours` (necesita el paquete `faster-coco-eval`).
    - **Ignora las anotaciones con `iscrowd: 1`**.
    - Si una fruta tiene **varios trozos**, los une en un solo polígono con `merge_multi_segment`, que conecta cada trozo con el siguiente por sus dos puntos más cercanos, mediante una "línea fina" de ida y vuelta.
    - **Los agujeros se pierden**: solo usa contornos exteriores, así que la máscara queda rellena.
    - Hay que pasar `cls91to80=False`. Si no, intenta traducir los ids de las 91 categorías de COCO original a sus 80 clases.
    - **No hace la separación train/val, no copia las imágenes y no crea el `.yaml`**. Eso lo tenemos que hacer nosotros.
  - `convert_segment_masks_to_yolo_seg` no nos sirve: espera máscaras semánticas (una imagen por foto, con la clase como valor de píxel), no una máscara por fruta.
- **Qué falta comprobar:** cuánto se parece la máscara convertida (polígono con trozos unidos y sin agujeros) a la original. Lo mediremos con el IoU entre ambas en todas las frutas: si sale alto, usamos `convert_coco` tal cual; si no, escribimos nuestra propia conversión. También lo revisaremos visualmente con `visualize.py`.
- **Sobre `iscrowd`:** en COCO original marca grupos de objetos anotados juntos (por ejemplo, una multitud), que la evaluación ignora. En este proyecto lo usan para los fragmentos de menos de 20 píxeles visibles: detectarlos no suma y no detectarlos no resta.
  - **Decisión: los incluimos en el entrenamiento como frutas.** Si los dejáramos fuera, sus píxeles contarían como fondo y el modelo aprendería que una fruta muy pequeña "no es fruta", justo lo contrario de lo que queremos para las de 20-30 px, que sí puntúan. Y como la evaluación ignora las detecciones sobre ellos, detectarlos no nos penaliza.
  - **Cómo:** `convert_coco` se salta las anotaciones con `iscrowd: 1`, así que antes de convertir hacemos una copia del JSON con `iscrowd: 0` en todas. Si un fragmento es tan pequeño que su contorno tiene menos de 3 puntos, `convert_coco` usa su caja como polígono.

### D7. ¿Cómo separamos train y validación? 🟢
- **Respuesta:** **80/20, igual para cada árbol**: 10 imágenes de cada uno de los 4 árboles para validación (40 en total) y las otras 160 para entrenar. Así las cuatro especies están igual de representadas.
- **Cómo se usa:** la validación sirve para elegir los ajustes (resolución, épocas, umbrales…). Para el modelo final que se entrega, se puede reentrenar con las 200 imágenes usando esos ajustes.
- **Limitación asumida:** las imágenes del mismo árbol comparten frutas vistas desde otro ángulo, así que la validación saldrá más alta que el test (ver D3).

### D8. ¿Dónde entrenamos? 🟢
- **Respuesta:** **en local, con la GPU de Darío: NVIDIA RTX 3060 de 12 GB.** Google Colab queda como plan B.
- **Por qué:** entrenar YOLO-seg a 1280 px en CPU es inviable. En local no hay límites de tiempo ni desconexiones, y no hay que subir el dataset.
- **A tener en cuenta:** la tarjeta de evaluación (RTX 2060) tiene **la mitad de memoria (6 GB)**. Que algo funcione en nuestra 3060 no garantiza que quepa allí: hay que vigilar la memoria y el tiempo de la inferencia.

### D9. ¿Conseguir el starter kit? 🟢
- **Respuesta:** está en `starter_kit/`, copiado de `Final project/Part 1/` del repositorio de la asignatura (https://github.com/jotaraul/uma_cv_2027).
- **Qué contiene:** `evaluate.py` (métricas oficiales; con `--images-file` puntúa solo las imágenes de validación), `check_submission.py` (simula la ejecución del profesor), `visualize.py` (dibuja anotaciones o predicciones) y `example_submission/` (un detector por umbral de color, solo para enseñar el formato).
- **Para actualizarlo:** hacer `git pull` en el clon del repositorio de la asignatura y volver a copiar la carpeta.

### D10. ¿Qué frutas cuentan en la evaluación? ¿Las de más de 20×20 px? 🟢
- **Respuesta:** **no es 20×20 px, son 20 píxeles de área visible.** El enunciado dice que las frutas con menos de 20 píxeles visibles se anotan con `iscrowd: 1` y la evaluación las ignora (ni suman ni restan).
- **Comprobado en `annotations.json`:** las frutas ignoradas tienen entre 1 y 19 píxeles de área, y las que cuentan empiezan exactamente en 20.
- **Por qué importa:** de las 11.076 frutas que se evalúan, 7.921 (el 71,5 %) miden menos de 400 px (menos que un cuadrado de 20×20). Si las descartáramos, perderíamos casi tres de cada cuatro frutas que puntúan.

### D11. ¿Cómo se llama el equipo? 🟢
- **Por qué importa:** la carpeta y el zip de la entrega se llaman `team_<nombre>/` y `team_<nombre>.zip`.
- **Respuesta:** **G17**. La carpeta de entrega es `team_G17/` y el zip, `team_G17.zip`.
