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
- **Resultado en validación:** el modelo ya acierta la especie en el 100 % de las detecciones, así que la votación por imagen no cambia nada (misma AP). La dejamos como opción (`--species-vote`), sin activar.
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
- **Respuesta:** **YOLO de segmentación** (variantes `-seg` de Ultralytics) con fine-tuning. En concreto **YOLO11s-seg**: los YOLO26-seg no caben en memoria al entrenar con tantas frutas por imagen (ver D12).
- **Por qué:** es la misma librería y el mismo flujo de trabajo que la práctica 7.5 (cargar un modelo preentrenado, `train`, `predict`), pero devuelve una máscara por objeto, que es lo que pide el proyecto.
- **Plan B:** si no funcionara bien, Mask R-CNN (torchvision).

### D5. ¿Cómo tratamos las frutas pequeñas? 🟢
- **Qué sabemos:** el 87,5 % de las frutas que se evalúan miden menos de 32×32 píxeles, y la mediana es de unos 13×13. YOLO redimensiona la imagen al tamaño de entrada (640 px por defecto), y a esa escala una fruta de 13×13 se queda en unos 6×6.
- **Respuesta:** **primero, la opción sencilla**: entrenar e inferir con la imagen a su tamaño real (`imgsz=1280`), y medir la AP y el tiempo.
- **Si no basta:** partir la imagen en trozos (*tiling*). Los trozos tienen que solaparse para no cortar frutas, y al juntarlos hay que eliminar las detecciones duplicadas. Es más complejo, así que solo lo haremos si la opción sencilla se queda corta.
- **A vigilar:** la memoria de GPU (la de evaluación tiene 6 GB) y el tiempo (ver D2).

### D6. ¿Cómo convertimos las máscaras RLE al formato de YOLO? 🟢
- **El problema:** las anotaciones están en RLE (COCO), y YOLO-seg entrena con un `.txt` por imagen con una línea por objeto: `<clase> x1 y1 x2 y2 ...`, un **único polígono** por objeto, con coordenadas entre 0 y 1 y al menos 3 puntos.
- **Cómo son nuestras máscaras** (las 11.076 frutas evaluadas): el **66,8 %** están partidas en varios trozos por hojas o ramas (el 18,3 %, en 5 o más) y el **19 %** tienen algún agujero.
- **Qué ofrece Ultralytics:** `convert_coco` admite RLE, saca los contornos exteriores y, si la fruta tiene varios trozos, los une en un solo polígono con `merge_multi_segment` (conecta cada trozo con el siguiente por sus puntos más cercanos). Se salta las anotaciones con `iscrowd: 1` y no separa train/val, no copia las imágenes ni crea el `.yaml`.
- **Qué hemos medido** con `team_G17/src/check_conversion.py`: cada polígono se vuelve a pintar como máscara con `polygon2mask` (la misma función que usa Ultralytics al entrenar) y se compara con la original.

  | | `convert_coco` tal cual | Nuestra conversión |
  |---|---|---|
  | IoU medio | 0,766 | **0,928** |
  | Frutas con IoU ≥ 0,75 | 61,0 % | **92,4 %** |
  | Frutas con IoU ≥ 0,95 | 9,2 % | **62,3 %** |
  | Frutas de 1 trozo, IoU medio | 0,874 | **0,995** |
  | Frutas de 5 o más trozos, IoU medio | 0,646 | **0,829** |

- **Por qué `convert_coco` perdía tanto:** no por la conversión del contorno (rellenarlo directamente da un IoU de 0,995), sino por un **problema de redondeo**. Las coordenadas se guardan normalizadas, y al volver a píxeles salen valores como 99,99996, que `polygon2mask` **trunca** a 99. La máscara se desplaza un píxel, y en una fruta de 10×10 eso baja el IoU a 0,75. Este truncado también ocurre al entrenar, así que el modelo aprendería máscaras desplazadas.
- **Respuesta:** **usamos nuestra propia conversión** (`team_G17/src/prepare_dataset.py`). Hace lo mismo que `convert_coco` (mismos contornos y misma unión de trozos con `merge_multi_segment` de Ultralytics), pero:
  1. escribe cada punto desplazado **medio píxel** (x + 0,5), para que al truncar caiga en el píxel correcto;
  2. incluye los fragmentos `iscrowd` (ver abajo).
- **Lo que queda:** las frutas de muchos trozos siguen perdiendo algo (IoU medio 0,83 con 5 o más trozos) porque YOLO exige un único polígono por fruta, y los agujeros se rellenan. Es un límite del formato de YOLO; si más adelante vemos que las máscaras son el punto débil, se puede revisar.
- **Sobre `iscrowd`:** en COCO original marca grupos de objetos anotados juntos (por ejemplo, una multitud), que la evaluación ignora. En este proyecto lo usan para los fragmentos de menos de 20 píxeles visibles: detectarlos no suma y no detectarlos no resta.
  - **Decisión: los incluimos en el entrenamiento como frutas.** Si los dejáramos fuera, sus píxeles contarían como fondo y el modelo aprendería que una fruta muy pequeña "no es fruta", justo lo contrario de lo que queremos para las de 20-30 px, que sí puntúan. Y como la evaluación ignora las detecciones sobre ellos, detectarlos no nos penaliza.
  - Si un fragmento es tan pequeño que su contorno tiene menos de 3 puntos, se usa el rectángulo de sus píxeles como polígono.

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

### D12. ¿Con qué configuración cabe el entrenamiento en la GPU? 🟢
- **Qué ha pasado:** con `imgsz=1280`, ninguna prueba ha llegado a entrenar una época en la RTX 3060 (12 GB, de los que otras aplicaciones ya usan ~1,2 GB):

  | Modelo | batch | mosaic | Resultado |
  |---|---|---|---|
  | `yolo26s-seg` | 4 | sí | sin memoria |
  | `yolo26s-seg` | 2 | sí | sin memoria |
  | `yolo26s-seg` | 2 | no | sin memoria (pide ~1 GB de golpe) |
  | `yolo11s-seg` | 4 | sí | sin memoria |

- **Por qué:** tenemos muchísimas frutas por imagen (unas 80 de media contando los fragmentos `iscrowd`, y más de 300 con mosaic, que junta 4 imágenes en una).
  - En **YOLO26-seg**, el paso que asigna qué predicción corresponde a cada fruta (`SegTaskAlignedAssigner`, en `ultralytics/utils/tal.py`) **calcula una máscara para muchas posiciones candidatas de cada fruta** (hasta 30 por fruta) a 320×320. Con cientos de frutas son miles de máscaras a la vez: más de 1 GB solo en ese paso. Lo hemos visto en el error.
  - **YOLO11-seg** usa el asignador normal, que solo compara cajas, así que debería gastar mucho menos. Con `batch=4` y mosaic tampoco cupo, pero `batch=2` no llegó a probarse limpio (ver el aviso de abajo).
- **Aviso importante:** cuando se queda sin memoria, Ultralytics reintenta solo con un batch más pequeño, pero en Windows CUDA queda en mal estado y el reintento falla con `CUDA error: unknown error` o `resource already mapped`. **Hay que lanzar cada prueba directamente con el batch que se quiere probar**, no fiarse del reintento automático.
- **Qué probar, en este orden** (1 época cada una; si cabe, se lanza el entrenamiento completo con esa configuración):
  1. `yolo11s-seg`, `batch=2`, mosaic activado.
  2. `yolo11s-seg`, `batch=2`, sin mosaic (`--mosaic 0`).
  3. Si nada cabe: dejar fuera los fragmentos `iscrowd` (rehacer el dataset con `prepare_dataset.py --no-crowd --overwrite`), que son el 30 % de los objetos, y repetir.
- **Antes de entrenar:** cerrar lo que use la gráfica (fondo animado de Lively Wallpaper, juegos, Overwolf…).
- **Respuesta:** **YOLO11s-seg entrenado con recortes de 640×640 de las imágenes a resolución nativa** (`prepare_dataset.py --crop 640`), con `imgsz=640`, `batch=8` y 2 *workers*. Cabe de sobra (~5 GB de GPU durante el entrenamiento) y 10 minutos dan una AP de 0,331 en validación. Al detectar se usa la imagen completa a 1280, así que las frutas tienen el mismo tamaño en píxeles que al entrenar.
- **Otro problema que apareció:** con los 8 *workers* por defecto, algunos procesos fallaban en Windows al cargar las DLL de PyTorch (`WinError 1114`) y el entrenamiento se quedaba colgado sin avanzar. Con `--workers 2` funciona.
