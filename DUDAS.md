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
- **Qué sabemos:** hay 20 minutos para todo el test, incluida la carga del modelo, en una RTX 2060 de 6 GB. No sabemos el número de imágenes.
- **Por qué importa:** determina cuánto tiempo podemos gastar por imagen (por ejemplo, si podemos permitirnos inferencia por trozos o refinar con SAM).
- **Cómo resolverla:** preguntar al profesor. Si no se puede saber, suponer un caso conservador y medir el tiempo por imagen.

### D3. ¿Las imágenes del train son de 4 árboles (uno por especie)? 🟢
- **Qué sabemos:** si unimos las imágenes que comparten algún `fruit_id`, salen exactamente 4 grupos de 50 imágenes, cada uno de una sola especie y sin frutas en común. Los datos no traen un identificador de árbol.
- **Por qué importa:** si solo hay un árbol por especie, no podemos validar con un árbol que el modelo no haya visto, y nuestra validación será optimista respecto al test (que tiene árboles distintos).
- **Respuesta:** **lo tomamos como 4 árboles, uno por especie.** Los únicos datos que tenemos son los que nos han dado, y en ellos salen 4 grupos. Trabajamos con eso.
- **Consecuencia:** la validación se hará con imágenes de los mismos 4 árboles que el entrenamiento, así que la puntuación de validación será más alta que la del test. Lo tendremos en cuenta al interpretar los resultados (ver D7).

---

## Para resolver nosotros

### D4. ¿Qué modelo de segmentación de instancias usamos? 🔴
- **Qué sabemos:** la práctica 7.5 usa YOLO de Ultralytics para detectar cajas. Ultralytics también tiene modelos de segmentación (variantes `-seg`), que es lo que pide este proyecto (máscara por fruta).
- **Por qué importa:** es la base de todo el método.
- **Cómo resolverla:** probar YOLO-seg con fine-tuning y medir con `evaluate.py`. Si no funciona bien, probar Mask R-CNN (torchvision).

### D5. ¿Cómo tratamos las frutas pequeñas? 🔴
- **Qué sabemos:** el 87,5 % de las frutas que se evalúan miden menos de 32×32 píxeles, y la mediana es de unos 13×13.
- **Por qué importa:** con la resolución típica de YOLO (640 px) muchas frutas quedarían en 2-3 píxeles.
- **Cómo resolverla:** comparar en validación una resolución de entrada más alta frente a inferencia por trozos (*tiling*), teniendo en cuenta la memoria (6 GB) y el tiempo.

### D6. ¿Cómo convertimos las máscaras RLE al formato de YOLO? 🔴
- **Qué sabemos:** las anotaciones están en RLE (COCO) y YOLO-seg entrena con polígonos. Algunas frutas aparecen partidas en varios trozos por las hojas.
- **Por qué importa:** si la conversión pierde trozos o deforma las máscaras pequeñas, el modelo aprende máscaras peores.
- **Cómo resolverla:** investigar la conversión y comprobarla visualmente con `visualize.py` en algunas imágenes.

### D7. ¿Cómo separamos train y validación? 🔴
- **Depende de:** D3 (resuelta: 4 árboles, uno por especie).
- **Por qué importa:** si las imágenes de validación son del mismo árbol que las de entrenamiento, el resultado sale más alto de lo que será en el test.
- **Qué sabemos ya:** no se puede separar por árboles sin quedarse sin una especie, así que la validación tendrá que ser un subconjunto de imágenes de cada uno de los 4 árboles.
- **Cómo resolverla:** decidir cuántas imágenes de cada árbol reservamos para validación, y dejarlo explicado en el README.

### D8. ¿Dónde entrenamos? 🔴
- **Por qué importa:** entrenar a alta resolución necesita GPU.
- **Cómo resolverla:** ver qué GPU tenemos cada uno o usar Google Colab.

### D9. ¿Conseguir el starter kit? 🔴
- **Qué sabemos:** `evaluate.py`, `check_submission.py`, `visualize.py` y `example_submission/` están en el repositorio de la asignatura `uma_cv_2027`, en `Final project/Part 1/`. No los tenemos todavía.
- **Cómo resolverla:** clonar o actualizar ese repositorio y copiar la carpeta del kit.

### D10. ¿Qué frutas cuentan en la evaluación? ¿Las de más de 20×20 px? 🟢
- **Respuesta:** **no es 20×20 px, son 20 píxeles de área visible.** El enunciado dice que las frutas con menos de 20 píxeles visibles se anotan con `iscrowd: 1` y la evaluación las ignora (ni suman ni restan).
- **Comprobado en `annotations.json`:** las frutas ignoradas tienen entre 1 y 19 píxeles de área, y las que cuentan empiezan exactamente en 20.
- **Por qué importa:** de las 11.076 frutas que se evalúan, 7.921 (el 71,5 %) miden menos de 400 px (menos que un cuadrado de 20×20). Si las descartáramos, perderíamos casi tres de cada cuatro frutas que puntúan.
