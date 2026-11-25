# Dudas del proyecto

Preguntas que tenemos abiertas sobre el proyecto. Cada una indica **por qué importa** y **cómo pensamos resolverla**: preguntando al profesor o comprobándolo nosotros con los datos o con experimentos.

Cuando una duda se resuelva, se marca como resuelta, se apunta la respuesta y, si afecta al método, se refleja en el [README.md](README.md).

**Estados:** 🔴 abierta · 🟢 resuelta

---

## Para preguntar al profesor

### D1. ¿En el test, cada imagen tiene una sola especie? 🔴
- **Qué sabemos:** en las 200 imágenes del train, todas las frutas de una imagen son de la misma especie (comprobado con `annotations.json`). El enunciado no dice nada sobre si esto se cumple también en el test.
- **Por qué importa:** si se cumple, podemos decidir la especie por votación entre todas las detecciones de la imagen, y eso eliminaría casi todos los errores de especie. Si no se cumple y lo usamos, las imágenes con varias especies saldrían mal.
- **Cómo resolverla:** preguntar al profesor.

### D2. ¿Cuántas imágenes tiene el test? 🔴
- **Qué sabemos:** hay 20 minutos para todo el test, incluida la carga del modelo, en una RTX 2060 de 6 GB. No sabemos el número de imágenes.
- **Por qué importa:** determina cuánto tiempo podemos gastar por imagen (por ejemplo, si podemos permitirnos inferencia por trozos o refinar con SAM).
- **Cómo resolverla:** preguntar al profesor. Si no se puede saber, suponer un caso conservador y medir el tiempo por imagen.

### D3. ¿Las imágenes del train son de 4 árboles (uno por especie)? 🔴
- **Qué sabemos:** si unimos las imágenes que comparten algún `fruit_id`, salen exactamente 4 grupos de 50 imágenes, cada uno de una sola especie y sin frutas en común. Los datos no traen un identificador de árbol, así que es una deducción.
- **Por qué importa:** si solo hay un árbol por especie, no podemos validar con un árbol que el modelo no haya visto, y nuestra validación será optimista respecto al test (que tiene árboles distintos).
- **Cómo resolverla:** preguntar al profesor. También sirve para saber si en el test habrá varios árboles por especie.

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
- **Depende de:** D3.
- **Por qué importa:** si las imágenes de validación son del mismo árbol que las de entrenamiento, el resultado sale más alto de lo que será en el test.
- **Cómo resolverla:** decidirlo cuando se resuelva D3, y dejarlo explicado en el README.

### D8. ¿Dónde entrenamos? 🔴
- **Por qué importa:** entrenar a alta resolución necesita GPU.
- **Cómo resolverla:** ver qué GPU tenemos cada uno o usar Google Colab.

### D9. ¿Conseguir el starter kit? 🔴
- **Qué sabemos:** `evaluate.py`, `check_submission.py`, `visualize.py` y `example_submission/` están en el repositorio de la asignatura `uma_cv_2027`, en `Final project/Part 1/`. No los tenemos todavía.
- **Cómo resolverla:** clonar o actualizar ese repositorio y copiar la carpeta del kit.
