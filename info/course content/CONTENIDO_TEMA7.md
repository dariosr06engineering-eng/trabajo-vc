# TEMA 7 — Reconocimiento de objetos: métodos clásicos

> Teoría: `TEORIA/L7_Object_Recognition_Classical_Methods_2026_v5.pdf`
> Prácticas: `PRACTICAS/P7 - Recognition (Discriminant, MLP, Detection)/` → 7.1 Funciones discriminantes · 7.4 Perceptrón multicapa (MLP) · 7.5 Detección de objetos con YOLO
> ⚠️ **No tienes el código de 7.2 y 7.3** (clasificador bayesiano/gaussiano y entrenamiento del perceptrón) → al final hay código de referencia hecho por mí.

---

## PARTE 1 — TEORÍA

### 1.1 Introducción
- Tema anterior: región → **vector de características x**. Ahora: asignar x a una de **M clases** C_i.
- Un **clasificador divide el espacio de características en regiones de decisión R_i**: si `x ∈ R_i ⇒ Ĉ = C_i`.
- **Fase de entrenamiento** (aprendizaje **supervisado**: muestras con clase conocida) → parámetros del clasificador w. **Fase de predicción**: se clasifican muestras nuevas con el modelo aprendido.

**Dos enfoques**:
| **Generativo** | **Discriminativo** |
|---|---|
| Modela **cómo genera cada clase las observaciones**: `p(x|C_k)` y `P(C_k)` | **Aprende directamente** la función de decisión `d_k(x)` a partir de datos etiquetados |
| Regla de Bayes → `P(C_k|x)` → `argmax` | `argmax d_k(x)` |
| Ej.: **clasificador bayesiano / gaussiano** | Ej.: **perceptrón**, redes neuronales |

**Ejemplo 1D (circularidad)**: círculo C = 1.00; cuadrado C = π/4 ≈ 0.785; triángulo equilátero C = π/(3√3) ≈ 0.605. Cada clase tiene una densidad `p(x|C_k)` → ¿dónde poner las fronteras para minimizar el error?

### 1.2 Funciones discriminantes
Una puntuación por clase `d_1(x), ..., d_M(x)`; se asigna la de **mayor puntuación**: `Ĉ = argmax_k d_k(x)`.

**Discriminante lineal**: `d_k(x) = w_kᵀ x + b_k = w_k1 x_1 + ... + w_kN x_N + b_k`
- `w_k` = vector de pesos (se aprende), `b_k` = sesgo (bias).
- **Representación aumentada**: `x̃ = [x_1 ... x_N, 1]ᵀ`, `w̃ = [w, b]` → `d_k(x) = w̃_kᵀ x̃` (el bias se mete como un peso más multiplicado por 1).

**Fronteras entre clases**: donde las puntuaciones son iguales:
`d_ij(x) = d_i(x) − d_j(x) = (w_i − w_j)ᵀ x̃`
- `> 0` → preferir C_i; `= 0` → **frontera**; `< 0` → preferir C_j.
- Regiones: `R_i = {x : d_i(x) > d_j(x) ∀ j ≠ i}`.

**Separabilidad**: depende de **las características elegidas** y de **la complejidad de la función discriminante**.
- Una característica puede separar sola y otra no; combinar varias ayuda.
- **Añadir una característica** puede hacer separables clases que se solapaban.
- Clases **no separables linealmente** (p.ej. una dentro de otra, en anillo) → hace falta una **frontera no lineal** (o transformar el espacio).

**¿Cuántas características?** Cuantos **menos datos de entrenamiento, menos características**.
**Maldición de la dimensionalidad**: al crecer la dimensión, el espacio se vuelve **disperso**; mantener la misma densidad de muestras exige **exponencialmente más datos** (ej. vecindario: 2 puntos en 1D, 8 en 2D, 26 en 3D).

### 1.3 Clasificador bayesiano
**Principio**: con **costes iguales** para todos los errores, elegir la clase de **máxima probabilidad a posteriori** **minimiza la probabilidad de error**:
`P(error | x, decidir C_k) = 1 − P(C_k | x)` → `Ĉ(x) = argmax_k P(C_k | x)`.

**Regla de Bayes**: `P(C_k | x) = p(x | C_k)·P(C_k) / p(x)`
- `P(C_k|x)` = **posterior**.
- `p(x|C_k)` = **verosimilitud** (likelihood). Como función de x es la **densidad condicionada a la clase**; para un x observado comparando clases es la **verosimilitud**.
- `P(C_k)` = **prior**: probabilidad de la clase **antes** de observar x.
- `p(x) = Σ_j p(x|C_j) P(C_j)` = **evidencia**: **no depende de k** → se puede ignorar para clasificar.

**Discriminantes equivalentes** (mismo orden entre clases):
`d_k = P(C_k|x)` ~ `p(x|C_k) P(C_k)` ~ **`ln p(x|C_k) + ln P(C_k)`** ← **discriminante MAP**
- ¿Por qué el logaritmo? Es **monótono**; cálculo más fácil y **numéricamente más estable** (sumas en vez de productos).
- **Priors iguales** → `d_k = ln p(x|C_k)` → clasificador de **máxima verosimilitud (ML)**.

**Región de rechazo**: si la mayor posterior no supera un umbral, **no se decide** (se rechaza la muestra) para evitar errores en zonas dudosas.

### 1.4 Modelo gaussiano
`p(x|C_k) = N(x; μ_k, Σ_k) = 1/((2π)^(N/2)|Σ_k|^(1/2)) · exp(−½ (x−μ_k)ᵀ Σ_k⁻¹ (x−μ_k))`
- **Vector de medias** μ_k y **matriz de covarianza** Σ_k = E[(x−μ)(x−μ)ᵀ].
- Σ **diagonal** → las características son **condicionalmente independientes** dada la clase (**Naive Bayes gaussiano**).
- `(x−μ)ᵀ Σ⁻¹ (x−μ)` = **distancia de Mahalanobis al cuadrado** `D_M²`.

**Caso general (Σ_k distintas) → QDA**:
`d_k(x) = −½ xᵀΣ_k⁻¹x + μ_kᵀΣ_k⁻¹x − ½ μ_kᵀΣ_k⁻¹μ_k − ½ ln|Σ_k| + ln P(C_k)`
→ polinomio **cuadrático** `xᵀ A_k x + w_kᵀ x + b_k` → **fronteras cuadráticas** (cónicas).

**Covarianza compartida (Σ_k = Σ) → LDA**:
- El término cuadrático `xᵀΣ⁻¹x` es **igual para todas las clases** → se cancela.
- `d_k(x) = μ_kᵀ Σ⁻¹ x − ½ μ_kᵀ Σ⁻¹ μ_k + ln P(C_k)` → **lineal** (`w_k = Σ⁻¹μ_k`).
- Si además **priors iguales** → **clasificador de mínima distancia de Mahalanobis al centroide**: `Ĉ = argmin_k (x−μ_k)ᵀΣ⁻¹(x−μ_k)`.

**Covarianza isótropa compartida (Σ = σ² I) y priors iguales**:
`d_k(x) = −||x − μ_k||²/(2σ²)` → **mínima distancia euclídea al centroide**.

| Supuesto | Frontera | Nombre |
|---|---|---|
| Σ_k distintas | **cuadrática** | QDA |
| Σ_k = Σ | **lineal** | LDA (Mahalanobis si priors iguales) |
| Σ = σ²I, priors iguales | lineal | Distancia **euclídea** al centroide |

**Entrenamiento** (estimación por máxima verosimilitud), con codificación **one-hot** `y_ik = 1 si c_i = k` (ej.: c_i = 3, M = 5 → y_i = [0,0,1,0,0]):
- `n_k = Σ_i y_ik` (nº de muestras de la clase)
- **Prior**: `P(C_k) = n_k / n`
- **Media**: `μ_k = (1/n_k) Σ_i y_ik x_i`
- **Covarianza**: `Σ_k = (1/n_k) Σ_i y_ik (x_i − μ_k)(x_i − μ_k)ᵀ`
- y_ik actúa como selector: solo contribuyen las muestras de la clase k.

**Ejemplo de las diapositivas**: μ1 = [2,2], μ2 = [5,2]; P(C1) = 4/10 = 0.4, P(C2) = 0.6; Σ compartida = [0.6 0.3; 0.3 0.6], |Σ| = 0.27, Σ⁻¹ = (20/9)·[1 −0.5; −0.5 1]. Frontera con priors iguales: **x2 = 2x1 − 5** (recta).

### 1.5 Perceptrón (aprendizaje de un discriminante lineal)
**Modelo binario**: `d(x) = wᵀx + b`
- `d > 0` → clase C1; `d = 0` → frontera; `d < 0` → clase C2.
- **w es la normal a la frontera** y **b controla su desplazamiento**.
- Etiquetas `y_i ∈ {−1, +1}`; predicción `ŷ_i = sign(d(x_i))`.

**Regla de aprendizaje**: si la muestra está **mal clasificada** (`y_i (wᵀx_i + b) ≤ 0`):
```
w ← w + η · y_i · x_i
b ← b + η · y_i          (η = tasa de aprendizaje)
```
- Si está bien clasificada, no se toca nada.
- **¿Por qué funciona?** Tras la actualización: `y_i·d_new(x_i) = y_i·d_old(x_i) + η(||x_i||² + 1)` → el término añadido es **> 0** → la puntuación con signo **aumenta** → x_i se mueve hacia el lado correcto de la frontera.
- Se repite para todas las muestras durante varias épocas.

**Limitaciones del perceptrón de una capa**:
- Solo implementa **fronteras lineales** → **no resuelve problemas no separables linealmente** (ej. **XOR**).
- Si los datos **no son linealmente separables**, el algoritmo **no converge**.
- Si son separables, hay **muchos hiperplanos** válidos y el perceptrón converge a **cualquiera**: **no maximiza el margen** (a diferencia de SVM).

**Perceptrón multiclase**: un discriminante lineal por clase `d_k(x) = w_kᵀx + b_k`, `Ĉ = argmax d_k` → sin regiones ambiguas, pero **sigue dando solo fronteras lineales**.

**Hacia el MLP**: añadir **capas ocultas** y **activaciones no lineales**:
- `h = σ(W_1 x̃)`, `d = W_2 h`, `ŷ_k = softmax(d)_k = e^{d_k} / Σ_j e^{d_j}`
- Dimensiones (ejemplo con N entradas, H = 3 ocultas, K = 3 clases): `W_1 ∈ R^{H×(N+1)}`, `W_2 ∈ R^{K×H}`.
- La **no linealidad viene de las activaciones ocultas**, **no del softmax**. El softmax solo convierte puntuaciones en probabilidades (0 < ŷ_k < 1, Σ ŷ_k = 1).

### Resumen
- Generativo: aprender `p(x|C_k)` y `P(C_k)` → Bayes → `P(C_k|x)`.
- Efecto de la covarianza: distintas → cuadráticas; iguales → lineales; σ²I + priors iguales → centroide euclídeo.
- Discriminativo: aprender `d_k(x)` directamente. El perceptrón de una capa solo da fronteras lineales.
- Las capas ocultas no lineales crean **fronteras de decisión no lineales**.

---

## PARTE 2 — PRÁCTICA 7.1 (Funciones discriminantes)

**Momentos de Hu del contorno como características** (círculos, cuadrados, triángulos; 20 imágenes de cada)
```python
contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
moments = cv2.moments(contours[0])                 # momentos del CONTORNO (no de la máscara)
hu[i,:] = cv2.HuMoments(moments).flatten()         # fila i = vector de 7 Hu de la imagen i
np.save("./data/hu_circles.npy", hu_circles)       # se guardan para las prácticas siguientes
```
- Matriz de datos 20×7: **una fila por muestra, una columna por característica**.
- Con el scatter (Hu1 vs Hu2) se analiza qué característica es más discriminativa y si las clases son separables linealmente.

**Discriminante lineal con representación aumentada**
```python
w = np.array([[1],[0],[-3]])                       # d(x) = 1·x1 + 0·x2 − 3
data = np.array([[2,2,1],[1,2,1],...])             # cada punto con un 1 al final (aumentado)
classification = np.dot(data, w)                   # d(x) para todos los puntos a la vez (6×3 · 3×1 = 6×1)
class_ = np.sign(classification)                   # +1 / −1
```
- Frontera: `x1 − 3 = 0` → la **recta vertical x1 = 3**. Puntos con x1 > 3 → +1.
- El 1 final es lo que permite meter el **bias** en w.

**Transformación a un espacio cuadrático** (truco para fronteras no lineales con un discriminante lineal)
```python
fx[i,:] = [x1**2, x1*x2, x2**2, x1, x2, 1]         # φ(x): 2D → 6D
w = np.array([[1],[0],[0],[0],[-5],[0]])           # d = x1² − 5·x2
classification = np.dot(transform_space(data), w)
```
- Frontera en el espacio original: `x1² − 5x2 = 0` → **parábola x2 = x1²/5** (no lineal).
- En el espacio transformado (eje x1², eje x2) la frontera es una **recta**: el discriminante es **lineal en φ(x)** pero **cuadrático en x**.
- Es la misma idea que un clasificador cuadrático / que usan las redes con capas ocultas (transformar el espacio para hacerlo separable).

---

## PARTE 3 — PRÁCTICAS 7.4 Y 7.5 (código)

### 7.4 Perceptrón y perceptrón multicapa (scikit-learn, dataset Iris)
> Aplica el perceptrón y el MLP de la sección 1.5 de este tema (y la teoría de redes del Tema 8).
```python
iris = load_iris(); X, y = iris.data, iris.target    # 150 muestras, 4 características, 3 clases
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)      # FIT solo con entrenamiento
X_test_scaled  = scaler.transform(X_test)           # test: solo TRANSFORM con la media/desv. de train
```
- `StandardScaler` → media 0, desviación 1 por característica. Necesario porque las características tienen **rangos distintos** (se ve en el boxplot) y los modelos basados en gradiente/pesos lo agradecen.
- ⚠️ **`fit_transform` sobre el test sería data leakage** (usar información del test). Tampoco se hace el fit antes de dividir.
- `random_state=42` → reproducible. `test_size=0.3` → 30 % test (45 muestras).

**Perceptrón**
```python
p = Perceptron(max_iter=1000, tol=1e-3, random_state=42).fit(X_train_scaled, y_train)
p.coef_.shape        # (3, 4): un vector de pesos por clase (one-vs-rest), 4 características
p.intercept_.shape   # (3,): un bias por clase
```
- Con **Breast Cancer** (binario, 30 características) → `coef_` (1, 30): un solo discriminante.
- Precisión limitada en Iris: **versicolor y virginica no son linealmente separables** → el perceptrón (lineal) falla en algunas.

**Métricas**
```python
accuracy_score(y_test, y_pred)
confusion_matrix(y_test, y_pred)        # filas = clase REAL, columnas = clase PREDICHA
classification_report(y_test, y_pred)
```
- Matriz de confusión perfecta → **solo la diagonal** distinta de 0.
- Fila "versicolor", columna "virginica" = versicolores clasificadas como virginica.
- **Precision** = TP/(TP+FP): de los que predije como clase k, cuántos lo eran.
- **Recall** = TP/(TP+FN): de los que eran clase k, cuántos encontré.
- **F1** = media armónica de precision y recall. **Support** = nº de muestras reales de cada clase.
- **Macro avg** = media simple entre clases; **weighted avg** = ponderada por support.

**MLP**
```python
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
mlp.loss_, mlp.n_iter_, mlp.loss_curve_            # pérdida final, iteraciones, curva de pérdida
[c.shape for c in mlp.coefs_]       # [(4,10), (10,5), (5,3)]  → (entradas, neuronas) de cada capa
[b.shape for b in mlp.intercepts_]  # [(10,), (5,), (3,)]       → un bias por neurona
mlp.out_activation_                 # 'softmax' (multiclase)
mlp.predict_proba([X_test_scaled[0,:]])   # probabilidades por clase (suman 1); ojo: 2D → lista o reshape
```
- `(10, 5)` = **2 capas ocultas** de 10 y 5 neuronas. Total: entrada (4) → 10 → 5 → salida (3).
- `len(mlp.coefs_) − 1` = nº de capas ocultas.
- Las matrices de sklearn tienen forma **(entradas, neuronas)** (traspuestas respecto a la notación de las diapositivas, W ∈ R^{neuronas×entradas}).
- Salida con varias neuronas → **softmax** y se elige la clase de mayor probabilidad.
- **¿Por qué activaciones no lineales?** Apilar capas lineales equivale a **una sola capa lineal** (composición de lineales = lineal) → sin no linealidad, un MLP sería otro perceptrón.
- `predict_proba` espera un array **2D** (n_muestras, n_características): por eso `[X_test_scaled[0,:]]`.

### 7.5 Detección de objetos con YOLO (Ultralytics)
> Usa conceptos del Tema 8 (CNN, backbone, fine-tuning, congelar capas). La detección de objetos no aparece en las diapositivas: solo en esta práctica.
**Detección** = clasificar **y localizar** (caja) cada objeto.
- **Dos etapas** (p.ej. **Faster R-CNN**): primero propone regiones, luego las clasifica → más lento.
- **Una etapa** (p.ej. **YOLO**, "You Only Look Once"): predice cajas y clases en **una sola pasada** → **tiempo real** (robótica móvil).

```python
model = YOLO('yolo26n.pt')             # 'n' = nano (el más pequeño), preentrenado en COCO
model.names                            # dict {índice: nombre}, 80 categorías COCO
result = model.predict(img_path, verbose=False)[0]   # predict devuelve una LISTA (una por imagen) → [0]
boxes = result.boxes.xyxy.cpu().numpy()   # N×4 [x_ul, y_ul, x_br, y_br] (esquinas), en píxeles
confidences = result.boxes.conf.cpu().numpy()
labels = [model.names[int(c)] for c in result.boxes.cls]   # cls son floats → int para indexar
```
- `.cpu().numpy()`: los tensores pueden estar en GPU; hay que pasarlos a CPU para convertirlos a NumPy.
- **xyxy** = esquina sup. izquierda + esquina inf. derecha. (Otros formatos: `xywh` centro + tamaño; SAM usa XYWH esquina + tamaño.)
- Un detector entrenado con COCO **solo conoce sus 80 categorías**: objetos fuera de COCO (puerta, ventana, lámpara, armario...) **no se detectan** o se confunden con la clase COCO más parecida.
- Ojo con los nombres: COCO llama **'couch'** al sofá y **'dining table'** a la mesa → por eso 'sofa' y 'table' "no están" en COCO al comparar cadenas.

**Umbral de confianza**
```python
model.predict(img, conf=threshold, classes=classes)   # classes = lista de índices a detectar (None = todas)
```
- Umbral **muy bajo** → muchas detecciones, **más falsos positivos**. **Muy alto** → pocas, **se pierden objetos** (falsos negativos).

**Fine-tuning con un dataset nuevo (HomeObjects-3K, 12 clases)**
```python
model_ft = YOLO('yolo26n.pt')
model_ft.train(data='HomeObjects-3K.yaml', epochs=10, imgsz=640, fraction=0.5, freeze=11, batch=16)
```
- `data`: YAML con rutas y nombres de clases. `epochs`: pasadas completas. `imgsz`: tamaño de entrada. `fraction=0.5`: usa el 50 % del dataset. `batch=16`: tamaño del mini-batch.
- **`freeze=11`**: congela las **11 primeras capas (backbone)** → solo se entrenan el resto y la cabeza. Ventajas: entrenamiento **más rápido**, menos memoria, **menos sobreajuste** con pocos datos, conserva las características generales aprendidas en COCO. Entrenar toda la red o más épocas → puede adaptarse más, pero con riesgo de sobreajuste y más coste.
- Se necesita **GPU** (`torch.cuda.is_available()`); si no, se carga un modelo ya fine-tuneado.
- La nueva **cabeza tiene 12 salidas** → el modelo fine-tuneado **pierde las categorías COCO** (ya no puede predecirlas; olvido).
- Evaluación con `model_ft.val()`: **mAP50** = mean Average Precision con IoU ≥ 0.5 (**IoU** = área de intersección / área de unión entre la caja predicha y la real).

**Combinar dos modelos** (COCO + nuevas categorías)
```python
new_categories = [i for i, n in model_ft.names.items() if n not in coco_categories]   # índices solo de clases nuevas
boxes = np.vstack([boxes_coco, boxes_new])          # apilar cajas (N1×4 + N2×4)
labels = labels_coco + labels_new                   # concatenar listas
confidences = np.concatenate([conf_coco, conf_new])
```
- Ventaja: se detectan **las 80 de COCO + las nuevas**. Inconvenientes: **dos inferencias** (más lento), umbrales/calibraciones distintas, y **cajas duplicadas** para el mismo objeto si una categoría está en ambos modelos (por eso solo se piden al segundo las categorías **nuevas**).

---

## PARTE 4 — CÓDIGO DE REFERENCIA (7.2 / 7.3, no están en tus prácticas)

**Clasificador gaussiano (generativo)**
```python
def train_gaussian(X, y, K):
    priors, means, covs = [], [], []
    for k in range(K):
        Xk = X[y == k]                                  # muestras de la clase k (el "selector" y_ik)
        priors.append(len(Xk) / len(X))                 # P(C_k) = n_k / n
        means.append(Xk.mean(axis=0))                   # μ_k
        covs.append(np.cov(Xk, rowvar=False, bias=True))  # Σ_k (bias=True → divide por n_k, estimador ML)
    return priors, means, covs

def predict_gaussian(x, priors, means, covs):
    d = []
    for P, mu, S in zip(priors, means, covs):
        diff = x - mu
        d.append(-0.5*diff @ np.linalg.inv(S) @ diff - 0.5*np.log(np.linalg.det(S)) + np.log(P))  # QDA
    return np.argmax(d)
```
- `rowvar=False`: cada **columna** es una variable (por defecto `np.cov` asume filas).
- Si se usa la **misma Σ** para todas las clases → LDA (frontera lineal).
- Con scikit-learn: `QuadraticDiscriminantAnalysis`, `LinearDiscriminantAnalysis`, `GaussianNB` (Σ diagonal).

**Perceptrón (regla de aprendizaje)**
```python
w = np.zeros(X.shape[1]); b = 0; eta = 0.1
for epoch in range(max_epochs):
    errors = 0
    for xi, yi in zip(X, y):                 # yi ∈ {−1, +1}
        if yi * (w @ xi + b) <= 0:           # mal clasificada (o sobre la frontera)
            w += eta * yi * xi
            b += eta * yi
            errors += 1
    if errors == 0: break                    # converge solo si es linealmente separable
```

---

## PARTE 5 — TRAMPAS TÍPICAS DE TEST
- MAP minimiza la probabilidad de error (con costes iguales).
- `p(x)` (evidencia) no depende de la clase → se ignora.
- Verosimilitud `p(x|C)` ≠ posterior `P(C|x)`.
- Priors iguales → máxima verosimilitud.
- Σ distintas → **cuadrática (QDA)**; Σ iguales → **lineal (LDA)**; Σ = σ²I + priors iguales → **euclídea**; Σ igual + priors iguales → **Mahalanobis**.
- Σ diagonal → Naive Bayes (independencia condicional).
- El log se usa por monotonía y estabilidad numérica.
- Perceptrón: actualiza **solo con errores**; `w ← w + η y x`; no converge si no es separable; **no maximiza el margen**; no resuelve XOR.
- w = normal a la frontera; b = desplazamiento.
- Maldición de la dimensionalidad: más dimensiones → exponencialmente más datos.
- En un MLP la no linealidad viene de las **activaciones ocultas**, no del softmax.
- Transformar x → φ(x) cuadrático permite fronteras no lineales con un discriminante lineal.
- StandardScaler: `fit` solo con train.
- Matriz de confusión: filas = real, columnas = predicha.
- YOLO: una etapa, tiempo real; `xyxy`; `conf` umbral; COCO = 80 clases; `freeze` congela el backbone; cabeza nueva → se pierden las clases antiguas.
