# TEMA 8 — Deep Learning para reconocimiento visual

> Teoría: `TEORIA/L8_Deep Learning for Visual Recognition_v3.pdf`
> ⚠️ **Tema solo de teoría: no tiene práctica propia.** Las prácticas de redes (7.4 MLP y 7.5 YOLO) están en el **Tema 7**, aunque usan conceptos de este tema (MLP, CNN, backbone, transfer learning y fine-tuning).

---

## PARTE 1 — TEORÍA

### 1.1 De características hechas a mano a características aprendidas
- El cambio clave es **cómo se obtienen las características**: diseñadas (handcrafted: SIFT, Hu, GLCM...) frente a **aprendidas** de los datos.
- La clasificación es el ejemplo didáctico; el objetivo general son las **representaciones aprendidas** (sirven para muchas tareas).
- Una red neuronal transforma la entrada en puntuaciones de clase mediante **capas sucesivas**.

### 1.2 Redes neuronales profundas
**Neurona**: suma ponderada + activación no lineal: `z = Σ w_i x_i + b`, `a = σ(z)`.
- La **activación σ introduce la no linealidad** → permite aprender funciones complejas.

**MLP (perceptrón multicapa)**: apila capas **totalmente conectadas (fully connected)**: `a^(l) = σ(W^(l) a^(l−1) + b^(l))`. La capa de salida da **una puntuación por clase**, normalmente seguida de **softmax**.

**Profundidad y anchura**:
- **Profundidad L** = nº de capas → representaciones más complejas y **jerárquicas**.
- **Anchura n** = neuronas por capa → conjunto más rico de características.
- Nº de pesos ≈ **L·n²** (L capas de n neuronas).
- Más profundidad/anchura → **más poder expresivo** pero **más coste computacional**.

**Aprendizaje jerárquico de características**:
- **Primeras capas**: características de bajo nivel (**bordes, orientaciones, contrastes de color**).
- **Capas intermedias**: combinan las anteriores → **texturas, formas, partes** de objetos.
- **Capas profundas**: patrones de alto nivel → **objetos y categorías**.

**Píxeles como entrada a una red totalmente conectada**:
- Imagen RGB 5×5 → 75 entradas; 3 neuronas → **225 pesos** (manejable).
- Imagen RGB 1000×1000 → 3·10⁶ entradas; 10³ neuronas → **3·10⁹ pesos** (inviable).
- Las capas FC **no explotan la estructura espacial** ni **reutilizan pesos** entre posiciones → **solución: CNNs**.

### 1.3 Entrenamiento
**Del dato a la pérdida (forward)**: dataset etiquetado `{(x_i, c_i)}` → red `f(x; W, b)` → **logits** d → **softmax** → probabilidades p̂ → **pérdida** comparando con la etiqueta real (one-hot).

**Softmax**: `p̂_k = e^{d_k} / Σ_j e^{d_j}` → `0 < p̂_k < 1`, `Σ p̂_k = 1`.
**Entropía cruzada (cross-entropy)**: `L(y, p̂) = −Σ_k y_k ln p̂_k` → con one-hot queda `L = −ln p̂_c` (c = clase real).
- Ejemplo: logits [1.3, 2.2, 0.7] → exp [3.67, 9.03, 2.01], suma 14.71 → p̂ = [0.25, **0.61**, 0.14]. Si la clase real es la 2 → `L = −ln(0.61) ≈ 0.49`.
- Pérdida menor cuando la probabilidad de la clase real es mayor; tiende a 0 cuando p̂_c → 1.

**Backpropagation** (aprendizaje): calcula los **gradientes de la pérdida** respecto a los parámetros con la **regla de la cadena** (`∇_W L`, `∇_b L`, del mismo tamaño que W y b).
**Descenso por gradiente**: `W ← W − η ∂L/∂W`, `b ← b − η ∂L/∂b` (η = **tasa de aprendizaje**, tamaño del paso; el signo menos → dirección de descenso). Optimizadores más avanzados: SGD con momento, **Adam**...

**Bucle de entrenamiento**:
1. Seleccionar un **mini-batch** → 2. forward → 3. pérdida → 4. backprop → 5. actualizar parámetros.
- **Una iteración** = un mini-batch + una actualización.
- **Una época** = una pasada completa por todo el conjunto de entrenamiento.
- Ej.: 1000 imágenes / mini-batch de 100 → **10 iteraciones por época**.
- **SGD estricto** = 1 ejemplo por actualización; **mini-batch SGD** = un grupo pequeño (gradiente estimado → ruidoso). Se suele **barajar** antes de cada época.

**Train / validación / test**:
| Conjunto | Uso |
|---|---|
| **Entrenamiento** | aprender los parámetros (pesos y sesgos) |
| **Validación** | ajustar **hiperparámetros**, vigilar la generalización, detectar sobreajuste, elegir el mejor modelo (**early stopping**) |
| **Test** | **solo** la evaluación final con datos nunca vistos |
- Reparto ejemplo: 70 / 15 / 15.
- **Evitar data leakage**: los tres conjuntos deben ser **independientes** (p.ej. imágenes casi duplicadas o fotogramas del mismo vídeo no pueden repartirse entre conjuntos).

**Sobreajuste (overfitting)**: la red aprende patrones **específicos del entrenamiento** que **no generalizan**.
- **Underfitting**: modelo demasiado simple. **Buen ajuste**: capta la tendencia sin el ruido. **Overfitting**: ajusta hasta el ruido.
- Señal típica: la **pérdida de validación sube** mientras la de entrenamiento sigue bajando → ahí está el mejor modelo / early stopping.
- Objetivo: no solo pérdida baja en entrenamiento, sino **buena generalización**.

**Regularización** (4 técnicas):
1. **Data augmentation**: crear ejemplos nuevos con transformaciones que **conservan la etiqueta** (volteo horizontal, recorte/zoom, rotación pequeña, brillo/color). Especialmente importante en visión.
2. **Weight decay (L2)**: `L_total = L_clasif + λ||W||²` → penaliza pesos grandes, desalienta soluciones demasiado complejas.
3. **Dropout**: **desactivar neuronas al azar durante el entrenamiento** (en test se usa la red completa) → reduce la co-adaptación y el sobreajuste.
4. **Early stopping**: parar cuando la validación deja de mejorar y quedarse con el modelo de menor pérdida de validación.

**Desvanecimiento del gradiente (vanishing gradients)**: en backprop se multiplican muchos términos → las primeras capas reciben gradientes **muy pequeños** → aprenden muy despacio. (**Exploding**: gradientes muy grandes → entrenamiento inestable.)
- Remedios: activaciones tipo **ReLU**, **buena inicialización** de pesos, **normalización** (batch normalization), **conexiones residuales / skip**.
- ⚠️ **Distinción clave**: el vanishing gradient es un problema de **optimización**; el overfitting es un problema de **generalización**.

### 1.4 Redes convolucionales (CNN)
**¿Por qué no FC para imágenes?** Demasiados parámetros e ignora la estructura espacial.
- **Conectividad local**: cada neurona mira solo un **campo receptivo** local (p.ej. 5×5×3): suma ponderada + bias + activación.
- **Compartición de pesos (weight sharing)**: el **mismo filtro** se aplica en **todas las posiciones** → un filtro = un canal de salida (**mapa de características**).
- → **Reducen enormemente los parámetros** y aprenden características visuales útiles.
- Jerarquía: 32×32×3 → 32×32×64 → 16×16×128 → 8×8×256 → clasificador.

**Pooling / submuestreo**:
- **Max pooling** 2×2 con stride 2: se queda con la **respuesta más fuerte** de cada vecindario (4×4 → 2×2). Ej.: [[1,3],[4,6]] → 6. **Average pooling**: la media.
- ¿Por qué? **Reduce tamaño y cómputo**; **robustez a pequeñas traslaciones**; **amplía el campo receptivo efectivo** en capas profundas.
- Bloque típico: **Conv + ReLU → Pooling → Conv + ReLU → Pooling → Clasificador (FC + softmax)**.

**De mapas a probabilidades**:
1. **Mapas de alto nivel** (salida del backbone), p.ej. 8×8×D.
2. **Global Average Pooling (GAP)**: cada mapa se reduce a **un valor** (media espacial) → vector 1×1×D. Ej.: mapa 2×2 [[1,2],[3,2]] → (1+2+3+2)/4 = 2.0.
3. **Cabeza de clasificación**: FC + softmax → probabilidades de K clases.

### 1.5 CNNs preentrenadas y transfer learning
- Una CNN **preentrenada** ya ha aprendido características en un dataset fuente (p.ej. **ImageNet**). **Transfer learning** reutiliza ese conocimiento para una **nueva tarea** → menos datos y menos entrenamiento.
- Entrenar desde cero: muchos datos etiquetados, mucho coste, pesos aleatorios.

| **Extracción de características (feature extraction)** | **Fine-tuning** |
|---|---|
| Backbone **congelado** (frozen) | Se **actualizan** algunas o todas las capas del backbone (las primeras suelen seguir congeladas) |
| **Solo se entrena el clasificador nuevo** | Se entrena el clasificador **y** parte del backbone |
| Rápido y simple; funciona con **pocos datos** | Más adaptable; necesita **más datos y cómputo**; entrenamiento más delicado |
| Cuando hay pocos datos o hay que entrenar rápido | Cuando hay más datos y se necesita más precisión |
"Feature extraction **reutiliza** representaciones; fine-tuning las **adapta**."

**CNN preentrenada como extractor**: imagen → backbone → mapas H×W×D → **GAP** → **embedding visual** `x ∈ R^D` → sirve para **clasificación** (clasificador lineal + softmax) **o** para **matching/búsqueda de imágenes** (comparar embeddings con **similitud coseno**). Da **descriptores reutilizables**, no solo predicciones.

### 1.6 Vision Transformers (ViT)
Pipeline: imagen → **parches** (p.ej. 16×16) → **embeddings de parche (tokens)** (+ **codificación posicional**) → **encoder Transformer** (**self-attention** + MLP) → predicción de clase.
- **CNN**: filtros convolucionales **locales** que combinan información espacial progresivamente.
- **ViT**: la **self-attention** modela relaciones entre parches, **incluso lejanos** (contexto global desde la primera capa).
- Igual que las CNN, los ViT sirven como **backbones preentrenados** (clasificación, segmentación...). (CLIP y SAM usan ViT.)

---

## PARTE 2 — TRAMPAS TÍPICAS DE TEST
- La no linealidad la dan las **activaciones**; sin ellas, varias capas = una lineal.
- Softmax: probabilidades que suman 1. Cross-entropy con one-hot = −ln p̂ de la clase real.
- Iteración = un mini-batch; época = todo el dataset. 1000/100 → 10 iteraciones por época.
- Validación → hiperparámetros y early stopping; test → solo evaluación final.
- Overfitting: validación sube y entrenamiento baja. Regularización: data augmentation, L2, dropout, early stopping.
- Dropout solo en **entrenamiento**.
- Vanishing gradient = optimización; overfitting = generalización. Remedios: ReLU, inicialización, batch norm, residuales.
- CNN: **conectividad local + compartición de pesos** → pocos parámetros.
- Max pooling → reduce tamaño, robustez a traslaciones, amplía el campo receptivo. GAP → un valor por canal.
- Feature extraction: backbone congelado, solo clasificador; fine-tuning: se adapta parte del backbone.
- ViT: parches + self-attention (relaciones globales).
