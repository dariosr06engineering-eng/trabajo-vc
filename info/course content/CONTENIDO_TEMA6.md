# TEMA 6 — Descripción de regiones y representaciones visuales

> Teoría: `TEORIA/L6_Region Description and Visual Representations_2026_v2.pdf`
> Prácticas: `PRACTICAS/P6 - Region description (Shape, Appearance, CLIP)/` → 6.1 Forma · 6.2 Apariencia · 6.3 CLIP

---

## PARTE 1 — TEORÍA

### 1.1 Introducción
Un **descriptor de región** representa una región segmentada con un **vector de características compacto** `f ∈ R^D`. Debe conservar la información necesaria para **emparejar, reconocer y clasificar**.
Propiedades deseadas: **invarianza a traslación, rotación y escala**.

Tres familias + aprendidas:
| Familia | Pregunta que responde | Ejemplos |
|---|---|---|
| **Forma** | ¿Qué forma tiene? | descriptores geométricos, momentos, Hu, Fourier |
| **Apariencia** | ¿Qué aspecto tiene? | estadísticos de intensidad/color, GLCM, textones |
| **Agregación de características locales** | ¿Qué patrones locales hay y con qué frecuencia? | descriptores locales, vocabulario visual, BoW |
| **Aprendidas** | El modelo aprende qué información es útil | embeddings de CNN/ViT, CLIP |

Todas producen un **vector de longitud fija**.

### 1.2 Forma
Se describe la geometría usando:
- el **contorno** (boundary): perímetro, forma del contorno, **descriptores de Fourier**.
- la **máscara** (todos los píxeles, valores 0/1, **no intensidades**): área, centroide, **momentos geométricos**.

**Descriptores geométricos simples (proporciones)**:
- **Circularidad**: `C = 4πA / P²` → **círculo perfecto C = 1**; formas menos circulares → C menor.
- **Elongación**: `E = L_max / L_min` del **rectángulo envolvente rotado** (mínima área) → ≥ 1.
- **Excentricidad**: `e = sqrt(1 − b²/a²)` (a, b semiejes de la elipse equivalente) → **e = 0 círculo**, mayor e → más alargado.

**Momentos geométricos** de una máscara binaria M(x,y):
`m_pq = Σ_x Σ_y x^p · y^q · M(x,y)` → resumen la distribución espacial de los píxeles.
- **Área**: `m00 = A`
- **Centroide**: `x̄ = m10/m00`, `ȳ = m01/m00`
- **Momentos centrales** (respecto al centroide): `μ_pq = Σ Σ (x − x̄)^p (y − ȳ)^q M(x,y)`
- **Matriz de segundo orden**: `C = (1/μ00)·[μ20 μ11; μ11 μ02]` → misma interpretación de autovalores/autovectores que **Harris**, pero describiendo la **dispersión espacial de los píxeles de la región**.
  - Autovalores λ1 ≥ λ2. **Orientación** `θ = atan2(v1y, v1x)` (autovector principal). **Elongación** = `sqrt(λ1/λ2)`.

**Momentos invariantes**:
1. **Traslación**: usar momentos **centrales** μ_pq (centrar las coordenadas).
2. **Escala**: normalizar por el área: `η_pq = μ_pq / μ00^(1 + (p+q)/2)`.
3. **Rotación**: **momentos de Hu** (`φ1 = η20 + η02`, ..., `φ7`) → **descriptor de 7 dimensiones** invariante a **traslación, escala y rotación**.

**Descriptores de Fourier** (del contorno):
- El contorno cerrado de N puntos se convierte en una **secuencia compleja 1D periódica**: `z_n = x_n + j·y_n`.
- **DFT**: `Z_k = Σ_n z_n · e^(−j2πkn/N)` → N coeficientes complejos (frecuencias espaciales).
- **Compacidad**: quedarse con **K << N coeficientes de baja frecuencia** (forma global; las altas frecuencias son detalle/ruido).
- **Invarianzas**:
  1. **Traslación**: el coeficiente DC `Z_0` depende de la posición → **Z_0 = 0** (o restar el centroide).
  2. **Escala**: dividir todos por una magnitud de referencia (p.ej. `|Z_1|`).
  3. **Rotación y punto de inicio**: solo afectan a las **fases** → usar solo los **módulos** `|Z_k|`.

### 1.3 Apariencia
**Estadísticos de intensidad (gris)**: media (brillo) `μ_I`, varianza (contraste) `σ_I²`, mín/máx, percentiles, histograma.
**Estadísticos de color (RGB)**: color medio `μ_c = [μR, μG, μB]`, **covarianza de color** `Σ_c`, histograma de color (por canal o conjunto), otros espacios (HSV, Lab...).

**Textura: la disposición espacial importa**. El histograma dice **qué valores hay, pero no dónde** → regiones grandes, rayas y ajedrez pueden tener el **mismo histograma** (50 % negro, 50 % blanco). La textura describe la **disposición espacial y repetición** de valores.

**GLCM (Gray-Level Co-occurrence Matrix)**:
- `A(i,j)` = nº de veces que un píxel de nivel **i** tiene un vecino de nivel **j** a un **desplazamiento d = (Δx, Δy)**.
- **Cuantización**: los 256 niveles se reducen a **L = 8, 16 o 32** → `dim(A) = L × L`.
- A diferencia del histograma, **sí contiene información espacial**.
- Se normaliza: `C = A / ΣA` (probabilidades).
- **Descriptores**:
  | Medida | Fórmula | Alta cuando... |
  |---|---|---|
  | **Contraste** | `Σ (i−j)² C(i,j)` | fuertes diferencias locales |
  | **Energía** | `Σ C(i,j)²` | texturas uniformes o muy regulares |
  | **Homogeneidad** | `Σ C(i,j) / (1 + |i−j|)` | vecinos con intensidades parecidas |
  | **Entropía** | `−Σ C log C` | texturas complejas / aleatorias |
- Problema práctico: depende del **offset d** y de la **cuantización L** elegidos.

**Textones**: la textura se describe por los **patrones locales que se repiten**. **Textón** = patrón local representativo.
1. Extraer parches/características locales.
2. Asignar cada uno al textón más cercano de un **diccionario** `T = {t1..tK}`.
3. **Descriptor = histograma de apariciones de textones**.
→ A diferencia del GLCM, **no se elige un offset d** explícito.

### 1.4 Agregación de características locales: Bag of Visual Words (BoW)
**Offline (crear vocabulario)**:
1. Imágenes de entrenamiento → descriptores locales (p.ej. **SIFT**, 128-D).
2. **K-means** sobre todos los descriptores.
3. **Vocabulario visual** `V = {w1..wK}`: cada palabra visual = un **centroide**.

**Online (codificar una región)**:
1. Región segmentada → keypoints → descriptores locales `d_i`.
2. Asignar cada descriptor a la palabra más cercana: `q(d_i) = argmin_k ||d_i − w_k||`.
3. **Descriptor BoW = histograma de palabras visuales** `h = [h1..hK]`.

| Ventajas | Limitación principal |
|---|---|
| Descriptor de **dimensión fija (K)** | **Se descarta la disposición espacial** |
| **Independiente del nº de keypoints** | (mismas palabras en distintas posiciones → mismo histograma) |
| Robusto a reordenamientos locales | |
| Simple e interpretable | |

"BoW sabe **qué** patrones locales aparecen y **cuántas veces**, pero **no dónde**."

### 1.5 Representaciones aprendidas
- En vez de diseñar el descriptor a mano, **se aprende de los datos**. "Handcrafted: **nosotros decidimos** qué medir. Aprendidas: **el modelo aprende** qué información visual es útil".
- **Vision backbone**: parte de la red que transforma la imagen en características aprendidas:
  - **CNN**: ResNet, EfficientNet, ConvNeXt...
  - **Transformers**: ViT, Swin Transformer.
- **Características de región a partir de feature maps**: imagen → backbone → mapas `F_c(x,y)` → **region pooling** (media dentro de la máscara): `f_c = (1/|R|) Σ_{(x,y)∈R} F_c(x,y)` → embedding de longitud fija.
- **Embeddings aprendidos**: el espacio de características se organiza según el **objetivo de entrenamiento** (clasificación, contrastivo, reconstrucción): regiones similares → **distancia pequeña**. Aplicaciones: **matching, retrieval, clasificación, reconocimiento**.

---

## PARTE 2 — PRÁCTICAS (código)

### 6.1 Descriptores de forma (caracteres de matrícula: 0, J, B, 6)
**Circularidad**
```python
contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
A = cv2.contourArea(cnt)          # área encerrada por el contorno
P = cv2.arcLength(cnt, True)      # perímetro; True = contorno cerrado
C = 4*np.pi*A / P**2
```
- `RETR_EXTERNAL`: solo contornos **exteriores** (ignora agujeros). `CHAIN_APPROX_NONE`: guarda **todos** los puntos del contorno (`SIMPLE` comprimiría los segmentos rectos).
- `cv2.findContours` devuelve `(contours, hierarchy)` en OpenCV 4.
- El **0** tiene la circularidad más alta (forma más parecida a un círculo).
- Circularidad: invariante a traslación, rotación y escala (salvo efectos de discretización).
- ⚠️ `contourArea` del contorno exterior **no resta los agujeros** (la B y el 0 "rellenos").

**Elongación**
```python
_, (w, h), _ = cv2.minAreaRect(cnt)     # rectángulo ROTADO de área mínima: (centro, (ancho, alto), ángulo)
E = max(w, h) / min(w, h)
```
- El **0 y la B tienen la misma elongación** porque tienen el **mismo rectángulo envolvente** (mismo tamaño de carácter).
- Con `minAreaRect` es invariante a rotación. Con `cv2.boundingRect` (rectángulo **recto**, alineado con los ejes) **NO sería invariante a rotación**.
- Con solo (C, E) no se distinguen bien todos los caracteres → poco discriminativo.

**Momentos de Hu**
```python
m = cv2.moments(region, binaryImage=True)   # binaryImage=True: todo píxel ≠ 0 cuenta como 1
h = cv2.HuMoments(m).flatten()              # 7 valores; flatten porque devuelve (7,1)
h_log = -np.sign(h) * np.log10(np.abs(h))   # escala log: los valores son de órdenes de magnitud muy distintos
```
- Prueba de invarianza (J): original, rotada (`np.rot90`), escalada ×2 (`cv2.resize` con `INTER_AREA`), espejo (`np.fliplr`).
  - Rotada y escalada → **mismos Hu** (salvo pequeñas diferencias por discretización).
  - **Espejo** → **h7 cambia de signo** (el resto igual). Problema para caracteres que son espejo de otro (p.ej. distinguir J de una "L"/"J" invertida), aunque puede ser útil para detectar reflejos.

**Solidez y agujeros (extra que hiciste)**
```python
hull = cv2.convexHull(contours[0])
solidity = np.count_nonzero(region) / cv2.contourArea(hull)   # área real / área de la envolvente convexa
contours, hierarchy = cv2.findContours(region, cv2.RETR_CCOMP, ...)   # CCOMP: jerarquía de 2 niveles (exteriores/agujeros)
n_holes = np.sum(hierarchy[0,:,3] != -1)   # hierarchy[..., 3] = padre; tener padre → es un agujero
```
- Agujeros: 0 → 1, B → 2, 6 → 1, J → 0. Muy discriminativo.

**Matrícula real**
```python
_, plate_bin = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # INV: caracteres negros → blancos
chars = [c for c in contours if 0.45*H < cv2.boundingRect(c)[3] < 0.7*H]   # filtrar por altura
chars.sort(key=lambda c: cv2.boundingRect(c)[0])                           # ordenar de izquierda a derecha por x
```
- `THRESH_BINARY_INV`: `findContours` busca objetos **blancos** sobre fondo negro.

### 6.2 Descriptores de apariencia
**Momentos centrales del histograma**
```python
hist = cv2.calcHist([image], [0], None, [256], [0,256]).flatten()   # [imagen], canal 0, sin máscara, 256 bins, rango
p = hist / hist.sum()                       # probabilidades
mu = np.sum(np.arange(256) * p)             # media
moment_n = Σ (z − mu)^n · p[z]              # n=0 → 1, n=1 → 0, n=2 → varianza, n=3 → asimetría...
```
- `calcHist` recibe **listas** (`[image]`, `[0]`, `[256]`). El `None` es la máscara (con una máscara solo se cuentan esos píxeles).
- μ0 = 1 siempre y μ1 = 0 siempre (momentos centrales).

**Cuantización y GLCM**
```python
quantized = (image.astype(np.uint16) * L // 256).astype(np.uint8)   # uint16 para que image*L no desborde
dx, dy = d
for y in range(max(0,-dy), min(rows, rows-dy)):       # rangos para no salirse de la imagen
    for x in range(max(0,-dx), min(cols, cols-dx)):
        A[image[y,x], image[y+dy, x+dx]] += 1          # (nivel del píxel, nivel del vecino)
```
- ⚠️ Si se multiplica `uint8 * L` sin convertir, se **desborda** (overflow).
- Indexación `image[y, x]` = (fila, columna).

**Características GLCM**
```python
C = A / A.sum()
i, j = np.indices(C.shape)
contrast = np.sum((i-j)**2 * C)
energy = np.sum(C**2)
homogeneity = np.sum(C / (1 + np.abs(i-j)))
entropy = -np.sum(C[C>0] * np.log(C[C>0]))     # solo C>0: log(0) = −inf
```

**Invarianzas (conclusión de la práctica)**:
- **Momentos del histograma**: invariantes a **rotación** (mismos píxeles) y prácticamente a **escala** (el histograma normalizado apenas cambia). No usan información espacial.
- **GLCM con un solo offset d=(1,1)**: **no** invariante a rotación (al rotar 90°, la diagonal cambia de dirección) ni a escala (al escalar ×2, el vecino a distancia 1 es más parecido → el contraste baja).
- **GLCM con 4 direcciones promediadas** (0°, 45°, 90°, 135° → offsets (1,0), (1,1), (0,1), (−1,1)): **invariante a rotaciones de 90°** (las direcciones se intercambian entre sí). Escala: sigue sin serlo, pero si la **distancia se escala igual que la imagen** (dist=2 para ×2) se recuperan los valores.

### 6.3 Representaciones aprendidas: CLIP
**CLIP**: entrenado con **400 millones de pares (imagen, texto)** de internet (sin etiquetas manuales). En cada batch de N pares calcula la matriz N×N de similitudes y una **pérdida contrastiva** aumenta la similitud de los N pares correctos y reduce la de los N²−N incorrectos. Resultado: imágenes y textos en el **mismo espacio de embeddings**.

```python
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai', device=device)
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model.eval()                                  # modo inferencia (desactiva dropout, etc.)

batch = torch.stack([preprocess(Image.fromarray(img)) for img in images]).to(device)
with torch.no_grad():                         # sin gradientes: no se entrena → menos memoria y más rápido
    F = model.encode_image(batch)
F = F / F.norm(dim=-1, keepdim=True)          # normalizar a norma 1 → producto escalar = coseno
```
- `preprocess`: redimensiona (224), **recorte central**, a tensor y normaliza con la media/desviación de CLIP.
- `Image.fromarray` porque `preprocess` espera una **imagen PIL** (en **RGB**).
- `torch.stack` → batch de forma (N, 3, 224, 224).
- Embedding de **512 dimensiones** (ViT-B/32), con **norma 1** tras normalizar.

**Clasificación zero-shot**
```python
T = embed_texts(['a photo of ' + label for label in labels])   # plantilla de prompt
S = 100 * F @ T.T                                              # similitudes coseno escaladas (temperatura)
S = np.exp(S - S.max(axis=1, keepdims=True))                  # softmax estable (restar el máximo)
P = S / S.sum(axis=1, keepdims=True)                          # N×K probabilidades
```
- **Zero-shot**: clasifica en categorías **dadas como texto**, **sin reentrenar**.
- `F @ T.T` = coseno porque ambos están normalizados. El ×100 es la escala/temperatura aprendida por CLIP (separa más las probabilidades).
- Se resta el máximo antes del `exp` para evitar overflow (no cambia el resultado).

**Recortes de regiones**
```python
crop = image[y:y+h, x:x+w]                    # bbox (x, y, w, h)
square = np.full((size,size,3), 255)          # se rellena a CUADRADO con blanco
```
- Se rellena a cuadrado porque `preprocess` hace **recorte central**: con un recorte alargado (lámpara, nevera) se perdería parte del objeto.
- Fallos típicos: regiones pequeñas o ambiguas, recortes con contexto engañoso, etiquetas poco descriptivas. Mejora: recortes con algo de contexto, prompts más descriptivos.

**Visualización del espacio de características**
```python
points = PCA(n_components=2).fit_transform(F)   # 512-D → 2-D
```
- Objetos del mismo tipo (taburetes, lámparas) quedan **cerca**.
- Diferencia con el espacio (circularidad, elongación) de 6.1: aquel tiene 2 dimensiones **diseñadas e interpretables**; el de CLIP tiene 512 **aprendidas**, con información **semántica**.

---

## PARTE 3 — TRAMPAS TÍPICAS DE TEST
- Circularidad = 4πA/P² (círculo = 1). Elongación con rectángulo **rotado** → invariante a rotación.
- m00 = área; centroide = (m10/m00, m01/m00).
- Momentos centrales → traslación; normalización por μ00 → escala; Hu → rotación. **Hu = 7 valores.** Espejo → cambia el signo de h7.
- Fourier: Z0 = 0 → traslación; dividir por |Z1| → escala; solo módulos → rotación y punto de inicio.
- Histograma: qué valores, **no dónde**. GLCM: sí incluye información espacial; depende de d y L; tamaño L×L.
- Textones: histograma de patrones; **no hace falta elegir d**.
- BoW: vocabulario con **K-means**; dimensión fija K; **pierde la disposición espacial**.
- CLIP: contrastivo, pares imagen-texto, zero-shot, embeddings normalizados, similitud coseno.
- Handcrafted: decidimos qué medir. Aprendidas: el modelo aprende qué es útil.
