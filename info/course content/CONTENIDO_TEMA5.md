# TEMA 5 — Segmentación de imágenes (Image Segmentation)

> Teoría: `TEORIA/L5_Segmentation 2026_v10.pdf`
> Prácticas: `PRACTICAS/P5 - Segmentation (Hough, SAM)/` → 5.1 Hough · 5.3 SAM
> ⚠️ **No tienes el código de la 5.2** (umbral/Otsu, umbral adaptativo, region growing, K-means) → al final hay código de referencia hecho por mí.

---

## PARTE 1 — TEORÍA

### 1.1 ¿Qué es segmentar?
Particionar la imagen en regiones **asignando una etiqueta a cada píxel**: `S(x,y) = k` ⇒ el píxel (x,y) pertenece a la región k.
Las regiones pueden representar:
- **Similitud visual** (intensidad, color, textura, proximidad) → segmentación **clásica**.
- **Entidades con significado** (clases semánticas: carretera, persona...; o instancias concretas) → segmentación **basada en aprendizaje**.

Mapa del tema (diapositiva resumen):
- **Clásica** (criterios diseñados a mano):
  - **Basada en contornos** (bordes, Hough)
  - **Umbralización** (global, Otsu, adaptativa)
  - **Basada en regiones** (region growing, K-means, GMM/EM)
  - **Basada en grafos** (grafo ponderado, corte, normalized cut)
- **Basada en aprendizaje** (redes neuronales):
  - **Semántica** (una clase por píxel)
  - **Instancias** (clase + ID de instancia)
  - **Panóptica** (semántica + instancias)
  - **Promptable / class-agnostic** (SAM)

### 1.2 Segmentación basada en contornos
- Identifica regiones por los **contornos** que las separan.
- Problema: los bordes son cambios **locales**, **no contornos cerrados** → hace falta un mecanismo para conectarlos/interpretarlos.

**Transformada de Hough**: encuentra **formas paramétricas** (rectas, circunferencias...) a partir de puntos de borde mediante **votación**:
- Cada punto de borde **vota por los parámetros de todas las formas que podrían pasar por él**.
- Los votos se acumulan en el **espacio de parámetros (acumulador)**.
- Los **picos** del acumulador = formas apoyadas por muchos puntos.
- Pipeline: bordes → Hough → forma detectada → contorno del objeto → máscara.

**Rectas**: forma normal `ρ = x·cos θ + y·sin θ`
- ¿Por qué no `y = mx + b`? Porque **m es infinita** en rectas verticales; (ρ, θ) está acotado.
- Cada punto (x, y) se convierte en una **sinusoide** en el espacio (ρ, θ).
- Puntos **colineales** → sus sinusoides se **cortan en el mismo (ρ0, θ0)** → pico.
- Procedimiento: para cada θ discreto se calcula ρ y se suma 1 en la celda (ρ, θ).
- θ ∈ [0, π), ρ ∈ [−d, d] con **d = diagonal de la imagen** = sqrt(alto² + ancho²) (máxima distancia posible al origen).

**Circunferencias**: `r² = (x − a)² + (y − b)²`
- **Radio conocido** → espacio de parámetros (a, b) **2D**.
- **Radio desconocido** → (a, b, r) **3D** (más caro).
- Con r conocido, cada punto de borde vota por una **circunferencia de centros candidatos** en el espacio (a, b).

Ventaja clave: Hough recupera el contorno aunque los bordes estén **incompletos** (sabemos qué familia de formas buscamos). Es robusto a huecos y ruido.

### 1.3 Segmentación por umbral
**Binaria**: `S(x,y) = 1 si I(x,y) > T; 0 en otro caso`.

**Umbral global**: un único T para toda la imagen. Asume **iluminación aproximadamente uniforme**.

**Método de Otsu** (selección automática de T):
- Para cada T posible se calculan:
  - pesos de clase `ω0(T) = Σ_{i≤T} p(i)`, `ω1(T) = Σ_{i>T} p(i)`
  - medias `μ0(T)`, `μ1(T)`
- **Varianza entre clases**: `σ_B²(T) = ω0(T)·ω1(T)·[μ0(T) − μ1(T)]²`
- **T\* = argmax σ_B²(T)** (búsqueda exhaustiva en los 256 valores).
- Equivale a hacer cada clase **lo más compacta posible** (mínima varianza intra-clase).
- Funciona bien con histogramas **bimodales**.

**Umbral adaptativo (local)**: `T(x,y) = f(N_xy) − C`
- `f` = estadístico local, p.ej. la **media** del vecindario (o media gaussiana).
- **C** = desplazamiento que controla lo permisiva que es la decisión fondo/objeto.
- Maneja **variaciones espaciales de iluminación** (ej. sudoku con sombras: el global falla, el adaptativo funciona).

**Adquisición controlada** (p.ej. **retroiluminación / backlighting**): en vez de complicar el algoritmo, **se hace la imagen más fácil de segmentar** → el umbral es muy robusto.

### 1.4 Segmentación basada en regiones
Agrupar píxeles por **similitud de características y/o relaciones espaciales** (p.ej. color RGB: píxeles de color similar están cerca en el espacio RGB).

**Region growing (crecimiento de regiones)**:
- Se parte de una o varias **semillas** y se añaden iterativamente los **vecinos conectados** que cumplan un criterio de similitud (p.ej. `|I(q) − μ_R| < T`).
- Decisiones clave: **selección de semillas** (cuántas, dónde), **conectividad** (4 u 8 vecinos), **criterio de similitud** (intensidad/color/textura; comparar con la semilla o con la **media actual** de la región).
- El resultado **depende del umbral** y **de la implementación/orden** (p.ej. qué semilla empieza).

**K-means**:
- Objetivo: dividir los datos en K clusters minimizando la suma de distancias al cuadrado a sus centroides: `min Σ_k Σ_{x∈C_k} ||x − μ_k||²`
- Algoritmo:
  1. Inicializar K centroides (aleatorio).
  2. **Asignar** cada punto al centroide más cercano.
  3. **Recalcular** cada centroide como la media de sus puntos.
  4. Repetir hasta convergencia.
- Para segmentar: cada píxel → vector de características (intensidad 1D, color 3D...) → cada píxel recibe la etiqueta de su cluster.
- Con solo apariencia, píxeles **alejados** pueden recibir la misma etiqueta (no tiene en cuenta la posición).
- **Color + posición**: `f(x,y) = [R, G, B, λx, λy]` → clusters espacialmente coherentes. Hay que **escalar** color y posición; λ controla su importancia relativa.
- **Ventajas**: simple, eficiente; **converge a un óptimo local** (no necesariamente global).
- **Limitaciones**: **hay que fijar K**; funciona mejor con clusters **compactos y esféricos**; sensible a **outliers**; sensible a la **inicialización** (inicializaciones distintas → clusters distintos).

**GMM (Gaussian Mixture Model) + EM (Expectation-Maximization)**:
- GMM: la distribución de características es una **mezcla de gaussianas**; cada cluster j = `N(μ_j, Σ_j)` con peso `π_j = P(C_j)`.
- `p(x) = Σ_j π_j · N(x; μ_j, Σ_j)`
- Intuición bayesiana: `P(C_j | x) = p(x | C_j)·P(C_j) / p(x)` (posterior ∝ verosimilitud × prior). ⚠️ **verosimilitud p(x|C) ≠ posterior P(C|x)**.
- **EM** estima los parámetros alternando:
  - **E-step**: responsabilidades / asignación **blanda** `γ_ij = P(C_j | x_i)` (cuánto explica la gaussiana j al punto i). Ej.: γ = 0.70, 0.25, 0.05.
  - **M-step**: re-estimar `N_j = Σ_i γ_ij`, `μ_j = (1/N_j) Σ γ_ij x_i`, `Σ_j = (1/N_j) Σ γ_ij (x_i−μ_j)(x_i−μ_j)ᵀ`, `π_j = N_j / N`.
  - Repetir hasta convergencia.
- Clusters de distintos tamaños, formas y orientaciones → bueno para clusters **elípticos**.

| | K-means | GMM con EM |
|---|---|---|
| Cluster | centroide μ_j | μ_j, Σ_j, π_j |
| Inicialización | K medias aleatorias | K gaussianas (μ_j, Σ_j) y P(C_j) |
| Asignación | **Dura** (centroide más cercano) | **Blanda / probabilística** (E-step) |
| Actualización | recalcular μ_j | M-step: μ_j, Σ_j, π_j |
| Forma | esférica, compacta | elíptica, distintas varianzas |
| Base | distancia | probabilidad |

**Basada en grafos**:
- Idea: modelar **explícitamente** qué píxeles están relacionados. Nodos = píxeles; aristas entre vecinos con peso
  `w_ij = exp(−||f_i − f_j||²/σ_f²) · exp(−||x_i − x_j||²/σ_x²)` (similitud de apariencia × cercanía).
  - Color parecido + cerca → peso alto; color distinto → peso bajo; lejos → sin arista.
- **Corte de grafo**: dividir los vértices en A y B: `cut(A,B) = Σ_{i∈A, j∈B} w_ij`. Varias regiones → particionar recursivamente.
- Una buena segmentación elimina conexiones **débiles** y conserva las **fuertes**.
- **Problema del corte mínimo**: tiende a aislar **conjuntos pequeños** poco conectados.
- **Normalized cut**: `Ncut(A,B) = cut(A,B)/assoc(A,V) + cut(A,B)/assoc(B,V)` → penaliza particiones diminutas (assoc pequeño → término grande).
- Ventajas: apariencia + relaciones espaciales; agrupamiento más **global**; regiones de forma irregular.
- Limitaciones: depende mucho de cómo se construye el grafo; **más caro que K-means**; el nº de segmentos sigue requiriendo decisiones.

### 1.5 Segmentación basada en aprendizaje
- La clásica agrupa píxeles en regiones; la **semántica aprendida segmenta y reconoce a la vez**.

**Semántica**: una **clase por píxel** de un conjunto predefinido (cielo, carretera, persona...). **No distingue instancias** de la misma clase.

**Bloques**: convolución (filtros aprendidos → mapas de características), **downsampling** (en el encoder, representaciones más abstractas), **upsampling** (en el decoder, recupera resolución para predecir por píxel). **Embedding** = representación aprendida que se pasa a otra parte del modelo.

**Encoder-decoder**: el encoder construye características de alto nivel (**QUÉ** hay); el decoder recupera la resolución espacial.
**Skip connections**: reutilizan características de alta resolución del encoder → ayudan a recuperar **DÓNDE** está.
**U-Net**: encoder-decoder con skip connections implementadas por **concatenación** de mapas de la misma resolución (`H×W×C1 + H×W×C2 → H×W×(C1+C2)`).

**Instancias**: clase + **ID de instancia** a los píxeles de objetos **contables** ("things"), no a regiones incontables (cielo, césped, carretera). Enfoque típico basado en detección: **Mask R-CNN** (detecta cada objeto y predice clase + máscara).

**Panóptica**: combina semántica + instancias → interpretación completa.
- **Things** (contables): clase + ID de instancia.
- **Stuff** (incontables): solo clase.

**Promptable / class-agnostic — SAM (Segment Anything Model)**:
- Genera máscaras a partir de **prompts**: **puntos, cajas o máscaras**.
- **No es segmentación semántica**: no asigna clases predefinidas → **class-agnostic** (sin etiqueta de clase).
- **Zero-shot**: segmenta objetos nuevos **sin reentrenar**.
- Arquitectura: **image encoder** (pesado, ViT) → **image embedding** ("qué hay en la imagen"); **prompt encoder** → **prompt embedding** ("qué parte me pides"); **mask decoder** (ligero) combina ambos.
  - El **image embedding se calcula una vez** y se reutiliza para muchos prompts.
  - El prompt **no clasifica**, guía al decoder hacia la región.
- En la práctica: prompts interactivos; rejilla de puntos para **segmentar todo automáticamente**; refinamiento interactivo con más prompts; **componible** (los prompts pueden venir de una persona o de otro sistema, p.ej. un detector).
- SAM es un **foundation model** de visión: preentrenado a gran escala con datos amplios y diversos, reutilizable para muchas tareas y dominios.

---

## PARTE 2 — PRÁCTICAS (código)

### 5.1 Hough
**Acumulador a mano**
```python
theta_values = [0, np.pi/4, np.pi/2, 3*np.pi/4]
rho_values = list(range(-d, d+1))                   # ρ con signo: de −d a d
acc = np.zeros([len(rho_values), len(theta_values)])
for (x, y) in puntos:
    for t_idx, theta in enumerate(theta_values):
        rho = int(np.round(x*np.cos(theta) + y*np.sin(theta)))
        acc[rho + d][t_idx] += 1                     # +d: desplaza el índice porque ρ puede ser negativo
```
- Puntos (1,4), (2,3), (3,2) son colineales (x+y=5) → en θ=π/4, ρ = 5/√2 ≈ 3.54 → redondeado a 4 para los tres → **pico con 3 votos**.
- Un acumulador **no puede tener una celda con más votos que puntos blancos** (cada punto vota **una vez por cada θ**, como mucho una vez a cada celda).
- ρ máximo = **diagonal de la imagen**.

**HoughLines (estándar)**
```python
edges = cv2.Canny(gray, 100, 200, apertureSize=3)
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=130)
for l in lines:
    rho, theta = l[0]
    a, b = cos(theta), sin(theta)
    x0, y0 = a*rho, b*rho                       # punto de la recta más cercano al origen
    pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*a))   # extender ±2000 px en la dirección de la recta
    pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*a))
```
- Entrada: imagen de **bordes binaria** (Canny), no la imagen en gris.
- `rho=1` (resolución en píxeles), `theta=π/180` (1°), `threshold` = **mínimo de votos**. Umbral alto → menos rectas.
- Devuelve (ρ, θ) → **rectas infinitas**: se dibujan atravesando zonas sin línea (cielo...). Para quedarse con los píxeles reales: hacer AND con la imagen de bordes o usar la probabilística.
- La dirección de la recta es `(−sin θ, cos θ)` (perpendicular a la normal (cos θ, sin θ)).

**HoughLinesP (probabilística)**
```python
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=250, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line.ravel()
```
- Usa un **subconjunto aleatorio** de puntos → **más rápida**.
- Devuelve **segmentos** con extremos (x1,y1,x2,y2).
- `minLineLength`: longitud mínima del segmento; `maxLineGap`: hueco máximo para unir dos segmentos.
- Sin restricción de tiempo, la estándar es más exacta (usa todos los puntos); la probabilística da segmentos y es más eficiente.
- Si no detecta nada, `lines` es **None** → `for line in lines` lanza un error.

### 5.3 SAM
```python
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)   # vit_b (91M), vit_l (308M), vit_h (636M)
sam.cuda()                                                     # necesita GPU (sin ella falla/lentísimo)
mask_generator = SamAutomaticMaskGenerator(sam, stability_score_thresh=0.96, min_mask_region_area=100)
masks = mask_generator.generate(image_RGB)                     # ⚠️ espera RGB, no BGR
```
- **Modelo más grande** → máscaras más precisas pero más lento y más memoria de GPU.
- `SamAutomaticMaskGenerator` = "segmentar todo" con una **rejilla de puntos** como prompts.
- `stability_score_thresh` alto → solo máscaras muy estables (menos máscaras). `min_mask_region_area` elimina regiones pequeñas.
- Cada máscara es un **diccionario**: `segmentation` (máscara booleana H×W), `area`, `bbox` (**formato XYWH**), `predicted_iou`, `point_coords`, `stability_score`, `crop_box`.
- La RGB se **redimensiona al tamaño de la NIR** (`cv2.resize(img, (nir_width, nir_height))`) para que las máscaras coincidan píxel a píxel con las bandas multiespectrales.

**NDVI** (índice de vegetación, agricultura de precisión)
```python
ndvi = (nir - red) / (nir + red + 1e-10)     # 1e-10 evita división por cero; se pasa a float antes
masked = np.ma.masked_where(~mask['segmentation'], ndvi)   # ~ invierte la máscara: se ocultan los píxeles FUERA
mask['ndvi_mean'] = masked.mean()
```
- NDVI ∈ [−1, 1]: alto (>0.3) → vegetación sana; ~0 → suelo; negativo → agua/no vegetación.
- Hay que convertir a **float** (`astype(float)`): con uint8 la resta se desborda.
- `np.ma.masked_where(cond, x)` **enmascara (excluye) donde la condición es True** → por eso se usa `~segmentation`.

**Visualización**: se ordenan las máscaras por `area` (descendente) y se pinta cada una con un color aleatorio con alfa 0.5 en una imagen RGBA.

---

## PARTE 3 — CÓDIGO DE REFERENCIA (5.2, no está en tus prácticas)
```python
# Umbral global fijo
_, bin1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# Otsu: el umbral que pases (0) se ignora; lo calcula y lo devuelve en T
T, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Umbral adaptativo: T(x,y) = media gaussiana de un bloque 15x15 − C (C=10). blockSize debe ser impar
bin_ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)

# K-means de color
Z = img.reshape(-1, 3).astype(np.float32)             # cada píxel = una fila [B,G,R]; kmeans exige float32
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # 10 inicializaciones
seg = centers.astype(np.uint8)[labels.flatten()].reshape(img.shape)   # cada píxel toma el color de su centroide

# GMM con EM (scikit-learn)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=K).fit(Z)
probs = gmm.predict_proba(Z)                          # asignación BLANDA (E-step)

# Region growing (esqueleto)
def region_growing(img, seed, T):
    region = np.zeros(img.shape, bool); region[seed] = True
    stack = [seed]; mean = float(img[seed]); n = 1
    while stack:
        y, x = stack.pop()
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:          # 4-conectividad
            ny, nx = y+dy, x+dx
            if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1] and not region[ny, nx]:
                if abs(float(img[ny, nx]) - mean) < T:       # criterio: parecido a la MEDIA de la región
                    region[ny, nx] = True; stack.append((ny, nx))
                    mean = (mean*n + float(img[ny, nx])) / (n+1); n += 1
    return region
```
- Con `THRESH_OTSU` el umbral pasado se ignora. Otsu y la imagen deben ser de 8 bits de un canal.
- `cv2.kmeans` exige `float32`; reshape a (N, 3) porque agrupa **vectores**, no imágenes.

---

## PARTE 4 — TRAMPAS TÍPICAS DE TEST
- Bordes ≠ contornos cerrados → hace falta Hough u otro mecanismo.
- Hough: punto → **sinusoide** en (ρ, θ); colineales → se cortan; se usa (ρ, θ) porque m→∞ en verticales.
- Circunferencias: radio conocido → 2D; desconocido → 3D.
- Otsu **maximiza la varianza entre clases** (= minimiza la intra-clase).
- Umbral adaptativo para **iluminación no uniforme**; `T = media local − C`.
- K-means: hay que fijar K, óptimo **local**, sensible a inicialización y outliers, clusters esféricos.
- EM: E-step = responsabilidades (blando); M-step = μ, Σ, π.
- Normalized cut evita los cortes de conjuntos diminutos del corte mínimo.
- Semántica no distingue instancias; instancias solo "things"; panóptica = things + stuff.
- U-Net: skip connections por **concatenación**.
- SAM: promptable, **class-agnostic**, **zero-shot**, foundation model; image embedding calculado una vez.
