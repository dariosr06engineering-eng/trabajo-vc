# TEMA 3 — Detección de bordes (Edge Detection)

> Teoría: `TEORIA/L3_Edge detection 2026_v8.pdf`
> Prácticas: `PRACTICAS/P3 - Edge detection/` → 3.1 Operadores de primera derivada · 3.2 Segunda derivada y Canny

---

## PARTE 1 — TEORÍA

### 1.1 ¿Qué es un borde y para qué sirve?
- Un **borde** es una transición entre dos regiones con niveles de gris (o color) muy distintos.
- Representación **compacta** de los contornos: una imagen RGB 750×531×24 bits (1.19 MB) frente a su mapa de bordes 750×531×1 bit (49.8 kB) → **24× menos datos** y conserva mucha información.
- Útil para **segmentación, reconocimiento de objetos y reconstrucción 3D**.
- Modelo ideal 1D (escalón con rampa): parámetros **incremento H = A − B**, **ángulo de la pendiente**, **coordenada x0** del punto medio. En imágenes reales no se cumple exactamente: son **discretas** y tienen **ruido**.

**Causas físicas de un borde** (no siempre es el límite de un objeto ⚠️):
1. Límite de objeto (objeto/fondo)
2. Cambio de orientación de la superficie (dos caras del mismo objeto)
3. Iluminación (sombras)
4. Material/color (reflectancia distinta)
5. Textura

**Tres tipos de error de un detector**:
- **Detección**: falsos negativos (bordes reales no detectados) y falsos positivos (bordes falsos por ruido).
- **Localización**: la respuesta no está en la posición exacta.
- **Respuesta múltiple**: varias respuestas para un mismo borde.
→ Son exactamente los **3 criterios de Canny**.

### 1.2 Operadores de primera derivada (gradiente)
En 1D, el borde está:
- en el **punto de inflexión** de f,
- en el **máximo de |f'|** (mayor variación),
- en el **cruce por cero de f''**.

En 2D, el **gradiente** es un vector: `∇f = (∂f/∂x, ∂f/∂y)`
- **Módulo** (fuerza del borde): `|∇f| = sqrt(fx² + fy²) ≈ |fx| + |fy|`
- **Orientación**: `θ = atan2(fy, fx)` → dirección de **máxima variación positiva**.
- ⚠️ **El gradiente es PERPENDICULAR al borde.**

**Aproximación discreta**: diferencias entre píxeles. Diferencia simétrica: `∂f/∂x ≈ (f(i, j+1) − f(i, j−1))/2` → kernel `[1 0 −1]/2` (ya volteado).
Gradiente = convolución con dos plantillas: `fx = f ⊗ hx`, `fy = f ⊗ hy`.

**Sobel** (convención: x = columnas crecientes, y = filas crecientes):
```
hx = 1/4·[1 0 -1; 2 0 -2; 1 0 -1]      hy = 1/4·[1 2 1; 0 0 0; -1 -2 -1]
```
Proceso: `fx`, `fy` → `|∇f|` → **umbral** → mapa de bordes. Umbral alto (120) → menos bordes; bajo (80) → más bordes (y más ruido).

**Tamaño del operador** = compromiso **detección vs. localización**:
- **Pequeño** (Sobel 3×3): estimación local → **mejor localización**, pero **más sensible al ruido**.
- **Grande** (Sobel 5×5): menos local → **peor localización**, pero **más robusto al ruido**.

### 1.3 El ruido importa → suavizar
La derivada amplifica el ruido: la respuesta al ruido puede ser tan grande como la del propio escalón.
**Solución**: suavizar con una gaussiana y luego derivar: `d/dx (f ⊗ g)`.

Por la propiedad de la derivada de la convolución: `d/dx (f ⊗ g) = f ⊗ (dg/dx)` → **una sola convolución** con la derivada de la gaussiana (se precalcula).

**DroG (Derivative of Gaussian)**:
- Combina suavizado y gradiente. **σ controla el suavizado**.
- `∂g/∂x = −(x/σ²)·g(x,y)` ; `∂g/∂y = −(y/σ²)·g(x,y)`
- Es separable: `DroG_x = g'(x)·g(y)` → derivada gaussiana en una dirección y suavizado gaussiano en la otra.
- Ejemplo con σ pequeño: `DroG_x ≈ 1/8·[1 0 −1; 2 0 −2; 1 0 −1]` (¡se parece a Sobel!).
- σ = 1 → bordes finos y textura; σ = 2 → estructuras intermedias; σ = 4 → solo estructuras grandes, respuestas más anchas.

### 1.4 Operadores de segunda derivada (Laplaciano)
- En 1D: f' tiene máximo/mínimo en el borde → f'' = 0 con **cambio de signo**.
- **Laplaciano**: `∇²f = ∂²f/∂x² + ∂²f/∂y²` → ⚠️ es un **ESCALAR**, no un vector (no da orientación).
- Discreto: `∂²f/∂x² ≈ f(i,j+1) − 2f(i,j) + f(i,j−1)` → kernel `[1 −2 1]`.
- Por la propiedad **distributiva**: `[1 −2 1]` horizontal + vertical = **`[0 1 0; 1 −4 1; 0 1 0]`** → una sola convolución.

**Cruce por cero (zero-crossing)** en un píxel p con vecino q si:
1. el Laplaciano **cambia de signo**: `L(p)·L(q) < 0`
2. el cambio es suficientemente grande: `|L(p) − L(q)| > T`

**Propiedades del Laplaciano**:
- **Invariante a rotación (isótropo)**.
- Bordes localizados por cruces por cero → bordes **finos**.
- **Muy sensible al ruido** (la segunda derivada amplifica alta frecuencia).

**LoG (Laplacian of Gaussian)**: `∇²(f ⊗ g) = f ⊗ ∇²g`
- Suavizado gaussiano + Laplaciano. Simetría radial → **isótropo**. Forma de "sombrero mexicano".
- **No da orientación del borde** (a diferencia del gradiente).
- σ grande: suprime ruido y detalle fino, pero peor localización y pierde estructuras pequeñas.

**DoG (Difference of Gaussians)**: aproximación del LoG **sin segundas derivadas**:
`DoG = G(kσ) − G(σ) ≈ (k−1)σ²·∇²G` → se suaviza a dos escalas cercanas (k > 1) y se restan.
Ventaja: cada gaussiana es **separable** → eficiente. (Se reutiliza en **SIFT**, tema 4.)

### 1.5 Detector de Canny
**Detector óptimo** para un escalón 1D con ruido gaussiano según 3 criterios:
1. **Buena detección** (alta probabilidad de bordes reales, baja de falsos).
2. **Buena localización** (cerca de la posición real).
3. **Respuesta única** (un borde → una respuesta).

El filtro óptimo se **aproxima muy bien por la primera derivada de una gaussiana (DroG)**.

**Algoritmo** (memorizar el orden):
1. **Gradiente** (módulo y orientación) con DroG (suavizado + derivada).
2. **Supresión de no máximos (NMS)** a lo largo de la dirección del gradiente.
3. **Doble umbral + histéresis**.

**NMS**:
- Cuantizar la orientación del gradiente a 4 direcciones: **0°, 45°, 90°, 135°**.
- Comparar `|∇f|` del píxel con sus **2 vecinos en la dirección del gradiente**: si es máximo local se mantiene; si no, **se pone a 0**.
- Resultado: bordes **finos, de ~1 píxel**. (Los bordes forman "crestas" en el módulo; se queda solo la cima.)
- Ej.: θ = 32° → se cuantiza a 45°.

**Histéresis**: un único umbral no sirve (alto → contornos rotos; bajo → mucho ruido). Con dos umbrales `T_low < T_high`:
- `|∇f| ≥ T_high` → **borde fuerte**: se mantiene.
- `|∇f| < T_low` → se **rechaza**.
- `T_low ≤ |∇f| < T_high` → **borde débil**: se mantiene **solo si está conectado a un borde fuerte**.

**Elección de σ**: σ pequeño → detalles finos pero más ruido/textura; σ grande → elimina ruido y textura fina, pierde bordes pequeños y localización.

### Resumen de L3
| | Qué es |
|---|---|
| Primer orden | DroG → módulo y dirección del gradiente tras suavizar |
| Segundo orden | Cruces por cero del LoG; DoG aproxima el LoG |
| Geometría | Gradiente apunta al mayor aumento, **normal al borde** |
| Canny | Suavizar → gradiente → NMS → histéresis |
| Color | Usar intensidad, combinar canales o gradiente de color; pasar a gris puede perder bordes definidos solo por color |

---

## PARTE 2 — PRÁCTICAS (código)

### 3.1 Operadores de primera derivada
```python
kernel_h = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])      # Prewitt horizontal
kernel_v = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])      # Prewitt vertical
d_h = cv2.filter2D(image, cv2.CV_16S, kernel_h)
d_v = cv2.filter2D(image, cv2.CV_16S, kernel_v)
```
- ⚠️ **¿Por qué `cv2.CV_16S`?** Las derivadas tienen **valores negativos** y pueden pasar de 255. Con `uint8` (`ddepth=-1`) los negativos se **saturan a 0** y se pierde la mitad de los bordes (los de transición claro→oscuro).
- La derivada horizontal detecta **bordes verticales** y viceversa.

**Gradiente + umbral**
```python
gradient = np.absolute(d_h) + np.absolute(d_v)                    # aproximación |fx|+|fy|
norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)     # reescala a [0,255]
ret, edges = cv2.threshold(norm, threshold, 255, cv2.THRESH_BINARY)
```
- Se usa el **valor absoluto** porque un borde de bajada da derivada negativa.
- Se **normaliza** para que el umbral 0–255 tenga sentido.
- Sobel dividido entre 4 (`*1/4`): para que la respuesta no se dispare (los pesos positivos suman 4) y quede en la escala de la intensidad.

**DroG**
```python
g1D = np.array([1/(sigma*np.sqrt(2*np.pi)) * np.e**(-(z**2)/(2*sigma**2)) for z in range(-w, w+1)])
g2D = signal.convolve2d(g1D.reshape(2*w+1,1), g1D.reshape(1,2*w+1))   # gaussiana 2D (separable)
x = np.arange(-w, w+1)       # vector fila
y = np.vstack(x)             # vector columna
DroG_h = g2D * (-x) / sigma**2      # ∂g/∂x
DroG_v = g2D * (-y) / sigma**2      # ∂g/∂y
edge_detection_chart(image, DroG_h, DroG_v, threshold)
```
- `x` (fila) multiplica por columnas → derivada en x; `np.vstack(x)` lo pone en columna → derivada en y (broadcasting de NumPy).
- σ grande → menos ruido, bordes más gruesos.

### 3.2 Segunda derivada y Canny
**Visualización del LoG**: se aplica el kernel Laplaciano a una gaussiana 2D → sombrero mexicano.
```python
laplace_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype="float")
LoG = cv2.filter2D(gauss_filter, -1, laplace_filter)
```

**Suavizado gaussiano normalizado**
```python
kernel_1D = np.array([np.exp(-(z**2)/(2*s**2)) for z in range(-w, w+1)])
kernel_1D = kernel_1D / np.sum(kernel_1D)          # normalizar → suma 1 (no cambia el brillo)
g2D = np.dot(vertical_kernel, horizontal_kernel)   # producto exterior columna·fila
smoothed = cv2.filter2D(image, -1, g2D)
smoothed_norm = cv2.normalize(smoothed, None, 0, 254, cv2.NORM_MINMAX, cv2.CV_8U)
```

**Laplaciano con y sin suavizado**
```python
lap = cv2.Laplacian(image, cv2.CV_16S, ksize=size_Laplacian)   # ksize impar (1,3,5,7)
lap_abs = cv2.convertScaleAbs(lap)                              # valor absoluto + convertir a uint8
lap_blur = cv2.Laplacian(blurred, cv2.CV_16S, ksize=size_Laplacian)   # ≈ LoG
```
- Sin suavizar, el Laplaciano saca **muchísimo ruido**.
- La salida del Laplaciano **no es todavía una imagen de bordes**: hace falta detectar **cruces por cero**.

**Zero-crossing (implementado a mano)**
```python
lap = cv2.normalize(lap, None, -255, 255, cv2.NORM_MINMAX)   # mantiene el signo
for x in range(1, width-1):
    for y in range(1, height-1):
        neighbors = [8 vecinos]
        if (lap[x,y] > T and any(n < -T for n in neighbors)) or \
           (lap[x,y] < -T and any(n > T for n in neighbors)):
            edges[x,y] = 255
```
- Busca **cambio de signo** con un vecino y magnitud por encima de T. Se normaliza a [−255, 255] (no a [0,255]) porque **hay que conservar el signo**.
- Los bucles empiezan en 1 y acaban en `n-1` para no salirse por los bordes de la imagen.

**Canny**
```python
canny        = cv2.Canny(image,   lower_threshold, upper_threshold)
canny_blurred = cv2.Canny(blurred, lower_threshold, upper_threshold)
```
- `cv2.Canny` calcula el gradiente con **Sobel** (apertureSize=3) y hace NMS + histéresis, **pero no aplica un suavizado gaussiano previo configurable** → suavizar antes da bordes más limpios.
- Los umbrales "buenos" dependen de cada imagen (no hay valores universales). Típicamente `upper ≈ 2–3 × lower`.
- Si `lower > upper`, OpenCV los intercambia.

---

## PARTE 3 — TRAMPAS TÍPICAS DE TEST
- Gradiente **perpendicular** al borde.
- Laplaciano = **escalar** (no da orientación); gradiente = vector (sí).
- LoG y Laplaciano son **isótropos**; DroG/Sobel son direccionales.
- Kernel pequeño → mejor localización/más ruido; grande → al revés.
- DoG ≈ LoG y es más eficiente (gaussianas separables).
- Canny ≈ DroG + NMS + histéresis. El filtro óptimo de Canny ≈ **primera** derivada de gaussiana (no LoG).
- NMS → bordes de 1 píxel; histéresis → conecta contornos.
- Borde débil se mantiene **solo si está conectado a uno fuerte**.
- Un borde no implica límite de objeto (sombras, textura, iluminación).
- Derivadas en OpenCV → usar `CV_16S`/`CV_64F`, nunca `uint8`.
- Pasar a gris puede perder bordes definidos solo por color.
