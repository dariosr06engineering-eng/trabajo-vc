# ALL — Visión por Computador (todos los temas)

Archivo generado uniendo CONTENIDO_TEMA2.md … CONTENIDO_TEMA8.md. Índice en INDICE.md.


---

# TEMA 2 — Procesamiento de imagen (Image Processing)

> Teoría: `TEORIA/L2_Image processing 2026_v4.pdf`
> Prácticas: `PRACTICAS/P2 - Image processing/` → 2.1 IP tools · 2.2 Smoothing · 2.3 Image enhancement

---

## PARTE 1 — TEORÍA

### 1.1 Objetivo del procesamiento de imagen
Mejorar la calidad de la imagen: **atenuar ruido, ajustar color, modificar brillo/contraste, reescalar**. En visión por computador suele ser el paso previo a extracción de características, segmentación o reconocimiento.

Una imagen es a la vez:
- una **matriz** `A[x, y]`
- una **función discreta** `f(x, y)` → son lo mismo.

### 1.2 Color
- Las cámaras (CMOS) y el ojo (conos) son sensibles a ciertas longitudes de onda.
- **RGB** es la representación estándar del curso. Un color es un vector 3D (R, G, B) y se puede manipular linealmente (sumar colores = sumar vectores).
- **R = G = B → nivel de gris** (la diagonal del cubo RGB).
- Imagen color = array 3D: `I(fila, columna, canal)`.
- **RGB → gris**: `I = αR + βG + γB`. Lo más simple: **media** `I = (R+G+B)/3` (proyección sobre la diagonal, α=β=γ=1/3).

### 1.3 Histograma
- Frecuencia con que aparece cada intensidad. Una imagen color tiene **3 histogramas** (uno por canal).
- Da **estadística** de intensidades (brillo, contraste) pero **pierde la información espacial** ⚠️ (dos imágenes muy distintas pueden tener el mismo histograma).
- **Brillo** = media de intensidades = **momento de orden 1**: `μ = Σ i·h(i)/N`
- **Contraste** = **desviación típica** = raíz del **2º momento central**: `σ = sqrt(Σ (i−μ)²·h(i)/N)`
- Sirve para: detectar **saturación** (en negro o blanco), conocer brillo/contraste, elegir **umbral de binarización**, ver si se usa todo el rango 0–255.

### 1.4 LUT (Look-Up Tables)
- `g(i,j) = LUT(f(i,j))`: sustituye cada intensidad de entrada por la que indica la tabla (un vector de 256 posiciones).
- **No cambia la geometría** de los objetos, **solo el histograma**.
- En RGB es como cambiar la paleta de colores.
- Muy eficientes (solo un acceso a tabla por píxel).

### 1.5 Convolución
Operación entre dos imágenes que produce otra; una de ellas es pequeña (el **kernel** con los pesos).

`g(i,j) = (f ⊗ h)(i,j) = Σ_m Σ_n f(i−m, j−n)·h(m,n) = Σ Σ f(i+m, j+n)·h(−m,−n)`

→ El kernel se **voltea** (flip horizontal + vertical = rotación 180°) antes de multiplicar y sumar.

Ejemplos de kernels:
- `[0 0 0; 0 1 0; 0 0 0]` → identidad (misma imagen)
- `[0 0 0; 0 0 1; 0 0 0]` (tras el flip) → imagen **desplazada 1 píxel**
- `1/9·ones(3,3)` → **suavizado** (media de los 8 vecinos + él)

**Propiedades** (muy preguntables):
| Propiedad | Expresión |
|---|---|
| Conmutativa | f⊗h = h⊗f |
| Asociativa | (f⊗h1)⊗h2 = f⊗(h1⊗h2) → base de la **separabilidad** |
| Distributiva | f⊗(h1+h2) = f⊗h1 + f⊗h2 → base del Laplaciano en 1 convolución |
| Asociativa con escalar | a(f⊗h) = (af)⊗h = f⊗(ah) |
| Derivada | D(f⊗h) = D(f)⊗h = f⊗D(h) → base del **DroG/LoG** |
| Teorema de convolución | F(f⊗h) = F(f)·F(h) (no se aplica en el curso) |

**Padding (p)** = ceros añadidos alrededor. **Stride (s)** = salto del kernel.
**Tamaño de salida**: `o = floor((i + 2p − k) / s) + 1`
- Ej.: entrada 6×6, kernel 3×3, p=0, s=1 → 4×4. Con p=1 → 6×6 (mismo tamaño). Con s=2, p=0 → 2×2.
- El stride es una forma de **submuestrear** (se usa mucho en CNNs).

### 1.6 Redimensionado
**Upsampling (interpolación)**: para cada píxel de salida, se mapea a la entrada y se combinan píxeles:
- **Vecino más cercano** (nearest): valor del píxel más próximo → pixelado, el más rápido.
- **Bilineal**: media ponderada de los **4 vecinos**.
- **Bicúbica**: media ponderada de los **16 vecinos** → el más suave y el más caro.

**Downsampling**:
- **Decimación**: quedarse con 1 de cada M muestras (`img[::M, ::M]`).
- **Remuestreo por interpolación** (nearest, bilineal, bicúbica).

**Aliasing**: al reducir resolución sin más aparecen **patrones falsos / moiré / bordes dentados**.
→ **Solución: suavizar (Gaussiano) ANTES de submuestrear** (anti-aliasing): se eliminan detalles finos que la baja resolución no puede representar.

### 1.7 Suavizado (Smoothing)
Objetivo: eliminar/reducir **ruido** (de sensor —más en CMOS—, digitalización, transmisión).

Tipos de ruido:
- **Impulsivo / sal y pimienta**: píxeles aislados con valores anómalos (mín./máx. → negro/blanco).
- **Gaussiano**: variaciones aleatorias aditivas de media cero.

| Método | Tipo | Bueno para | Inconveniente |
|---|---|---|---|
| **Media de vecindario** | Lineal (convolución, kernel 1/p) | Ruido gaussiano | **Emborrona bordes** |
| **Gaussiano** | Lineal, **separable** | Ruido gaussiano | Emborrona (controlado por σ) |
| **Mediana** | **NO lineal** | **Sal y pimienta** | Coste alto (hay implementaciones eficientes) |
| **Media temporal** | Media de M imágenes | Ruido gaussiano, **preserva bordes** | Solo escena **estática** sin cambios de luz |

**Filtro Gaussiano**: `g(x,y) = 1/(2πσ²)·exp(−(x²+y²)/2σ²) = g(x)·g(y)` → **separable**.
- Una convolución 2D N×N cuesta **O(N²)** por píxel; dos 1D (fila + columna) cuestan **O(2N)** → mismo resultado, mucho más eficiente (por la propiedad asociativa).
- **σ mayor → más suavizado → imagen más borrosa**. El tamaño del kernel `w` debe ser proporcional a σ (suficiente para recoger los valores no despreciables).

**Mediana**: ordena el vecindario y toma el valor central. Ej.: vecindario {10,10,10,10,11,11,12,12,50} → mediana = 11 (el 50 "ruido" desaparece). **Preserva bordes** y elimina sal y pimienta.

**Media temporal**: `g = (1/M) Σ f_i = f_sin_ruido + (1/M) Σ η_i` → el término de ruido tiende a 0 (media cero). **No funciona con sal y pimienta.**

### 1.8 Realce (Image Enhancement)
Mejorar brillo y contraste estirando/desplazando el histograma mediante **LUTs**.
- **LUT lineal a trozos**: estira un rango de intensidades.
- **Corrección gamma** `g = f^γ` (f normalizada a [0,1]):
  - **γ < 1** → aumenta el contraste de los **oscuros** (estira intensidades bajas, aclara). Ej. γ=0.5 (raíz).
  - **γ > 1** → aumenta el contraste de los **claros** (comprime los bajos, oscurece). Ej. γ=3.
- **Ecualización del histograma**: intenta un histograma **uniforme**. LUT = **histograma acumulado** (CDF) escalado: `T(i) = 255·Σ_{k=0..i} p(k)`. Útil para imágenes oscuras / bajo contraste.
  - En la práctica **no sale perfectamente uniforme** porque hay pocos niveles discretos (256) y no se pueden "partir" las barras.
- **CLAHE** (Contrast Limited Adaptive HE): ecualiza **localmente** por regiones (tiles) y **limita el contraste** (clip limit) para no amplificar el ruido.

### Resumen de L2
- Color = array 3D `I(x, y, c)`.
- Histograma → brillo/contraste, pero sin información espacial.
- Convolución: operación lineal clave en IP y CNNs.
- Ruido gaussiano → kernel gaussiano. Sal y pimienta → mediana (no lineal).
- LUTs: eficaces y eficientes para brillo/contraste.

---

## PARTE 2 — PRÁCTICAS (código)

### 2.1 IP tools
```python
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # leer en gris (equivale a flag 0)
image_cropped = image[300:400, 200:475]          # recorte: [filas, columnas] = [y, x]
plt.hist(image.ravel(), 256, range=[0,256])      # ravel() aplana la matriz a 1D para el histograma
```
**Binarización**
```python
ret, binarized = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
```
- Devuelve **una tupla** (umbral usado, imagen). Si haces `img = cv2.threshold(...)` sin desempaquetar, `img` es una tupla → falla al mostrar.
- `> threshold` → 255, resto → 0.

**LUT lineal con recorte**
```python
lut = np.arange(0, 256)
lut = np.clip((lut - a) * b, 0, 255)   # desplaza a, estira por b, recorta a [0,255]
im_lut = cv2.LUT(image, lut)
```
- `np.clip` es imprescindible: sin él habría valores <0 o >255.
- ⚠️ `cv2.LUT` espera una LUT de **256 elementos**; lo correcto es que sea `uint8` (`lut.astype(np.uint8)`).

**Convolución manual (widget)** — el notebook muestra exactamente el algoritmo:
```python
flipped_kernel = np.flip(kernel)                         # 1. voltear el kernel
padded = np.pad(img, pad_width=p, mode='constant')       # 2. padding con ceros
patch = padded[r*s : r*s+3, c*s : c*s+3]                 # 3. ventana según stride
result = np.sum(patch * flipped_kernel)                  # 4. multiplicar y sumar
out_h = (3 - 3 + 2*p)//s + 1                             # tamaño de salida
```
**Convolución con OpenCV**
```python
kernel = np.ones((3,3))/9
im_conv = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
```
- `ddepth=-1` → la salida tiene el **mismo tipo** que la entrada (uint8).
- ⚠️ **`cv2.filter2D` hace CORRELACIÓN, no convolución (no voltea el kernel)**. Para kernels simétricos (media, gaussiano) da igual; para derivadas cambia el signo.

**Interpolación**
```python
dim = (new_width, new_height)                              # ⚠️ cv2.resize usa (ancho, alto) = (cols, filas)
nearest  = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
bilinear = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
bicubic  = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
```
- `image.shape` = (filas, cols) = (alto, ancho) pero `cv2.resize` pide `(ancho, alto)`. Confundirlo es error típico.

**Downsampling con y sin anti-aliasing**
```python
decimated    = image[::step, ::step]                        # decimación directa → aliasing/moiré
smoothed     = cv2.GaussianBlur(image, (0,0), sigmaX=sigma) # (0,0) → OpenCV calcula el tamaño a partir de sigma
anti_aliased = smoothed[::step, ::step]                     # suavizar y luego decimar
```

### 2.2 Smoothing
**Filtro de media**
```python
kernel = np.ones((2*w+1, 2*w+1), np.float32) / ((2*w+1)**2)   # w=1 → 3x3, w=2 → 5x5
smoothed = cv2.filter2D(image, cv2.CV_8U, kernel)
```
- `w_kernel` es la "apertura": tamaño = 2w+1 (siempre impar → hay píxel central).
- Los pesos suman 1 → no cambia el brillo medio.

**Filtro Gaussiano (separabilidad)**
```python
kernel_1D = np.float32([gauss(z, sigma) for z in range(-w, w+1)])
vertical   = kernel_1D.reshape(2*w+1, 1)     # columna
horizontal = kernel_1D.reshape(1, 2*w+1)     # fila
kernel_2D  = signal.convolve2d(horizontal, vertical)   # = producto exterior np.outer(k1, k1)
smoothed = cv2.filter2D(image, cv2.CV_8U, kernel_2D)
```
- Columna ⊗ fila = matriz (producto exterior) → separabilidad.
- Si no se **normaliza** el kernel (suma ≠ 1) la imagen se oscurece o aclara. Con la fórmula continua 1/(σ√2π) y pocas muestras la suma no es exactamente 1.

**Mediana**
```python
ksize = w if w % 2 != 0 else w + 1     # ⚠️ medianBlur exige tamaño IMPAR y ≥ 3
smoothed = cv2.medianBlur(img_uint8, ksize)
```
- `cv2.medianBlur` con tamaño par → **error**. (Lo comentaste tú mismo en el notebook: "falla en lo de impar".)

**Media temporal**
```python
selected = burst[:burst_length]           # primeras N imágenes de la secuencia
avg = np.average(selected, axis=0)        # media píxel a píxel a lo largo del eje 0 (tiempo)
avg = avg.astype(np.uint8)                # la media devuelve float64 → convertir para mostrar
```
- `axis=0` es el eje de las imágenes. Con `axis=1` o `2` promediarías filas/columnas (mal).
- Conclusión práctica: más imágenes → menos ruido, pero solo con escena estática.

**Medición de tiempos** con `perf_counter_ns()`: la mediana y la media temporal suelen ser las más costosas; media y gaussiano son convoluciones rápidas.

### 2.3 Image enhancement
**Gamma sobre la luminancia (YCrCb)**
```python
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
lut = np.array([((i/255.0)**gamma)*255 for i in np.arange(256)]).astype("uint8")
out[:,:,0] = cv2.LUT(image[:,:,0], lut)       # solo canal Y (luminancia)
out_rgb = cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)
```
- **¿Por qué YCrCb?** Se modifica solo la **luminancia (Y)** y se conservan los canales de **crominancia** (Cr, Cb) → **no se alteran los colores**. Si se ecualizan R, G y B por separado, los colores cambian.
- Se normaliza a [0,1] antes de elevar a γ y luego se reescala a 255.

**Ecualización**
```python
out[:,:,0] = cv2.equalizeHist(image_ycrcb[:,:,0])   # equalizeHist solo acepta 1 canal uint8
```
**CLAHE**
```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out[:,:,0] = clahe.apply(image_ycrcb[:,:,0])
```
- Mayor `clipLimit` → más contraste local pero **más ruido** amplificado.
- CLAHE da un resultado más natural que la ecualización global.

---

## PARTE 3 — TRAMPAS TÍPICAS DE TEST
- El histograma **no** tiene información espacial.
- Una LUT **no** cambia la geometría, solo el histograma.
- La mediana **no es lineal** → no se puede implementar como convolución.
- Mediana → sal y pimienta. Gaussiano/media → ruido gaussiano. Media temporal → gaussiano, no sal y pimienta.
- La media temporal **preserva bordes**; la media espacial los **emborrona**.
- Gaussiano separable: O(N²) → O(2N) por píxel.
- γ<1 aclara/realza oscuros; γ>1 oscurece/realza claros.
- La ecualización no da un histograma perfectamente uniforme (niveles discretos).
- Antes de submuestrear hay que **suavizar** (aliasing).
- Bicúbica = 16 vecinos; bilineal = 4; nearest = 1.
- `cv2.imread` carga en **BGR**; `plt.imshow` espera **RGB** → colores cambiados si no se convierte.
- `cv2.filter2D` = correlación. `ddepth=-1` = mismo tipo que la entrada.
- Tamaño de salida de la convolución: `(i + 2p − k)/s + 1`.


---

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


---

# TEMA 4 — Detección y descripción de puntos clave (Keypoints)

> Teoría: `TEORIA/L4_Keypoint detection_desciption_2026_v7.pdf`
> Prácticas: `PRACTICAS/P4 - Keypoints (Harris, NCC)/` → 4.1 Harris y NCC
> ⚠️ **No tienes código de SIFT** → al final hay un código de referencia hecho por mí.

---

## PARTE 1 — TEORÍA

### 1.1 ¿Qué es un keypoint?
Puntos de interés / *feature points*: **posiciones distintivas** de la imagen que se pueden **detectar y emparejar de forma fiable** entre imágenes distintas.

Tres problemas: **DETECTAR** (¿dónde?), **DESCRIBIR** (¿cómo es su vecindario?), **EMPAREJAR** (¿cuál corresponde a cuál?).

- **Buen detector**: puntos **precisos, repetibles y distintivos**, de forma **eficiente**.
- **Buen descriptor**:
  - **Invarianza/robustez**: el mismo punto de la escena da descriptores parecidos aunque cambie la imagen (rotación, escala, iluminación...).
  - **Distintividad**: puntos distintos → descriptores distintos.

Aplicaciones: (1) recuperar **relaciones geométricas** (stitching/panorámicas, estéreo, SLAM, seguimiento); (2) **reconocimiento y recuperación de imágenes**.

### 1.2 Detector de esquinas de Harris
**Esquina** = punto donde la intensidad varía mucho **en dos direcciones**.
Intuición: ventana W que se desplaza:
- **Región plana**: no cambia en ninguna dirección.
- **Borde**: cambia solo al moverse en una dirección (perpendicular al borde).
- **Esquina**: cambia en todas las direcciones.

**Formulación**: cambio al desplazar la ventana (Δx, Δy):
`E(Δx,Δy) = Σ_W w(x,y)·[I(x+Δx, y+Δy) − I(x,y)]²`
Con Taylor (aproximación lineal) `I(x+Δx, y+Δy) ≈ I + Ix·Δx + Iy·Δy`:
`E(Δx,Δy) ≈ [Δx Δy] · M · [Δx Δy]ᵀ`

**Matriz M (tensor de estructura)**, con ponderación gaussiana w centrada en el punto:
```
M = Σ_W w(x,y) · | Ix²    IxIy |
                  | IxIy   Iy²  |
```
- M es **simétrica** → 2 autovectores ortogonales (direcciones principales de variación) y 2 autovalores λ1, λ2 (cuánto varía en cada una).
- Al **rotar** la imagen, los autovectores rotan pero los **autovalores no cambian** → invarianza a rotación.

**Clasificación por autovalores**:
| λ1, λ2 | Región |
|---|---|
| ambos pequeños | **Plana** |
| uno grande, otro pequeño | **Borde** |
| ambos grandes | **Esquina** |

**Respuesta de Harris** (evita calcular autovalores):
`R = det(M) − k·(traza M)² = λ1·λ2 − k·(λ1+λ2)²` con **k = 0.04 – 0.06** (empírico)
- **R > 0 grande → esquina**
- **R < 0 → borde**
- **|R| ≈ 0 → región plana**

**Algoritmo**:
1. Derivadas Ix, Iy (p.ej. Sobel).
2. Suavizado gaussiano de las 3 imágenes Ix², Iy², IxIy.
3. Imagen R con det y traza.
4. Umbral: R > T.
5. **Supresión de no máximos** → máximo local de cada región.

**Propiedades de Harris**:
- Simple y eficiente.
- Robusto al ruido (por el suavizado).
- **Invariante a rotación** (depende de autovalores).
- Invariante a cambios **aditivos** de brillo; **solo parcialmente** a cambios de contraste.
- ⚠️ **NO es invariante a escala** (una esquina vista de cerca parece un borde curvo).

**Shi-Tomasi**: mismo principio y misma M, pero criterio **min(λ1, λ2) > umbral**, calculando los autovalores explícitamente. Mejor comportamiento ante **deformaciones afines** (las rectas paralelas siguen paralelas). Se usa para seleccionar puntos en el tracker **Lucas-Kanade / KLT**.

### 1.3 Emparejamiento con parches (patches)
¿Qué punto de la imagen g corresponde al keypoint de f? Se compara el parche alrededor del punto.

- **SSD** (distancia L2²): `Σ (f − g)²`
- **SAD** (distancia L1): `Σ |f − g|` → más barata computacionalmente.
- ⚠️ SSD y SAD **NO son invariantes a brillo ni contraste**. Aun así SAD se usa en **tracking en secuencias** (entre fotogramas consecutivos el brillo apenas cambia).

**Correlación cruzada (CC)**: `CC(i,j) = Σ f(i+m, j+n)·h(m,n)` → como la convolución **pero sin voltear** el kernel. (Convolución = correlación con el kernel volteado.)
- Tampoco es invariante a brillo/contraste → hay que **normalizar**.

**NCC (Normalized Cross-Correlation)**:
1. Quitar el **brillo**: restar la media del parche → media 0.
2. Quitar el **contraste**: dividir por la norma → norma 1.
3. `NCC = Σ f̂·ĝ`.

**Interpretación geométrica**: cada parche es un vector (dimensión = nº de píxeles); NCC = **producto escalar de dos vectores unitarios = cos θ** del ángulo entre ellos.
- Rango **[−1, 1]**. ≈1 → muy parecidos.
- ⚠️ **NCC NO es invariante a rotación ni a escala** del parche (sí a brillo y contraste).

### 1.4 SIFT (Scale Invariant Feature Transform) — Lowe, 2004
Detecta estructuras distintivas que siguen detectándose cuando cambia su **tamaño aparente**. Da **detector** (dónde y a qué escala) **+ descriptor** (cómo es el vecindario).

**Pipeline**: Imagen → Detector (x, y, σ) → Orientación (x, y, σ, θ) → Descriptor **128-D**.

**Espacio de escalas**: `L(x,y,σ) = G(x,y,σ) ⊗ I(x,y)`. Aumentar σ elimina detalle fino y resalta estructuras grandes. Estructuras pequeñas se detectan a escala fina; grandes, a escala gruesa.

**Muestreo**:
- **Octava** = rango de escalas de factor 2 a la **misma resolución**.
- Con **s = 3** → cada octava tiene **s+3 = 6 imágenes gaussianas** → **5 DoG**.
- La siguiente octava = imagen **submuestreada por 2** (pirámide gaussiana). Unas 4–6 octavas útiles.

**DoG**: se restan niveles gaussianos adyacentes: `D(x,y,σ) = L(x,y,kσ) − L(x,y,σ)` (aproximación del LoG).

**Detección de extremos**: cada píxel candidato de una DoG se compara con sus **26 vecinos**:
- **8** en su misma escala
- **9** en la escala superior
- **9** en la inferior
→ Es keypoint si es **mayor o menor que los 26**.
Se descartan los de **bajo contraste** y los **mal localizados sobre bordes**.

¿Por qué detecta **blobs**? Recordar: cruces por cero del LoG → bordes. Aquí: **extremos de la DoG en posición y escala → centros de blobs**. Cada keypoint tiene una **escala característica σ** (el tamaño del círculo dibujado).

**Paso 1 – Orientación dominante**:
- En la imagen gaussiana de la escala del keypoint, se calculan módulo y orientación del gradiente en un vecindario (~16×16).
- **Histograma de 36 bins** (cada 10°, 360°). El **pico más alto** = orientación del keypoint.
- Picos **≥ 80 % del máximo** → keypoints **adicionales** (misma posición y escala, otra orientación).
- Este histograma de 36 bins es **solo para la orientación**; el descriptor usa **8 bins**.

**Paso 2 – Descriptor**:
- Ventana a la escala del keypoint y **rotada según su orientación dominante** (→ invarianza a rotación).
- Ponderación gaussiana.
- Se divide en **4×4 celdas**, cada una con un **histograma de 8 orientaciones**.
- **16 × 8 = 128 valores**.
- Se **normaliza** → robustez a iluminación.

**Matching de SIFT**: distancia **euclídea** entre descriptores de 128 dimensiones; se elige el más cercano. **Ratio test de Lowe**: aceptar si `d1/d2 < ratio` (d1 = más cercano, d2 = segundo). Ratio 0.95 → muchos emparejamientos (y más falsos); **0.75 → menos pero más fiables**.

SIFT funciona con imágenes **rotadas, a distinta escala y con distinta iluminación**.

### Resumen comparativo (tabla del final de las diapositivas)
| Método | Idea | Invariancia / robustez | Papel |
|---|---|---|---|
| **Harris** | Esquinas a partir del tensor de estructura | Rotación ✔ · Escala ✘ | Detección |
| **Shi-Tomasi / KLT** | Esquinas con min de los dos autovalores | Rotación ✔ · Escala ✘ | Detección / tracking |
| **NCC** | Comparar parches | Brillo y contraste ✔ · Escala ✘ · Rotación ✘ | Emparejamiento |
| **SIFT** | Extremos DoG + descriptor de orientaciones de gradiente | Escala ✔ · Rotación ✔ · Robusto a iluminación | Detección + descripción + matching |

---

## PARTE 2 — PRÁCTICA 4.1 (código)

### Demo del tensor M (ejemplos sintéticos plano / borde / esquina)
```python
Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)     # dx=1, dy=0 → derivada en x
Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)     # derivada en y
g = cv2.getGaussianKernel(n, sigma); W = g @ g.T    # ventana gaussiana 2D (producto exterior)
M = [[ΣW·Ix², ΣW·IxIy], [ΣW·IxIy, ΣW·Iy²]]
E = d.T @ M @ d                                     # cambio estimado para el desplazamiento d
eigvals, eigvecs = np.linalg.eigh(M)                # eigh: para matrices simétricas
R = np.linalg.det(M) - k * np.trace(M)**2
```
- Plano: λ1 ≈ λ2 ≈ 0. Borde: un λ grande. Esquina: los dos grandes.
- Al rotar el patrón cambian los elementos de M pero **no los autovalores**.
- `cv2.cornerHarris` usa **media de vecindario** (no gaussiana) para la ventana.

### Harris con OpenCV
```python
gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))   # cornerHarris exige float32
R = cv2.cornerHarris(gray, blockSize, ksize, k)              # blockSize = tamaño de W, ksize = apertura de Sobel
threshold = const_k * R.max()                                # umbral relativo al máximo
```
- Argumentos: `blockSize` (tamaño de la ventana W), `ksize` (Sobel), `k` (0.04–0.06).
- **W = 1** → no hay vecindario: M tiene determinante ≈ 0 (rango 1) y no se distinguen esquinas. **W grande** → esquinas menos localizadas, se fusionan.
- La imagen se convirtió a RGB antes, pero en `apply_harris` se usa `COLOR_BGR2GRAY` → intercambia los pesos de R y B (efecto pequeño, pero es un "error" detectable).
- El umbral depende de la imagen y de los parámetros, por eso se toma **relativo a R.max()**.

### Supresión de no máximos
```python
mx = signal.order_filter(cim, np.ones([sze,sze]), sze**2 - 1)   # filtro de MÁXIMO en ventana sze×sze
bordermask[radius:rows-radius, radius:cols-radius] = 1          # ignora los bordes de la imagen
r, c = np.where((cim == mx) & (cim > thresh) & (bordermask == 1))
```
- `order_filter` con rango `sze²−1` = el **último** del orden = el máximo (filtro de dilatación).
- Un píxel es keypoint si es **igual al máximo de su vecindario** (máximo local) **y** supera el umbral.
- `np.where` devuelve **(filas, columnas)** = (y, x).
```python
kps = [cv2.KeyPoint(float(c[i]), float(r[i]), 1.0) ...]    # ⚠️ KeyPoint(x=columna, y=fila, size)
```

### Correlación vs NCC a mano
```python
corr = np.sum(patch * template)                             # CC: depende del brillo
patch_n = (patch - patch.mean()) / np.sqrt(np.sum((patch - patch.mean())**2))
ncc = np.sum(patch_n * template_n)                          # ∈ [−1, 1]
```
- La CC puede dar **más valor a un parche más brillante aunque no se parezca**; la NCC corrige eso.

### Template matching
```python
template = image[r-w:r+w+1, c-w:c+w+1]                      # parche (2w+1)×(2w+1) centrado en el keypoint
ncc = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)   # TM_CCOEFF_NORMED = NCC
max_y, max_x = np.where(ncc == np.amax(ncc))
```
- La salida de `matchTemplate` tiene tamaño `(H−h+1, W−w+1)` y cada posición es la **esquina superior izquierda** del parche → hay que sumar `w` para el centro (o usar padding).
- La respuesta NCC es un mapa de similitud; el máximo (≈1) está donde se tomó el template.
- Con un template de **otra imagen** similar, el máximo es menor que 1 y puede haber falsos.

### Emparejamiento entre dos imágenes + homografía (stitching)
```python
image_l_pad = cv2.copyMakeBorder(image_l, w, w, w, w, cv2.BORDER_REFLECT)   # padding por reflexión
for cada keypoint izquierdo:
    template = image_l_pad[...]
    ncc = cv2.matchTemplate(image_r_pad, template, cv2.TM_CCOEFF_NORMED)
    ncc_at_kps = [ncc[r_r[j], c_r[j]] for j in ...]     # solo se evalúa en los keypoints derechos
    if max(ncc_at_kps) > 0.65: matches.append((i, argmax))
matches = [cv2.DMatch(queryIdx, trainIdx, distance) ...]
H, mask = cv2.findHomography(pts_r, pts_l, cv2.RANSAC, 5.0)   # RANSAC descarta outliers; 5 px de tolerancia
result = cv2.warpPerspective(image_r, H, (2*w, h))             # deforma la derecha al plano de la izquierda
result[0:h, 0:w] = image_l                                     # pega la izquierda
```
- **Padding** para poder sacar parches de keypoints cercanos al borde. `BORDER_REFLECT` es mejor que ceros (no mete bordes artificiales).
- Gracias al padding, las coordenadas del keypoint en `ncc` coinciden directamente.
- Umbral 0.65: si baja → más emparejamientos y más falsos.
- **RANSAC**: estima la homografía robustamente aunque haya emparejamientos erróneos.
- Homografía: transformación proyectiva 3×3 (8 grados de libertad, mínimo **4 correspondencias**).
- `warpPerspective` recibe el tamaño como **(ancho, alto)**.

---

## PARTE 3 — CÓDIGO DE REFERENCIA (no está en tus prácticas): SIFT
```python
sift = cv2.SIFT_create()                              # (OpenCV ≥ 4.4)
kp1, des1 = sift.detectAndCompute(gray1, None)        # kp: (x, y, size=escala, angle=orientación); des: N×128
kp2, des2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher(cv2.NORM_L2)                       # distancia euclídea
knn = bf.knnMatch(des1, des2, k=2)                    # los 2 más cercanos para el ratio test
good = [m for m, n in knn if m.distance < 0.75 * n.distance]   # ratio test de Lowe
img = cv2.drawMatches(img1, kp1, img2, kp2, good, None)
```
- `des` tiene **128 columnas** (4×4×8). `NORM_L2` porque son vectores reales (para ORB/BRIEF binarios sería `NORM_HAMMING`).
- `k=2` es necesario para el ratio test.

---

## PARTE 4 — TRAMPAS TÍPICAS DE TEST
- Harris: invariante a **rotación**, **NO a escala**. R>0 esquina, R<0 borde, R≈0 plano. k = 0.04–0.06.
- Harris usa `R` (det y traza) para **no calcular autovalores**; Shi-Tomasi sí los calcula (`min(λ1,λ2)`).
- M es simétrica; sus autovalores no cambian al rotar.
- SSD/SAD/CC **no** invariantes a brillo/contraste; NCC **sí**, pero NCC **no** a rotación/escala.
- NCC = coseno entre vectores normalizados, ∈ [−1, 1].
- Correlación = convolución sin voltear el kernel.
- SIFT: s=3 → 6 gaussianas → 5 DoG por octava; 26 vecinos; orientación 36 bins (10°); picos ≥80 % → keypoints extra; descriptor 4×4×8 = **128**; ratio de Lowe 0.75–0.8.
- SIFT detecta **blobs** (extremos en DoG), no esquinas.
- `cv2.KeyPoint(x, y)` → x = columna; `np.where` → (fila, columna).
- `cv2.cornerHarris` exige `float32`.


---

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


---

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


---

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


---

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

