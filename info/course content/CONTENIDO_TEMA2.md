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
