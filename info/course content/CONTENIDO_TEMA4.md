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
