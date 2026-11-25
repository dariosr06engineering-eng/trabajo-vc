# ÍNDICE — Examen de Visión por Computador

Cada `CONTENIDO_TEMAx.md` junta **teoría + prácticas (código explicado) + trampas típicas de test**.
`ALL.md` une todos los temas en un único archivo.

| Tema | Archivo | Teoría (PDF) | Prácticas (código) | Código que falta (hay código de referencia en el .md) |
|---|---|---|---|---|
| 2. Procesamiento de imagen | [CONTENIDO_TEMA2.md](CONTENIDO_TEMA2.md) | L2_Image processing | 2.1 IP tools · 2.2 Smoothing · 2.3 Image enhancement | — |
| 3. Detección de bordes | [CONTENIDO_TEMA3.md](CONTENIDO_TEMA3.md) | L3_Edge detection | 3.1 Primera derivada (Sobel, DroG) · 3.2 Laplaciano/LoG + Canny | — |
| 4. Keypoints | [CONTENIDO_TEMA4.md](CONTENIDO_TEMA4.md) | L4_Keypoint detection | 4.1 Harris + NCC + homografía | **SIFT** |
| 5. Segmentación | [CONTENIDO_TEMA5.md](CONTENIDO_TEMA5.md) | L5_Segmentation | 5.1 Hough · 5.3 SAM | **5.2** (Otsu, umbral adaptativo, K-means, region growing) |
| 6. Descripción de regiones | [CONTENIDO_TEMA6.md](CONTENIDO_TEMA6.md) | L6_Region Description | 6.1 Forma · 6.2 Apariencia (GLCM) · 6.3 CLIP | — |
| 7. Reconocimiento clásico | [CONTENIDO_TEMA7.md](CONTENIDO_TEMA7.md) | L7_Object Recognition | 7.1 Funciones discriminantes · 7.4 MLP · 7.5 Detección con YOLO | **7.2 y 7.3** (clasificador gaussiano, perceptrón) |
| 8. Deep Learning | [CONTENIDO_TEMA8.md](CONTENIDO_TEMA8.md) | L8_Deep Learning | **Solo teoría** (no tiene práctica propia) | — |
| Todo junto | [ALL.md](ALL.md) | | | |

## Contenido de cada tema
- **Tema 2**: color y RGB→gris · histograma (brillo/contraste) · LUTs · convolución (propiedades, padding, stride) · interpolación y aliasing · suavizado (media, gaussiano, mediana, media temporal) · gamma, ecualización, CLAHE.
- **Tema 3**: qué es un borde · gradiente y Sobel · DroG · Laplaciano, cruces por cero, LoG, DoG · Canny (NMS + histéresis).
- **Tema 4**: Harris (matriz M, R, autovalores) · Shi-Tomasi · SSD/SAD/CC/NCC · SIFT (espacio de escalas, DoG, 26 vecinos, orientación, descriptor de 128) · matching y homografía.
- **Tema 5**: Hough (rectas y circunferencias) · Otsu y umbral adaptativo · region growing · K-means · GMM/EM · normalized cut · segmentación semántica, de instancias y panóptica · U-Net · SAM.
- **Tema 6**: circularidad, elongación, excentricidad · momentos y Hu · descriptores de Fourier · estadísticos de color · GLCM · textones · Bag of Visual Words · embeddings y CLIP.
- **Tema 7**: funciones discriminantes · separabilidad y maldición de la dimensionalidad · Bayes/MAP · clasificador gaussiano (QDA/LDA/Mahalanobis/euclídeo) · perceptrón · hacia el MLP · prácticas de perceptrón/MLP con scikit-learn (métricas, matriz de confusión) y detección con YOLO.
- **Tema 8 (solo teoría)**: MLP · softmax y cross-entropy · backprop y SGD · train/val/test · overfitting y regularización · vanishing gradient · CNN, pooling, GAP · transfer learning · ViT.

## Estructura de carpetas
- `TEORIA/` → PDFs de teoría (L2–L8). No existe L1.
- `PRACTICAS/Px - .../codigo/` → notebooks `.ipynb`.
- `PRACTICAS/Px - .../pdf_lectura/` → notebooks exportados de Colab con sus salidas.
- `ZIPS_ORIGINALES/` → los .zip tal y como venían.

## Plan de estudio recomendado
1. Temas con más peso y más trampas: **4, 5 y 7**.
2. Temas intermedios: **3 y 6**.
3. Repaso rápido: **2 y 8**.
4. Al final, la sección "TRAMPAS TÍPICAS DE TEST" de cada tema.
