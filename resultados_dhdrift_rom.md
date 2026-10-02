# DHDrift ROM-ML: qué hemos hecho, qué decidimos y qué concluimos

Documento para presentar resultados. Está escrito en orden de presentación: primero la pregunta, luego el método, luego los problemas que fueron saliendo (incluidos los fallos), luego los resultados y por último lo que se puede y no se puede afirmar.

**Nota sobre el estado de los resultados.** Todo lo de las secciones 5 y 6 sale de ejecuciones reales sobre DHDrift (900 series). Lo marcado como *hipótesis* no está comprobado todavía.

---

## 0. Guion sugerido para la reunión (10 minutos)

1. La pregunta y por qué importa (sección 1).
2. Datos y unidad de análisis (sección 2).
3. Método en una diapositiva: perfil de 24 h → PCA → clasificador (sección 3).
4. Decisiones de diseño que protegen la validez: split por realizaciones, sin fugas, control `pre` (sección 4).
5. Lo que salió mal y cómo lo detectamos (sección 5). Aquí se gana credibilidad: el fallo de las etiquetas cambió los resultados.
6. Resultado principal: tabla de 10 semillas (sección 6.1).
7. Resultado sobre NS (sección 6.3) y efecto de la transición y de la severidad (6.7 y 6.8).
8. Lo que dicen las matrices de confusión (sección 6.4).
9. Limitaciones y qué no se puede decir (sección 7).
10. Próximos pasos (sección 8).

---

## 1. Pregunta de investigación

> ¿Se puede reducir un perfil diario de demanda de calefacción urbana (24 dimensiones) a unas pocas componentes y seguir identificando qué tipo de cambio estructural ha ocurrido?

Los tres tipos de cambio del benchmark DHDrift son:

- **EXP**: expansión de la red. Es un cambio súbito que sube la magnitud de la carga.
- **REF**: rehabilitación de edificios. Es un cambio incremental que baja la carga.
- **NS**: *night setback* (reducción nocturna de la temperatura de consigna). Es un cambio incremental que altera la **forma** del perfil diario, sobre todo por la noche.

Cada uno tiene severidad baja, media y alta (L, M, H).

Lo que **no** es este estudio: no es otro benchmark de detectores de drift. No detectamos cuándo ocurre el cambio (esa información la damos como conocida); identificamos de qué tipo es. Esto hay que decirlo explícitamente en la reunión, porque un clasificador supervisado que conoce el evento no compite con detectores no supervisados.

## 2. Datos y unidad de análisis

- **Dataset:** DHDrift 1.0.0, DOI 10.5281/zenodo.18128257. Se descargó y se verificó con su md5 antes de descomprimirlo; `data/raw/` no se modifica.
- **Contenido usado:** 900 series horarias sintéticas (2020-2023), repartidas en 9 grupos de 100 realizaciones (3 escenarios × 3 severidades). Se excluyó la serie `baseline`, porque no tiene evento de drift.
- **Unidad de observación:** el **día** (vector de 24 horas). No tratamos cada hora como muestra independiente.
- **Etiquetas:** se usa el `ground_truth_labels.csv` del dataset en vez de inferir los instantes del evento desde la carga. El formato real tiene 11.173 filas para 900 ficheros: una fila `onset` por serie (900) y filas `step` (10.273) que marcan los escalones de los drifts incrementales.
- **Fases por día**, definidas respecto al `onset`:
  - `pre`: antes del `onset`.
  - `transition`: del `onset` al último `step`. En EXP, que es súbito y no tiene escalones, usamos una ventana fija de 180 días.
  - `developed`: después de la transición.

## 3. Método

1. Cargar y validar las series (formato horario, 2020-2023, sin duplicados ni nulos, etiquetas que apuntan a ficheros existentes).
2. Construir perfiles diarios de 24 h y asignar fase respecto al evento.
3. Dividir en train y test por **realizaciones completas** (80 % / 20 %).
4. Ajustar `StandardScaler` y PCA **solo con las realizaciones de entrenamiento**.
5. Entrenar dos clasificadores: regresión logística y random forest (200 árboles).
6. Evaluar en test, por separado en las fases `developed`, `transition` y `pre`.

Representaciones comparadas: perfil crudo (24 dimensiones, solo escalado) y PCA con 2, 5 y 10 componentes.

Métricas: Macro-F1, balanced accuracy, F1 por clase (en especial NS), matrices de confusión, y RMSE de reconstrucción en vatios (unidades originales) para el ROM.

## 4. Decisiones de diseño y por qué

| Decisión | Motivo |
|---|---|
| Split por realizaciones completas y estratificado por escenario × severidad | Los días de una misma realización están muy correlacionados. Si se mezclaran días entre train y test, el resultado sería optimista. Los IDs de train y test se guardan y se comprueba que no se solapan. |
| Escalado y PCA ajustados solo con train | Evita fuga de información. Hay un test automático que comprueba que los parámetros del ajuste no cambian aunque se modifique el test. |
| Entrenar con días `developed` y evaluar también en `transition` | La pregunta científica más interesante es si el cambio se identifica **antes** de estar totalmente establecido. |
| Evaluar también la fase `pre` como control | Antes del drift los tres escenarios deberían ser indistinguibles, así que el acierto debe estar en el azar (0,33). Si no lo estuviera, el clasificador estaría reconociendo otra cosa (el distrito de partida o la época del año). |
| Dos modelos simples, sin redes neuronales ni búsqueda de hiperparámetros | Una línea base interpretable. No se introducen métodos nuevos hasta entender este resultado. |
| Semilla fija y resultados con 10 semillas | Reproducibilidad y una idea de la estabilidad frente al reparto train/test. |
| PCA con solver determinista (`svd_solver="full"`) | Resultados reproducibles. |
| En `full`, el entrenamiento se limita a 300.000 filas elegidas al azar (con semilla) | Mantener tiempos razonables con el random forest. |
| Código único para demo, mini y full; solo cambia el YAML | Lo que se prueba en el modo pequeño es lo mismo que se ejecuta en el final. |
| Tests automáticos con datos sintéticos que imitan la estructura de DHDrift | Permiten verificar la pipeline sin descargar el dataset. Los resultados sintéticos no significan nada científicamente; solo prueban que el código funciona. |

**Desviaciones respecto a la guía original que se nos pasó:**
- No se usa seaborn (no hacía falta).
- Los splits y `run_metadata.json` se guardan dentro de `results/<modo>/`.
- `make verify` ejecuta los tests unitarios y de integración.
- Añadimos la fase `pre` como control, el análisis con varias semillas y las métricas por clase.
- La comprobación de severidad (L=25 %, M=50 %, H=100 %) no se verifica automáticamente: no sabíamos el formato de `metadata/`. Queda como **no verificada**.

## 5. Lo que salió mal y cómo lo resolvimos

Esta sección es la que más conviene contar con claridad.

### 5.1 Problemas de entorno (Windows)
- `python` no se encontraba porque Windows lo redirigía a un alias de la Microsoft Store. Solución: usar el lanzador `py` para crear el entorno virtual y, a partir de ahí, llamar siempre al Python del entorno (`.venv\Scripts\python.exe`).
- PowerShell bloqueaba la activación del entorno por la política de ejecución de scripts. Solución: no activar el entorno y llamar directamente a su Python.

### 5.2 Un check de validación mal planteado
La validación del dataset dio 12 comprobaciones correctas y 1 fallo: "EXP es súbito, REF y NS son incrementales" para las 900 series. El fallo era **nuestro**, no del dataset: el código tomó la columna `event_type` (que vale `onset` o `step`) como si fuera el tipo de drift. Se corrigió para que ese check se omita cuando el CSV no trae el tipo de drift. Hay una segunda comprobación que se omite (severidades 25/50/100 %).

### 5.3 El error importante: el inicio del drift estaba mal
Nuestro código asumía **una fila por serie** en el CSV de etiquetas. En realidad hay un `onset` y muchas filas `step`, y el código se quedaba con la **última fila** de cada fichero. En REF y NS eso significaba que el "inicio del drift" era el **último escalón**, no el comienzo.

Consecuencias:
- Las fases estaban desplazadas: lo que llamábamos `transition` eran en realidad días casi completamente desarrollados, y la rampa real caía dentro de `pre`.
- Los primeros resultados eran demasiado optimistas. Por ejemplo, con random forest y perfil crudo en la fase de transición, el Macro-F1 salía 0,972; tras corregir, sale 0,842-0,844.

Cómo se detectó: el check fallido de 5.2 nos llevó a mirar el contenido real del CSV de etiquetas, y al contar filas por tipo de evento salieron 11.173 filas para 900 ficheros (900 `onset` y 10.273 `step`). Además, los resultados casi perfectos (0,998) ya nos hacían desconfiar.

Corrección: usar el `onset` como inicio y el último `step` como fin de la transición. **Todos los resultados que se presentan son posteriores a esta corrección.** Los anteriores se descartaron.

### 5.4 Otros fallos menores
- Un error de arrays de solo lectura con pandas 3 al construir máscaras de filas (lo detectaron los tests).
- La figura de resumen tenía la leyenda tapando una línea, y las barras de error no se ven porque son más pequeñas que los puntos (ver sección 7).

### 5.5 Falsa alarma útil
Pensamos que el 0,998 del random forest podía deberse a que todas las series comparten el mismo tiempo meteorológico (cada fecha tendría una "huella") o a que EXP parte de otro nivel de base. Se comprobó con el control `pre`, que dio azar (ver 6.2). Eso descarta esa explicación.

## 6. Resultados

### 6.1 Resultado principal: ¿cuánta dimensionalidad se puede quitar?

Random forest, Macro-F1 en test, media ± desviación en 10 semillas (semillas 42 a 51; cada una cambia el reparto de realizaciones):

| Representación | Componentes | `developed` | `transition` | RMSE de reconstrucción (W) |
|---|---:|---|---|---:|
| Crudo | 24 | 0,998 ± 0,000 | 0,842 ± 0,003 | -- |
| PCA | 10 | 0,995 ± 0,001 | 0,847 ± 0,003 | 4.665 |
| PCA | 5 | 0,989 ± 0,000 | 0,828 ± 0,003 | 10.849 |
| PCA | 2 | 0,949 ± 0,001 | 0,770 ± 0,003 | 39.314 |

Lectura:
- **De 24 a 10 dimensiones no se pierde información útil.** En `transition`, PCA-10 queda 0,005 por encima del crudo, pero esa diferencia es pequeña: lo prudente es decir "equivalente" y no "mejor".
- **Con 5 componentes** la pérdida es de unos 1,4 puntos en `transition` y menos de 1 punto en `developed`.
- **Con 2 componentes** la caída ya es clara: unos 5 puntos en `developed` y unos 7 en `transition`.
- La fase `transition` es la que discrimina entre representaciones. En `developed` el random forest está casi en el techo.

### 6.2 Control `pre`: no hay fuga
Con random forest, el Macro-F1 en `pre` es 0,329-0,335 con todas las representaciones, es decir, el azar para tres clases. Con regresión logística es 0,287-0,294 (balanced accuracy 0,33). Por tanto, el clasificador no está reconociendo el distrito de partida ni la época del año. **Está identificando el cambio ocurrido.**

### 6.3 Sobre NS (el cambio de forma)
- El F1 de NS con random forest en `transition` es 0,871 con perfil crudo y 0,879 con PCA-10. En `developed` es 1,000 con crudo, PCA-5 y PCA-10, y 0,993 con PCA-2.
- Es decir: **la información sobre el cambio de forma de NS se conserva con pocas componentes.**
- Matiz importante: el clasificador es supervisado y conoce el tipo de evento en entrenamiento. Esto **no contradice** que los detectores convencionales tengan dificultades con drift de forma. Solo dice que la información sigue disponible en un espacio reducido.

### 6.4 Matrices de confusión (random forest, test)

Con perfil crudo en `developed` casi todo es diagonal: EXP 43.139 de 43.233 días bien, REF 45.416 de 45.569 y NS 44.988 de 44.988.

En la fase `transition` (perfil crudo):
- EXP se reconoce casi siempre: 10.712 de 10.800 días (99 %).
- Pero 3.041 días de REF y 2.601 de NS se clasifican como EXP. **El error principal tiene una sola dirección:** los días de transición de REF y NS tienden a caer en EXP. Por eso el F1 de EXP en `transition` es el más bajo (0,789 con crudo): es un problema de **precisión**, no de recall.

Con PCA-2 en `developed`:
- NS sigue reconociéndose casi perfecto (44.782 de 44.988).
- Lo que se pierde es la separación entre EXP y REF (2.818 días de EXP clasificados como REF y 3.370 de REF como EXP).
- Esto es contraintuitivo respecto a lo que esperábamos: no se pierde la forma de NS, sino la **dirección** del cambio de magnitud (subida en EXP, bajada en REF).

### 6.5 Regresión logística
Queda muy por debajo del random forest: 0,718 en `developed` y 0,649 en `transition` con perfil crudo. Confunde mucho EXP con REF incluso sin reducir. La separación necesita no linealidad, así que sirve poco como línea base fuerte. Con PCA-10 se iguala al crudo (0,710 y 0,631); con PCA-2 y PCA-5 cae a unos 0,50 en `transition`.

### 6.6 Comprobación visual de los datos
La figura de perfiles de ejemplo confirma lo esperado:
- EXP: el perfil medio sube (de unos 270.000 W a unos 500.000 W) y mantiene la forma.
- REF: baja (de unos 220.000 W a unos 55.000 W) y casi no cambia de forma.
- NS: la carga nocturna (horas 0-4 y 21-23) cae mucho y la de las primeras horas de la mañana sube algo.
- Se ve además una **estacionalidad muy fuerte** (la carga casi desaparece en verano).

### 6.7 ¿Cuándo se identifica el cambio durante la transición?

Se divide cada transición en 5 tramos (0-20 %, 20-40 %, ..., 80-100 % del recorrido desde el `onset` hasta el último escalón) y se mide la balanced accuracy en cada tramo. Un solo split (semilla 42), sin barras de error.

Random forest:

| Tramo | Crudo | PCA-10 | PCA-5 | PCA-2 |
|---|---:|---:|---:|---:|
| 0-20 % | 0,560 | 0,585 | 0,559 | 0,508 |
| 20-40 % | 0,821 | 0,813 | 0,791 | 0,703 |
| 40-60 % | 0,992 | 0,983 | 0,969 | 0,925 |
| 60-80 % | 0,986 | 0,992 | 0,978 | 0,919 |
| 80-100 % | 0,990 | 0,993 | 0,985 | 0,900 |

Regresión logística:

| Tramo | Crudo | PCA-10 | PCA-5 | PCA-2 |
|---|---:|---:|---:|---:|
| 0-20 % | 0,492 | 0,459 | 0,324 | 0,326 |
| 20-40 % | 0,671 | 0,658 | 0,487 | 0,469 |
| 40-60 % | 0,588 | 0,588 | 0,569 | 0,559 |
| 60-80 % | 0,693 | 0,669 | 0,626 | 0,636 |
| 80-100 % | 0,865 | 0,855 | 0,690 | 0,680 |

Lectura:
- **El cambio se identifica bien antes de que la transición termine.** Con random forest se llega a ≈0,99 con el perfil crudo ya en el tramo 40-60 %. En el primer 20 % el acierto es 0,56, por encima del azar (0,33) pero lejos de ser fiable.
- **PCA-10 es prácticamente igual que el perfil crudo en todos los tramos**, incluidos los iniciales. PCA-5 pierde un poco en los tramos tempranos (0,791 frente a 0,821 en 20-40 %). PCA-2 pierde bastante pronto (0,703 en 20-40 %) y además **no mejora al final**: baja de 0,925 a 0,900 en los últimos tramos mientras las demás se mantienen. Esto es coherente con las matrices de confusión (con 2 componentes se pierde la distinción entre EXP y REF), pero la causa no está comprobada.
- **Regresión logística:** mejora con el avance (de 0,49 a 0,87 con el crudo), con una bajada en el tramo 40-60 % que no sabemos explicar. Con PCA-2 y PCA-5 solo llega a ≈0,69.

Cautelas específicas:
- **Los tramos no son comparables entre escenarios.** En EXP el 100 % del recorrido son 180 días fijos y el cambio es súbito; en REF y NS es la rampa real hasta el último escalón. El primer tramo mezcla una clase probablemente ya fácil (EXP) con dos difíciles, y la balanced accuracy global no lo separa. Para afirmar cuánto tarda en identificarse cada escenario hace falta el recall por clase y tramo, que todavía no está calculado.

### 6.8 Efecto de la severidad

Balanced accuracy por severidad, un solo split (semilla 42). Random forest:

| Fase | Severidad | Crudo | PCA-10 | PCA-5 | PCA-2 |
|---|---|---:|---:|---:|---:|
| `developed` | H | 1,000 | 1,000 | 1,000 | 0,999 |
| `developed` | M | 0,998 | 0,994 | 0,991 | 0,946 |
| `developed` | L | 0,996 | 0,992 | 0,980 | 0,905 |
| `transition` | H | 0,902 | 0,907 | 0,896 | 0,854 |
| `transition` | M | 0,865 | 0,869 | 0,857 | 0,794 |
| `transition` | L | 0,835 | 0,836 | 0,809 | 0,716 |

Lectura:
- **La severidad baja es la más difícil, como cabía esperar, pero con random forest sigue siendo identificable:** 0,996 en `developed` y 0,835 en `transition` con el perfil crudo.
- **PCA-10 iguala al perfil crudo en las tres severidades**, también en la baja.
- **La pérdida de reducir mucho se concentra en las severidades bajas.** Con PCA-2, en `developed` se pierden apenas 0,001 puntos en H, pero ≈0,09 en L; en `transition` se pierden ≈0,05 en H y ≈0,12 en L. Con PCA-5 la pérdida es de ≈0,02-0,03 en L.
- **Incluso con severidad alta, la fase `transition` queda en ≈0,90.** El efecto de la transición (los primeros días de la rampa se parecen a otras clases) es independiente de la severidad.
- Con regresión logística el patrón es el mismo, a un nivel más bajo (por ejemplo, en `transition` con el crudo: H 0,781, M 0,660, L 0,536).

No se ha desglosado la confusión por clase y severidad, así que no se puede atribuir el error de EXP en `transition` a una severidad concreta.

## 7. Qué se puede y qué no se puede afirmar

**Se puede afirmar:**
- En este benchmark, un perfil de 24 dimensiones puede reducirse a 10 componentes sin pérdida apreciable de la capacidad de identificar el tipo de cambio, y a 5 con una pérdida pequeña.
- La capacidad de identificar NS se mantiene con pocas componentes.
- Antes del drift no hay señal de identificación (control `pre`).

**No se puede afirmar (todavía):**
- Que PCA-10 sea *mejor* que el perfil crudo. La diferencia (0,005) es pequeña y no está contrastada por pares.
- Por qué los días de transición de REF y NS se confunden con EXP. **Es una hipótesis:** los primeros días de una rampa tienen poco drift y se parecen a un cambio de magnitud. Además, la ventana de transición de REF y NS es más larga (aproximadamente 1,7 veces, estimado por el número de días de test), lo que baja mecánicamente la precisión de EXP. Hay que medirlo.
- Qué clase falla en cada severidad. El desglose por severidad (6.8) es global; no se ha calculado la confusión por clase y severidad.
- Que sea válido en redes reales: DHDrift es un benchmark **sintético**.

**Cautelas metodológicas:**
- Las desviaciones entre semillas son muy pequeñas porque solo miden la variabilidad del reparto train/test. Las 10 particiones salen de las mismas 900 realizaciones y sus conjuntos de test se solapan. Hay que describirlo como "variabilidad entre particiones", no como incertidumbre estadística completa.
- La fase `transition` mezcla dos definiciones: del `onset` al último escalón en REF y NS, y 180 días fijos en EXP. Hay que decirlo en la metodología.
- Este estudio es de **identificación**, no de detección.
- La estacionalidad es muy fuerte y no hemos hecho un experimento que la elimine. El control `pre` indica que no sesga la identificación, pero no demuestra que no afecte a lo que capturan las primeras componentes.
- La comprobación de las severidades (25/50/100 %) no está verificada.

## 8. Próximos pasos

1. **Completar el análisis por avance de la transición** (script `experiments/06_progress.py`, ya ejecutado; ver 6.7 y 6.8): añadir el recall por clase, tramo y severidad, que es lo que permite decir cuánto tarda en identificarse cada escenario y explicar el error de EXP.
2. Comparación por pares entre PCA-10 y el perfil crudo, para ver si la diferencia es real o ruido.
3. Figura final: mover la leyenda y mostrar la dispersión de otra forma (por ejemplo, bandas o puntos por semilla).
4. Como robustez, repetir normalizando cada perfil por su media diaria para quitar parte de la estacionalidad y la magnitud.
5. Solo si el resultado lo justifica: otros métodos de reducción (autoencoders u otros). La guía original pide no añadirlos hasta entender esta línea base.

## 9. Apéndice: tablas completas (10 semillas)

### logistic_regression — developed

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.718 ± 0.002 | 0.713 ± 0.002 | 0.814 ± 0.003 | -- |
| PCA | 2 | 0.643 ± 0.001 | 0.638 ± 0.001 | 0.708 ± 0.002 | 39314.446 |
| PCA | 5 | 0.659 ± 0.001 | 0.654 ± 0.001 | 0.718 ± 0.002 | 10849.059 |
| PCA | 10 | 0.710 ± 0.002 | 0.705 ± 0.002 | 0.809 ± 0.002 | 4664.711 |

### logistic_regression — transition

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.649 ± 0.007 | 0.659 ± 0.009 | 0.644 ± 0.004 | -- |
| PCA | 2 | 0.502 ± 0.006 | 0.526 ± 0.008 | 0.481 ± 0.008 | 39314.446 |
| PCA | 5 | 0.504 ± 0.006 | 0.531 ± 0.008 | 0.495 ± 0.008 | 10849.059 |
| PCA | 10 | 0.631 ± 0.006 | 0.643 ± 0.008 | 0.638 ± 0.007 | 4664.711 |

### logistic_regression — pre

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.289 ± 0.002 | 0.332 ± 0.001 | 0.169 ± 0.003 | -- |
| PCA | 2 | 0.287 ± 0.001 | 0.329 ± 0.001 | 0.056 ± 0.002 | 39314.446 |
| PCA | 5 | 0.288 ± 0.001 | 0.332 ± 0.001 | 0.050 ± 0.002 | 10849.059 |
| PCA | 10 | 0.294 ± 0.002 | 0.335 ± 0.001 | 0.168 ± 0.004 | 4664.711 |

### random_forest — developed

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.998 ± 0.000 | 0.998 ± 0.000 | 1.000 ± 0.000 | -- |
| PCA | 2 | 0.949 ± 0.001 | 0.949 ± 0.001 | 0.993 ± 0.001 | 39314.446 |
| PCA | 5 | 0.989 ± 0.000 | 0.989 ± 0.000 | 1.000 ± 0.000 | 10849.059 |
| PCA | 10 | 0.995 ± 0.001 | 0.995 ± 0.001 | 1.000 ± 0.000 | 4664.711 |

### random_forest — transition

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.842 ± 0.003 | 0.868 ± 0.002 | 0.871 ± 0.005 | -- |
| PCA | 2 | 0.770 ± 0.003 | 0.791 ± 0.003 | 0.833 ± 0.005 | 39314.446 |
| PCA | 5 | 0.828 ± 0.003 | 0.853 ± 0.002 | 0.876 ± 0.005 | 10849.059 |
| PCA | 10 | 0.847 ± 0.003 | 0.870 ± 0.002 | 0.879 ± 0.005 | 4664.711 |

### random_forest — pre

| representation | components | macro_f1 | balanced_accuracy | f1_NS | recon_rmse |
|---|---|---|---|---|---|
| Raw | 24 | 0.334 ± 0.002 | 0.340 ± 0.001 | 0.308 ± 0.002 | -- |
| PCA | 2 | 0.329 ± 0.001 | 0.335 ± 0.001 | 0.316 ± 0.002 | 39314.446 |
| PCA | 5 | 0.335 ± 0.001 | 0.338 ± 0.001 | 0.316 ± 0.002 | 10849.059 |
| PCA | 10 | 0.333 ± 0.002 | 0.338 ± 0.002 | 0.307 ± 0.002 | 4664.711 |

### F1 por clase en `transition` (un solo split, semilla 42)

| modelo | representación | componentes | F1 EXP | F1 REF | F1 NS |
|---|---|---:|---:|---:|---:|
| logistic_regression | Raw | 24 | 0.671 | 0.640 | 0.644 |
| logistic_regression | PCA | 2 | 0.492 | 0.549 | 0.483 |
| logistic_regression | PCA | 5 | 0.485 | 0.547 | 0.498 |
| logistic_regression | PCA | 10 | 0.642 | 0.621 | 0.635 |
| random_forest | Raw | 24 | 0.789 | 0.868 | 0.876 |
| random_forest | PCA | 2 | 0.714 | 0.756 | 0.835 |
| random_forest | PCA | 5 | 0.772 | 0.841 | 0.881 |
| random_forest | PCA | 10 | 0.814 | 0.854 | 0.883 |

## 10. Reproducibilidad

- Dataset: DHDrift 1.0.0, DOI 10.5281/zenodo.18128257 (md5 del zip verificado en la descarga).
- Semilla base 42; las 10 semillas son 42 a 51.
- Cada ejecución guarda en `results/<modo>/`: `run_metadata.json` (modo, semilla, versión de Python, dataset, componentes, modelos e IDs de realizaciones), las listas de train y test en `splits/`, las tablas (`results_table.csv`, `results_seeds_summary.md`) y las figuras.
- Los resultados de la sección 6 corresponden a la configuración `full` (900 series, 80 % de realizaciones para entrenar, 20 % para test, PCA con 2, 5 y 10 componentes).
- Para dejar trazabilidad completa, conviene inicializar Git en la carpeta del proyecto y guardar el commit junto a los resultados que se publiquen. En la copia local el campo `git_commit` de `run_metadata.json` sale como `unknown` mientras no haya repositorio.
