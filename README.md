# DHDrift ROM-ML

Pipeline reproducible: perfiles diarios de 24 h → PCA (ROM) → LR / RF → identificar EXP / REF / NS.

**Dataset:** DHDrift 1.0.0, DOI [10.5281/zenodo.18128257](https://doi.org/10.5281/zenodo.18128257)
(zip md5 `6b681bdebae41174d6d833e2d50e1a3f`). `data/raw/` nunca se modifica.

## Uso

```bash
git init && git add -A && git commit -m "init"
uv sync
make data      # descarga + verifica md5 + descomprime en data/raw (~570 MB)
make check     # valida el dataset
make demo      # humo (resultados en results/demo/)
make mini
make full
make verify    # tests con datos sintéticos (no necesita el dataset)
```
Si no usas `uv`: `python -m venv .venv && source .venv/bin/activate && pip install -e . pytest` y `make demo`.

## Decisiones de diseño

- Unidad: día (24 h). Se excluye `baseline` (no tiene evento de drift).
- Split 80/20 por **realizaciones completas**, estratificado por escenario×severidad. IDs en `results/<mode>/splits/`.
- Fases (`ground_truth_labels.csv`: una fila `onset` + filas `step` por serie): `pre` (antes del onset), `transition` (onset → último `step`), `developed` (después). En drift súbito (EXP, sin `step`) la transición es una ventana fija de `phases.transition_days` días. Se evalúa también `pre` como control (no debería ser identificable).
- Scaler y PCA se ajustan solo con las realizaciones de train (todos sus días). Los clasificadores se entrenan con días `developed` de train y se evalúan en test por separado en `developed` y `transition`.
- Métricas: Macro-F1, balanced accuracy, matriz de confusión, RMSE de reconstrucción (W), y F1 de NS (análisis de drift que cambia la forma).
- `results/<mode>/`: `01…05_*.png`, `results_table.csv`, `summary_table.md`, `summary.json`, `run_metadata.json`, `splits/`.

## Si `make check` falla por columnas

`ground_truth_labels.csv` se autodetecta (columna de fichero y de inicio de drift). Si no acierta,
fija `labels.file_col` y `labels.start_col` en `configs/*.yaml`.

## Varias semillas (media ± std)

```powershell
.\.venv\Scripts\python.exe -m experiments.05_seeds --config configs/full.yaml --n-seeds 10
```

Genera `results/full/results_seeds_summary.md` y `06_components_vs_performance_seeds.png`.

## Regenerar solo las figuras (sin reentrenar)

```powershell
.\.venv\Scripts\python.exe -m experiments.replot --config configs/full.yaml --data
```

Sin `--data` solo rehace 05, 06 y 07 desde los CSV (segundos). Con `--data` rehace también 01, 02 y 03 (recarga las series, ~1,5 min). Con `--confusion` reentrena una vez para regenerar 04 y guarda `confusion_matrices.json`; a partir de ahí 04 también sale sin reentrenar.

---

# Revisión (guía computacional de Kai)

Todo lo de `computational_guide.md` (Prioridades 0–15) se ejecuta con **un solo comando**, sin supervisión:

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install psutil
.\.venv\Scripts\python.exe -m experiments.run_all --config configs/full.yaml
```

- Antes de irte: evita que el portátil se suspenda (`powercfg /change standby-timeout-ac 0`; deshaz con el valor que tuvieras).
- Es **reanudable**: si se corta, vuelve a lanzar el mismo comando; salta las etapas y las semillas ya terminadas (`--no-resume` para repetir todo).
- Cada etapa está aislada: si una falla, se guarda `results/full/revision/error_<etapa>.txt` y sigue con las demás.
- Opciones: `--only P3,P5` (solo esos puntos), `--skip P9,P10` (omitir los más lentos), `--n-seeds 10`, `--skip-tests`.
- Prueba rápida con 2 realizaciones por celda: `.\.venv\Scripts\python.exe -m experiments.run_all --config configs/demo.yaml --skip-tests`.
- **Tiempo**: no está medido con los datos reales. Estimación orientativa: de 8 a 24 h según CPU (lo más pesado son `05_seeds` con 12 representaciones × 2 modelos × 10 semillas, la validación cruzada del RF y el banco de ruido). Los parámetros de coste están en el bloque `revision:` de `configs/full.yaml`.
- Al terminar: `results/full/revision/REVISION_REPORT.md` (estado de cada etapa + índice de figuras), `output_checksums.csv` (sha256 de todas las salidas) y `run_log.txt`.

## Figuras: cada imagen lleva su `.txt`

Todas las figuras (`.png`, `.svg`, `.pdf`) se guardan con `src/figio.save_fig`, que escribe al lado un `.txt` con el propósito, cómo leerla y los datos dibujados (ejes, leyendas, valores de cada serie, anotaciones de las matrices). `figures_manifest.json` (en cada carpeta) enlaza figura → descripción → script → commit de git. `replot` también regenera los `.txt`.

## Qué hace cada punto y dónde queda

| Punto | Qué se ejecuta | Salidas (en `results/full/`) |
|---|---|---|
| P0 | tests, validación del dataset, md5 del zip frente a Zenodo, `04_final`, `05_seeds`, `06_progress`, comprobación de que train/test no se solapan en ninguna semilla, consistencia con el manuscrito, checksums | `revision/P0_freeze/` (`freeze_metadata.json` con commit, versiones, hardware y config; `manuscript_consistency.csv`), `splits/seed_*_{train,test}.csv`, `run_metadata.json`, `output_checksums.csv` |
| P1 | rejilla prefijada k=2..12 (Raw aparte, nunca "PCA-24"); media ± SD en 10 splits; **delta pareado por split** frente a Raw (media, p2.5, p97.5 — sensibilidad a la partición); **bootstrap por realización** dentro de cada split (IC del delta) | `results_seeds_summary.*`, `paired_deltas_vs_raw.*`, `bootstrap_deltas_*.csv`, `06_*.png`, `14_rf_pca_curve.png` |
| P2 | RMSE (W y kW), NRMSE-SD por perfil, NRMSE-media y NRMSE del subconjunto, por k × escenario × severidad × fase (nunca con una única escala global) | `nrmse_by_group.csv` |
| P3 | precisión/recall/F1/soporte por clase × severidad × fase × tramo de progreso, en los 10 splits (esquema exacto de la guía + `n_pred`); análisis de falsos positivos de EXP | `progress_class_severity_*.csv`, `exp_false_positive_analysis.md`, `08_*.png`, `09_*.png` |
| P4 | comprobación de trayectoria meteorológica compartida; líneas base solo-fecha / solo-temperatura; perfil vs perfil+temperatura; rendimiento por terciles de temperatura (definidos con train), estación, escenario×estación y progreso×estación | `revision/P4_weather/` |
| P5 | modos de representación `daily_mean_normalized` y `daily_zscore_shape`, con scaler/PCA/clasificadores propios | `revision/P5_normalization/` |
| P6 | cargas PC1–PC6, correlaciones de Spearman y parciales (dada la temperatura), indicadores NS predefinidos, línea base de un indicador (umbral elegido en train; AUC, F1, Brier, ECE, calibración), efecto horario por escenario | `revision/P6_pca_ns_interpretation/` |
| P7 | Hellinger pre vs fases posteriores por realización (1-D con bins de train; gaussiana aproximada en PCA-5), IC por bootstrap de realizaciones, estratificado por escenario/severidad/estación/progreso | `revision/P7_hellinger/` |
| P8 | BL por los modelos de 3 clases (probabilidades, entropía; sin accuracy) y **modelo de 4 clases** con NONE = días `pre` | `revision/P8_baseline_negative_control/` |
| P9 | banco de ruido (gaussiano, sesgo, huecos aleatorios y cortes con 3 imputaciones, desfase ±1 h, picos) y **fallos simulados** (medida vs físico) con flujo jerárquico | `revision/P9_noise_robustness/`, `revision/P9_fault_anomaly/` |
| P10 | validación cruzada agrupada solo con train para la rejilla de RF, selección congelada y test; banco de recursos (tiempos, latencia, tamaño, memoria, nodos, profundidad) | `revision/P10_rf_sensitivity/`, `revision/P10_resource_benchmark/` |
| P11 | ventanas de seguimiento (90/180/365 d), ventanas de calendario emparejadas, buffers de frontera, y reentrenado con EXP de 90 y 360 d | `revision/P11_transition_sensitivity/` |
| P12 | PCA ajustada con todos los días / solo pre / solo developed (siempre solo train) | `revision/P12_pca_fit_regimes/` |
| P13 | **NO implementado** (ver abajo) | `revision/P13_overlapping_interventions/NOT_IMPLEMENTED.md` |
| P14 | inglés, fuentes y tamaños coherentes, RF como figura principal (`14_rf_pca_curve.png`), curvas con incertidumbre y crudo separado, matrices de confusión normalizadas con texto legible (conteos en `confusion_counts.csv`), `revision/figures/00_workflow.{svg,pdf,png}` | — |
| P15 | `tests/test_revision.py` (13 tests nuevos) | `make verify` |

## Decisiones y desviaciones respecto a la guía (léelas)

- **`max_features=1.0` añadido a la rejilla del RF.** Con PCA-2, `sqrt` y `0.5` dan ambos 1 variable por división, así que no distinguen nada; 1.0 es lo que contrasta la hipótesis de que parte de la caída de PCA-2 sea un artefacto del bosque.
- **`daily_energy_normalized` no se ejecuta**: tras `StandardScaler` es idéntico a `x/media` (difiere en una constante ×24). Hay un test que lo verifica. La versión z-score sí se ejecuta.
- **Precisión por clase en una celda**: dentro de una celda (severidad, fase, tramo) la precisión de la clase *c* es TP / (todos los días predichos como *c* en esa celda, sea cual sea su clase real). Si se calculase solo dentro de una clase real no tendría sentido.
- **Tramo de progreso en EXP**: es avance a lo largo de una ventana fija de 180 d, no avance físico de la implementación. Aparece en el título/descripción de las figuras.
- **NRMSE**: se reportan tres versiones (por perfil / sobre el subconjunto / respecto a la media) por grupo; el agregado global es solo descriptivo.
- **Bootstrap**: remuestrea realizaciones (estratificado por escenario×severidad) dentro de cada split de test fijo; las réplicas son comunes a todas las representaciones. Sigue siendo sensibilidad de la partición + muestreo de realizaciones, no incertidumbre del simulador.
- **Horas de noche (P6/P7)**: 22–05 noche, 06–21 día, 06–09 mañana, tal como pide la guía, **pero asumen que el timestamp está en hora local**. Revisa `P6_hourly_effect.png` (efecto real por hora) y los metadatos de DHDrift; si no coincide, cambia `night_hours` etc. en el YAML.
- **Tiempo (`weather.csv`)**: la columna de temperatura se autodetecta (contiene "temp"); si falla, fija `revision.weather.temp_col`/`time_col`. Si no hay temperatura, P4 se omite con un `SKIPPED.txt`. Hemisferio norte para las estaciones, HDD con base 15.5 °C (configurable).
- **Hellinger**: solo 1-D (media diaria, night/day, morning rebound) y gaussiana aproximada en PCA-5; nunca en 24-D. Es descriptivo (usa todas las realizaciones, bins definidos con train) y no afirma equivalencia física de L/M/H.
- **Cuarta clase (P8)**: NONE = días `pre` de las 900 realizaciones (muchas trayectorias independientes), no una sola serie BL. BL solo se usa como control: nunca entra en entrenamiento (hay un test).
- **Fallos (P9)**: son simulaciones transparentes, no calibradas con datos reales (`src/perturb.py`, con sus parámetros en la docstring). Los de medida y los físicos se mantienen separados. El flujo jerárquico fusiona "evento/no evento" y "fallo vs intervención estructural" en un único enrutador de 4 clases (none / measurement_anomaly / physical_fault / structural) y después aplica el clasificador EXP/REF/NS.
- **P13 (intervenciones solapadas) no se implementa**: exige simular datos nuevos con el generador de DHDrift (que no está en este repo) y un rediseño multietiqueta; mezclar series ya existentes no sería una intervención simulada. La propia guía lo marca como extensión.
- **`reference/manuscript_numbers.csv`**: contiene las cifras de la ejecución `full` (semilla 42, pegadas en la conversación) y las medias de 10 semillas del resumen del proyecto. **Verifícalas contra el manuscrito y añade las que falten**; `P0_manuscript_consistency` fallará (y lo dirá) si alguna no coincide con lo generado.
- `configs/full.yaml` usa ahora `rom.components: [2..12]`. Las figuras de confusión y de progreso solo muestran `figures.confusion_components` / `progress_components` (2, 5, 10) para que sigan siendo legibles; las tablas llevan todas las dimensiones. Los resultados de k = 2, 5, 10 se reproducen exactamente (verificado en datos sintéticos contra el código original: diferencia 0.0).
- La caché del dataset está en `data/processed/` (ignorada por git). Si cambias los datos de `data/raw`, borra esa carpeta.
