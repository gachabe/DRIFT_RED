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
