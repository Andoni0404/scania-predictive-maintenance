# Mantenimiento predictivo – SCANIA Component X

Analítica de datos para la Industria · Universidad de Deusto · Curso 2026-2027
Grupo 1: Andoni Garrido, Jon Larrañaga

¿Se puede anticipar, a partir de las lecturas de sensores de un camión, si su componente X va a fallar pronto y en qué ventana de tiempo, con el menor coste total?

## Dataset
SCANIA Component X Dataset (v3), Scania CV AB y Universidad de Estocolmo. CC BY 4.0.
Descarga: https://doi.org/10.5878/bnh5-ka77

Los datos no se suben al repositorio (1,5 GB; GitHub no admite ficheros de más de 100 MB).

1. Descarga el dataset desde https://doi.org/10.5878/bnh5-ka77
2. Copia los 9 CSV en `data/raw/`:
   - `train_operational_readouts.csv`, `train_tte.csv`, `train_specifications.csv`
   - `validation_operational_readouts.csv`, `validation_labels.csv`, `validation_specifications.csv`
   - `test_operational_readouts.csv`, `test_labels.csv`, `test_specifications.csv`
3. Comprueba que están y genera los Parquet:
   ```bash
   python -m src.data --check   # solo comprueba
   python -m src.data           # comprueba y convierte a Parquet en data/processed/
   ```

## Estructura
```
data/raw/         CSV originales (fuera de git)
data/processed/   Parquet unidos y limpios (fuera de git)
notebooks/        01_ingesta, 02_limpieza, 03_eda, 04_features, 05_modelado, 06_evaluacion
src/              Código reutilizable (data.py: rutas, carga, Parquet, matriz de costes)
models/           Modelos entrenados (fuera de git)
app/              Dashboard Streamlit
service/          Servicio de predicción BentoML
reports/          Figuras y entregables
```

## Instalación
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Fases
1. Ingesta (CSV → Parquet, unión por `vehicle_id`)
2. Verificación y limpieza
3. Análisis exploratorio
4. Ingeniería de características
5. Modelado
6. Evaluación con la matriz de costes de Scania
7. Aplicación (Streamlit + BentoML)
