"""Carga y conversión del SCANIA Component X Dataset.

Uso desde la raíz del proyecto:
    python -m src.data            # comprueba los CSV y genera los Parquet
    python -m src.data --check    # solo comprueba que están los CSV
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

SPLITS = ("train", "validation", "test")

# Filas esperadas (sin cabecera) según la documentación del dataset
EXPECTED_ROWS = {
    "train_operational_readouts.csv": 1_122_452,
    "train_tte.csv": 23_550,
    "train_specifications.csv": 23_550,
    "validation_operational_readouts.csv": 196_227,
    "validation_labels.csv": 5_046,
    "validation_specifications.csv": 5_046,
    "test_operational_readouts.csv": 198_140,
    "test_labels.csv": 5_045,
    "test_specifications.csv": 5_045,
}

# Variables de sensor
COUNTERS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]
HISTOGRAMS = {"167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36}
HIST_COLUMNS = [f"{h}_{i}" for h, n in HISTOGRAMS.items() for i in range(n)]
SENSOR_COLUMNS = COUNTERS + HIST_COLUMNS  # 105 columnas
SPEC_COLUMNS = [f"Spec_{i}" for i in range(8)]

# Matriz de costes de Scania: COST_MATRIX[real][predicho]
COST_MATRIX = [
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
]


def label_file(split: str) -> str:
    return "train_tte.csv" if split == "train" else f"{split}_labels.csv"


def check_raw(count_rows: bool = False) -> pd.DataFrame:
    """Comprueba que los 9 CSV están en data/raw (y opcionalmente cuenta filas)."""
    rows = []
    for name, expected in EXPECTED_ROWS.items():
        path = RAW / name
        exists = path.exists()
        n = None
        if exists and count_rows:
            with open(path, "rb") as f:
                n = sum(1 for _ in f) - 1
        rows.append({
            "fichero": name,
            "existe": exists,
            "MB": round(path.stat().st_size / 1e6, 1) if exists else None,
            "filas": n,
            "esperadas": expected,
            "ok": exists and (n is None or n == expected),
        })
    return pd.DataFrame(rows)


def _readouts_schema() -> dict:
    # float64: algunos contadores e intervalos superan 2^24 y en float32 perderían precisión
    types = {"vehicle_id": pa.int32(), "time_step": pa.float64()}
    types.update({c: pa.float64() for c in SENSOR_COLUMNS})
    return types


def csv_to_parquet(csv_path: Path, parquet_path: Path, column_types: dict | None = None) -> int:
    """Convierte un CSV a Parquet por bloques, sin cargarlo entero en memoria."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    convert = pv.ConvertOptions(column_types=column_types) if column_types else None
    reader = pv.open_csv(csv_path, convert_options=convert,
                         read_options=pv.ReadOptions(block_size=64 << 20))
    n = 0
    with pq.ParquetWriter(parquet_path, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            n += batch.num_rows
    return n


def _parquet_rows(path: Path) -> int | None:
    """Filas de un Parquet, o None si está incompleto o dañado."""
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except Exception:
        return None


def build_parquet(overwrite: bool = False) -> None:
    """Genera los Parquet de los 9 CSV en data/processed."""
    for split in SPLITS:
        files = {
            f"{split}_operational_readouts": _readouts_schema(),
            f"{split}_specifications": {"vehicle_id": pa.int32()},
            label_file(split).removesuffix(".csv"): {"vehicle_id": pa.int32()},
        }
        for stem, types in files.items():
            src, dst = RAW / f"{stem}.csv", PROCESSED / f"{stem}.parquet"
            if dst.exists() and not overwrite and _parquet_rows(dst) == EXPECTED_ROWS[src.name]:
                print(f"  ya existe  {dst.name}")
                continue
            n = csv_to_parquet(src, dst, types)
            print(f"  {n:>9,} filas  {src.name} -> {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def load_split(split: str, float32: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Devuelve (lecturas, especificaciones, etiquetas) de un conjunto.

    Las lecturas vienen ordenadas por vehicle_id y time_step.
    Las especificaciones se devuelven como categóricas.
    float32=True reduce la memoria a la mitad (train: ~1 GB -> ~0,5 GB) a costa de precisión.
    """
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}")
    table = pq.read_table(PROCESSED / f"{split}_operational_readouts.parquet")
    if float32:
        table = table.cast(pa.schema([
            pa.field(f.name, pa.float32() if f.name in SENSOR_COLUMNS else f.type)
            for f in table.schema
        ]))
    readouts = table.to_pandas()
    del table
    readouts = readouts.sort_values(["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)
    specs = pd.read_parquet(PROCESSED / f"{split}_specifications.parquet")
    specs[SPEC_COLUMNS] = specs[SPEC_COLUMNS].astype("category")
    labels = pd.read_parquet(PROCESSED / f"{label_file(split).removesuffix('.csv')}.parquet")
    return readouts, specs, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="solo comprobar los CSV")
    parser.add_argument("--overwrite", action="store_true", help="regenerar los Parquet")
    args = parser.parse_args()

    status = check_raw()
    print(status.to_string(index=False))
    missing = status.loc[~status["existe"], "fichero"].tolist()
    if missing:
        raise SystemExit(f"\nFaltan {len(missing)} ficheros en {RAW}. Descárgalos de "
                         "https://doi.org/10.5878/bnh5-ka77 (ver README).")
    if not args.check:
        print("\nGenerando Parquet en data/processed ...")
        build_parquet(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
