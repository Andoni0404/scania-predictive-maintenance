"""Limpieza del SCANIA Component X Dataset.

La misma limpieza se aplica a train, validación, test y a las lecturas que reciba la
API de BentoML. Todo lo que se "aprende" (medianas, umbrales, categorías) se ajusta
solo con train para no filtrar información de validación/test.

Uso:
    from src.cleaning import Cleaner
    cleaner = Cleaner().fit(train_readouts, train_specs)
    readouts_clean, specs_clean = cleaner.transform(readouts, specs)
    cleaner.report_  # resumen de cambios de la última llamada a transform
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as _data
from src.data import COUNTERS, HIST_COLUMNS, HISTOGRAMS, SENSOR_COLUMNS, SPEC_COLUMNS

# Cada sensor (variable) agrupa sus columnas: los contadores tienen 1 y los histogramas 10-36
SENSOR_GROUPS = {c.split("_")[0]: [c] for c in COUNTERS}
SENSOR_GROUPS.update({h: [f"{h}_{i}" for i in range(n)] for h, n in HISTOGRAMS.items()})

OTHER = "Otras"


def sort_readouts(readouts: pd.DataFrame) -> pd.DataFrame:
    """Copia ordenada por vehicle_id y time_step."""
    if _data._is_sorted(readouts):
        return readouts.reset_index(drop=True).copy()
    return readouts.sort_values(["vehicle_id", "time_step"], kind="stable").reset_index(drop=True)


def null_pattern(readouts: pd.DataFrame) -> pd.DataFrame:
    """Para cada sensor: lecturas con el sensor entero nulo y lecturas con nulos parciales."""
    rows = []
    for sensor, cols in SENSOR_GROUPS.items():
        n_null = readouts[cols].isna().sum(axis=1)
        rows.append({
            "sensor": sensor,
            "columnas": len(cols),
            "lecturas sensor completo nulo": int((n_null == len(cols)).sum()),
            "lecturas nulo parcial": int(((n_null > 0) & (n_null < len(cols))).sum()),
            "% lecturas": round((n_null > 0).mean() * 100, 3),
        })
    return pd.DataFrame(rows).sort_values("% lecturas", ascending=False).reset_index(drop=True)


# Las funciones siguientes recorren las columnas de una en una para no duplicar en
# memoria la tabla completa (train ocupa ~1 GB en float64).

def _prev_valid(col: pd.Series, vid: pd.Series) -> pd.Series:
    """Último valor no nulo anterior dentro del mismo camión."""
    return col.groupby(vid, sort=False).shift().groupby(vid, sort=False).ffill()


def monotonic_violations(readouts: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Lecturas en las que una variable acumulada baja respecto a su último valor válido.

    Se compara con el último valor no nulo (no solo con la lectura anterior) para
    detectar también los descensos que quedan "escondidos" detrás de un hueco de nulos.
    Las lecturas deben estar ordenadas por vehicle_id y time_step.
    """
    cols = cols or SENSOR_COLUMNS
    vid = readouts["vehicle_id"]
    parts = []
    for c in cols:
        x = readouts[c]
        prev = _prev_valid(x, vid)
        d = x - prev
        idx = np.flatnonzero((d < 0).to_numpy())
        if len(idx) == 0:
            continue
        after_gap = x.groupby(vid, sort=False).shift().isna().to_numpy()[idx]
        pv = prev.to_numpy()[idx]
        parts.append(pd.DataFrame({
            "fila": idx,
            "vehicle_id": vid.to_numpy()[idx],
            "time_step": readouts["time_step"].to_numpy()[idx],
            "variable": c,
            "sensor": c.split("_")[0],
            "valor anterior": pv,
            "descenso": d.to_numpy()[idx],
            "descenso relativo": -d.to_numpy()[idx] / np.where(pv == 0, np.nan, pv),
            "tras hueco de nulos": after_gap,
        }))
    if not parts:
        return pd.DataFrame(columns=["fila", "vehicle_id", "time_step", "variable", "sensor",
                                     "valor anterior", "descenso", "descenso relativo",
                                     "tras hueco de nulos"])
    return pd.concat(parts, ignore_index=True).sort_values("fila", ignore_index=True)


def fix_resets(readouts: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Corrige EN SITIO los descensos de variables acumuladas sumando un desfase.

    Si una variable baja respecto a su último valor válido (reinicio de la ECU o del
    sensor), se suma la diferencia a esa lectura y a todas las siguientes del camión,
    de forma que la serie continúa sin bajar y se conservan los incrementos reales.
    Los nulos siguen siendo nulos. Devuelve una matriz booleana (lecturas x columnas)
    con los valores corregidos.
    """
    vid = readouts["vehicle_id"]
    events = np.zeros((len(readouts), len(cols)), dtype=bool)
    for j, c in enumerate(cols):
        x = readouts[c]
        drop = (_prev_valid(x, vid) - x).clip(lower=0).fillna(0)
        if not (drop > 0).any():
            continue
        events[:, j] = (drop > 0).to_numpy()
        corrected = x + drop.groupby(vid, sort=False).cumsum()
        # el máximo acumulado elimina restos de redondeo (p. ej. -5e-14) de la suma
        readouts[c] = corrected.groupby(vid, sort=False).cummax()
    return events


def growth_rates(readouts: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Incremento por unidad de time_step entre lecturas consecutivas del mismo camión."""
    g = readouts.groupby("vehicle_id", sort=False)
    dt = g["time_step"].diff()
    return g[cols].diff().div(dt, axis=0)


@dataclass
class Cleaner:
    """Limpieza ajustada en train y aplicable a cualquier conjunto.

    Pasos de transform():
      1. Ordena por vehicle_id y time_step.
      2. Corrige descensos de las variables acumuladas (reinicios) sumando un desfase,
         para que la serie siga creciendo sin perder los incrementos reales.
      3. Marca (no corrige) saltos atípicos en los contadores: log(1 + ritmo de
         crecimiento) por encima de Q3 + k·IQR calculado en train.
      4. Imputa nulos: lectura anterior del mismo camión (ffill), si no hay, la siguiente
         (bfill) y, si el camión no tiene ningún valor, la mediana de train.
      5. Agrupa categorías raras de las especificaciones (< min_count camiones en train)
         y las no vistas en train en "Otras".
    """

    min_count: int = 50          # camiones mínimos para mantener una categoría
    iqr_k: float = 3.0           # multiplicador del IQR (escala log) para marcar saltos
    medians_: dict = field(default_factory=dict)
    rate_limits_: dict = field(default_factory=dict)
    categories_: dict = field(default_factory=dict)
    report_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(self, readouts: pd.DataFrame, specs: pd.DataFrame) -> "Cleaner":
        self.medians_ = readouts[SENSOR_COLUMNS].median().astype(float).to_dict()

        counters = sort_readouts(readouts[["vehicle_id", "time_step"] + COUNTERS])
        fix_resets(counters, COUNTERS)
        rates = np.log1p(growth_rates(counters, COUNTERS).clip(lower=0))
        q1, q3 = rates.quantile(0.25), rates.quantile(0.75)
        self.rate_limits_ = (q3 + self.iqr_k * (q3 - q1)).astype(float).to_dict()

        self.categories_ = {}
        for c in SPEC_COLUMNS:
            counts = specs[c].astype(str).value_counts()
            keep = sorted(counts[counts >= self.min_count].index.tolist())
            self.categories_[c] = keep + [OTHER]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, readouts: pd.DataFrame, specs: pd.DataFrame | None = None):
        if not self.medians_:
            raise RuntimeError("Llama antes a fit() con los datos de train")
        r = sort_readouts(readouts)
        rep = {"lecturas": len(r), "camiones": int(r["vehicle_id"].nunique())}

        vid = r["vehicle_id"]
        isna = r[SENSOR_COLUMNS].isna()
        flags = pd.DataFrame(index=r.index)

        # 1. Descensos en variables acumuladas (reinicios) ----------------
        events = fix_resets(r, SENSOR_COLUMNS)
        flags["flag_reinicio"] = events.any(axis=1)
        rep["descensos corregidos (valores)"] = int(events.sum())
        rep["camiones con descensos"] = int(vid[flags["flag_reinicio"]].nunique())
        del events

        # 2. Saltos atípicos (solo se marcan) ----------------------------
        # Ritmo de crecimiento antes de imputar, para no crear saltos artificiales.
        rates = np.log1p(growth_rates(r, COUNTERS).clip(lower=0))
        flags["flag_salto"] = rates.gt(pd.Series(self.rate_limits_), axis=1).any(axis=1)
        rep["lecturas con salto atípico"] = int(flags["flag_salto"].sum())
        del rates

        # 3. Nulos -------------------------------------------------------
        flags["n_sensores_imputados"] = sum(
            isna[cols].any(axis=1).astype("int8") for cols in SENSOR_GROUPS.values()
        ).astype("int8")
        rep["valores nulos"] = int(isna.to_numpy().sum())
        rep["lecturas con nulos"] = int(isna.any(axis=1).sum())
        n_prev = n_next = n_median = 0
        for c in isna.columns[isna.any()]:
            x = r[c]
            x1 = x.groupby(vid, sort=False).ffill()
            x2 = x1.groupby(vid, sort=False).bfill()
            n_prev += int(x.isna().sum() - x1.isna().sum())
            n_next += int(x1.isna().sum() - x2.isna().sum())
            n_median += int(x2.isna().sum())
            r[c] = x2.fillna(self.medians_[c])
        rep["imputados con lectura anterior"] = n_prev
        rep["imputados con lectura siguiente"] = n_next
        rep["imputados con mediana de train"] = n_median
        del isna

        r = pd.concat([r, flags], axis=1)
        self.report_ = rep
        if specs is None:
            return r
        return r, self.transform_specs(specs)

    def transform_specs(self, specs: pd.DataFrame) -> pd.DataFrame:
        s = specs.copy()
        n_grouped = 0
        for c in SPEC_COLUMNS:
            cats = self.categories_[c]
            values = s[c].astype(str)
            mask = ~values.isin(cats)
            n_grouped += int(mask.sum())
            s[c] = pd.Categorical(values.where(~mask, OTHER), categories=cats)
        self.report_["especificaciones agrupadas en Otras"] = n_grouped
        return s

    # ------------------------------------------------------- persistencia
    def save(self, path: Path) -> None:
        params = {k: getattr(self, k) for k in ("min_count", "iqr_k", "medians_", "rate_limits_", "categories_")}
        Path(path).write_text(json.dumps(params, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Cleaner":
        return cls(**json.loads(Path(path).read_text()))


# ---------------------------------------------------------------- ficheros limpios

CLEAN = _data.PROCESSED / "clean"
CLEANER_PATH = _data.ROOT / "models" / "cleaner.json"


def build_clean(min_count: int = 50, iqr_k: float = 3.0) -> pd.DataFrame:
    """Ajusta el Cleaner en train, limpia los tres conjuntos y guarda los Parquet.

    Devuelve una tabla con el resumen de cambios por conjunto.
    """
    CLEAN.mkdir(parents=True, exist_ok=True)
    reports = {}
    cleaner = None
    for split in _data.SPLITS:
        readouts, specs, labels = _data.load_split(split)
        if cleaner is None:  # train va primero
            cleaner = Cleaner(min_count=min_count, iqr_k=iqr_k).fit(readouts, specs)
            cleaner.save(CLEANER_PATH)
        r, s = cleaner.transform(readouts, specs)
        r.to_parquet(CLEAN / f"{split}_readouts.parquet", index=False)
        s.to_parquet(CLEAN / f"{split}_specifications.parquet", index=False)
        labels.to_parquet(CLEAN / f"{split}_labels.parquet", index=False)
        reports[split] = cleaner.report_
        del readouts, r
    return pd.DataFrame(reports)


def load_clean(split: str, float32: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(lecturas, especificaciones, etiquetas) ya limpias de un conjunto."""
    r = pd.read_parquet(CLEAN / f"{split}_readouts.parquet")
    if float32:
        r[SENSOR_COLUMNS] = r[SENSOR_COLUMNS].astype("float32")
    s = pd.read_parquet(CLEAN / f"{split}_specifications.parquet")
    lb = pd.read_parquet(CLEAN / f"{split}_labels.parquet")
    return r, s, lb
