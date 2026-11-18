"""Reading the raw data pack and reading/writing interim tables."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Paths

log = logging.getLogger("superr.io")

RAW_FILES = {
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "matches": "matches.csv",
    "playing_eleven": "playing_eleven.csv",
    "mapping": "competition_team_player_mapping.csv",
    "bbb": "ball_by_ball.csv",
}


def read_raw(paths: Paths, which: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read the data-pack CSVs. ``which`` restricts to a subset of RAW_FILES keys."""
    out = {}
    for key, fname in RAW_FILES.items():
        if which is not None and key not in which:
            continue
        fp = paths.raw / fname
        if not fp.exists():
            raise FileNotFoundError(
                f"{fp} not found. Put the six data-pack CSVs in {paths.raw} (see data/README.md).")
        df = pd.read_csv(fp, low_memory=False)
        if key == "matches":
            df["match_date"] = pd.to_datetime(df["match_date"])
        out[key] = df
    return out


def save(df: pd.DataFrame, name: str, paths: Paths) -> Path:
    fp = paths.interim / f"{name}.parquet"
    df.to_parquet(fp, index=False)
    log.info("wrote %s (%d rows)", fp.relative_to(paths.interim.parent.parent), len(df))
    return fp


def load(name: str, paths: Paths) -> pd.DataFrame:
    fp = paths.interim / f"{name}.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} missing: run the stage that produces '{name}' first.")
    return pd.read_parquet(fp)


def save_output(df: pd.DataFrame, name: str, paths: Paths, sub: str | None = None,
                index: bool = False) -> Path:
    d = paths.outputs / sub if sub else paths.outputs
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{name}.csv"
    df.to_csv(fp, index=index)
    return fp


def save_json(obj, name: str, paths: Paths, sub: str | None = None) -> Path:
    d = paths.outputs / sub if sub else paths.outputs
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{name}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return fp


def load_json(name: str, paths: Paths, sub: str | None = None):
    d = paths.outputs / sub if sub else paths.outputs
    with open(d / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
