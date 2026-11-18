"""Configuration loading and project paths."""
from __future__ import annotations

import copy
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"

log = logging.getLogger("superr")


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def deep_update(base: dict, upd: dict | None) -> dict:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config/config.yaml (or ``path``), then apply ``overrides`` (nested dict)."""
    cfg = load_yaml(Path(path) if path else CONFIG_DIR / "config.yaml")
    cfg["aliases"] = load_yaml(CONFIG_DIR / "team_aliases.yaml")
    cfg["tournaments"] = load_yaml(CONFIG_DIR / "tournaments.yaml")
    env_over = os.environ.get("SUPERR_CONFIG_OVERRIDES")
    if env_over:
        cfg = deep_update(cfg, yaml.safe_load(env_over))
    return deep_update(cfg, overrides)


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    jev_cache: Path
    outputs: Path
    report: Path

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Paths":
        def p(key: str) -> Path:
            q = Path(cfg["paths"][key])
            return q if q.is_absolute() else ROOT / q

        out = cls(raw=p("raw"), interim=p("interim"), jev_cache=p("jev_cache"),
                  outputs=p("outputs"), report=p("report"))
        for d in (out.interim, out.jev_cache, out.outputs, out.report, out.report / "figures",
                  out.outputs / "validation"):
            d.mkdir(parents=True, exist_ok=True)
        return out


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
