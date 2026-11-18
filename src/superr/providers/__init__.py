"""Situation providers: win probability (call A), next-ball distribution (call B), stakes (call C).

Two interchangeable back-ends:
  * jev       - TypeSafe AI's Jev, through the cached client (superr.jev)
  * empirical - models trained on the state table itself (plan section 12.5): a cross-fitted
                monotone gradient-boosted wp model, an empirical next-ball table, and a Monte
                Carlo of remaining fixtures for stakes.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger("superr.providers")


def resolve(cfg: dict, which: str) -> str:
    """Return 'jev' or 'empirical' for provider slot ``which`` (wp | p_next | stakes)."""
    choice = cfg["provider"].get(which, "auto")
    has_key = bool(os.environ.get(cfg["jev"].get("api_key_env", "TYPESAFE_API_KEY")))
    if choice == "auto":
        choice = "jev" if has_key else "empirical"
    if choice == "jev" and not has_key:
        raise RuntimeError(f"provider.{which} = jev but ${cfg['jev'].get('api_key_env')} is not set")
    return choice
