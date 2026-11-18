"""Stage 02: Jev calls A, B, C with cache (or the empirical fallback).

Usage: python src/02_jev.py [--set key=value ...]   (same as: python -m superr run jev)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "jev", *sys.argv[1:]])
