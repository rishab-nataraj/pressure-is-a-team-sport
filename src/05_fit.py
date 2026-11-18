"""Stage 05: fit_pt, CV grid, cluster bootstrap (plan sections 5-9).

Usage: python src/05_fit.py [--set key=value ...]   (same as: python -m superr run fit)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "fit", *sys.argv[1:]])
