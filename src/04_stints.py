"""Stage 04: stint table (spec section 6).

Usage: python src/04_stints.py [--set key=value ...]   (same as: python -m superr run stints)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "stints", *sys.argv[1:]])
