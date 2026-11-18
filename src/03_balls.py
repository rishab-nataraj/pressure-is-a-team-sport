"""Stage 03: WPA, LI, pressure, attribution (spec section 5).

Usage: python src/03_balls.py [--set key=value ...]   (same as: python -m superr run balls)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "balls", *sys.argv[1:]])
