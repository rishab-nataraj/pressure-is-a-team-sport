"""Stage 01: state table, one row per delivery (spec section 1).

Usage: python src/01_states.py [--set key=value ...]   (same as: python -m superr run states)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "states", *sys.argv[1:]])
