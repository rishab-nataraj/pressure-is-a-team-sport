"""Stage 00: team_canon, player names, enriched matches.

Usage: python src/00_canon.py [--set key=value ...]   (same as: python -m superr run canon)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "canon", *sys.argv[1:]])
