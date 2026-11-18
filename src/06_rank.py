"""Stage 06: scores, eligibility, top-10 lists (plan section 10).

Usage: python src/06_rank.py [--set key=value ...]   (same as: python -m superr run rank)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "rank", *sys.argv[1:]])
