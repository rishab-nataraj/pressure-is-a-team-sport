"""Stage 08: methodology report and figures.

Usage: python src/08_report.py [--set key=value ...]   (same as: python -m superr run report)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "report", *sys.argv[1:]])
