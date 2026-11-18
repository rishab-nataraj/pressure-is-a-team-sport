"""Stage 07: spec section 7 and plan section 11 checks.

Usage: python src/07_validate.py [--set key=value ...]   (same as: python -m superr run validate)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from superr.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["run", "validate", *sys.argv[1:]])
