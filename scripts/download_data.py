"""Download BTS months named on the command line (default: all of 2024)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.download import download_month


def main(argv: list[str]) -> None:
    year = int(argv[0]) if argv else 2024
    months = [int(m) for m in argv[1:]] or list(range(1, 13))
    for m in months:
        path = download_month(year, m)
        size_mb = path.stat().st_size / 1e6
        print(f"{year}-{m:02d}  {path}  {size_mb:,.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1:])
