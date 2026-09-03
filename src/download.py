"""Fetch monthly BTS On-Time Performance files.

The files are served as plain zips with no authentication, so the whole data
step is reproducible: `make data` gets you exactly what this analysis ran on.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

from src.config import BTS_URL

RAW_DIR = Path("data/raw")


def month_url(year: int, month: int) -> str:
    """BTS uses no leading zero on the month."""
    return BTS_URL.format(year=year, month=month)


def download_month(year: int, month: int, dest: Path = RAW_DIR,
                   timeout: int = 900) -> Path:
    """Download one month and extract its CSV. Returns the CSV path.

    Skips the network entirely if the CSV is already on disk, so re-running
    `make data` after a partial run is cheap.
    """
    dest.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest.glob(f"*{year}_{month}.csv"))
    if existing:
        return existing[0]

    resp = requests.get(month_url(year, month), timeout=timeout)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV inside the {year}-{month} archive")
        src_name = names[0]
        out = dest / f"bts_{year}_{month}.csv"
        with zf.open(src_name) as fh, out.open("wb") as target:
            target.write(fh.read())
    return out
