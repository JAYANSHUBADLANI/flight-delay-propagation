"""Train the as-of-cutoff model and write reports/prediction_metrics.json.

Always runs a within-month split. If a second month is present it also runs a
true out-of-time evaluation, which is the one worth believing: a within-month
split still shares weather systems and schedule quirks across the boundary.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import build_features, time_split
from src.model import evaluate
from src.rotations import build_rotations, clean, load_month

REPORTS = Path("reports")


def _month_key(path: str) -> tuple[int, int]:
    m = re.search(r"(\d{4})_(\d{1,2})\.csv$", path)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def prepare(path: str) -> pd.DataFrame:
    return build_features(build_rotations(clean(load_month(path))))


def main() -> None:
    files = sorted(glob.glob("data/raw/*.csv"), key=_month_key)
    if not files:
        raise SystemExit("no BTS files in data/raw/; run `make data` first")

    first = prepare(files[0])
    out: dict = {
        "lead_minutes": 120,
        "train_month": Path(files[0]).stem,
        "chained_legs": int(len(first)),
        "inbound_landed_by_cutoff_pct": round(float(first["inbound_landed_by_cutoff"].mean() * 100), 2),
        "inbound_departed_by_cutoff_pct": round(float(first["inbound_departed_by_cutoff"].mean() * 100), 2),
    }

    train, test = time_split(first, cut_day=25)
    out["within_month"] = {
        "train_n": int(len(train)), "test_n": int(len(test)),
        **evaluate(train, test),
    }

    if len(files) > 1:
        later = prepare(files[1])
        out["out_of_time"] = {
            "test_month": Path(files[1]).stem,
            "train_n": int(len(first)), "test_n": int(len(later)),
            **evaluate(first, later),
        }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "prediction_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
