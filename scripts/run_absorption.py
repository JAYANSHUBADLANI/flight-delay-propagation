"""Rebuild rotations and write the absorption evidence into reports/."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.absorption import (absorption_by_buffer, absorption_by_buffer_controlled,
                            absorption_by_inbound_delay, additive_penalty,
                            baseline_departure_delay)
from src.rotations import build_rotations, chain_quality, clean, load_month

REPORTS = Path("reports")


def main(pattern: str = "data/raw/*.csv") -> None:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no BTS files matched {pattern!r}; run `make data` first")

    REPORTS.mkdir(exist_ok=True)
    raw = load_month(files[0]) if len(files) == 1 else None
    if raw is None:
        import pandas as pd
        raw = pd.concat([load_month(f) for f in files], ignore_index=True)

    d = build_rotations(clean(raw))
    quality = chain_quality(d)
    baseline = baseline_departure_delay(d)
    quality["baseline_departure_delay_min"] = round(baseline, 2)
    quality["months_loaded"] = len(files)

    by_delay = absorption_by_inbound_delay(d, baseline)
    by_buffer = absorption_by_buffer(d, baseline=baseline)
    import pandas as pd
    controlled = pd.concat(
        [absorption_by_buffer_controlled(d, lo, hi, baseline) for lo, hi in [(30, 60), (60, 90)]],
        ignore_index=True,
    )

    (REPORTS / "chain_quality.json").write_text(json.dumps(quality, indent=2))
    by_delay.to_json(REPORTS / "absorption_by_inbound_delay.json", orient="records", indent=2)
    by_buffer.to_json(REPORTS / "absorption_by_buffer.json", orient="records", indent=2)
    controlled.to_json(REPORTS / "absorption_by_buffer_controlled.json", orient="records", indent=2)
    additive = additive_penalty(d, baseline)
    additive.to_json(REPORTS / "additive_penalty.json", orient="records", indent=2)

    print(json.dumps(quality, indent=2))
    print("\nABSORPTION BY INBOUND DELAY\n", by_delay.to_string(index=False))
    print("\nABSORPTION BY SCHEDULED TURN (inbound > 15 min late)\n",
          by_buffer.to_string(index=False))
    print("\nSAME, WITH INBOUND DELAY HELD IN A NARROW BAND\n",
          controlled.to_string(index=False))
    print("\nADDED MINUTES ON TOP OF THE INBOUND DELAY\n",
          additive.pivot(index="turn_bucket", columns="inbound_bucket",
                         values="added_minutes").to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/*.csv")
