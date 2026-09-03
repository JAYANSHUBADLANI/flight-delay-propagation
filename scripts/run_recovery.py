"""Solve the recovery MILP across several real disruptions, write reports/.

Each scenario takes one carrier's worst-exposed aircraft out of a hub, holds it
on the ground for a fixed number of minutes, and asks what the rest of its day
should look like.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.recovery import (_connections, build_scenario, do_nothing, solve)
from src.rotations import build_rotations, clean, load_month

REPORTS = Path("reports")

SCENARIOS = [
    ("2024-01-15", "UA", "ORD"),
    ("2024-01-15", "AA", "DFW"),
    ("2024-01-22", "DL", "ATL"),
    ("2024-01-08", "WN", "DEN"),
]
DISRUPTIONS = [60.0, 120.0, 180.0]


def main() -> None:
    files = sorted(glob.glob("data/raw/*.csv"))
    if not files:
        raise SystemExit("no BTS files in data/raw/; run `make data` first")
    d = build_rotations(clean(load_month(files[0])))

    rows = []
    for date, carrier, station in SCENARIOS:
        for minutes in DISRUPTIONS:
            try:
                sc = build_scenario(d, date, carrier, station, disruption_minutes=minutes)
            except ValueError:
                continue
            base, opt = do_nothing(sc), solve(sc)
            if opt.status != "optimal":
                rows.append({"date": date, "carrier": carrier, "station": station,
                             "disruption_minutes": minutes, "status": opt.status})
                continue
            covered = set(opt.assignment) | set(opt.cancelled)
            rows.append({
                "date": date, "carrier": carrier, "station": station,
                "disruption_minutes": minutes,
                "legs": int(len(sc.legs)), "aircraft": int(len(sc.aircraft)),
                "connections": int(len(_connections(sc))),
                "do_nothing_delay_minutes": base.total_delay_minutes,
                "recovered_delay_minutes": opt.total_delay_minutes,
                "delay_saved_minutes": round(base.total_delay_minutes
                                             - opt.total_delay_minutes, 1),
                "swaps": int(opt.swaps), "cancellations": int(len(opt.cancelled)),
                # A leg reachable from no aircraft would mean a subtour survived.
                "unassigned_legs": int(len(sc.legs) - len(covered)),
                "status": opt.status,
            })

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "recovery_scenarios.json").write_text(json.dumps(rows, indent=2))

    print(f"{'scenario':<24}{'disr':>6}{'legs':>6}{'nothing':>9}{'recovered':>11}"
          f"{'saved':>8}{'swaps':>7}{'cancels':>9}{'orphan':>8}")
    for r in rows:
        if r.get("status") != "optimal":
            print(f"{r['date']} {r['carrier']}/{r['station']:<8}{r['disruption_minutes']:>6.0f}  {r['status']}")
            continue
        print(f"{r['date']} {r['carrier']}/{r['station']:<8}{r['disruption_minutes']:>6.0f}"
              f"{r['legs']:>6}{r['do_nothing_delay_minutes']:>9.0f}"
              f"{r['recovered_delay_minutes']:>11.0f}{r['delay_saved_minutes']:>8.0f}"
              f"{r['swaps']:>7}{r['cancellations']:>9}{r['unassigned_legs']:>8}")


if __name__ == "__main__":
    main()
