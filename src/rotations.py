"""Reconstruct aircraft rotations from BTS On-Time Performance rows.

A single aircraft flies a sequence of legs through the day. BTS publishes one
row per leg with a tail number, so the rotation can be rebuilt by sorting a
tail's legs by scheduled departure. Everything downstream -- delay absorption,
the prediction features, the recovery model -- depends on those chains being
right, so this module is deliberately strict about what counts as a chain.
"""
from __future__ import annotations

import pandas as pd

from src.config import MAX_TURN_MINUTES, USECOLS

_MINUTES_PER_DAY = 1440


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert BTS 'HHMM' integers to minutes after midnight.

    BTS writes midnight as 2400 rather than 0000, which would otherwise sort
    after every other departure of the day and silently reverse a rotation.
    """
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.where(vals != 2400, 0)
    return (vals // 100) * 60 + (vals % 100)


def load_month(path, usecols=None) -> pd.DataFrame:
    """Read one monthly BTS CSV, keeping only the columns this project uses."""
    return pd.read_csv(path, usecols=usecols or USECOLS, low_memory=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the legs that actually flew and can be placed on a timeline.

    Cancelled and diverted legs are dropped: a cancelled leg has no arrival to
    propagate, and a diverted one ends somewhere the schedule does not know
    about, so it breaks the chain by definition rather than by data error.
    """
    out = df.copy()
    out["Tail_Number"] = out["Tail_Number"].astype("string").str.strip()
    out.loc[out["Tail_Number"].isin(["", "UNKNOW", "UNKNOWN"]), "Tail_Number"] = pd.NA

    keep = (
        (out["Cancelled"] == 0)
        & (out["Diverted"] == 0)
        & out["Tail_Number"].notna()
        & out["DepDelay"].notna()
        & out["ArrDelay"].notna()
    )
    out = out.loc[keep].copy()

    out["sched_dep_min"] = hhmm_to_minutes(out["CRSDepTime"])

    # Arrival is departure plus scheduled elapsed time, NOT the published
    # arrival clock time.
    #
    # BTS timestamps every leg in the *local* time of its own airport, so a
    # westbound flight can land at an earlier clock reading than it departed
    # without any midnight involved: ATL-HSV leaves at 08:25 Eastern and lands
    # at 08:24 Central after 59 minutes in the air. Treating "arrival before
    # departure" as an overnight and adding a day, which is the obvious rule,
    # is wrong in both directions -- in January 2024 it shifts 2,575 legs by a
    # spurious 1,440 minutes and still misses about 8,800 genuine overnights.
    # Elapsed time is timezone-free and gives one consistent clock.
    out["sched_arr_min"] = out["sched_dep_min"] + pd.to_numeric(
        out["CRSElapsedTime"], errors="coerce")
    out["overnight"] = out["sched_arr_min"] >= _MINUTES_PER_DAY
    out = out[out["sched_arr_min"].notna()].copy()

    out["actual_arr_min"] = out["sched_arr_min"] + out["ArrDelay"]
    out["actual_dep_min"] = out["sched_dep_min"] + out["DepDelay"]
    return out.reset_index(drop=True)


def build_rotations(df: pd.DataFrame) -> pd.DataFrame:
    """Order each aircraft's legs and attach the preceding leg to every row.

    Rotations are cut at the calendar day. An aircraft that flies past midnight
    does continue the next morning, but BTS keys the row by FlightDate and the
    overnight gap is long enough that the delay does not survive it, so a
    same-day chain is the honest unit here. This is stated in the README as a
    limitation rather than hidden.
    """
    d = df.sort_values(["Tail_Number", "FlightDate", "sched_dep_min"]).copy()
    g = d.groupby(["Tail_Number", "FlightDate"], sort=False)

    d["leg_index"] = g.cumcount()
    d["legs_in_day"] = g["Origin"].transform("size")

    d["prev_dest"] = g["Dest"].shift(1)
    d["prev_arr_delay"] = g["ArrDelay"].shift(1)
    d["prev_sched_arr_min"] = g["sched_arr_min"].shift(1)
    d["prev_actual_arr_min"] = g["actual_arr_min"].shift(1)
    d["prev_sched_dep_min"] = g["sched_dep_min"].shift(1)
    d["prev_actual_dep_min"] = g["actual_dep_min"].shift(1)
    d["prev_dep_delay"] = g["DepDelay"].shift(1)
    d["prev_origin"] = g["Origin"].shift(1)

    d["has_predecessor"] = d["prev_dest"].notna()
    # A genuine turn: the aircraft left from where the previous leg put it.
    d["chain_ok"] = d["has_predecessor"] & (d["prev_dest"] == d["Origin"])
    d["sched_turn_min"] = d["sched_dep_min"] - d["prev_sched_arr_min"]
    d["usable_turn"] = (
        d["chain_ok"]
        & d["sched_turn_min"].between(0, MAX_TURN_MINUTES)
    )
    return d.reset_index(drop=True)


def chain_quality(d: pd.DataFrame) -> dict:
    """Summary of how well the rotations reconstructed, for reports/."""
    succ = d[d["has_predecessor"]]
    return {
        "legs": int(len(d)),
        "tail_days": int(d.groupby(["Tail_Number", "FlightDate"]).ngroups),
        "mean_legs_per_tail_day": float(d["legs_in_day"].mean()),
        "successor_legs": int(len(succ)),
        "chain_continuous_pct": float(succ["chain_ok"].mean() * 100) if len(succ) else float("nan"),
        "usable_turn_legs": int(d["usable_turn"].sum()),
        "median_sched_turn_min": float(d.loc[d["usable_turn"], "sched_turn_min"].median()),
        "overnight_legs_pct": float(d["overnight"].mean() * 100),
    }
