"""Features for predicting a departure delay, built as of a fixed lead time.

The failure mode this module exists to avoid is leakage. The obvious features
for "will this leg leave late" are the inbound aircraft's arrival delay and the
turn that actually happened -- but at the moment a prediction is useful, the
inbound may still be in the air, and its arrival delay is not a number anyone
has. A model trained on it scores well and cannot be run.

So every feature here is stamped with an explicit cutoff, `LEAD_MINUTES` before
scheduled departure, and a fact about the inbound leg enters only if it had
already happened by then. Where it had not, the feature is missing rather than
filled, and the model is left to handle the missingness -- which is itself the
signal that the aircraft was still out.
"""
from __future__ import annotations

import pandas as pd

LEAD_MINUTES = 120

NUMERIC_FEATURES = [
    "sched_dep_min",
    "sched_turn_min",
    "leg_index",
    "legs_in_day",
    "Distance",
    "inbound_sched_arr_slack",
    "known_inbound_arr_delay",
    "known_inbound_dep_delay",
    "origin_departures_that_hour",
]
BOOLEAN_FEATURES = ["inbound_landed_by_cutoff", "inbound_departed_by_cutoff"]
CATEGORICAL_FEATURES = ["Reporting_Airline", "day_of_week"]

FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
TARGET = "DepDelay"


def build_features(d: pd.DataFrame, lead_minutes: int = LEAD_MINUTES) -> pd.DataFrame:
    """Return the chained legs with an as-of-cutoff feature block attached."""
    f = d[d["usable_turn"]].copy()
    cutoff = f["sched_dep_min"] - lead_minutes
    f["cutoff_min"] = cutoff

    # What had actually happened to the inbound aircraft by the cutoff.
    f["inbound_landed_by_cutoff"] = f["prev_actual_arr_min"] <= cutoff
    f["inbound_departed_by_cutoff"] = f["prev_actual_dep_min"] <= cutoff

    # Its arrival delay is knowable only once it has landed; its departure delay
    # only once it has left. Anything else stays missing on purpose.
    f["known_inbound_arr_delay"] = f["prev_arr_delay"].where(f["inbound_landed_by_cutoff"])
    f["known_inbound_dep_delay"] = f["prev_dep_delay"].where(f["inbound_departed_by_cutoff"])

    # Schedule-only, so always known: how long after the cutoff the inbound was
    # *planned* to land. Positive means the schedule itself says we will be
    # deciding before the aircraft is due.
    f["inbound_sched_arr_slack"] = f["prev_sched_arr_min"] - cutoff

    dates = pd.to_datetime(f["FlightDate"])
    f["day_of_week"] = dates.dt.dayofweek.astype("category")
    f["Reporting_Airline"] = f["Reporting_Airline"].astype("category")

    # Scheduled congestion at the origin, from the published schedule only.
    f["dep_hour"] = (f["sched_dep_min"] // 60).astype(int)
    counts = f.groupby(["FlightDate", "Origin", "dep_hour"])["Origin"].transform("size")
    f["origin_departures_that_hour"] = counts

    return f


def time_split(f: pd.DataFrame, cut_day: int):
    """Split by calendar day: earlier days train, later days test.

    A random split would put a leg's own rotation-mates on both sides and let
    the model see the day it is being asked about.
    """
    day = pd.to_datetime(f["FlightDate"]).dt.day
    return f[day < cut_day].copy(), f[day >= cut_day].copy()


def xy(f: pd.DataFrame):
    return f[FEATURES], f[TARGET]
