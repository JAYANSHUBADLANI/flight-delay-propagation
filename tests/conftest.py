import pandas as pd
import pytest


def _hhmm(v):
    v = 0 if v == 2400 else v
    return (v // 100) * 60 + (v % 100)


def _leg(date, tail, flt, org, dst, dep, arr, dep_delay, arr_delay,
         cancelled=0, diverted=0, elapsed=None):
    """One BTS-shaped row.

    `elapsed` is scheduled block time in minutes. It defaults to the clock
    difference between dep and arr, rolling past midnight when needed, which is
    what a same-timezone leg looks like. Pass it explicitly to build a leg that
    crosses a timezone or a day boundary.
    """
    if elapsed is None:
        elapsed = _hhmm(arr) - _hhmm(dep)
        if elapsed <= 0:
            elapsed += 1440
    return {
        "FlightDate": date, "Reporting_Airline": "XX", "Tail_Number": tail,
        "Flight_Number_Reporting_Airline": flt, "Origin": org, "Dest": dst,
        "CRSDepTime": dep, "DepTime": dep, "DepDelay": dep_delay,
        "CRSArrTime": arr, "ArrTime": arr, "ArrDelay": arr_delay,
        "CRSElapsedTime": elapsed,
        "Cancelled": cancelled, "Diverted": diverted, "Distance": 500,
        "CarrierDelay": 0, "WeatherDelay": 0, "NASDelay": 0,
        "SecurityDelay": 0, "LateAircraftDelay": 0,
    }


@pytest.fixture
def toy() -> pd.DataFrame:
    """A hand-built day: one clean 3-leg rotation, one broken chain, plus rows
    that must be dropped (cancelled, diverted, missing tail)."""
    rows = [
        # N1: ORD -> DFW -> DEN, continuous. Second leg gets a 60 min late
        # aircraft against a 60 min scheduled turn.
        _leg("2024-01-02", "N1", 1, "ORD", "DFW", 800, 1000, 0, 60),
        _leg("2024-01-02", "N1", 2, "DFW", "DEN", 1100, 1300, 30, 25),
        _leg("2024-01-02", "N1", 3, "DEN", "ORD", 1400, 1700, 0, 0),
        # N2: chain break, the second leg does not start where the first ended.
        _leg("2024-01-02", "N2", 4, "LAX", "SFO", 900, 1000, 0, 20),
        _leg("2024-01-02", "N2", 5, "SEA", "PDX", 1200, 1300, 5, 0),
        # N3: overnight leg, scheduled arrival before scheduled departure.
        _leg("2024-01-02", "N3", 6, "JFK", "LAX", 2300, 200, 0, 10),
        # Rows that must never reach the rotation builder.
        _leg("2024-01-02", "N4", 7, "ATL", "MCO", 700, 830, 0, 0, cancelled=1),
        _leg("2024-01-02", "N5", 8, "BOS", "PHL", 700, 830, 0, 0, diverted=1),
        _leg("2024-01-02", None, 9, "IAH", "AUS", 700, 830, 0, 0),
    ]
    return pd.DataFrame(rows)
