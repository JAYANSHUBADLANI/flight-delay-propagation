import pandas as pd

from src.rotations import (build_rotations, chain_quality, clean,
                           hhmm_to_minutes)


def test_hhmm_basic():
    got = hhmm_to_minutes(pd.Series([0, 15, 100, 830, 1435, 2359]))
    assert list(got) == [0, 15, 60, 510, 875, 1439]


def test_hhmm_treats_2400_as_midnight():
    # BTS writes midnight as 2400. Left alone it sorts after every other
    # departure and silently reverses a rotation.
    assert hhmm_to_minutes(pd.Series([2400])).iloc[0] == 0


def test_clean_drops_unusable_rows(toy):
    out = clean(toy)
    assert len(out) == 6  # cancelled, diverted and null-tail rows are gone
    assert out["Cancelled"].sum() == 0
    assert out["Diverted"].sum() == 0
    assert out["Tail_Number"].notna().all()


def test_overnight_arrival_lands_after_midnight(toy):
    out = clean(toy)
    night = out[out["Flight_Number_Reporting_Airline"] == 6].iloc[0]
    assert night["overnight"]
    # 23:00 departure, 3 hours in the air -> 02:00 the next day on one clock.
    assert night["sched_arr_min"] == 1380 + 180
    assert night["sched_arr_min"] > night["sched_dep_min"]


def test_a_westbound_leg_is_not_mistaken_for_an_overnight():
    # ATL 08:25 Eastern to HSV 08:24 Central is 59 minutes in the air, not a
    # flight that lands the next day. Reading the clock instead of the elapsed
    # time adds a spurious 1,440 minutes and destroys the turnaround arithmetic.
    from tests.conftest import _leg
    row = pd.DataFrame([_leg("2024-01-02", "N7", 1, "ATL", "HSV", 825, 824, 0, 0,
                             elapsed=59)])
    out = clean(row).iloc[0]
    assert not out["overnight"]
    assert out["sched_arr_min"] == out["sched_dep_min"] + 59


def test_chain_continuity_and_break(toy):
    d = build_rotations(clean(toy))
    n1 = d[d["Tail_Number"] == "N1"].sort_values("leg_index")
    assert list(n1["chain_ok"]) == [False, True, True]  # first leg has no predecessor

    n2 = d[d["Tail_Number"] == "N2"].sort_values("leg_index")
    # SFO -> SEA is not a turn; the aircraft cannot have flown it
    assert not n2["chain_ok"].any()


def test_scheduled_turn_and_previous_delay(toy):
    d = build_rotations(clean(toy))
    leg2 = d[(d["Tail_Number"] == "N1") & (d["leg_index"] == 1)].iloc[0]
    assert leg2["prev_arr_delay"] == 60
    assert leg2["sched_turn_min"] == 60  # 10:00 arrive, 11:00 depart
    assert leg2["usable_turn"]


def test_chain_quality_shape(toy):
    q = chain_quality(build_rotations(clean(toy)))
    assert q["legs"] == 6
    assert 0 <= q["chain_continuous_pct"] <= 100
    assert q["tail_days"] == 3
