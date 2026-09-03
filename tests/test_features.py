import pandas as pd

from src.features import build_features, time_split
from src.rotations import build_rotations, clean


def _chain(prev_arr_delay, prev_dep_delay, sched_turn, dep_delay=0.0):
    """One two-leg rotation with the timings the cutoff logic keys off.

    Leg 1: 08:00 -> 10:00 scheduled. Leg 2 departs `sched_turn` after that.
    """
    from tests.conftest import _leg

    dep2_min = 600 + sched_turn
    dep2 = (dep2_min // 60) * 100 + dep2_min % 60
    rows = [
        _leg("2024-01-02", "N9", 1, "ORD", "DFW", 800, 1000, prev_dep_delay, prev_arr_delay),
        _leg("2024-01-02", "N9", 2, "DFW", "DEN", int(dep2), 1500, dep_delay, 0),
    ]
    return build_rotations(clean(pd.DataFrame(rows)))


def test_arrival_delay_is_hidden_when_the_inbound_has_not_landed():
    # 3 hour turn: cutoff is 2h before an 13:00 departure, i.e. 11:00. The
    # inbound was scheduled in at 10:00 but ran 90 minutes late, landing 11:30 --
    # after the cutoff. Its arrival delay cannot be known.
    f = build_features(_chain(prev_arr_delay=90, prev_dep_delay=60, sched_turn=180))
    row = f.iloc[0]
    assert not row["inbound_landed_by_cutoff"]
    assert pd.isna(row["known_inbound_arr_delay"])
    # It had departed by then, so that much is fair game.
    assert row["inbound_departed_by_cutoff"]
    assert row["known_inbound_dep_delay"] == 60


def test_arrival_delay_is_used_when_the_inbound_landed_in_time():
    # Same 3 hour turn, but the inbound was only 10 minutes late, landing 10:10,
    # comfortably before the 11:00 cutoff.
    f = build_features(_chain(prev_arr_delay=10, prev_dep_delay=5, sched_turn=180))
    row = f.iloc[0]
    assert row["inbound_landed_by_cutoff"]
    assert row["known_inbound_arr_delay"] == 10


def test_nothing_about_the_inbound_leaks_on_a_short_turn():
    # 45 minute turn: the leg departs 10:45, so the cutoff is 08:45 -- before the
    # inbound has even left at 08:00 + 60 late = 09:00. Nothing is knowable.
    f = build_features(_chain(prev_arr_delay=70, prev_dep_delay=60, sched_turn=45))
    row = f.iloc[0]
    assert not row["inbound_landed_by_cutoff"]
    assert not row["inbound_departed_by_cutoff"]
    assert pd.isna(row["known_inbound_arr_delay"])
    assert pd.isna(row["known_inbound_dep_delay"])


def test_schedule_only_features_are_always_available():
    # The slack figure is built from published times alone, so it survives even
    # the case where every actual is hidden.
    f = build_features(_chain(prev_arr_delay=70, prev_dep_delay=60, sched_turn=45))
    row = f.iloc[0]
    assert not pd.isna(row["inbound_sched_arr_slack"])
    assert not pd.isna(row["sched_turn_min"])
    # 10:45 dep, cutoff 08:45, inbound due 10:00 -> 75 minutes after the cutoff
    assert row["inbound_sched_arr_slack"] == 75


def test_cutoff_moves_with_the_lead_time():
    d = _chain(prev_arr_delay=90, prev_dep_delay=60, sched_turn=180)
    tight = build_features(d, lead_minutes=30).iloc[0]
    wide = build_features(d, lead_minutes=240).iloc[0]
    # A 30 minute lead sees the landing; a 4 hour lead cannot.
    assert tight["inbound_landed_by_cutoff"]
    assert not wide["inbound_landed_by_cutoff"]


def test_time_split_does_not_mix_days():
    rows = []
    from tests.conftest import _leg
    for day in range(1, 9):
        rows += [
            _leg(f"2024-01-0{day}", "N9", 1, "ORD", "DFW", 800, 1000, 0, 10),
            _leg(f"2024-01-0{day}", "N9", 2, "DFW", "DEN", 1200, 1500, 5, 0),
        ]
    f = build_features(build_rotations(clean(pd.DataFrame(rows))))
    tr, te = time_split(f, cut_day=6)
    tr_days = set(pd.to_datetime(tr["FlightDate"]).dt.day)
    te_days = set(pd.to_datetime(te["FlightDate"]).dt.day)
    assert tr_days and te_days
    assert not (tr_days & te_days)
    assert max(tr_days) < min(te_days)
