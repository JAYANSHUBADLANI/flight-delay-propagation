import pandas as pd

from src.absorption import (absorption_by_buffer_controlled,
                            absorption_by_inbound_delay,
                            baseline_departure_delay)


def _turns(n_baseline=200, n_late=200, baseline_delay=5.0,
           inbound=50.0, dep_when_late=30.0, turn=60.0):
    """Chained legs with delays chosen so the corrected answer is known by hand."""
    rows = []
    for _ in range(n_baseline):
        rows.append({"usable_turn": True, "prev_arr_delay": -10.0,
                     "DepDelay": baseline_delay, "sched_turn_min": turn})
    for _ in range(n_late):
        rows.append({"usable_turn": True, "prev_arr_delay": inbound,
                     "DepDelay": dep_when_late, "sched_turn_min": turn})
    return pd.DataFrame(rows)


def test_baseline_is_measured_on_non_late_inbounds_only():
    d = _turns(baseline_delay=5.0, dep_when_late=30.0)
    assert baseline_departure_delay(d) == 5.0


def test_adjusted_passthrough_nets_out_the_baseline():
    d = _turns(baseline_delay=5.0, inbound=50.0, dep_when_late=30.0)
    row = absorption_by_inbound_delay(d).query("bucket == '30-60'").iloc[0]
    assert row["raw_passthrough"] == 0.6           # 30 / 50, the confounded figure
    assert row["adjusted_passthrough"] == 0.5      # (30 - 5) / 50
    assert row["absorbed_pct"] == 50.0


def test_raw_ratio_can_exceed_one_while_corrected_does_not():
    # The confound that motivates the correction: a small inbound delay sitting
    # on top of an ordinary baseline looks like amplification.
    d = _turns(baseline_delay=6.0, inbound=8.0, dep_when_late=12.0)
    row = absorption_by_inbound_delay(d).query("bucket == '0-15'").iloc[0]
    assert row["raw_passthrough"] > 1.0
    assert row["adjusted_passthrough"] < row["raw_passthrough"]


def test_early_inbound_bucket_reports_no_passthrough():
    # Dividing by a negative mean inbound delay would produce a number that
    # looks like absorption but means nothing.
    d = _turns()
    row = absorption_by_inbound_delay(d).query("bucket == 'early / on time'").iloc[0]
    # pandas coerces the None to NaN because the column holds floats elsewhere;
    # what matters is that no ratio is reported, not which null it is.
    assert pd.isna(row["adjusted_passthrough"])
    assert pd.isna(row["raw_passthrough"])


def test_small_buckets_are_dropped_not_reported_noisily():
    d = _turns(n_late=10)  # below the 50-row floor
    out = absorption_by_inbound_delay(d)
    assert "30-60" not in set(out["bucket"])


def test_controlled_buffer_holds_inbound_delay_flat():
    a = _turns(n_late=200, inbound=45.0, dep_when_late=50.0, turn=30.0)
    b = _turns(n_late=200, inbound=45.0, dep_when_late=20.0, turn=120.0)
    d = pd.concat([a, b], ignore_index=True)
    out = absorption_by_buffer_controlled(d, 30, 60)
    means = out["mean_inbound_delay"].tolist()
    assert max(means) - min(means) < 1.0        # the band did its job
    short = out.query("bucket == '<40 min'").iloc[0]
    long_ = out.query("bucket == '90-150'").iloc[0]
    assert short["adjusted_passthrough"] > long_["adjusted_passthrough"]


def test_additive_penalty_separates_the_two_regimes():
    from src.absorption import additive_penalty

    # A tight turn that adds a fixed 10 minutes whatever arrives, against a long
    # turn that hands back half of whatever arrives.
    rows = []
    for inbound, dep_tight in [(8.0, 18.0), (45.0, 55.0)]:
        for _ in range(200):
            rows.append({"usable_turn": True, "prev_arr_delay": inbound,
                         "DepDelay": dep_tight, "sched_turn_min": 30.0})
            rows.append({"usable_turn": True, "prev_arr_delay": inbound,
                         "DepDelay": inbound / 2, "sched_turn_min": 120.0})
    for _ in range(200):
        rows.append({"usable_turn": True, "prev_arr_delay": -5.0,
                     "DepDelay": 0.0, "sched_turn_min": 60.0})
    d = pd.DataFrame(rows)

    out = additive_penalty(d)
    tight = out[out["turn_bucket"] == "<40 min"]["added_minutes"].tolist()
    # Flat and positive: the penalty does not scale with what arrived.
    assert all(abs(v - 10.0) < 0.5 for v in tight)

    long_turn = out[out["turn_bucket"] == "90-150"].sort_values("mean_inbound_delay")
    added = long_turn["added_minutes"].tolist()
    # Proportional: a bigger inbound delay gives back more, so this decreases.
    assert added == sorted(added, reverse=True)
