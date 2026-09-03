import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, TARGET
from src.model import (LEAKY_FEATURES, baseline_carry_forward, baseline_mean,
                       evaluate, fit_model, scores)


def _frame(n=400, seed=0):
    """A synthetic feature block where the target genuinely depends on the
    schedule-only features, so a model has something honest to learn."""
    rng = np.random.default_rng(seed)
    turn = rng.choice([25.0, 50.0, 80.0, 130.0], size=n)
    inbound = rng.gamma(2.0, 15.0, size=n)
    landed = rng.random(n) < 0.3
    df = pd.DataFrame({
        "sched_dep_min": rng.integers(300, 1300, n).astype(float),
        "sched_turn_min": turn,
        "leg_index": rng.integers(1, 5, n).astype(float),
        "legs_in_day": rng.integers(2, 7, n).astype(float),
        "Distance": rng.integers(200, 2500, n).astype(float),
        "inbound_sched_arr_slack": rng.normal(30, 40, n),
        "known_inbound_arr_delay": np.where(landed, inbound, np.nan),
        "known_inbound_dep_delay": rng.normal(10, 20, n),
        "origin_departures_that_hour": rng.integers(1, 40, n).astype(float),
        "inbound_landed_by_cutoff": landed,
        "inbound_departed_by_cutoff": rng.random(n) < 0.8,
        "Reporting_Airline": pd.Categorical(rng.choice(["AA", "DL", "UA"], n)),
        "day_of_week": pd.Categorical(rng.integers(0, 7, n)),
        "prev_arr_delay": inbound,
    })
    # Tight turns pass more through; this is the signal the model should find.
    df[TARGET] = inbound * np.where(turn < 60, 1.1, 0.4) + rng.normal(0, 5, n)
    return df


def test_scores_match_hand_computation():
    got = scores([0.0, 10.0], [2.0, 8.0])
    assert got["mae"] == 2.0
    assert got["rmse"] == 2.0
    assert got["n"] == 2


def test_baseline_mean_predicts_the_training_mean_not_the_test_mean():
    train, test = _frame(seed=1), _frame(seed=2)
    expected = float(np.mean(np.abs(test[TARGET] - train[TARGET].mean())))
    # `scores` rounds to three decimals by design, so compare at that resolution.
    assert baseline_mean(train, test)["mae"] == pytest.approx(expected, abs=1e-3)


def test_carry_forward_falls_back_to_the_mean_when_nothing_landed():
    train = _frame(seed=1)
    test = _frame(seed=2)
    test["known_inbound_arr_delay"] = np.nan     # nothing was knowable
    assert baseline_carry_forward(train, test)["mae"] == pytest.approx(
        baseline_mean(train, test)["mae"], abs=1e-3)


def test_carry_forward_never_predicts_a_negative_delay():
    train, test = _frame(seed=1), _frame(seed=2)
    test["known_inbound_arr_delay"] = -30.0      # inbound arrived early
    # An early inbound cannot imply a negative departure delay prediction.
    assert baseline_carry_forward(train, test)["mae"] == pytest.approx(
        float(np.mean(np.abs(test[TARGET] - 0.0))), abs=1e-3)


def test_the_honest_feature_set_excludes_the_unconditional_inbound_delay():
    # The structural guarantee behind the whole cutoff argument.
    assert "prev_arr_delay" not in FEATURES
    assert "prev_arr_delay" in LEAKY_FEATURES
    assert set(FEATURES).issubset(set(LEAKY_FEATURES))


def test_model_learns_something_beyond_the_mean():
    train, test = _frame(seed=1), _frame(seed=2)
    model = fit_model(train)
    from src.model import _prepare
    pred = model.predict(_prepare(test[FEATURES]))
    assert scores(test[TARGET], pred)["mae"] < baseline_mean(train, test)["mae"]


def test_evaluate_reports_all_four_comparisons():
    train, test = _frame(seed=1), _frame(seed=2)
    out = evaluate(train, test)
    assert set(out) == {"predict_training_mean", "carry_inbound_delay_forward",
                        "model_as_of_cutoff", "model_with_leakage"}
    for block in out.values():
        assert block["n"] == len(test)
        assert block["mae"] > 0
