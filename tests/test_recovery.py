import pandas as pd

from src.recovery import (MIN_TURN_MINUTES, Scenario, _connections, do_nothing,
                          solve)


def _legs(rows):
    return pd.DataFrame(rows, columns=["Tail_Number", "Origin", "Dest",
                                       "sched_dep_min", "sched_arr_min",
                                       "prev_actual_arr_min", "prev_arr_delay"])


def _scenario(disruption=120.0):
    """Two aircraft, each with a two-leg out-and-back from ORD.

    A flies ORD-DFW-ORD, B flies ORD-DEN-ORD on a similar cycle, so B's legs are
    genuine swap candidates for A's.
    """
    legs = _legs([
        ("A", "ORD", "DFW", 480, 600, None, None),
        ("A", "DFW", "ORD", 660, 780, 600, 0),
        ("B", "ORD", "DEN", 500, 620, None, None),
        ("B", "DEN", "ORD", 680, 800, 620, 0),
    ])
    ready = {"A": 480 - MIN_TURN_MINUTES + disruption, "B": 500 - MIN_TURN_MINUTES}
    return Scenario(legs=legs, aircraft=["A", "B"], ready_time=ready,
                    station="ORD", carrier="XX", date="2024-01-02",
                    disrupted_tail="A", disruption_minutes=disruption)


def test_an_undisrupted_aircraft_carries_no_delay():
    base = do_nothing(_scenario())
    b_legs = [2, 3]
    assert all(base.delays[j] == 0.0 for j in b_legs)


def test_the_disrupted_aircraft_propagates_and_decays():
    base = do_nothing(_scenario(disruption=120.0))
    assert base.delays[0] == 120.0
    # The scheduled turn gives some of it back before the second leg.
    assert 0 < base.delays[1] < base.delays[0]


def test_connections_are_strictly_forward_in_time():
    sc = _scenario()
    for i, j in _connections(sc):
        assert sc.legs.at[j, "sched_dep_min"] > sc.legs.at[i, "sched_dep_min"]
        assert sc.legs.at[j, "Origin"] == sc.legs.at[i, "Dest"]


def test_no_leg_is_orphaned_by_a_subtour():
    # Without a strictly-forward connection rule the formulation admits a closed
    # loop i -> j -> i that satisfies every constraint while being flown by no
    # aircraft, and those legs silently disappear from the plan.
    sc = _scenario()
    sol = solve(sc)
    assert sol.status == "optimal"
    covered = set(sol.assignment) | set(sol.cancelled)
    assert covered == set(range(len(sc.legs)))


def test_recovery_is_never_worse_than_doing_nothing():
    sc = _scenario()
    assert solve(sc).total_delay_minutes <= do_nothing(sc).total_delay_minutes


def test_a_scheduled_turn_tighter_than_the_minimum_is_still_feasible():
    # Airlines schedule turns below any round-number minimum -- 10% of real
    # January turns are under 40 minutes. A leg's own scheduled succession must
    # not be charged delay for being tighter than the model's floor.
    tight = _legs([
        ("A", "ORD", "DFW", 480, 600, None, None),
        ("A", "DFW", "ORD", 610, 730, 600, 0),   # a 10 minute scheduled turn
    ])
    sc = Scenario(legs=tight, aircraft=["A"],
                  ready_time={"A": 480 - MIN_TURN_MINUTES},
                  station="ORD", carrier="XX", date="2024-01-02")
    assert do_nothing(sc).total_delay_minutes == 0.0


def test_delay_does_not_cross_a_broken_chain():
    # About 1.2% of recorded successions do not connect. The aircraft reached the
    # new origin by a route the file does not show, so it starts clean there.
    broken = _legs([
        ("A", "ORD", "DFW", 480, 600, None, None),
        ("A", "IAH", "ORD", 620, 740, 600, 0),   # not where the first leg landed
    ])
    sc = Scenario(legs=broken, aircraft=["A"],
                  ready_time={"A": 480 - MIN_TURN_MINUTES + 200},
                  station="ORD", carrier="XX", date="2024-01-02")
    base = do_nothing(sc)
    assert base.delays[0] == 200.0
    assert base.delays[1] == 0.0


def test_swaps_are_not_free():
    # With no swap cost the solver permutes interchangeable aircraft at will and
    # reports the permutation as a recommendation.
    sc = _scenario(disruption=0.0)
    free = solve(sc, swap_penalty=0.0)
    priced = solve(sc, swap_penalty=60.0)
    assert priced.swaps <= free.swaps


def test_cancellation_is_chosen_only_when_delay_is_worse():
    sc = _scenario(disruption=280.0)
    cheap = solve(sc, cancel_penalty=10.0)
    dear = solve(sc, cancel_penalty=100_000.0)
    assert len(cheap.cancelled) >= len(dear.cancelled)
