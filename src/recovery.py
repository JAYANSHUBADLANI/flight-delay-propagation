"""Aircraft recovery over a disrupted rotation, as a mixed integer program.

When an aircraft arrives late into a hub, the legs it was due to fly are at
risk, and so are the legs of every aircraft it could be swapped with. The
choices are to hold a leg (let the delay through), swap it onto a different
aircraft, or cancel it. This module builds that decision as a MILP and solves it
with HiGHS through `scipy.optimize.milp`.

The formulation is connection-based. Each leg is a task; a binary variable turns
on when one leg is flown immediately after another by the same aircraft; a
continuous variable carries each leg's departure delay, pushed forward along
whichever connections the solver switches on. That is the standard shape of the
aircraft recovery problem, reduced to the part a public dataset can support.

**What this deliberately is not.** Real recovery is constrained by crew legality,
gate and slot availability, maintenance routing, and passenger reaccommodation.
None of that is in the BTS file. Neither is aircraft type, so "can this tail fly
this leg" is approximated by same-carrier-same-station rather than by fleet
compatibility. The objective is therefore in flight delay-minutes, not passenger
delay-minutes, and the result is an aircraft-rotation recovery, not an airline
operations recovery. Treated as anything more it would be wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

# Derived, not invented. Across the 372,750 real chained turns in January 2024
# the 1st percentile of scheduled turnaround is 28 minutes and the 5th is 33, so
# 25 excludes essentially nothing the airlines actually schedule. The obvious
# round number, 40, would have declared 37,422 turns that were genuinely flown
# to be impossible, and charged delay to aircraft that were never disrupted.
MIN_TURN_MINUTES = 25
MAX_DELAY_MINUTES = 300
CANCEL_PENALTY_MINUTES = 240
SWAP_PENALTY_MINUTES = 15
MAX_SWAP_CANDIDATES = 8
SOLVER_TIME_LIMIT = 60.0


@dataclass
class Scenario:
    """One carrier's departures from one station inside one time window."""
    legs: pd.DataFrame              # indexed 0..n-1
    aircraft: list[str]
    ready_time: dict[str, float]    # tail -> minute it is available at the station
    station: str
    carrier: str
    date: str
    disrupted_tail: str | None = None
    disruption_minutes: float = 0.0


@dataclass
class Solution:
    status: str
    total_delay_minutes: float
    cancelled: list[int] = field(default_factory=list)
    assignment: dict[int, str] = field(default_factory=dict)
    delays: dict[int, float] = field(default_factory=dict)
    swaps: int = 0


def build_scenario(d: pd.DataFrame, date: str, carrier: str, station: str,
                   disruption_minutes: float = 120.0,
                   max_candidates: int = MAX_SWAP_CANDIDATES) -> Scenario:
    """Carve one recoverable situation out of the rotation table.

    Two scoping mistakes are worth naming, because both produce a model that
    runs and answers the wrong question.

    The first is taking every departure from a hub inside a time window. At a
    big station almost every aircraft flies one leg in a few hours, so there is
    no rotation left to re-sequence -- only a large, highly symmetric assignment
    of interchangeable aircraft, which stalls branch and bound and would not
    answer the question anyway.

    The second is keeping only the legs that depart the chosen station. An
    aircraft's two departures from one hub are separated by the rest of its
    rotation, and dropping those middle legs means the time between them has to
    be *guessed*. Guessing it manufactures delay on aircraft that were never
    disrupted. So the scenario keeps each selected tail's **whole day**, and the
    station only decides which tails are in scope.
    """
    day = d[(d["FlightDate"] == date) & (d["Reporting_Airline"] == carrier)].copy()
    at_station = day[day["Origin"] == station]
    if at_station.empty:
        raise ValueError(f"no {carrier} departures from {station} on {date}")

    per_tail = at_station.groupby("Tail_Number").agg(
        legs=("Dest", "size"), worst_inbound=("prev_arr_delay", "max"),
        first_dep=("sched_dep_min", "min"),
    ).sort_values(["legs", "worst_inbound"], ascending=False)
    disrupted = str(per_tail.index[0])

    lo = float(per_tail.at[disrupted, "first_dep"])
    hi = float(at_station[at_station["Tail_Number"] == disrupted]["sched_dep_min"].max())
    others = (at_station[(at_station["Tail_Number"] != disrupted)
                         & (at_station["sched_dep_min"].between(lo - 120, hi + 120))]
              .groupby("Tail_Number").size().sort_values(ascending=False))
    candidates = [disrupted] + [str(t) for t in others.index[:max_candidates]]

    # Keep only each tail's longest *connected* run of legs.
    #
    # About 1.2% of recorded successions do not connect, and a tail with a break
    # in its day needs two separate strings to describe it. The formulation lets
    # each aircraft open one string, so a broken rotation cannot be reproduced
    # at all -- not even the do-nothing plan -- and the solver returns something
    # worse than doing nothing while looking optimal. Restricting the scenario
    # to connected segments keeps the model and the data consistent. The cost is
    # that a scenario covers part of a tail's day rather than all of it, which
    # is stated in the README rather than hidden.
    selected = []
    for tail in candidates:
        rows = (day[day["Tail_Number"] == tail]
                .sort_values("sched_dep_min").reset_index(drop=True))
        runs, current = [], [0]
        for k in range(1, len(rows)):
            if rows.at[k, "Origin"] == rows.at[k - 1, "Dest"]:
                current.append(k)
            else:
                runs.append(current); current = [k]
        runs.append(current)
        best = max(runs, key=len)
        selected.append(rows.loc[best])

    legs = (pd.concat(selected, ignore_index=True)
            [["Tail_Number", "Origin", "Dest", "sched_dep_min", "sched_arr_min",
              "prev_actual_arr_min", "prev_arr_delay"]]
            .sort_values(["Tail_Number", "sched_dep_min"]).reset_index(drop=True))

    ready: dict[str, float] = {}
    for tail in candidates:
        rows = legs[legs["Tail_Number"] == tail]
        # An undisrupted aircraft is, by construction, ready for its own first
        # departure of the day. Reaching into `prev_actual_arr_min` here is
        # tempting and wrong: its minimum is the arrival of a *later* inbound,
        # which charges an on-time tail for a landing that has not happened yet.
        ready[tail] = float(rows["sched_dep_min"].min()) - MIN_TURN_MINUTES

    first_dep = float(legs[legs["Tail_Number"] == disrupted]["sched_dep_min"].min())
    ready[disrupted] = first_dep - MIN_TURN_MINUTES + disruption_minutes

    return Scenario(legs=legs, aircraft=candidates, ready_time=ready,
                    station=station, carrier=carrier, date=date,
                    disrupted_tail=disrupted, disruption_minutes=disruption_minutes)


def _connections(sc: Scenario) -> list[tuple[int, int]]:
    """Ordered leg pairs an aircraft could physically fly back to back.

    Now that whole rotations are in scope, this is the real condition: the first
    leg must land where the second one starts, and the schedule must leave at
    least a minimum turn between them once the allowed delay is taken into
    account. Nothing is estimated.
    """
    legs = sc.legs
    out = []
    for i in range(len(legs)):
        arr_i = float(legs.at[i, "sched_arr_min"])
        dest_i = legs.at[i, "Dest"]
        for j in range(len(legs)):
            if i == j:
                continue
            if legs.at[j, "Origin"] != dest_i:
                continue
            # Strictly forward in time. This is physics -- an aircraft cannot
            # fly a leg that departs earlier than the one it is already on --
            # and it is also what keeps the connection graph acyclic. Without
            # it the formulation admits subtours: a closed loop i -> j -> i
            # satisfies "every leg is covered" and "each leg has one successor"
            # while being flown by no aircraft at all. Those legs then vanish
            # from the plan and the reported delay is too low.
            if float(legs.at[j, "sched_dep_min"]) <= float(legs.at[i, "sched_dep_min"]):
                continue
            if float(legs.at[j, "sched_dep_min"]) + MAX_DELAY_MINUTES >= arr_i + MIN_TURN_MINUTES:
                out.append((i, j))
    return out


def solve(sc: Scenario, cancel_penalty: float = CANCEL_PENALTY_MINUTES,
          swap_penalty: float = SWAP_PENALTY_MINUTES,
          min_turn: int = MIN_TURN_MINUTES) -> Solution:
    """Minimise total departure delay, plus penalties for cancelling and swapping.

    The swap penalty is not cosmetic. Without it the aircraft in a scenario are
    interchangeable wherever their timings allow, so the solver permutes them at
    zero cost and returns a plan that "swaps" almost every leg for no gain. That
    is a symmetry artifact, not a recommendation, and a real operation pays for
    every tail change in ground crew, catering and passenger communication.
    """
    legs = sc.legs
    n = len(legs)
    conns = _connections(sc)
    # An aircraft can only begin at the airport it is standing at, and not
    # before it is ready. Enumerating all aircraft x all legs instead adds
    # hundreds of physically impossible variables and the symmetry between them
    # is what makes the search stall.
    # An aircraft may open a string at any airport its own schedule puts it at,
    # not only where its day begins. Restricting starts to the first origin
    # looks tighter but is wrong: where the recorded chain breaks -- about 1.2%
    # of successions -- the legs beyond the break become reachable from no
    # aircraft at all, and the solver is forced to cancel them. Those show up as
    # confident "cancel this flight" recommendations that are nothing of the
    # sort. The file does not show how the aircraft got there; it does show that
    # it was there.
    own_origins = {a: set(legs.loc[legs["Tail_Number"] == a, "Origin"])
                   for a in sc.aircraft}
    first_origin = {a: legs.loc[legs["Tail_Number"] == a, "Origin"].iloc[0]
                    for a in sc.aircraft}
    starts = [
        (a, j) for a in sc.aircraft for j in range(n)
        if legs.at[j, "Origin"] in own_origins[a]
        and float(legs.at[j, "sched_dep_min"]) + MAX_DELAY_MINUTES
        >= sc.ready_time[a] + min_turn
    ]

    n_y, n_s, n_c, n_d = len(conns), len(starts), n, n
    total = n_y + n_s + n_c + n_d
    oy, os_, oc, od = 0, n_y, n_y + n_s, n_y + n_s + n_c

    cost = np.zeros(total)
    cost[oc:oc + n_c] = cancel_penalty
    cost[od:od + n_d] = 1.0
    # A start counts as a swap when it puts a leg on a tail that was not its own.
    for k, (a, j) in enumerate(starts):
        if a != legs.at[j, "Tail_Number"]:
            cost[os_ + k] += swap_penalty
    for k, (i, j) in enumerate(conns):
        if legs.at[i, "Tail_Number"] != legs.at[j, "Tail_Number"]:
            cost[oy + k] += swap_penalty

    integrality = np.ones(total)
    integrality[od:od + n_d] = 0            # the delay variables stay continuous
    lower = np.zeros(total)
    upper = np.ones(total)
    upper[od:od + n_d] = MAX_DELAY_MINUTES

    A, lo, hi = [], [], []

    # Every leg is flown exactly once, or cancelled.
    for j in range(n):
        row = np.zeros(total)
        for k, (i_, j_) in enumerate(conns):
            if j_ == j:
                row[oy + k] = 1
        for k, (a_, j_) in enumerate(starts):
            if j_ == j:
                row[os_ + k] = 1
        row[oc + j] = 1
        A.append(row); lo.append(1); hi.append(1)

    # A leg can hand on to at most one successor, and only if it was flown.
    for i in range(n):
        row = np.zeros(total)
        for k, (i_, j_) in enumerate(conns):
            if i_ == i:
                row[oy + k] = 1
        row[oc + i] = 1
        A.append(row); lo.append(-np.inf); hi.append(1)

    # Each aircraft opens at most one string.
    for a in sc.aircraft:
        row = np.zeros(total)
        for k, (a_, j_) in enumerate(starts):
            if a_ == a:
                row[os_ + k] = 1
        A.append(row); lo.append(-np.inf); hi.append(1)

    # Starting delay: if aircraft a opens with leg j, that leg cannot leave
    # before a is ready.  d_j >= ready(a) - sched_dep(j) - M(1 - s[a,j])
    for k, (a, j) in enumerate(starts):
        # Only the aircraft's own first station can hold it on the ground. Deeper
        # in its rotation it reached the airport by a route the file omits, so
        # charging it the disruption there would invent delay.
        if legs.at[j, "Origin"] != first_origin[a]:
            continue
        need = sc.ready_time[a] + min_turn - float(legs.at[j, "sched_dep_min"])
        if need <= 0:
            continue
        # M must be exactly `need`: any smaller and the constraint still forces
        # delay onto leg j when this aircraft does *not* start it, which is how
        # a feasible recovery gets reported as infeasible. Any larger only
        # weakens the relaxation.
        row = np.zeros(total)
        row[od + j] = 1
        row[os_ + k] = -need
        A.append(row); lo.append(0.0); hi.append(np.inf)

    # Propagation: if j follows i, j inherits i's delay less whatever slack the
    # schedule left between them.  d_j >= d_i - slack - M(1 - y[i,j])
    for k, (i, j) in enumerate(conns):
        gap = float(legs.at[j, "sched_dep_min"]) - float(legs.at[i, "sched_arr_min"])
        same_tail = legs.at[i, "Tail_Number"] == legs.at[j, "Tail_Number"]
        # An existing scheduled succession keeps the turn it was given; a new
        # one created by a swap has to clear the minimum.
        required = min(min_turn, gap) if same_tail else min_turn
        slack = gap - required
        # With y = 0 this must be vacuous. d_j - d_i is at worst -MAX_DELAY, and
        # the right hand side is -slack - m, so m >= MAX_DELAY - slack; a
        # negative slack (a tight connection) makes M *larger*, not smaller.
        m = MAX_DELAY_MINUTES - min(0.0, slack)
        row = np.zeros(total)
        row[od + j] = 1
        row[od + i] = -1
        row[oy + k] = -m
        A.append(row); lo.append(-slack - m); hi.append(np.inf)

    res = milp(c=cost, integrality=integrality,
               bounds=Bounds(lower, upper),
               constraints=LinearConstraint(np.array(A), lo, hi),
               options={"time_limit": SOLVER_TIME_LIMIT})

    if not res.success:
        return Solution(status=res.message, total_delay_minutes=float("nan"))

    x = res.x
    cancelled = [j for j in range(n) if x[oc + j] > 0.5]
    delays = {j: round(float(x[od + j]), 1) for j in range(n) if j not in cancelled}

    assignment: dict[int, str] = {}
    for k, (a, j) in enumerate(starts):
        if x[os_ + k] > 0.5:
            assignment[j] = a
    changed = True
    while changed:
        changed = False
        for k, (i, j) in enumerate(conns):
            if x[oy + k] > 0.5 and i in assignment and j not in assignment:
                assignment[j] = assignment[i]
                changed = True

    swaps = sum(1 for j, a in assignment.items()
                if a != legs.at[j, "Tail_Number"])
    return Solution(status="optimal",
                    total_delay_minutes=round(float(sum(delays.values())), 1),
                    cancelled=cancelled, assignment=assignment, delays=delays,
                    swaps=swaps)


def do_nothing(sc: Scenario, min_turn: int = MIN_TURN_MINUTES) -> Solution:
    """The counterfactual: every leg stays on its original aircraft.

    Delay is pushed along each tail's real scheduled rotation, so an aircraft
    that was never disrupted comes out at zero rather than absorbing a guess.
    """
    legs = sc.legs
    delays: dict[int, float] = {}
    for tail in sc.aircraft:
        rows = legs[legs["Tail_Number"] == tail].sort_values("sched_dep_min")
        available = sc.ready_time[tail] + min_turn
        prev_dest = None
        for j, row in rows.iterrows():
            # Only propagate across a turn the aircraft actually made. About 1.2%
            # of recorded successions do not connect -- tail reassignments, ferry
            # and maintenance movements the file does not distinguish. Pushing
            # delay across one of those invents it: the aircraft reached the new
            # origin by some route the data does not show, so it starts clean.
            if prev_dest is not None and row["Origin"] != prev_dest:
                available = float(row["sched_dep_min"]) - min_turn
            delay = min(max(0.0, available - float(row["sched_dep_min"])),
                        MAX_DELAY_MINUTES)
            delays[j] = round(delay, 1)
            # The turn this aircraft was *scheduled* to make is feasible by
            # definition -- the airline planned it and the aircraft flew it. A
            # minimum turn is a constraint on connections a recovery plan would
            # newly create, not a licence to overrule the published schedule.
            # About 1% of scheduled turns are under 25 minutes and a few are
            # recorded as negative; imposing a floor on those invents delay on
            # aircraft that were never disrupted.
            prev_dest = row["Dest"]
            nxt = rows.index[rows.index.get_loc(j) + 1] if rows.index.get_loc(j) + 1 < len(rows) else None
            required = min_turn
            if nxt is not None and legs.at[nxt, "Origin"] == prev_dest:
                required = min(min_turn,
                               float(legs.at[nxt, "sched_dep_min"])
                               - float(row["sched_arr_min"]))
            available = float(row["sched_arr_min"]) + delay + required
    return Solution(status="baseline",
                    total_delay_minutes=round(float(sum(delays.values())), 1),
                    assignment={j: legs.at[j, "Tail_Number"] for j in range(len(legs))},
                    delays=delays)
