"""How much of an inbound aircraft's delay survives into the next departure.

The naive way to measure this is mean(departure delay) / mean(inbound delay)
over the chained pairs. That ratio is confounded: a leg departs a few minutes
late on average even when its aircraft arrived early, because boarding, gate
and crew produce delay of their own. Left uncorrected the confound inflates
every bucket, and it inflates the small-delay buckets most, to the point where
a 0-15 minute inbound delay appears to be *amplified* rather than absorbed.

Everything here therefore nets out that baseline before dividing.
"""
from __future__ import annotations

import pandas as pd

from src.config import BUFFER_BUCKETS, DELAY_BUCKETS


def baseline_departure_delay(d: pd.DataFrame) -> float:
    """Mean departure delay on legs whose inbound aircraft was not late.

    This is the delay a leg carries for reasons that have nothing to do with
    propagation, and it is the quantity every bucket below is corrected by.
    """
    clean = d[d["usable_turn"] & (d["prev_arr_delay"] <= 0)]
    return float(clean["DepDelay"].mean())


def _rows(d: pd.DataFrame, baseline: float, buckets, column: str) -> list[dict]:
    out = []
    for lo, hi, label in buckets:
        s = d[(d[column] > lo) & (d[column] <= hi)]
        if len(s) < 50:
            continue
        mean_in = float(s["prev_arr_delay"].mean())
        mean_dep = float(s["DepDelay"].mean())
        row = {
            "bucket": label,
            "n": int(len(s)),
            "mean_inbound_delay": round(mean_in, 2),
            "mean_departure_delay": round(mean_dep, 2),
        }
        if mean_in > 0:
            raw = mean_dep / mean_in
            adj = (mean_dep - baseline) / mean_in
            row["raw_passthrough"] = round(raw, 3)
            row["adjusted_passthrough"] = round(adj, 3)
            row["absorbed_pct"] = round((1 - adj) * 100, 1)
        else:
            row["raw_passthrough"] = None
            row["adjusted_passthrough"] = None
            row["absorbed_pct"] = None
        out.append(row)
    return out


def absorption_by_inbound_delay(d: pd.DataFrame, baseline: float | None = None) -> pd.DataFrame:
    """Absorption curve across inbound-delay buckets."""
    turns = d[d["usable_turn"]]
    if baseline is None:
        baseline = baseline_departure_delay(d)
    return pd.DataFrame(_rows(turns, baseline, DELAY_BUCKETS, "prev_arr_delay"))


def absorption_by_buffer(d: pd.DataFrame, min_inbound_delay: float = 15.0,
                         baseline: float | None = None) -> pd.DataFrame:
    """Absorption by scheduled turnaround time, holding the delay case fixed.

    This is the mechanism test. If schedule buffer is what absorbs delay, then
    among legs that all received a genuinely late aircraft, the ones with a
    longer scheduled turn should pass through less of it.
    """
    turns = d[d["usable_turn"] & (d["prev_arr_delay"] > min_inbound_delay)]
    if baseline is None:
        baseline = baseline_departure_delay(d)
    return pd.DataFrame(_rows(turns, baseline, BUFFER_BUCKETS, "sched_turn_min"))


def absorption_by_buffer_controlled(d: pd.DataFrame, delay_lo: float, delay_hi: float,
                                    baseline: float | None = None) -> pd.DataFrame:
    """Absorption by scheduled turn, with inbound delay held to a narrow band.

    `absorption_by_buffer` compares turn lengths across every late-inbound case
    at once, and its buckets do not receive the same size of delay: the longest
    turns sit on a mean inbound delay roughly twice that of the shortest. Some
    of the absorption it reports is therefore the delay-size effect measured in
    `absorption_by_inbound_delay`, not the buffer.

    Restricting to a narrow inbound band equalises that -- inside a band the
    mean inbound delay agrees to about a minute across every turn bucket -- so
    whatever gradient remains is attributable to scheduled turnaround.
    """
    turns = d[d["usable_turn"] & d["prev_arr_delay"].between(delay_lo, delay_hi)]
    if baseline is None:
        baseline = baseline_departure_delay(d)
    out = pd.DataFrame(_rows(turns, baseline, BUFFER_BUCKETS, "sched_turn_min"))
    out.insert(0, "inbound_band", f"{delay_lo:.0f}-{delay_hi:.0f}")
    return out


def additive_penalty(d: pd.DataFrame, baseline: float | None = None) -> pd.DataFrame:
    """Minutes of departure delay added on top of the inbound delay itself.

    The passthrough ratio in this module is the natural summary only if the turn
    scales its effect with the size of the delay it receives. It does not, and
    the two regimes differ:

    * A **tight** turn adds a roughly fixed penalty -- about 7 to 10 minutes --
      more or less regardless of how late the inbound was. Dividing that fixed
      quantity by a small inbound delay is what produces the apparent
      "amplification" of small delays; the same penalty against a 73 minute
      inbound reads as a ratio near 1.1. Nothing is being amplified.
    * A **long** turn absorbs roughly in proportion, because there is buffer
      available to give back, up to the size of that buffer.

    So the ratio is a fair summary for long turns and a misleading one for tight
    turns, and this function is the view that separates them.
    """
    turns = d[d["usable_turn"]]
    if baseline is None:
        baseline = baseline_departure_delay(d)
    out = []
    for tlo, thi, tlab in BUFFER_BUCKETS:
        band = turns[(turns["sched_turn_min"] > tlo) & (turns["sched_turn_min"] <= thi)]
        for dlo, dhi, dlab in DELAY_BUCKETS:
            if dhi <= 0:
                continue
            s = band[(band["prev_arr_delay"] > dlo) & (band["prev_arr_delay"] <= dhi)]
            if len(s) < 50:
                continue
            excess = float((s["DepDelay"] - s["prev_arr_delay"]).mean() - baseline)
            out.append({
                "turn_bucket": tlab,
                "inbound_bucket": dlab,
                "n": int(len(s)),
                "mean_inbound_delay": round(float(s["prev_arr_delay"].mean()), 2),
                "added_minutes": round(excess, 2),
            })
    return pd.DataFrame(out)
