"""Render the README figures from the committed reports/ JSON.

Reads only what `run_absorption.py` already wrote, so a figure can never show a
number that is not also in the evidence files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dcdcd6"
SERIES = ["#2a78d6", "#eb6834"]   # categorical slots 1 and 2, fixed order
TURN_ORDER = ["<40 min", "40-60", "60-90", "90-150", "150+"]
MUTED = "#9a9a92"     # for the bar that is not an available option


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)


def buffer_figure(out=FIGURES / "absorption_by_buffer.png"):
    rows = json.loads((REPORTS / "absorption_by_buffer_controlled.json").read_text())
    bands = sorted({r["inbound_band"] for r in rows})

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    width = 0.38

    for i, band in enumerate(bands):
        by_bucket = {r["bucket"]: r for r in rows if r["inbound_band"] == band}
        xs, ys = [], []
        for j, b in enumerate(TURN_ORDER):
            if b not in by_bucket:
                continue
            xs.append(j + (i - 0.5) * width)
            ys.append(by_bucket[b]["adjusted_passthrough"])
        bars = ax.bar(xs, ys, width=width - 0.02, color=SERIES[i], zorder=3,
                      label=f"inbound {band} min late")
        for rect, val in zip(bars, ys):
            ax.text(rect.get_x() + rect.get_width() / 2, val + 0.03, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK_SOFT, zorder=4)

    ax.axhline(1.0, color=INK, linewidth=1.4, linestyle="--", zorder=2)
    ax.text(len(TURN_ORDER) - 0.62, 1.03, "delay passes through unchanged",
            ha="right", va="bottom", fontsize=8.5, color=INK)

    ax.set_xticks(range(len(TURN_ORDER)))
    ax.set_xticklabels(TURN_ORDER)
    ax.set_xlabel("scheduled turnaround", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_ylabel("share of inbound delay passed on", fontsize=9.5, color=INK_SOFT)
    ax.set_title("A tight turn amplifies an inbound delay; a long one absorbs it",
                 fontsize=12, color=INK, pad=12, loc="left")
    ax.set_ylim(0, 1.42)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in leg.get_texts():
        text.set_color(INK_SOFT)
    _style(ax)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def delay_figure(out=FIGURES / "absorption_by_inbound_delay.png"):
    rows = [r for r in json.loads((REPORTS / "absorption_by_inbound_delay.json").read_text())
            if r["adjusted_passthrough"] is not None]

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    xs = [r["bucket"] for r in rows]
    ys = [r["adjusted_passthrough"] for r in rows]

    bars = ax.bar(xs, ys, width=0.6, color=SERIES[0], zorder=3)
    for rect, val, r in zip(bars, ys, rows):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=INK_SOFT, zorder=4)
        ax.text(rect.get_x() + rect.get_width() / 2, 0.04, f"n={r['n']:,}",
                ha="center", va="bottom", fontsize=7.5, color=SURFACE, zorder=4)

    ax.axhline(1.0, color=INK, linewidth=1.4, linestyle="--", zorder=2)
    ax.set_xlabel("how late the inbound aircraft arrived (minutes)",
                  fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_ylabel("share of inbound delay passed on", fontsize=9.5, color=INK_SOFT)
    ax.set_title("Small inbound delays are amplified; only larger ones get absorbed",
                 fontsize=12, color=INK, pad=12, loc="left")
    ax.set_ylim(0, 1.32)
    _style(ax)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def prediction_figure(out=FIGURES / "prediction_mae.png"):
    """The four comparisons, with the leaky bar marked as unavailable.

    It is drawn hatched and in muted ink rather than as a fourth series colour,
    because it is not a method anyone can choose -- it is the size of a mistake.
    """
    metrics = json.loads((REPORTS / "prediction_metrics.json").read_text())
    block = metrics.get("out_of_time") or metrics["within_month"]

    rows = [
        ("predict the training mean", block["predict_training_mean"]["mae"], False),
        ("carry the inbound delay forward", block["carry_inbound_delay_forward"]["mae"], False),
        ("model, as of the cutoff", block["model_as_of_cutoff"]["mae"], False),
        ("model with leakage\n(cannot be run)", block["model_with_leakage"]["mae"], True),
    ]

    fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(rows)))[::-1]

    for y, (label, mae, leaky) in zip(ys, rows):
        ax.barh(y, mae, height=0.58, zorder=3,
                color=MUTED if leaky else SERIES[0],
                hatch="//" if leaky else None,
                edgecolor=SURFACE if leaky else "none", linewidth=0)
        ax.text(mae + 0.35, y, f"{mae:.2f}", va="center", ha="left",
                fontsize=9.5, color=INK_SOFT, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("mean absolute error, minutes  (lower is better)",
                  fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title("What the leakage is worth, in minutes of apparent accuracy",
                 fontsize=12, color=INK, pad=12, loc="left")
    ax.set_xlim(0, max(r[1] for r in rows) * 1.18)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    figures = [buffer_figure(), delay_figure()]
    if (REPORTS / "prediction_metrics.json").exists():
        figures.append(prediction_figure())
    for path in figures:
        print(f"wrote {path}")
