"""
Load benchmark results and generate publication-quality figures.

Data source, in order of precedence:
  1. --run-id  : single run loaded from the configured store (SQLite or CSV).
  2. output.store (auto): all runs for the benchmark name, concatenated.
  3. results_csv in [[benchmark]]: legacy CSV path fallback.

This module is now invoked via the unified CLI:
  python cli.py analyze
  python cli.py analyze --run-id <run_id>
  python cli.py analyze --format json
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data.config import load_config


# ALGORITHM VISUAL IDENTITY

_PALETTE: dict[str, str] = {
    "SQP (no field)":                "#1f77b4",   # blue
    "SQP (flex bounds, no field)":   "#6baed6",   # light blue
    "SQP (QRiskField)":              "#ff7f0e",   # orange  (Geo/Ours)
    "SQP (flex bounds, QRiskField)": "#9467bd",   # purple  (Geo + Relax Const./Ours)
    "SQP (field)":                   "#e6550d",   # darker orange (WMR Geo)
    "iLQR (no field)":               "#2ca02c",   # green
    "iLQR (field)":                  "#74c476",   # light green (Geo)
    "DDP (no field)":                "#d62728",   # red
    "DDP (field)":                   "#fc8d59",   # salmon (Geo)
}

_FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def has_geo(algo_name: str) -> bool:
    """True when the algorithm uses a geometric (risk-field) constraint."""
    low = algo_name.lower()
    return ("field" in low or "qriskfield" in low) and "no field" not in low


def algo_style(name: str, idx: int = 0) -> tuple[str, str]:
    color  = _PALETTE.get(name, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
    marker = "^" if has_geo(name) else "o"
    return color, marker


def resolve_label(name: str, aliases: dict) -> str:
    return aliases.get(name, name)


# MATPLOTLIB STYLE

def apply_style(style: str = "paper"):
    if style == "paper":
        plt.rcParams.update({
            "font.family":      "serif",
            "font.size":        11,
            "axes.titlesize":   12,
            "axes.labelsize":   11,
            "legend.fontsize":  9,
            "xtick.labelsize":  10,
            "ytick.labelsize":  10,
            "axes.grid":        True,
            "grid.linestyle":   "--",
            "grid.alpha":       0.45,
            "lines.linewidth":  2.0,
            "figure.dpi":       100,
        })


# FIGURE SAVE / SHOW

def save_fig(fig: plt.Figure, stem: str, fig_cfg: dict, analyze_cfg: dict):
    if fig_cfg.get("save", False):
        out_dir = Path(fig_cfg.get("output_dir", "figures"))
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt  = fig_cfg.get("format", "pdf")
        dpi  = fig_cfg.get("dpi", 150)
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(str(path), format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {path}")

    if analyze_cfg.get("show", False) or not fig_cfg.get("save", False):
        plt.show()
    else:
        plt.close(fig)


# DATA HELPERS

def ordered_algos(df: pd.DataFrame, algo_list: list) -> list:
    """Return algorithms present in df, preserving the TOML-specified order."""
    present = set(df["Algorithm"].unique())
    if not algo_list:
        return list(present)
    ordered = [a for a in algo_list if a in present]
    # Append any in df but not in algo_list
    ordered += [a for a in present if a not in ordered]
    return ordered


def filter_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "Success" not in df.columns:
        return df
    return df[df["Success"] == True]


def filter_common_successes(df: pd.DataFrame, algos: list) -> pd.DataFrame:
    """
    Keep (City, Segment, Nominal Speed) where every algorithm succeeded.
    """
    if "Success" not in df.columns:
        return df
    key = ["City", "Segment", "Nominal Speed"]
    key = [c for c in key if c in df.columns]
    mask = (
        df[df["Algorithm"].isin(algos)]
        .groupby(key)["Success"]
        .all()
    )
    valid = mask[mask].reset_index()[key]
    return df.merge(valid, on=key, how="inner")


def agg_by_pace(df: pd.DataFrame, metric: str, algos: list) -> dict:
    """
    Aggregate metric by (Algorithm, Nominal Speed). Successful runs only.
    """
    df_ok = filter_ok(df)
    if "Nominal Speed" not in df_ok.columns or metric not in df_ok.columns:
        return {}
    result = {}
    for algo in algos:
        sub = df_ok[df_ok["Algorithm"] == algo]
        if sub.empty:
            continue
        grp    = sub.groupby("Nominal Speed")[metric]
        speeds = sorted(grp.groups.keys())
        means  = np.array([grp.get_group(s).mean() for s in speeds])
        stds   = np.array([grp.get_group(s).std(ddof=1) for s in speeds])
        stds   = np.nan_to_num(stds, nan=0.0)
        result[algo] = (np.array(speeds), means, stds)
    return result


def agg_by_pace_penalized(df: pd.DataFrame, metric: str, algos: list) -> dict:
    """
    Includes ALL attempts: failed/crashed runs are assigned metric = 0.
    """
    if "Nominal Speed" not in df.columns or metric not in df.columns:
        return {}
    result = {}
    for algo in algos:
        sub = df[df["Algorithm"] == algo].copy()
        if sub.empty:
            continue
        if "Success" in sub.columns:
            sub.loc[sub["Success"] != True, metric] = 0.0
        grp    = sub.groupby("Nominal Speed")[metric]
        speeds = sorted(grp.groups.keys())
        means  = np.array([grp.get_group(s).mean() for s in speeds])
        stds   = np.array([grp.get_group(s).std(ddof=1) for s in speeds])
        stds   = np.nan_to_num(stds, nan=0.0)
        result[algo] = (np.array(speeds), means, stds)
    return result


def agg_per_algo(df: pd.DataFrame, metric: str, algos: list,
                 ok_only: bool = True) -> dict:
    """Returns {algo: (mean, std, values_array)} for a given metric."""
    sub = filter_ok(df) if ok_only else df
    result = {}
    for algo in algos:
        vals = sub[sub["Algorithm"] == algo][metric].dropna().values if metric in sub.columns else np.array([])
        result[algo] = (vals.mean() if len(vals) else np.nan,
                        vals.std(ddof=1) if len(vals) > 1 else 0.0,
                        vals)
    return result


# INDIVIDUAL PLOT FUNCTIONS

def fig_size(fig_cfg: dict, scale: float = 1.0) -> tuple:
    w = fig_cfg.get("width", 7.0) * scale
    h = fig_cfg.get("height", 4.5) * scale
    return w, h


def plot_pace_vs_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    title: str = "",
    hline: float = None,
    hline_label: str = None,
    show_band: bool = True,
    ax: plt.Axes = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    agg = agg_by_pace(df, metric, algos)
    for i, (algo, (speeds, means, stds)) in enumerate(agg.items()):
        color, marker = algo_style(algo, i)
        label = resolve_label(algo, aliases)
        ax.plot(speeds, means, label=label, color=color, marker=marker,
                markersize=5, linewidth=2.0)
        if show_band:
            ax.fill_between(speeds, means - stds, means + stds, color=color, alpha=0.12)

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0,
                   label=hline_label or f"y={hline}")

    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    speeds_all = sorted({s for (sp, _, _) in agg.values() for s in sp}) if agg else []
    if speeds_all:
        ax.set_xticks(speeds_all)

    if standalone:
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_pace_vs_clearance(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    central: str = "median",
    log_scale: bool = True,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Min Clearance vs pace on paired routes.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    if central == "median":
        agg_fn = "median"
    elif central == "min":
        agg_fn = "min"
    elif central in ("p5", "p10"):
        agg_fn = central
    else:
        agg_fn = "mean"

    df_paired = filter_common_successes(df, algos)
    if "Nominal Speed" not in df_paired.columns or "Min Clearance" not in df_paired.columns:
        return fig

    for i, algo in enumerate(algos):
        sub = df_paired[df_paired["Algorithm"] == algo]
        if sub.empty:
            continue
        grp    = sub.groupby("Nominal Speed")["Min Clearance"]
        speeds = sorted(grp.groups.keys())
        if agg_fn == "p5":
            vals = np.array([grp.get_group(s).quantile(0.05) for s in speeds])
        elif agg_fn == "p10":
            vals = np.array([grp.get_group(s).quantile(0.10) for s in speeds])
        else:
            vals = np.array([getattr(grp.get_group(s), agg_fn)() for s in speeds])
        vals   = np.clip(vals, 1e-4, None)

        color, marker = algo_style(algo, i)
        label = resolve_label(algo, aliases)
        lw = 2.5 if has_geo(algo) else 1.6
        ax.plot(speeds, vals, label=label, color=color, marker=marker,
                markersize=5, linewidth=lw)

    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    _label_map = {"p5": "5th-pct", "p10": "10th-pct", "min": "Min",
                  "median": "Median", "mean": "Mean"}
    agg_label = _label_map.get(agg_fn, agg_fn.capitalize())
    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel(f"{agg_label} Min Clearance (m)")
    ax.set_title(f"Pace vs. Min Clearance (paired routes, {agg_label})")

    speeds_all = sorted(df_paired["Nominal Speed"].unique()) if not df_paired.empty else []
    if speeds_all:
        ax.set_xticks(speeds_all)

    if standalone:
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_pace_vs_rate(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    log_scale: bool = True,
    ymin_zero: bool = False,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Pace vs. Success Rate and Clear Rate.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    if "Nominal Speed" not in df.columns:
        return fig

    speeds = sorted(df["Nominal Speed"].unique())

    for i, algo in enumerate(algos):
        color, marker = algo_style(algo, i)
        label = resolve_label(algo, aliases)
        sub   = df[df["Algorithm"] == algo]

        sr, cr = [], []
        for s in speeds:
            grp = sub[sub["Nominal Speed"] == s]
            sr.append(grp["Success"].mean() * 100 if "Success" in grp.columns and len(grp) else np.nan)
            cr.append(grp["Is Clear"].mean() * 100 if "Is Clear" in grp.columns and len(grp) else np.nan)

        lw = 2.5 if has_geo(algo) else 1.6
        ax.plot(speeds, sr, label=label, color=color, marker=marker,
                markersize=5, linewidth=lw, linestyle="-")
        ax.plot(speeds, cr, color=color, marker=marker,
                markersize=4, linewidth=lw * 0.7, linestyle="--", alpha=0.7)

    ax.plot([], [], color="gray", linestyle="-",  linewidth=1.5, label="Success rate")
    ax.plot([], [], color="gray", linestyle="--", linewidth=1.2, label="Clear rate", alpha=0.7)

    if log_scale:
        # Clip to small positive so log scale doesn't break on 0%
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    if ymin_zero and not log_scale:
        ax.set_ylim(bottom=0)

    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(speeds)
    ax.set_title("Pace vs. Success & Clear Rate")

    if standalone:
        ax.legend(loc="lower left", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_success_rate(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    title: str = "Success Rate by Algorithm",
    ax: plt.Axes = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(fig_cfg.get("width", 7.0),
                                        max(3.0, len(algos) * 0.55)))
    else:
        fig = ax.figure

    rates, colors, labels = [], [], []
    for i, algo in enumerate(algos):
        sub = df[df["Algorithm"] == algo]
        rates.append(sub["Success"].mean() * 100 if len(sub) else 0.0)
        colors.append(algo_style(algo, i)[0])
        labels.append(resolve_label(algo, aliases))

    y    = np.arange(len(labels))
    bars = ax.barh(y, rates, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Success Rate (%)")
    ax.set_xlim(0, 108)
    ax.axvline(100, color="black", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    if title:
        ax.set_title(title)

    if standalone:
        fig.tight_layout()
    return fig


def plot_algo_bar(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    title: str = "",
    hline: float = None,
    ok_only: bool = True,
    ax: plt.Axes = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    agg = agg_per_algo(df, metric, algos, ok_only=ok_only)
    xs, means, errs, colors, labels = [], [], [], [], []
    for i, algo in enumerate(algos):
        mu, sd, _ = agg[algo]
        if np.isnan(mu):
            continue
        xs.append(i)
        means.append(mu)
        errs.append(sd)
        colors.append(algo_style(algo, i)[0])
        labels.append(resolve_label(algo, aliases))

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, color=colors, capsize=4,
           alpha=0.85, edgecolor="black", linewidth=0.5)
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if standalone:
        fig.tight_layout()
    return fig


def plot_violin(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    title: str = "",
    hline: float = None,
    ok_only: bool = True,
    ax: plt.Axes = None,
) -> plt.Figure:
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    agg = agg_per_algo(df, metric, algos, ok_only=ok_only)
    data, colors, labels = [], [], []
    for i, algo in enumerate(algos):
        _, _, vals = agg[algo]
        if len(vals) > 1:
            data.append(vals)
            colors.append(algo_style(algo, i)[0])
            labels.append(resolve_label(algo, aliases))

    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    if hline is not None:
        ax.axhline(hline, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    if standalone:
        fig.tight_layout()
    return fig


def plot_clearance_box_by_pace(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Grouped box plot per (algorithm, pace).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    df_ok  = filter_ok(df)
    speeds = sorted(df_ok["Nominal Speed"].unique()) if "Nominal Speed" in df_ok.columns else []
    if not speeds or "Min Clearance" not in df_ok.columns:
        return fig

    n      = len(algos)
    width  = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, algo in enumerate(algos):
        color, marker = algo_style(algo, i)
        label  = resolve_label(algo, aliases)
        data   = [
            df_ok[(df_ok["Algorithm"] == algo) & (df_ok["Nominal Speed"] == s)
                  ]["Min Clearance"].dropna().values
            for s in speeds
        ]
        positions = [j + 1 + offsets[i] for j, d in enumerate(data) if len(d) > 0]
        data      = [d for d in data if len(d) > 0]
        if not data:
            continue

        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.82,
            patch_artist=True, showfliers=False, whis=[5, 95],
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=0.8),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.70)

        ax.plot([], [], color=color, marker=marker, linestyle="-",
                linewidth=2, label=label)

    ax.set_xticks(range(1, len(speeds) + 1))
    ax.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Clearance Distribution by Pace")

    if standalone:
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_clearance_violin_by_pace(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Grouped violin plot per (algorithm, pace).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    df_paired = filter_common_successes(df, algos)
    speeds = sorted(df_paired["Nominal Speed"].unique()) if "Nominal Speed" in df_paired.columns else []
    if not speeds or "Min Clearance" not in df_paired.columns:
        return fig

    n       = len(algos)
    width   = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, algo in enumerate(algos):
        color, marker = algo_style(algo, i)
        label = resolve_label(algo, aliases)
        positions, data = [], []
        for j, s in enumerate(speeds):
            vals = df_paired[
                (df_paired["Algorithm"] == algo) & (df_paired["Nominal Speed"] == s)
            ]["Min Clearance"].dropna().values
            if len(vals) > 1:
                positions.append(j + 1 + offsets[i])
                data.append(vals)

        if not data:
            continue

        parts = ax.violinplot(data, positions=positions, widths=width * 0.88,
                              showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        for key in ("cmedians", "cmaxes", "cmins", "cbars"):
            if key in parts:
                parts[key].set_color(color)
                parts[key].set_linewidth(1.0)
        if "cmedians" in parts:
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        ax.plot([], [], color=color, marker=marker, linestyle="-",
                linewidth=2, label=label)

    ax.set_xticks(range(1, len(speeds) + 1))
    ax.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Clearance Distribution by Pace (paired routes)")

    if standalone:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_clearance_aggregate_vs_pace(
    df: pd.DataFrame,
    fig_cfg: dict,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Mean ± 0.1 std band pooling all algorithms at each pace.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=fig_size(fig_cfg))
    else:
        fig = ax.figure

    df_ok = filter_ok(df)
    if "Nominal Speed" not in df_ok.columns or "Min Clearance" not in df_ok.columns:
        return fig

    speeds = sorted(df_ok["Nominal Speed"].unique())
    means, stds = [], []
    for s in speeds:
        vals = df_ok[df_ok["Nominal Speed"] == s]["Min Clearance"].dropna().values
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1))

    means = np.array(means)
    stds  = np.array(stds) * 0.1

    color = "#2c7bb6"
    ax.plot(speeds, means, color=color, lw=2.5, marker="o", markersize=5, label="Mean")
    ax.fill_between(speeds, means - stds, means + stds,
                    color=color, alpha=0.25, label="±0.1 std")

    ax.set_xlabel("Pace (m/s)")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Clearance vs. Pace — Cumulative (all algorithms, successful runs)")
    ax.set_xticks(speeds)

    if standalone:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


def plot_clearance_box_by_city(
    df: pd.DataFrame,
    algos: list,
    aliases: dict,
    fig_cfg: dict,
    ax: plt.Axes = None,
) -> plt.Figure:
    """
    Grouped box plot of Min Clearance by city.
    """
    standalone = ax is None
    if standalone:
        w, h = fig_size(fig_cfg)
        fig, ax = plt.subplots(figsize=(w * 1.4, h))
    else:
        fig = ax.figure

    df_ok  = filter_ok(df)
    if "City" not in df_ok.columns or "Min Clearance" not in df_ok.columns:
        return fig

    cities = df_ok["City"].unique()
    n      = len(algos)
    width  = 0.8 / n
    offsets = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, algo in enumerate(algos):
        color, marker = algo_style(algo, i)
        label  = resolve_label(algo, aliases)
        data   = [
            df_ok[(df_ok["Algorithm"] == algo) & (df_ok["City"] == city)
                  ]["Min Clearance"].dropna().values
            for city in cities
        ]
        positions = [j + 1 + offsets[i] for j, d in enumerate(data) if len(d) > 0]
        data      = [d for d in data if len(d) > 0]
        if not data:
            continue

        bp = ax.boxplot(
            data, positions=positions, widths=width * 0.82,
            patch_artist=True, showfliers=False, whis=[5, 95],
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            boxprops=dict(linewidth=0.8),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
        ax.plot([], [], color=color, marker=marker, linestyle="-",
                linewidth=2, label=label)

    ax.set_xticks(range(1, len(cities) + 1))
    ax.set_xticklabels(cities, rotation=30, ha="right")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation_mode="anchor")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("City")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title("Min Clearance Distribution by City")

    if standalone:
        ax.legend(loc="best", frameon=True, framealpha=0.9)
        fig.tight_layout()
    return fig


# PER-BENCHMARK FIGURE SETS

def figures_quad(df: pd.DataFrame, bench: dict, fig_cfg: dict, analyze_cfg: dict):
    algos   = ordered_algos(df, bench.get("algorithms", []))
    aliases  = analyze_cfg.get("aliases", {})
    name     = bench.get("name", "Quadcopter").replace(" ", "_")
    central   = analyze_cfg.get("clearance_central", "median")
    log_scale = analyze_cfg.get("clearance_log_scale", True)

    non_ddp = [a for a in algos if not a.startswith("DDP")]
    fig = plot_pace_vs_clearance(df, non_ddp, aliases, fig_cfg, central=central, log_scale=log_scale)
    save_fig(fig, f"{name}_pace_vs_clearance", fig_cfg, analyze_cfg)

    fig = plot_pace_vs_metric(df, "Avg Solve Time", "Avg Solve Time (s)",
                              algos, aliases, fig_cfg, title="Pace vs. Average Solve Time")
    save_fig(fig, f"{name}_pace_vs_solve_time", fig_cfg, analyze_cfg)

    fig = plot_pace_vs_metric(df, "Path Length", "Path Length (m)",
                              algos, aliases, fig_cfg, title="Pace vs. Path Length",
                              show_band=False)
    save_fig(fig, f"{name}_pace_vs_path_length", fig_cfg, analyze_cfg)

    fig = plot_pace_vs_metric(df, "Control Effort", "Control Effort (‖u‖)",
                              algos, aliases, fig_cfg, title="Pace vs. Control Effort")
    save_fig(fig, f"{name}_pace_vs_ctrl_effort", fig_cfg, analyze_cfg)

    fig = plot_success_rate(df, algos, aliases, fig_cfg)
    save_fig(fig, f"{name}_success_rate", fig_cfg, analyze_cfg)

    fig = plot_pace_vs_rate(df, non_ddp, aliases, fig_cfg,
                            log_scale=analyze_cfg.get("rate_log_scale", True),
                            ymin_zero=analyze_cfg.get("rate_ymin_zero", False))
    save_fig(fig, f"{name}_pace_vs_rate", fig_cfg, analyze_cfg)

    fig = plot_violin(df, "Min Clearance", "Min Clearance (m)", algos, aliases, fig_cfg,
                      title="Min Clearance Distribution", hline=0.0)
    save_fig(fig, f"{name}_clearance_dist", fig_cfg, analyze_cfg)

    fig = plot_violin(df, "Avg Solve Time", "Avg Solve Time (s)", algos, aliases, fig_cfg,
                      title="Solve Time Distribution")
    save_fig(fig, f"{name}_solve_time_dist", fig_cfg, analyze_cfg)

    fig = plot_clearance_box_by_pace(df, algos, aliases, fig_cfg)
    save_fig(fig, f"{name}_clearance_box_by_pace", fig_cfg, analyze_cfg)

    fig = plot_clearance_violin_by_pace(df, non_ddp, aliases, fig_cfg)
    save_fig(fig, f"{name}_clearance_violin_by_pace", fig_cfg, analyze_cfg)

    fig = plot_clearance_aggregate_vs_pace(df, fig_cfg)
    save_fig(fig, f"{name}_clearance_aggregate_vs_pace", fig_cfg, analyze_cfg)

    if "City" in df.columns:
        fig = plot_clearance_box_by_city(df, algos, aliases, fig_cfg)
        save_fig(fig, f"{name}_clearance_box_by_city", fig_cfg, analyze_cfg)

    sqp_algos = [a for a in algos if a.startswith("SQP")]
    if sqp_algos:
        fig = plot_pace_vs_metric(df, "Avg Solve Time", "Avg Solve Time (s)",
                                  sqp_algos, aliases, fig_cfg,
                                  title="SQP Variants — Pace vs. Solve Time")
        save_fig(fig, f"{name}_sqp_solve_time", fig_cfg, analyze_cfg)

    if analyze_cfg.get("per_city", False) and "City" in df.columns:
        for city in df["City"].unique():
            df_city = df[df["City"] == city]
            city_slug = city.replace(" ", "_")
            fig = plot_pace_vs_clearance(df_city, algos, aliases, fig_cfg, central=central, log_scale=log_scale)
            save_fig(fig, f"{name}_{city_slug}_pace_vs_clearance", fig_cfg, analyze_cfg)

    w = fig_cfg.get("width", 7.0)
    h = fig_cfg.get("height", 4.5)
    fig, axes = plt.subplots(2, 4, figsize=(w * 2.1, h * 1.6))
    fig.suptitle(f"{bench.get('name', 'Quadcopter')} — Benchmark Summary",
                 fontsize=13, fontweight="bold")

    plot_pace_vs_clearance(df, non_ddp, aliases, fig_cfg, central=central, ax=axes[0, 0])
    plot_pace_vs_metric(df, "Avg Solve Time", "Avg Solve Time (s)",
                        algos, aliases, fig_cfg, ax=axes[0, 1])
    plot_pace_vs_metric(df, "Path Length", "Path Length (m)",
                        algos, aliases, fig_cfg, ax=axes[0, 2])
    plot_pace_vs_metric(df, "Control Effort", "Control Effort",
                        algos, aliases, fig_cfg, ax=axes[0, 3])

    plot_success_rate(df, algos, aliases, fig_cfg, title="Success Rate", ax=axes[1, 0])
    plot_pace_vs_metric(df, "Travel Time", "Travel Time (s)",
                        algos, aliases, fig_cfg, ax=axes[1, 1])
    plot_violin(df, "Min Clearance", "Min Clearance (m)", algos, aliases, fig_cfg,
                title="Clearance Dist.", hline=0.0, ax=axes[1, 2])
    plot_violin(df, "Avg Solve Time", "Solve Time (s)", algos, aliases, fig_cfg,
                title="Solve Time Dist.", ax=axes[1, 3])

    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc="lower center",
                   ncol=min(len(algos), 4), fontsize=8,
                   bbox_to_anchor=(0.5, -0.02), frameon=True)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, f"{name}_summary", fig_cfg, analyze_cfg)


def figures_wmr(df: pd.DataFrame, bench: dict, fig_cfg: dict, analyze_cfg: dict):
    algos   = ordered_algos(df, bench.get("algorithms", []))
    aliases = analyze_cfg.get("aliases", {})
    name    = bench.get("name", "WMR").replace(" ", "_")

    fig = plot_algo_bar(df, "Min Clearance", "Min Clearance (m)", algos, aliases, fig_cfg,
                        title="Min Clearance by Algorithm", hline=0.0)
    save_fig(fig, f"{name}_clearance", fig_cfg, analyze_cfg)

    fig = plot_algo_bar(df, "Avg Solve Time", "Avg Solve Time (s)", algos, aliases, fig_cfg,
                        title="Avg Solve Time by Algorithm")
    save_fig(fig, f"{name}_solve_time", fig_cfg, analyze_cfg)

    fig = plot_success_rate(df, algos, aliases, fig_cfg)
    save_fig(fig, f"{name}_success_rate", fig_cfg, analyze_cfg)

    fig = plot_algo_bar(df, "Path Length", "Path Length (m)", algos, aliases, fig_cfg,
                        title="Path Length by Algorithm")
    save_fig(fig, f"{name}_path_length", fig_cfg, analyze_cfg)

    w = fig_cfg.get("width", 7.0)
    h = fig_cfg.get("height", 4.5)
    fig, axes = plt.subplots(2, 3, figsize=(w * 1.6, h * 1.4))
    fig.suptitle(f"{bench.get('name', 'WMR')} — Benchmark Summary",
                 fontsize=13, fontweight="bold")

    plot_algo_bar(df, "Min Clearance", "Min Clearance (m)", algos, aliases, fig_cfg,
                  title="Min Clearance", hline=0.0, ax=axes[0, 0])
    plot_algo_bar(df, "Avg Solve Time", "Avg Solve Time (s)", algos, aliases, fig_cfg,
                  title="Avg Solve Time", ax=axes[0, 1])
    plot_success_rate(df, algos, aliases, fig_cfg, title="Success Rate", ax=axes[0, 2])
    plot_algo_bar(df, "Path Length", "Path Length (m)", algos, aliases, fig_cfg,
                  title="Path Length", ax=axes[1, 0])
    plot_algo_bar(df, "Control Effort", "Control Effort", algos, aliases, fig_cfg,
                  title="Control Effort", ax=axes[1, 1])
    plot_violin(df, "Min Clearance", "Min Clearance (m)", algos, aliases, fig_cfg,
                title="Clearance Dist.", hline=0.0, ax=axes[1, 2])

    fig.tight_layout()
    save_fig(fig, f"{name}_summary", fig_cfg, analyze_cfg)


# MAIN

_FIGURE_DISPATCH = {
    "quad": figures_quad,
    "wmr":  figures_wmr,
}

_COLUMN_MAP = {
    "city": "City", "scenario": "Scenario", "algorithm": "Algorithm",
    "segment": "Segment", "nominal_speed": "Nominal Speed",
    "success": "Success", "is_clear": "Is Clear", "crash_reason": "Crash Reason",
    "avg_solve_time": "Avg Solve Time", "std_solve_time": "Std Solve Time",
    "min_clearance": "Min Clearance", "ref_min_clearance": "Ref Min Clearance",
    "travel_time": "Travel Time", "path_length": "Path Length",
    "converge_rate": "Converge Rate", "control_effort": "Control Effort",
    "steps": "Steps",
}


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename lowercase store column names to Title Case used by figure functions."""
    return df.rename(columns=_COLUMN_MAP)


def _load_from_store(store, benchmark_name: str, run_id: str | None) -> pd.DataFrame:
    """Load results from store — one run if run_id given, all runs otherwise."""
    if run_id is not None:
        df = store.load_run(run_id)
        return _normalize_df(df) if not df.empty else df

    runs = store.list_runs(benchmark_name)
    if runs.empty:
        return pd.DataFrame()
    frames = []
    for rid in runs["run_id"]:
        part = store.load_run(rid)
        if not part.empty:
            frames.append(_normalize_df(part))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _export_json(df: pd.DataFrame, bench: dict, fig_cfg: dict, analyze_cfg: dict) -> None:
    """Write a machine-readable JSON summary of key metrics per algorithm."""
    algos    = ordered_algos(df, bench.get("algorithms", []))
    out_dir  = Path(fig_cfg.get("output_dir", "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    name     = bench.get("name", "benchmark").replace(" ", "_")

    def _safe(v):
        if v is None:
            return None
        try:
            return None if math.isnan(v) else float(v)
        except (TypeError, ValueError):
            return v

    summary: dict = {"benchmark": bench.get("name"), "algorithms": {}}

    for algo in algos:
        sub    = df[df["Algorithm"] == algo]
        sub_ok = sub[sub["Success"] == True] if "Success" in sub.columns else sub

        metrics: dict = {}
        for col in ["Avg Solve Time", "Min Clearance", "Travel Time",
                    "Path Length", "Control Effort"]:
            vals = sub_ok[col].dropna().values if col in sub_ok.columns else np.array([])
            metrics[col] = {
                "mean": _safe(float(np.mean(vals))) if len(vals) else None,
                "std":  _safe(float(np.std(vals, ddof=1))) if len(vals) > 1 else None,
                "n":    int(len(vals)),
            }

        sr = _safe(sub["Success"].mean() * 100) if "Success" in sub.columns and len(sub) else None
        cr = _safe(sub["Is Clear"].mean() * 100) if "Is Clear" in sub.columns and len(sub) else None
        summary["algorithms"][algo] = {"success_rate": sr, "clear_rate": cr, "metrics": metrics}

    out_path = out_dir / f"{name}_summary.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved: {out_path}")


def analyze_cli(
    config_path: str,
    only: str,
    run_id: str | None = None,
    fmt: str = "figure",
) -> None:
    from data.store import open_store

    cfg         = load_config(Path(config_path))
    fig_cfg     = cfg.get("figure", {})
    analyze_cfg = cfg.get("analyze", {})
    output_cfg  = cfg.get("output", {})

    apply_style(fig_cfg.get("style", "paper"))

    store_path = output_cfg.get("store", "")
    if not store_path or not Path(store_path).exists():
        print(f"No store found at '{store_path}'. Run a benchmark first.")
        return

    store = open_store(store_path)

    for bench in cfg.get("benchmark", []):
        if only and bench.get("name") != only:
            continue

        name  = bench.get("name", "")
        df    = _load_from_store(store, name, run_id)
        if df.empty:
            label = f"run_id={run_id}" if run_id else f"benchmark='{name}'"
            print(f"[{name}] No results found for {label} — skipping.")
            continue

        label = f"run_id={run_id}" if run_id else "all runs"
        print(f"\nAnalyzing [{name}] — {len(df)} rows from store ({label})")

        if fmt == "json":
            _export_json(df, bench, fig_cfg, analyze_cfg)
        else:
            btype    = bench.get("type", "quad")
            dispatch = _FIGURE_DISPATCH.get(btype, figures_quad)
            dispatch(df, bench, fig_cfg, analyze_cfg)


if __name__ == "__main__":
    print("Please use the unified CLI: python cli.py analyze")
