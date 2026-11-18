"""Report figures (static PNG). Palette: the validated reference categorical order
(blue, orange, aqua), one-hue sequential blue, diverging blue <-> red with a gray midpoint.
Thin marks, recessive grid, text in ink colours (never the series colour)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger("superr.plots")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POS, NEG, MID = "#2a78d6", "#e34948", "#bdbcb6"
ROLE_COLOR = {"bat": S1, "bowl": S2}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def calibration(cal: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=MID, lw=1, ls="--", label="perfect calibration")
    for inn, col in ((1, S1), (2, S2)):
        c = cal[cal["innings"] == inn]
        ax.plot(c["wp_mean"], c["won"], marker="o", ms=6, color=col, label=f"innings {inn}")
    ax.set_xlabel("predicted win probability (decile mean)")
    ax.set_ylabel("share of batting sides that won")
    ax.set_title("Win-probability calibration")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, out)


def leverage_by_over(prof: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for inn, col in ((1, S1), (2, S2)):
        p = prof[prof["innings"] == inn]
        ax.plot(p["overid"], p["mean"], color=col, marker="o", ms=4, label=f"innings {inn}")
    ax.axhline(1, color=MID, lw=1, ls="--")
    ax.set_xlabel("over")
    ax.set_ylabel("mean leverage index (LI)")
    ax.set_title("Leverage rises through a chase; LI = 1 is an average ball")
    ax.legend()
    ax.set_xticks(range(1, 21))
    return _save(fig, out)


def stakes_hist(stakes: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(0, 1, 21)
    for kind, col in (("league", S1), ("bilateral", S2), ("knockout", S3)):
        s = stakes.loc[stakes["kind"] == kind, "stakes"]
        ax.hist(s, bins=bins, color=col, alpha=0.75, label=f"{kind} (n={len(s)})", edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("stakes = p_advance_if_win - p_advance_if_lose")
    ax.set_ylabel("(match, team) rows")
    ax.set_title("Match stakes by context")
    ax.legend()
    return _save(fig, out)


def top10(df: pd.DataFrame, role: str, out: Path) -> Path:
    d = df.sort_values("rank", ascending=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(d))
    col = ROLE_COLOR[role]
    if d["P_lo"].notna().any():
        ax.hlines(y, d["P_lo"], d["P_hi"], color=col, lw=2, alpha=0.45)
    ax.scatter(d["P"], y, s=48, color=col, zorder=3, edgecolor=SURFACE, linewidth=2)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(104, yi, f"P(top 10) {r['p_top']:.2f}" if pd.notna(r["p_top"]) else "", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{int(r)}. {n}" for r, n in zip(d["rank"], d["player_name"])])
    ax.axvline(50, color=MID, lw=1, ls="--")
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("P (0-100; 50 = no change under pressure). Bar: bootstrap 90% interval")
    ax.set_title(f"Top 10 {'batters' if role == 'bat' else 'bowlers'} under pressure")
    return _save(fig, out)


def team_T(t: pd.DataFrame, out: Path) -> Path:
    d = t.sort_values("T")
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    cols = [POS if v >= 50 else NEG for v in d["T"]]
    ax.barh(d["team_canon"], d["T"] - 50, left=50, color=cols, height=0.7, edgecolor=SURFACE, linewidth=1)
    ax.axvline(50, color=INK2, lw=1)
    for i, (_, r) in enumerate(d.iterrows()):
        if r["weak_identification"]:
            ax.text(101, i, "no movers", va="center", fontsize=7, color=INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("T (0-100; 50 = average team response to pressure)")
    ax.set_title("Team clutch T")
    ax.tick_params(axis="y", labelsize=8)
    return _save(fig, out)


def csk(csk_df: pd.DataFrame, out: Path) -> Path:
    d = csk_df[csk_df["format"] == "IPL"].dropna(subset=["T"])
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(d["distinct_per_win"], d["T"], s=60, color=S1, edgecolor=SURFACE, linewidth=2, zorder=3)
    placed: list[tuple[float, float]] = []
    xr = max(d["distinct_per_win"].max() - d["distinct_per_win"].min(), 1e-9)
    yr = max(d["T"].max() - d["T"].min(), 1e-9)
    for _, r in d.sort_values("T", ascending=False).iterrows():
        x, y = r["distinct_per_win"], r["T"]
        dy = 4
        # nudge a label down if it would sit on top of an earlier one
        for px, py in placed:
            if abs(px - x) / xr < 0.25 and abs(py - y) / yr < 0.06:
                dy = -11
        placed.append((x, y))
        ax.annotate(r["team_canon"], (x, y), xytext=(5, dy), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("distinct man-of-the-match winners per win (IPL 2022-26)")
    ax.set_ylabel("team clutch T")
    ax.set_title("The CSK claim: spread of match-winners vs T")
    return _save(fig, out)


def shrinkage(scored: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, role in zip(axes, ("bat", "bowl")):
        d = scored[scored["role"] == role]
        ax.scatter(d["exposure"], d["c"], s=10, color=MID, label="all player-roles", zorder=2)
        e = d[d["eligible"]]
        ax.scatter(e["exposure"], e["c"], s=18, color=ROLE_COLOR[role], label="eligible", zorder=3)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("pressure exposure E = sum(omega n h^2)")
        ax.set_ylabel("c (shrunk pressure slope)")
        ax.set_title(f"{'Batters' if role == 'bat' else 'Bowlers'}: opportunity weighting")
        ax.legend(loc="lower right")
    return _save(fig, out)


def synthetic(syn: pd.DataFrame, out: Path) -> Path:
    g = syn.groupby("effect_multiplier")[["spearman_c_eligible", "corr_tau"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["effect_multiplier"], g["spearman_c_eligible"], marker="o", ms=7, color=S1, label="player c (eligible, Spearman)")
    ax.plot(g["effect_multiplier"], g["corr_tau"], marker="o", ms=7, color=S2, label="team tau (Pearson)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(g["effect_multiplier"])
    ax.set_xticklabels([f"{m:g}x" for m in g["effect_multiplier"]])
    ax.set_ylim(0, 1)
    ax.set_xlabel("planted effect size (x CV-implied prior sd)")
    ax.set_ylabel("planted vs recovered correlation")
    ax.set_title("Synthetic recovery on the real schedule")
    ax.legend(loc="upper left")
    return _save(fig, out)


def oot(j: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, role in zip(axes, ("bat", "bowl")):
        d = j[j["role"] == role]
        ax.scatter(d["c"], d["test_slope"], s=12, color=ROLE_COLOR[role], alpha=0.8)
        ax.axhline(0, color=MID, lw=1)
        ax.axvline(0, color=MID, lw=1)
        ax.set_xlabel("c fitted on 2022-2024")
        ax.set_ylabel("pressure slope in 2025-26")
        ax.set_title(f"{'Batters' if role == 'bat' else 'Bowlers'}: out-of-time")
    return _save(fig, out)


def ablations(ab: pd.DataFrame, out: Path) -> Path:
    piv = ab.pivot(index="ablation", columns="role", values="top10_overlap").fillna(0)
    piv = piv.reindex(ab["ablation"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(piv))
    hgt = 0.38
    ax.barh(y + hgt / 2 + 0.01, piv.get("bat", 0), height=hgt, color=S1, label="batters", edgecolor=SURFACE)
    ax.barh(y - hgt / 2 - 0.01, piv.get("bowl", 0), height=hgt, color=S2, label="bowlers", edgecolor=SURFACE)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the main top 10 kept")
    ax.set_title("Ablations and sensitivities: top-10 overlap")
    ax.legend(loc="lower right")
    return _save(fig, out)


def make_all(paths, cfg) -> dict[str, str]:
    from . import io
    fig = paths.report / "figures"
    val = paths.outputs / "validation"
    out: dict[str, str] = {}

    def rd(name):
        fp = val / f"{name}.csv"
        return pd.read_csv(fp) if fp.exists() else None

    try:
        if (c := rd("data_7_1_calibration")) is not None:
            out["calibration"] = str(calibration(c, fig / "fig_calibration.png"))
        if (p := rd("data_leverage_by_over")) is not None:
            out["leverage"] = str(leverage_by_over(p, fig / "fig_leverage_by_over.png"))
        out["stakes"] = str(stakes_hist(io.load("stakes", paths), fig / "fig_stakes.png"))
        for role, name in (("bat", "top10_batters"), ("bowl", "top10_bowlers")):
            fp = paths.outputs / f"{name}.csv"
            if fp.exists():
                out[f"top10_{role}"] = str(top10(pd.read_csv(fp), role, fig / f"fig_top10_{role}.png"))
        out["team_T"] = str(team_T(pd.read_csv(paths.outputs / "team_scores.csv"), fig / "fig_team_T.png"))
        out["shrinkage"] = str(shrinkage(io.load("scored_players", paths), fig / "fig_shrinkage.png"))
        if (x := rd("model_11_9_csk_claim")) is not None:
            out["csk"] = str(csk(x, fig / "fig_csk.png"))
        if (x := rd("model_11_1_synthetic_recovery")) is not None:
            out["synthetic"] = str(synthetic(x, fig / "fig_synthetic.png"))
        if (x := rd("model_11_5_out_of_time")) is not None:
            out["oot"] = str(oot(x, fig / "fig_oot.png"))
        if (x := rd("model_11_8_ablations")) is not None:
            out["ablations"] = str(ablations(x, fig / "fig_ablations.png"))
    finally:
        plt.close("all")
    log.info("figures: %s", ", ".join(out))
    return out
