"""Plausibility checks for the trained win-probability model.

1. Monotonicity: mean predicted win probability should rise with victory
   points (and with VP-vs-best-opponent lead).
2. Trajectories: for a few sampled games, predicted win probability over turns
   should converge toward 1 for the eventual winner.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..features.extract import FEATURE_COLUMNS, LABEL_COLUMN
from ..inference.predict import load_model


def _raw_probs(bundle, df: pd.DataFrame) -> np.ndarray:
    X = df[bundle.get("features", FEATURE_COLUMNS)].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        return bundle["model"].predict_proba(X)[:, 1]


def monotonicity_check(bundle, df: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["pred"] = _raw_probs(bundle, df)
    table = (df.groupby("vp_total")
               .agg(mean_pred=("pred", "mean"),
                    observed_win=(LABEL_COLUMN, "mean"),
                    n=("pred", "size"))
               .reset_index())
    corr = df[["vp_total", "pred"]].corr().iloc[0, 1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["vp_total"], table["mean_pred"], "o-", label="mean predicted")
    ax.plot(table["vp_total"], table["observed_win"], "s--", label="observed win rate")
    ax.set_xlabel("victory points (total)")
    ax.set_ylabel("win probability")
    ax.set_title(f"Win prob vs VP (corr={corr:.3f})")
    ax.legend()
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / "vp_monotonicity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return table, corr


def trajectory_plots(bundle, df: pd.DataFrame, report_dir: Path,
                     n_games=4, seed=0) -> list:
    rng = np.random.default_rng(seed)
    # prefer games with a decent number of logged turns
    counts = df.groupby("game_id")["turn_number"].nunique()
    eligible = counts[counts >= 15].index.to_numpy()
    chosen = rng.choice(eligible, size=min(n_games, len(eligible)), replace=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()
    for ax, gid in zip(axes, chosen):
        g = df[df["game_id"] == gid].copy()
        g["pred"] = _raw_probs(bundle, g)
        # normalize across players within each turn
        g["norm"] = g.groupby("turn_number")["pred"].transform(
            lambda s: s / s.sum() if s.sum() > 0 else 1.0 / len(s))
        winner = g.loc[g[LABEL_COLUMN] == 1, "player_id"]
        winner = int(winner.iloc[0]) if len(winner) else -1
        for pid, sub in g.groupby("player_id"):
            sub = sub.sort_values("turn_number")
            style = "-" if pid == winner else "--"
            lbl = f"p{pid}" + (" (winner)" if pid == winner else "")
            ax.plot(sub["turn_number"], sub["norm"], style, label=lbl)
        ax.set_title(f"game {gid}")
        ax.set_xlabel("turn")
        ax.set_ylabel("win prob")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("Predicted win probability over turns (solid = eventual winner)")
    report_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(report_dir / "trajectories.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return list(chosen)


def run(features_path="data/features.parquet", model_path="models/gbt.joblib",
        report_dir="reports"):
    bundle = load_model(model_path)
    df = pd.read_parquet(features_path)
    report_dir = Path(report_dir)

    table, corr = monotonicity_check(bundle, df, report_dir)
    print(f"Monotonicity: corr(vp_total, predicted win prob) = {corr:.3f}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    chosen = trajectory_plots(bundle, df, report_dir)
    print(f"Saved vp_monotonicity.png and trajectories.png (games {chosen}) "
          f"-> {report_dir}/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Model sanity checks + plots")
    ap.add_argument("--features", default="data/features.parquet")
    ap.add_argument("--model", default="models/gbt.joblib")
    ap.add_argument("--report-dir", default="reports")
    args = ap.parse_args()
    run(args.features, args.model, args.report_dir)
