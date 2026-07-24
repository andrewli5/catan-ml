"""Train + evaluate win-probability models.

Baseline: logistic regression (scaled). Main: gradient boosted trees
(sklearn HistGradientBoostingClassifier). Split is BY GAME so correlated turns
never leak across train/test. Reports log loss (primary), accuracy, ROC-AUC,
and Brier score, and saves a calibration (reliability) plot.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features.extract import FEATURE_COLUMNS, LABEL_COLUMN


def load_xy(features_path: str | Path):
    df = pd.read_parquet(features_path)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[LABEL_COLUMN].to_numpy(dtype=int)
    groups = df["game_id"].to_numpy()
    return df, X, y, groups


def split_by_game(X, y, groups, test_frac=0.2, seed=0):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx


def evaluate(name: str, model, X_te, y_te) -> dict:
    p = np.clip(model.predict_proba(X_te)[:, 1], 1e-6, 1 - 1e-6)
    return {
        "model": name,
        "log_loss": float(log_loss(y_te, p)),
        "accuracy": float(accuracy_score(y_te, (p >= 0.5).astype(int))),
        "roc_auc": float(roc_auc_score(y_te, p)),
        "brier": float(brier_score_loss(y_te, p)),
    }


def save_calibration_plot(models: dict, X_te, y_te, out_path: Path, n_bins=15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for name, model in models.items():
        p = model.predict_proba(X_te)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=n_bins,
                                                strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Observed win frequency")
    ax.set_title("Calibration (reliability) — held-out games")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(features_path="data/features.parquet", model_dir="models",
          report_dir="reports", test_frac=0.2, seed=0, verbose=True):
    df, X, y, groups = load_xy(features_path)
    train_idx, test_idx = split_by_game(X, y, groups, test_frac, seed)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    n_games = len(np.unique(groups))
    n_train_games = len(np.unique(groups[train_idx]))
    n_test_games = len(np.unique(groups[test_idx]))

    # naive reference: constant base-rate predictor
    base_rate = float(y_tr.mean())
    base_ll = float(log_loss(y_te, np.full_like(y_te, base_rate, dtype=float),
                             labels=[0, 1]))

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0),
    )
    gbt = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=True, random_state=seed,
    )
    # numpy>=2 raises spurious divide/overflow FP flags inside BLAS matmul
    # during LR fit/predict; results are valid, so silence the noise.
    with np.errstate(all="ignore"):
        logreg.fit(X_tr, y_tr)
        gbt.fit(X_tr, y_tr)

        results = [
            {"model": "base_rate", "log_loss": base_ll, "accuracy": None,
             "roc_auc": None, "brier": None},
            evaluate("logreg", logreg, X_te, y_te),
            evaluate("gbt", gbt, X_te, y_te),
        ]

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": logreg, "features": FEATURE_COLUMNS},
                    model_dir / "logreg.joblib")
        joblib.dump({"model": gbt, "features": FEATURE_COLUMNS},
                    model_dir / "gbt.joblib")

        save_calibration_plot(
            {"logreg": logreg, "gbt": gbt}, X_te, y_te,
            Path(report_dir) / "calibration.png",
        )

    meta = {
        "n_games": int(n_games),
        "n_train_games": int(n_train_games),
        "n_test_games": int(n_test_games),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "test_base_rate_win": float(y_te.mean()),
        "features": FEATURE_COLUMNS,
        "metrics": results,
    }
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"games: {n_games} (train {n_train_games} / test {n_test_games}), "
              f"rows: train {len(train_idx)} / test {len(test_idx)}")
        print(f"test win base rate: {y_te.mean():.3f}")
        print(f"{'model':<12}{'log_loss':>10}{'accuracy':>10}"
              f"{'roc_auc':>10}{'brier':>10}")
        for r in results:
            acc = "-" if r["accuracy"] is None else f"{r['accuracy']:.4f}"
            auc = "-" if r["roc_auc"] is None else f"{r['roc_auc']:.4f}"
            br = "-" if r["brier"] is None else f"{r['brier']:.4f}"
            print(f"{r['model']:<12}{r['log_loss']:>10.4f}{acc:>10}"
                  f"{auc:>10}{br:>10}")
        print(f"saved models -> {model_dir}/  calibration -> {report_dir}/calibration.png")
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train Catan win-prob models")
    ap.add_argument("--features", default="data/features.parquet")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.features, args.model_dir, args.report_dir,
          args.test_frac, args.seed)
