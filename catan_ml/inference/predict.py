"""Load a trained model and return per-player win probabilities for a state.

A "state" here is the dict produced by ``GameState.to_row()`` (also what the
simulator logs). The model scores each seated player independently, then the
probabilities are normalized to sum to 1 (exactly one player wins).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from ..engine.state import GameState
from ..features.extract import FEATURE_COLUMNS, row_to_player_features


def load_model(path: str | Path = "models/gbt.joblib"):
    return joblib.load(path)


def predict_from_row(bundle: dict, row: dict, normalize: bool = True) -> dict:
    """Return {player_id: win_probability} for one logged-state dict."""
    model = bundle["model"]
    cols = bundle.get("features", FEATURE_COLUMNS)
    n = row["n_players"]
    X = np.array([
        [row_to_player_features(row, pid)[c] for c in cols]
        for pid in range(n)
    ], dtype=float)
    with np.errstate(all="ignore"):
        raw = model.predict_proba(X)[:, 1]
    if normalize:
        s = raw.sum()
        probs = raw / s if s > 0 else np.full(n, 1.0 / n)
    else:
        probs = raw
    return {pid: float(probs[pid]) for pid in range(n)}


def predict_from_state(bundle: dict, state: GameState, normalize: bool = True) -> dict:
    return predict_from_row(bundle, state.to_row(), normalize=normalize)


def _demo(model_path: str, seed: int) -> None:
    """Simulate a random game to a random point and predict from that state."""
    import random

    from ..engine.actions import active_player, apply_action
    from ..sim.bots import make_bots

    rng = random.Random(seed)
    n = rng.choice((2, 3, 4))
    state = GameState.new_game(n, rng)
    bots = make_bots(n, rng)
    stop_turn = rng.randint(20, 120)
    while state.winner == -1 and state.turn_number < stop_turn:
        pid = active_player(state)
        action = bots[pid].choose_action(state, pid)
        apply_action(state, action)

    bundle = load_model(model_path)
    probs = predict_from_row(bundle, state.to_row())
    print(f"Demo game: {n} players, stopped at turn {state.turn_number}")
    print(f"{'player':<8}{'VP(total)':>10}{'settle':>8}{'city':>6}"
          f"{'win_prob':>10}")
    for pid in range(n):
        p = state.players[pid]
        print(f"{pid:<8}{p.total_vp():>10}{p.settlements:>8}{p.cities:>6}"
              f"{probs[pid]:>10.3f}")
    print("(win_prob sums to 1 across players)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Predict Catan win probabilities")
    ap.add_argument("--model", default="models/gbt.joblib")
    ap.add_argument("--state", help="path to a JSON state row (GameState.to_row())")
    ap.add_argument("--demo", action="store_true",
                    help="simulate a random game state and predict")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.state:
        with open(args.state) as f:
            row = json.load(f)
        bundle = load_model(args.model)
        probs = predict_from_row(bundle, row)
        print(json.dumps({str(k): round(v, 4) for k, v in probs.items()}, indent=2))
    else:
        _demo(args.model, args.seed)
