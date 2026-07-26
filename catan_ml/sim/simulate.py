"""Game simulator: setup, turn loop, and dataset logging."""
from __future__ import annotations

import random
from pathlib import Path

from ..engine.actions import active_player, apply_action
from ..engine.state import GameState
from .bots import make_bots


def run_game(n_players: int, rng: random.Random, game_id: int,
             log_every: int = 1, max_turns: int = 400, state_hook=None,
             check_conservation: bool = False) -> tuple:
    """Play one game via the Action API. Returns (rows, winner)."""
    state = GameState.new_game(n_players, rng)
    state.check_conservation = check_conservation
    bots = make_bots(n_players, rng)

    rows = []
    turn = 1
    while turn <= max_turns:
        pid = active_player(state)
        bot = bots[pid]
        action = bot.choose_action(state, pid)
        apply_action(state, action)

        if state_hook is not None:
            state_hook(state)

        # log once per turn, and always log the terminal winning state
        if state.winner != -1 or (action.kind == "end_turn" and turn % log_every == 0):
            row = state.to_row()
            row["game_id"] = game_id
            row["bot_styles"] = [b.name for b in bots]
            rows.append(row)

        if state.winner != -1:
            break

        if action.kind == "end_turn":
            turn += 1

    winner = state.winner
    for row in rows:
        row["winner"] = winner
    return rows, winner


def run_games(n_games: int, out_path: str | Path, n_players_choices=(2, 3, 4),
              seed: int = 0, log_every: int = 1, max_turns: int = 600,
              drop_unfinished: bool = True, verbose: bool = True) -> "pd.DataFrame":
    """Simulate ``n_games`` games and write logged states to parquet."""
    import pandas as pd

    rng = random.Random(seed)
    all_rows = []
    finished = 0
    for g in range(n_games):
        n_players = rng.choice(n_players_choices)
        rows, winner = run_game(n_players, rng, game_id=g,
                                log_every=log_every, max_turns=max_turns)
        if winner == -1 and drop_unfinished:
            continue
        finished += 1
        all_rows.extend(rows)
        if verbose and (g + 1) % 250 == 0:
            print(f"  simulated {g + 1}/{n_games} games "
                  f"({finished} finished, {len(all_rows)} rows)")

    df = pd.DataFrame(all_rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    if verbose:
        print(f"Wrote {len(df)} rows from {finished}/{n_games} finished games "
              f"-> {out_path}")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Simulate Catan games -> parquet")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--out", type=str, default="data/games.parquet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=600)
    args = ap.parse_args()
    run_games(args.games, args.out, seed=args.seed,
              log_every=args.log_every, max_turns=args.max_turns)
