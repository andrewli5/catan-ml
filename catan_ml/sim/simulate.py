"""Game simulator: setup, turn loop, and dataset logging."""
from __future__ import annotations

import random
from pathlib import Path

from ..engine import rules as R
from ..engine.board import RESOURCES, TOPO
from ..engine.state import GameState
from .bots import make_bots


def _setup_phase(state: GameState, bots: list) -> None:
    """Snake-draft: each player places 2 settlements + 2 roads.
    The second settlement yields starting resources (standard rule)."""
    n = state.n_players
    order = list(range(n)) + list(range(n - 1, -1, -1))
    for k, pid in enumerate(order):
        bot = bots[pid]
        vid = bot.setup_settlement(state, pid)
        R.place_settlement(state, pid, vid, free=True)
        eid = bot.setup_road(state, pid, vid)
        R.place_road(state, pid, eid, free=True, setup_vertex=vid)
        # second settlement (second half of snake) grants resources
        if k >= n:
            for h in TOPO.vertex_hexes[vid]:
                res = state.board.hex_resource[h]
                if res in RESOURCES:
                    state.players[pid].gain(res, 1)


def run_game(n_players: int, rng: random.Random, game_id: int,
             log_every: int = 1, max_turns: int = 400, state_hook=None) -> tuple:
    """Play one game. Returns (rows, winner). rows is a list of state dicts
    tagged with game_id/turn; winner attached to each row after game ends.
    ``state_hook`` (if given) is called with the live GameState each turn,
    used by tests to assert board-legality invariants."""
    state = GameState.new_game(n_players, rng)
    bots = make_bots(n_players, rng)
    _setup_phase(state, bots)

    rows = []
    turn = 0
    winner = -1
    while turn < max_turns:
        pid = state.current_player
        state.turn_number = turn
        roll = R.roll_dice(state)
        if roll == 7:
            R.handle_robber(state, pid, bots[pid].choose_robber(state, pid))
        else:
            R.produce(state, roll)
        bots[pid].act(state, pid)

        if state_hook is not None:
            state_hook(state)

        if turn % log_every == 0:
            row = state.to_row()
            row["game_id"] = game_id
            row["bot_styles"] = [b.name for b in bots]
            rows.append(row)

        if R.check_winner(state) != -1:
            winner = state.winner
            break
        state.current_player = (pid + 1) % n_players
        turn += 1

    for row in rows:
        row["winner"] = winner
    return rows, winner


def run_games(n_games: int, out_path: str | Path, n_players_choices=(2, 3, 4),
              seed: int = 0, log_every: int = 1, max_turns: int = 600,
              drop_unfinished: bool = True, verbose: bool = True) -> "pd.DataFrame":
    """Simulate ``n_games`` games and write logged states to parquet.

    Returns the raw logged DataFrame (one row per logged state snapshot).
    """
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
