"""Engine smoke + legality invariants.

Run: python -m tests.test_engine   (or pytest tests/test_engine.py)
Confirms 50 games complete with no crashes and no illegal board states.
"""
from __future__ import annotations

import random
import time

from catan_ml.engine.board import TOPO
from catan_ml.engine.state import CITY, EMPTY, SETTLEMENT
from catan_ml.engine import rules as R
from catan_ml.sim.simulate import run_game


def assert_board_legal(state) -> None:
    # 1. no vertex both empty-typed yet owned, or vice versa
    for v in range(TOPO.n_vertices):
        owned = state.vertex_owner[v] != -1
        typed = state.vertex_type[v] != EMPTY
        assert owned == typed, f"vertex {v} owner/type mismatch"
    # 2. distance rule: no two occupied vertices adjacent
    for v in range(TOPO.n_vertices):
        if state.vertex_type[v] == EMPTY:
            continue
        for nb in TOPO.vertex_neighbors[v]:
            assert state.vertex_type[nb] == EMPTY, f"adjacent builds {v},{nb}"
    # 3. resource counts non-negative
    for p in state.players:
        for r, c in p.resources.items():
            assert c >= 0, f"negative {r} for player {p.pid}"
    # 4. VP accounting consistent with pieces
    for p in state.players:
        assert p.total_vp() >= p.settlements + 2 * p.cities
    # 5. piece supply limits never exceeded
    for p in state.players:
        assert p.roads <= R.MAX_ROADS
        assert p.settlements <= R.MAX_SETTLEMENTS
        assert p.cities <= R.MAX_CITIES


def test_topology_counts():
    assert TOPO.n_hexes == 19
    assert TOPO.n_vertices == 54
    assert TOPO.n_edges == 72
    # every edge references two distinct valid vertices
    for a, b in TOPO.edges:
        assert a != b and 0 <= a < 54 and 0 <= b < 54


def test_fifty_games_complete_and_legal():
    rng = random.Random(42)
    finished = 0
    for g in range(50):
        n_players = rng.choice((2, 3, 4))
        rows, winner = run_game(n_players, rng, game_id=g, max_turns=600,
                                state_hook=assert_board_legal)
        assert rows, "no states logged"
        # winner reached 10 VP (or game was capped)
        if winner != -1:
            finished += 1
            # reconstruct final winner VP from last row
            last = rows[-1]
            assert last[f"p{winner}_total_vp"] >= 10
    # Weak greedy bots plateau ~10% of the time (board saturates + dev deck
    # empties -> no legal move); those games are dropped during data gen.
    assert finished >= 40, f"only {finished}/50 games finished"


def test_no_hang_many_games():
    """Regression: 300 games at a high turn cap must finish quickly.

    Previously, unbounded road-building blew up the longest-road trail search
    and hung for tens of minutes. Piece limits + a DFS budget bound it.
    """
    rng = random.Random(1)
    t0 = time.time()
    for g in range(300):
        n = rng.choice((2, 3, 4))
        run_game(n, rng, game_id=g, max_turns=600)
    elapsed = time.time() - t0
    assert elapsed < 60, f"300 games took {elapsed:.1f}s (possible hang regression)"


def _run_all():
    test_topology_counts()
    print("topology OK (19 hexes / 54 vertices / 72 edges)")
    test_fifty_games_complete_and_legal()
    print("50-game smoke + legality OK")
    t0 = time.time()
    test_no_hang_many_games()
    print(f"no-hang regression OK (300 games in {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    _run_all()
