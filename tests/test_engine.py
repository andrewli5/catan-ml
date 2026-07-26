"""Engine smoke + legality invariants.

Run: pytest
Confirms 50 games complete with no crashes and no illegal board states.
"""
from __future__ import annotations

import pickle
import random
import subprocess
import sys
import time

from catan_ml.engine import invariant
from catan_ml.engine import rules as R
from catan_ml.engine.actions import Action, apply_action, legal_actions
from catan_ml.engine.board import TOPO
from catan_ml.engine.state import CITY, EMPTY, SETTLEMENT, GameState
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


def assert_conservation(state) -> None:
    invariant.check(state)


def _both_hooks(state) -> None:
    assert_board_legal(state)
    assert_conservation(state)


def test_clone_independence():
    state = GameState.new_game(4, random.Random(123))
    state.trade_offer = {
        "offerer": 0,
        "responder": 1,
        "give": {"wood": 1, "brick": 0, "sheep": 2, "wheat": 0, "ore": 0},
        "want": {"wood": 0, "brick": 1, "sheep": 0, "wheat": 0, "ore": 0},
    }
    clone = state.clone()
    original = state.to_row()
    assert clone.to_row() == original

    # mutate every mutable field of the clone
    clone.board.hex_resource[0] = "changed"
    clone.board.hex_number[0] = 99
    clone.board.robber_hex = 5
    clone.board.vertex_port[0] = "2:1_wood"
    clone.players[0].resources["wood"] = 999
    clone.players[0].settlements += 1
    clone.vertex_owner[0] = 3
    clone.vertex_type[0] = SETTLEMENT
    clone.edge_owner[0] = 3
    clone.dev_deck.append("extra")
    clone.rng_dice.randint(1, 6)
    clone.trade_offer["give"]["wood"] = 999

    assert state.to_row() == original, "mutating clone changed original"
    assert clone.to_row() != original
    assert state.trade_offer["give"]["wood"] == 1, "mutating clone trade_offer changed original"

    # mutate original, clone must stay as it was after the first mutation set
    clone_snapshot = clone.to_row()
    state.board.hex_resource[1] = "changed"
    state.board.hex_number[1] = 99
    state.board.robber_hex = 7
    state.board.vertex_port[1] = "3:1"
    state.players[1].resources["brick"] = 111
    state.players[1].cities += 1
    state.vertex_owner[1] = 2
    state.vertex_type[1] = CITY
    state.edge_owner[1] = 2
    state.dev_deck.append("other")
    state.rng_dice.randint(1, 6)
    state.trade_offer["want"]["brick"] = 999

    assert clone.to_row() == clone_snapshot, "mutating original changed clone"
    assert clone.trade_offer["want"]["brick"] == 1, "mutating original trade_offer changed clone"


def test_topology_determinism():
    # ids and structure must be identical in a fresh interpreter process.
    here = pickle.dumps(TOPO)
    cmd = [
        sys.executable,
        "-c",
        "import pickle, sys; from catan_ml.engine.board import TOPO; "
        "sys.stdout.buffer.write(pickle.dumps(TOPO))",
    ]
    there = subprocess.check_output(cmd, timeout=30)
    assert here == there, "topology differs across interpreter invocations"


def test_topology_counts():
    assert TOPO.n_hexes == 19
    assert TOPO.n_vertices == 54
    assert TOPO.n_edges == 72
    # every edge references two distinct valid vertices
    for a, b in TOPO.edges:
        assert a != b and 0 <= a < 54 and 0 <= b < 54
    # adjacency is symmetric
    for v in range(TOPO.n_vertices):
        for nb in TOPO.vertex_neighbors[v]:
            assert v in TOPO.vertex_neighbors[nb]


def test_fifty_games_complete_and_legal():
    rng = random.Random(42)
    finished = 0
    for g in range(50):
        n_players = rng.choice((2, 3, 4))
        rows, winner = run_game(n_players, rng, game_id=g, max_turns=600,
                                state_hook=_both_hooks, check_conservation=True)
        assert rows, "no states logged"
        if winner != -1:
            finished += 1
            last = rows[-1]
            assert last[f"p{winner}_total_vp"] >= 10
    assert finished >= 40, f"only {finished}/50 games finished"


def test_conservation_fuzz():
    """Fuzz: 1000 games with conservation invariant enabled after every action."""
    rng = random.Random(7)
    for g in range(1000):
        n = rng.choice((2, 3, 4))
        run_game(n, rng, game_id=g, max_turns=600,
                 state_hook=assert_conservation, check_conservation=True)


def test_no_hang_many_games():
    """Regression: 300 games at a high turn cap must finish quickly."""
    rng = random.Random(1)
    t0 = time.time()
    for g in range(300):
        n = rng.choice((2, 3, 4))
        run_game(n, rng, game_id=g, max_turns=600,
                 state_hook=_both_hooks, check_conservation=True)
    elapsed = time.time() - t0
    assert elapsed < 60, f"300 games took {elapsed:.1f}s (possible hang regression)"


