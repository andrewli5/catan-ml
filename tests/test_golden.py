"""Golden scenario tests for subtle full-rules behaviour."""
from __future__ import annotations

import random

from catan_ml.engine import invariant
from catan_ml.engine import rules as R
from catan_ml.engine.actions import Action, apply_action, legal_actions
from catan_ml.engine.board import TOPO
from catan_ml.engine.state import GameState, SETTLEMENT


def _find_path(start: int, length: int, used: set | None = None):
    """Return a list of edge ids forming a simple vertex path."""
    if used is None:
        used = {start}
    if length == 0:
        return []
    for e in TOPO.vertex_edges[start]:
        a, b = TOPO.edges[e]
        nxt = b if a == start else a
        if nxt in used:
            continue
        sub = _find_path(nxt, length - 1, used | {nxt})
        if sub is not None:
            return [e] + sub
    return None


def _path_vertices(edges: list) -> list:
    adj = {}
    for e in edges:
        a, b = TOPO.edges[e]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = next(v for v, nbs in adj.items() if len(nbs) == 1)
    seq = [start]
    prev, cur = None, start
    while len(seq) <= len(edges):
        nxt = [v for v in adj[cur] if v != prev][0]
        seq.append(nxt)
        prev, cur = cur, nxt
    return seq


def test_monopoly_takes_from_all_opponents():
    state = GameState.new_game(3, random.Random(0))
    state.phase = "main"
    p0, p1, p2 = state.players
    p0.dev_cards["monopoly"] = 1
    state.dev_deck.remove("monopoly")
    p1.resources["wheat"] = 3
    p2.resources["wheat"] = 2
    state.bank["wheat"] -= 5
    before = p0.resources["wheat"]
    apply_action(state, Action("play_monopoly", ("wheat",)))
    assert p0.resources["wheat"] == before + 5
    assert p1.resources["wheat"] == 0
    assert p2.resources["wheat"] == 0
    assert state.dev_card_played_this_turn
    invariant.check(state)


def test_dev_card_bought_this_turn_is_unplayable():
    state = GameState.new_game(3, random.Random(1))
    state.phase = "main"
    p0 = state.players[0]
    for r, n in R.COST_DEV.items():
        p0.resources[r] = n
        state.bank[r] -= n
    apply_action(state, Action("buy_dev", ()))
    assert state.dev_card_bought_this_turn
    # the card may be a knight but cannot be played the same turn
    acts = legal_actions(state)
    assert not any(a.kind.startswith("play_") for a in acts)


def test_discard_exactly_floor_n_over_2():
    state = GameState.new_game(3, random.Random(2))
    state.phase = "discard"
    state.discard_queue = [0]
    p0 = state.players[0]
    hand = {"wood": 2, "brick": 2, "sheep": 2, "wheat": 2, "ore": 1}
    p0.resources.update(hand)
    for r, c in hand.items():
        state.bank[r] -= c
    assert p0.hand_size() == 9
    discard = {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1, "ore": 0}
    apply_action(state, Action("discard", (discard,)))
    assert p0.hand_size() == 5
    assert sum(discard.values()) == 4
    assert all(p0.resources[r] >= 0 for r in p0.resources)
    invariant.check(state)


def test_bank_shortfall_nobody_gets_resource():
    state = GameState.new_game(4, random.Random(4))
    wheat_hexes = [h for h, res in enumerate(state.board.hex_resource) if res == "wheat"]
    h1, h2 = wheat_hexes[:2]
    state.board.hex_number[h1] = 8
    state.board.hex_number[h2] = 8
    # only these two hexes produce on an 8
    for h in range(TOPO.n_hexes):
        if h not in (h1, h2):
            state.board.hex_number[h] = 0
    robber = 0
    while robber in (h1, h2):
        robber += 1
    state.board.robber_hex = robber
    v1 = v2 = None
    for cand1 in TOPO.hex_vertices[h1]:
        for cand2 in TOPO.hex_vertices[h2]:
            if cand1 != cand2:
                v1, v2 = cand1, cand2
                break
        if v1 is not None:
            break
    assert v1 is not None
    state.vertex_owner[v1] = 0
    state.vertex_type[v1] = SETTLEMENT
    state.vertex_owner[v2] = 1
    state.vertex_type[v2] = SETTLEMENT
    state.bank["wheat"] = 1
    before0 = state.players[0].resources["wheat"]
    before1 = state.players[1].resources["wheat"]
    R.produce(state, 8)
    # two players owed -> official shortfall: nobody gets it
    assert state.players[0].resources["wheat"] == before0
    assert state.players[1].resources["wheat"] == before1
    assert state.bank["wheat"] == 1


def test_largest_army_tie_and_incumbent():
    state = GameState.new_game(3, random.Random(3))
    p0, p1, p2 = state.players
    p0.played_dev["knight"] = 3
    p1.played_dev["knight"] = 3
    R._recompute_largest_army(state)
    assert not any(p.has_largest_army for p in state.players)
    p0.played_dev["knight"] = 4
    R._recompute_largest_army(state)
    assert p0.has_largest_army
    # tie at 4: incumbent keeps it
    p1.played_dev["knight"] = 4
    R._recompute_largest_army(state)
    assert p0.has_largest_army
    assert not p1.has_largest_army
    # p0 drops below, p1 takes it
    p0.played_dev["knight"] = 3
    R._recompute_largest_army(state)
    assert p1.has_largest_army
    assert not p0.has_largest_army


def test_largest_army_tie_clears_non_leader():
    """A tie among new leaders should strip the card from a holder who is not tied."""
    state = GameState.new_game(4, random.Random(9))
    p0, p1, p2 = state.players[0], state.players[1], state.players[2]
    p0.played_dev["knight"] = 4
    R._recompute_largest_army(state)
    assert p0.has_largest_army
    # Two other players tie above the current holder; card goes to nobody.
    p1.played_dev["knight"] = 5
    p2.played_dev["knight"] = 5
    R._recompute_largest_army(state)
    assert not any(p.has_largest_army for p in state.players)
    # Once the tie is broken, a single leader takes the card.
    p2.played_dev["knight"] = 4
    R._recompute_largest_army(state)
    assert p1.has_largest_army
    assert not p0.has_largest_army


def test_longest_road_broken_by_opponent_settlement():
    state = GameState.new_game(4, random.Random(5))
    edges = None
    for v in range(TOPO.n_vertices):
        edges = _find_path(v, 4)
        if edges:
            break
    assert edges is not None
    seq = _path_vertices(edges)
    R.place_settlement(state, 0, seq[0], free=True)
    for e in edges:
        R.place_road(state, 0, e, free=True)
    assert state.players[0].longest_road_len == 4
    middle = seq[2]
    R.place_settlement(state, 1, middle, free=True)
    # road is split into two halves
    assert state.players[0].longest_road_len == 2


def test_road_building_with_only_one_legal_road():
    state = GameState.new_game(4, random.Random(6))
    # Player 0 has a lone settlement with exactly one reachable road spot.
    v0 = 0
    R.place_settlement(state, 0, v0, free=True)
    eid = next(e for e in TOPO.vertex_edges[v0] if state.edge_owner[e] == -1)
    p0 = state.players[0]
    p0.dev_cards["road_building"] = 1
    state.dev_deck.remove("road_building")
    state.phase = "main"
    before = p0.roads
    apply_action(state, Action("play_road_building", (eid, eid)))
    assert p0.roads == before + 1
    assert p0.played_dev["road_building"] == 1
    invariant.check(state)
