"""Exact longest-road computation using bounded subset DP.

The standard board limits a player to 15 road pieces, so the longest trail can be
found exactly by dynamic programming over edge subsets (2^m states, m<=15). This
replaces the exponential DFS with a silent wrong-answer budget.
"""
from __future__ import annotations

from .board import TOPO
from .state import GameState


def _longest_trail(state: GameState, pid: int) -> int:
    """Longest trail (no repeated edges) for pid's roads, blocked by opponent
    settlements/cities at intermediate vertices.
    """
    my_edges = [(eid, TOPO.edges[eid]) for eid in range(TOPO.n_edges)
                if state.edge_owner[eid] == pid]
    m = len(my_edges)
    if m == 0:
        return 0

    # can pass through v unless an opponent building sits there
    passable = [True] * TOPO.n_vertices
    for v, owner in enumerate(state.vertex_owner):
        if owner != -1 and owner != pid:
            passable[v] = False

    # vertex -> list of (edge_bit, other_vertex) for this player's roads
    inc = [[] for _ in range(TOPO.n_vertices)]
    for bit, (eid, (a, b)) in enumerate(my_edges):
        ebit = 1 << bit
        inc[a].append((ebit, b))
        inc[b].append((ebit, a))

    nmasks = 1 << m
    dp = [0] * nmasks
    bits = [0] * nmasks
    for mask in range(1, nmasks):
        bits[mask] = bits[mask >> 1] + (mask & 1)

    for bit, (eid, (a, b)) in enumerate(my_edges):
        dp[1 << bit] |= (1 << a) | (1 << b)

    best = 0
    for mask in range(1, nmasks):
        verts = dp[mask]
        if not verts:
            continue
        pc = bits[mask]
        if pc > best:
            best = pc
        vmask = verts
        while vmask:
            lsb = vmask & -vmask
            v = lsb.bit_length() - 1
            if passable[v]:
                for ebit, other in inc[v]:
                    if mask & ebit:
                        continue
                    dp[mask | ebit] |= 1 << other
            vmask ^= lsb
    return best


def _assign_longest_road(state: GameState) -> None:
    """Award the Longest Road special card using official tie/incumbent rules."""
    lengths = [p.longest_road_len for p in state.players]
    current = next((i for i, p in enumerate(state.players) if p.has_longest_road), -1)
    best = max(lengths)
    if best < 5:
        for p in state.players:
            p.has_longest_road = False
        return
    leaders = [i for i, L in enumerate(lengths) if L == best]
    if current in leaders:
        return
    if len(leaders) == 1:
        for i, p in enumerate(state.players):
            p.has_longest_road = i == leaders[0]
    else:
        # tie with no incumbent: card returns to supply
        for p in state.players:
            p.has_longest_road = False


def recompute(state: GameState, pid: int) -> None:
    """Recompute one player's longest road and refresh card assignment."""
    state.players[pid].longest_road_len = _longest_trail(state, pid)
    _assign_longest_road(state)


def on_road(state: GameState, pid: int) -> None:
    """Call after a road is placed for pid."""
    recompute(state, pid)


def on_settlement(state: GameState, pid: int, vid: int) -> None:
    """Call after a settlement is placed; an opponent settlement may break roads."""
    affected = set()
    for eid in TOPO.vertex_edges[vid]:
        owner = state.edge_owner[eid]
        if owner != -1 and owner != pid:
            affected.add(owner)
    for opid in affected:
        state.players[opid].longest_road_len = _longest_trail(state, opid)
    # owner placing their own settlement may also split their own road?  No: own
    # settlements do not block your own road network.
    _assign_longest_road(state)
