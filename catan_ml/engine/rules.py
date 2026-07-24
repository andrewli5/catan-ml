"""Core Catan rules: legality, production, building, robber, dev cards.

Simplifications (v1, agreed in PLAN.md):
- Dev cards: only knights (move robber + largest army) and VP cards. No
  year-of-plenty / monopoly / road-building.
- Trading: bank/port only (4:1, 3:1 generic, 2:1 resource). No player trades.
- Robber discard on 7: players with >7 cards discard a random half.
"""
from __future__ import annotations

from .board import PIPS, RESOURCES, TOPO
from .state import (
    CITY,
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    EMPTY,
    SETTLEMENT,
    WIN_VP,
    GameState,
)

# Standard piece supply per player. Enforced so road networks stay bounded
# (an unbounded road count makes the longest-trail search blow up).
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4

# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

def can_build_settlement(state: GameState, pid: int, vid: int, setup: bool = False) -> bool:
    if state.vertex_type[vid] != EMPTY:
        return False
    # distance rule: no adjacent vertex may be occupied
    for nb in TOPO.vertex_neighbors[vid]:
        if state.vertex_type[nb] != EMPTY:
            return False
    p = state.players[pid]
    if p.settlements >= MAX_SETTLEMENTS:
        return False
    if setup:
        return True
    # must connect to one of the player's own roads
    if not any(state.edge_owner[e] == pid for e in TOPO.vertex_edges[vid]):
        return False
    return p.can_afford(COST_SETTLEMENT)


def can_build_road(state: GameState, pid: int, eid: int, setup_vertex: int | None = None) -> bool:
    if state.edge_owner[eid] != -1:
        return False
    a, b = TOPO.edges[eid]
    if setup_vertex is not None:
        # setup road must touch the just-placed settlement
        return setup_vertex in (a, b)
    p = state.players[pid]
    if p.roads >= MAX_ROADS or not p.can_afford(COST_ROAD):
        return False
    return _road_connects(state, pid, a) or _road_connects(state, pid, b)


def _road_connects(state: GameState, pid: int, v: int) -> bool:
    """A new road may attach at vertex v if the player has a building there,
    or a road there and no opponent building blocks the vertex."""
    owner = state.vertex_owner[v]
    if owner == pid:
        return True
    if owner != -1:
        return False  # opponent building blocks connection through this vertex
    return any(state.edge_owner[e] == pid for e in TOPO.vertex_edges[v])


def can_build_city(state: GameState, pid: int, vid: int) -> bool:
    if state.vertex_type[vid] != SETTLEMENT or state.vertex_owner[vid] != pid:
        return False
    p = state.players[pid]
    if p.cities >= MAX_CITIES:
        return False
    return p.can_afford(COST_CITY)


def legal_settlement_spots(state: GameState, pid: int, setup: bool = False) -> list:
    return [v for v in range(TOPO.n_vertices) if can_build_settlement(state, pid, v, setup)]


def legal_road_edges(state: GameState, pid: int, setup_vertex: int | None = None) -> list:
    return [e for e in range(TOPO.n_edges) if can_build_road(state, pid, e, setup_vertex)]


def legal_city_spots(state: GameState, pid: int) -> list:
    return [v for v in range(TOPO.n_vertices) if can_build_city(state, pid, v)]


# ---------------------------------------------------------------------------
# Building (mutations). ``free`` skips cost/limits (used in setup).
# ---------------------------------------------------------------------------

def place_settlement(state: GameState, pid: int, vid: int, free: bool = False) -> None:
    assert can_build_settlement(state, pid, vid, setup=free), "illegal settlement"
    p = state.players[pid]
    if not free:
        p.pay(COST_SETTLEMENT)
    state.vertex_owner[vid] = pid
    state.vertex_type[vid] = SETTLEMENT
    p.settlements += 1
    _recompute_longest_road(state)


def place_road(state: GameState, pid: int, eid: int, free: bool = False,
               setup_vertex: int | None = None) -> None:
    assert can_build_road(state, pid, eid, setup_vertex if free else None), "illegal road"
    p = state.players[pid]
    if not free:
        p.pay(COST_ROAD)
    state.edge_owner[eid] = pid
    p.roads += 1
    _recompute_longest_road(state)


def upgrade_city(state: GameState, pid: int, vid: int) -> None:
    assert can_build_city(state, pid, vid), "illegal city"
    p = state.players[pid]
    p.pay(COST_CITY)
    state.vertex_type[vid] = CITY
    p.settlements -= 1
    p.cities += 1


# ---------------------------------------------------------------------------
# Dice / production / robber
# ---------------------------------------------------------------------------

def roll_dice(state: GameState) -> int:
    d = state.rng.randint(1, 6) + state.rng.randint(1, 6)
    state.dice_last_roll = d
    return d


def produce(state: GameState, roll: int) -> None:
    """Distribute resources for a non-7 roll to adjacent settlements/cities."""
    b = state.board
    for h in range(TOPO.n_hexes):
        if b.hex_number[h] != roll or h == b.robber_hex:
            continue
        res = b.hex_resource[h]
        if res not in RESOURCES:
            continue
        for v in TOPO.hex_vertices[h]:
            owner = state.vertex_owner[v]
            if owner == -1:
                continue
            amount = 2 if state.vertex_type[v] == CITY else 1
            state.players[owner].gain(res, amount)


def handle_robber(state: GameState, pid: int, target_hex: int) -> None:
    """Move robber and steal one random resource from an adjacent opponent."""
    # discard for anyone holding >7 (random half, rounded down)
    for p in state.players:
        if p.hand_size() > 7:
            _discard_half(state, p)
    state.board.robber_hex = target_hex
    victims = set()
    for v in TOPO.hex_vertices[target_hex]:
        o = state.vertex_owner[v]
        if o != -1 and o != pid and state.players[o].hand_size() > 0:
            victims.add(o)
    if victims:
        victim = state.rng.choice(sorted(victims))
        _steal_one(state, robber=pid, victim=victim)


def _discard_half(state: GameState, p) -> None:
    n = p.hand_size() // 2
    pool = []
    for r, c in p.resources.items():
        pool += [r] * c
    state.rng.shuffle(pool)
    for r in pool[:n]:
        p.resources[r] -= 1


def _steal_one(state: GameState, robber: int, victim: int) -> None:
    vp = state.players[victim]
    pool = []
    for r, c in vp.resources.items():
        pool += [r] * c
    if not pool:
        return
    r = state.rng.choice(pool)
    vp.resources[r] -= 1
    state.players[robber].gain(r, 1)


# ---------------------------------------------------------------------------
# Dev cards (knights + VP only)
# ---------------------------------------------------------------------------

def buy_dev_card(state: GameState, pid: int) -> str | None:
    p = state.players[pid]
    if not state.dev_deck or not p.can_afford(COST_DEV):
        return None
    p.pay(COST_DEV)
    kind = state.dev_deck.pop()
    if kind == "knight":
        p.dev_knight += 1
    else:
        p.dev_vp += 1
    return kind


def play_knight(state: GameState, pid: int, target_hex: int) -> None:
    p = state.players[pid]
    assert p.dev_knight > 0, "no knight to play"
    p.dev_knight -= 1
    p.knights_played += 1
    handle_robber(state, pid, target_hex)
    _recompute_largest_army(state)


# ---------------------------------------------------------------------------
# Trading (bank/port only)
# ---------------------------------------------------------------------------

def trade_ratio(state: GameState, pid: int, resource: str) -> int:
    """Best bank ratio for giving up ``resource`` (4, 3, or 2)."""
    ratio = 4
    for v in range(TOPO.n_vertices):
        if state.vertex_owner[v] != pid:
            continue
        port = state.board.vertex_port.get(v)
        if port is None:
            continue
        if port == "3:1":
            ratio = min(ratio, 3)
        elif port == f"2:1_{resource}":
            ratio = min(ratio, 2)
    return ratio


def bank_trade(state: GameState, pid: int, give: str, get: str) -> bool:
    p = state.players[pid]
    ratio = trade_ratio(state, pid, give)
    if p.resources[give] < ratio:
        return False
    p.resources[give] -= ratio
    p.gain(get, 1)
    return True


# ---------------------------------------------------------------------------
# Longest road / largest army / win
# ---------------------------------------------------------------------------

def _recompute_longest_road(state: GameState) -> None:
    lengths = [_longest_trail(state, pid) for pid in range(state.n_players)]
    for pid, p in enumerate(state.players):
        p.longest_road_len = lengths[pid]
    # holder: strictly-longest with >=5; keep current holder on ties
    current = next((i for i, p in enumerate(state.players) if p.has_longest_road), -1)
    best = max(lengths)
    if best < 5:
        for p in state.players:
            p.has_longest_road = False
        return
    leaders = [i for i, L in enumerate(lengths) if L == best]
    if current in leaders:
        return  # incumbent keeps it on tie
    if len(leaders) == 1:
        for i, p in enumerate(state.players):
            p.has_longest_road = i == leaders[0]


def _longest_trail(state: GameState, pid: int) -> int:
    """Longest trail (no repeated edges) in the player's road network,
    broken by opponent buildings at intermediate vertices."""
    my_edges = [e for e in range(TOPO.n_edges) if state.edge_owner[e] == pid]
    if not my_edges:
        return 0
    # vertex -> list of (neighbor_vertex, edge_id) usable by this player
    adj = {}
    for e in my_edges:
        a, b = TOPO.edges[e]
        adj.setdefault(a, []).append((b, e))
        adj.setdefault(b, []).append((a, e))

    def passable(v: int) -> bool:
        # can traverse through v unless an opponent building sits there
        o = state.vertex_owner[v]
        return o == -1 or o == pid

    best = 0
    # Piece limits already bound the road graph; this step cap is just a guard
    # so the trail search can't run away if those limits ever change.
    budget = 200_000

    def dfs(v: int, used: set) -> None:
        nonlocal best, budget
        best = max(best, len(used))
        if budget <= 0 or not passable(v):
            return
        for nb, e in adj.get(v, ()):
            if e in used:
                continue
            budget -= 1
            used.add(e)
            dfs(nb, used)
            used.remove(e)

    for start in list(adj.keys()):
        dfs(start, set())
    return best


def _recompute_largest_army(state: GameState) -> None:
    counts = [p.knights_played for p in state.players]
    current = next((i for i, p in enumerate(state.players) if p.has_largest_army), -1)
    best = max(counts)
    if best < 3:
        return
    leaders = [i for i, c in enumerate(counts) if c == best]
    if current in leaders:
        return
    if len(leaders) == 1:
        for i, p in enumerate(state.players):
            p.has_largest_army = i == leaders[0]


def check_winner(state: GameState) -> int:
    for pid, p in enumerate(state.players):
        if p.total_vp() >= WIN_VP:
            state.winner = pid
            return pid
    return -1
