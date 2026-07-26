"""Core Catan rules: legality, production, building, robber, dev cards, trading."""
from __future__ import annotations

from .board import RESOURCES, TOPO
from . import longest_road as lr
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

# Standard piece supply per player.
MAX_ROADS = 15
MAX_SETTLEMENTS = 5
MAX_CITIES = 4


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

def can_build_settlement(state: GameState, pid: int, vid: int, setup: bool = False) -> bool:
    if state.vertex_type[vid] != EMPTY:
        return False
    for nb in TOPO.vertex_neighbors[vid]:
        if state.vertex_type[nb] != EMPTY:
            return False
    p = state.players[pid]
    if p.settlements >= MAX_SETTLEMENTS:
        return False
    if setup:
        return True
    if not any(state.edge_owner[e] == pid for e in TOPO.vertex_edges[vid]):
        return False
    return p.can_afford(COST_SETTLEMENT)


def can_build_road(state: GameState, pid: int, eid: int,
                   setup_vertex: int | None = None, free: bool = False) -> bool:
    if state.edge_owner[eid] != -1:
        return False
    a, b = TOPO.edges[eid]
    if setup_vertex is not None:
        return setup_vertex in (a, b)
    p = state.players[pid]
    if p.roads >= MAX_ROADS or (not free and not p.can_afford(COST_ROAD)):
        return False
    return _road_connects(state, pid, a) or _road_connects(state, pid, b)


def _road_connects(state: GameState, pid: int, v: int) -> bool:
    """A new road may attach at vertex v if the player has a building there,
    or a road there and no opponent building blocks the vertex."""
    owner = state.vertex_owner[v]
    if owner == pid:
        return True
    if owner != -1:
        return False
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
# Building (mutations). ``free`` skips cost/limits (used in setup and cards).
# ---------------------------------------------------------------------------

def _spend(state: GameState, pid: int, cost: dict) -> None:
    """Take resources from player and return them to the bank."""
    p = state.players[pid]
    p.pay(cost)
    for r, n in cost.items():
        state.bank[r] += n


def place_settlement(state: GameState, pid: int, vid: int, free: bool = False) -> None:
    assert can_build_settlement(state, pid, vid, setup=free), "illegal settlement"
    p = state.players[pid]
    if not free:
        _spend(state, pid, COST_SETTLEMENT)
    state.vertex_owner[vid] = pid
    state.vertex_type[vid] = SETTLEMENT
    p.settlements += 1
    lr.on_settlement(state, pid, vid)


def place_road(state: GameState, pid: int, eid: int, free: bool = False,
               setup_vertex: int | None = None) -> None:
    sv = setup_vertex if free else None
    assert can_build_road(state, pid, eid, sv, free=free), "illegal road"
    p = state.players[pid]
    if not free:
        _spend(state, pid, COST_ROAD)
    state.edge_owner[eid] = pid
    p.roads += 1
    lr.on_road(state, pid)


def upgrade_city(state: GameState, pid: int, vid: int) -> None:
    assert can_build_city(state, pid, vid), "illegal city"
    p = state.players[pid]
    _spend(state, pid, COST_CITY)
    state.vertex_type[vid] = CITY
    p.settlements -= 1
    p.cities += 1


# ---------------------------------------------------------------------------
# Dice / production / robber / discards
# ---------------------------------------------------------------------------

def roll_dice(state: GameState) -> int:
    d = state.rng_dice.randint(1, 6) + state.rng_dice.randint(1, 6)
    state.dice_last_roll = d
    state.turn_has_rolled = True
    return d


def produce(state: GameState, roll: int) -> None:
    """Distribute resources for a non-7 roll, honouring bank shortfall rules."""
    b = state.board
    claims = {r: [] for r in RESOURCES}
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
            claims[res].append((owner, amount))

    for res, entitlements in claims.items():
        if not entitlements:
            continue
        total = sum(a for _, a in entitlements)
        if total <= state.bank[res]:
            for owner, amount in entitlements:
                state.players[owner].gain(res, amount)
                state.bank[res] -= amount
        else:
            # official shortfall: nobody gets it unless exactly one player is owed,
            # who receives whatever remains.
            if len({o for o, _ in entitlements}) == 1:
                owner = entitlements[0][0]
                give = min(state.bank[res], total)
                state.players[owner].gain(res, give)
                state.bank[res] -= give
            # otherwise: no distribution of this resource


def discard_cards(state: GameState, pid: int, discard: dict) -> None:
    """Discard chosen resources from a hand > 7.``discard`` has floor(n/2) cards."""
    p = state.players[pid]
    n = p.hand_size() // 2
    total = sum(discard.values())
    assert total == n, f"must discard {n}, got {total}"
    for r, c in discard.items():
        assert p.resources[r] >= c, f"cannot discard {c} {r}"
        p.resources[r] -= c
        state.bank[r] += c


def move_robber(state: GameState, pid: int, target_hex: int, victim_pid: int) -> None:
    """Move robber to a different hex and steal one card from victim."""
    assert target_hex != state.board.robber_hex, "robber must move to a different hex"
    state.board.robber_hex = target_hex
    if victim_pid == -1:
        return
    victim = state.players[victim_pid]
    pool = []
    for r, c in victim.resources.items():
        pool += [r] * c
    if not pool:
        return
    r = state.rng_dice.choice(pool)
    victim.resources[r] -= 1
    state.players[pid].gain(r, 1)


# ---------------------------------------------------------------------------
# Dev cards (all five types)
# ---------------------------------------------------------------------------

def buy_dev_card(state: GameState, pid: int) -> str | None:
    p = state.players[pid]
    if not state.dev_deck or not p.can_afford(COST_DEV):
        return None
    _spend(state, pid, COST_DEV)
    kind = state.dev_deck.pop()
    p.dev_cards[kind] += 1
    return kind


def play_knight(state: GameState, pid: int, target_hex: int, victim_pid: int) -> None:
    p = state.players[pid]
    assert p.dev_cards["knight"] > 0, "no knight to play"
    assert not state.dev_card_bought_this_turn and not state.dev_card_played_this_turn
    p.dev_cards["knight"] -= 1
    p.played_dev["knight"] += 1
    state.dev_card_played_this_turn = True
    move_robber(state, pid, target_hex, victim_pid)
    _recompute_largest_army(state)


def play_road_building(state: GameState, pid: int, eids: tuple) -> int:
    p = state.players[pid]
    assert p.dev_cards["road_building"] > 0, "no road building card"
    assert not state.dev_card_bought_this_turn and not state.dev_card_played_this_turn
    p.dev_cards["road_building"] -= 1
    p.played_dev["road_building"] += 1
    state.dev_card_played_this_turn = True
    placed = 0
    for eid in eids:
        if eid is None:
            continue
        if not can_build_road(state, pid, eid, free=True):
            break
        place_road(state, pid, eid, free=True)
        placed += 1
    return placed


def play_year_of_plenty(state: GameState, pid: int, res1: str, res2: str) -> None:
    p = state.players[pid]
    assert p.dev_cards["year_of_plenty"] > 0, "no year of plenty card"
    assert not state.dev_card_bought_this_turn and not state.dev_card_played_this_turn
    p.dev_cards["year_of_plenty"] -= 1
    p.played_dev["year_of_plenty"] += 1
    state.dev_card_played_this_turn = True
    for res in (res1, res2):
        if state.bank[res] > 0:
            state.bank[res] -= 1
            p.gain(res, 1)


def play_monopoly(state: GameState, pid: int, resource: str) -> None:
    p = state.players[pid]
    assert p.dev_cards["monopoly"] > 0, "no monopoly card"
    assert not state.dev_card_bought_this_turn and not state.dev_card_played_this_turn
    p.dev_cards["monopoly"] -= 1
    p.played_dev["monopoly"] += 1
    state.dev_card_played_this_turn = True
    for opp in state.players:
        if opp.pid == pid:
            continue
        n = opp.resources[resource]
        if n:
            opp.resources[resource] -= n
            p.gain(resource, n)


def reveal_vp(state: GameState, pid: int) -> None:
    """Reveal hidden VP cards (public from now on)."""
    p = state.players[pid]
    n = p.dev_cards["vp"]
    if n:
        p.dev_cards["vp"] -= n
        p.played_dev["vp"] += n


# ---------------------------------------------------------------------------
# Trading (bank/port and player-to-player)
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
    if state.bank[get] <= 0:
        return False
    p.resources[give] -= ratio
    state.bank[give] += ratio
    state.bank[get] -= 1
    p.gain(get, 1)
    return True


def _normalize_bundle(bundle: dict) -> dict:
    return {r: max(0, int(bundle.get(r, 0))) for r in RESOURCES}


def can_offer_trade(state: GameState, offerer: int, responder: int,
                    give: dict, want: dict) -> bool:
    if responder == offerer or not (0 <= responder < state.n_players):
        return False
    if offerer < 0 or offerer >= state.n_players:
        return False
    give = _normalize_bundle(give)
    want = _normalize_bundle(want)
    if sum(give.values()) == 0 or sum(want.values()) == 0:
        return False
    o = state.players[offerer]
    r = state.players[responder]
    for res in RESOURCES:
        if o.resources[res] < give[res]:
            return False
        if r.resources[res] < want[res]:
            return False
    return True


def apply_player_trade(state: GameState, offerer: int, responder: int,
                       give: dict, want: dict) -> None:
    """Execute a validated player trade: offerer gives ``give`` and receives ``want``."""
    give = _normalize_bundle(give)
    want = _normalize_bundle(want)
    assert can_offer_trade(state, offerer, responder, give, want)
    o = state.players[offerer]
    r = state.players[responder]
    for res in RESOURCES:
        o.resources[res] -= give[res]
        r.resources[res] += give[res]
        r.resources[res] -= want[res]
        o.resources[res] += want[res]


# ---------------------------------------------------------------------------
# Largest army / win
# ---------------------------------------------------------------------------

def _recompute_largest_army(state: GameState) -> None:
    counts = [p.knights_played for p in state.players]
    current = next((i for i, p in enumerate(state.players) if p.has_largest_army), -1)
    best = max(counts)
    if best < 3:
        for p in state.players:
            p.has_largest_army = False
        return
    leaders = [i for i, c in enumerate(counts) if c == best]
    if current in leaders:
        # Incumbent keeps the card on a tie; clear any stale holders (should be none).
        for i, p in enumerate(state.players):
            p.has_largest_army = i == current
        return
    # No incumbent, or incumbent has dropped below the leaders.
    for p in state.players:
        p.has_largest_army = False
    if len(leaders) == 1:
        state.players[leaders[0]].has_largest_army = True


def check_winner(state: GameState) -> int:
    for pid, p in enumerate(state.players):
        if p.total_vp() >= WIN_VP:
            state.winner = pid
            return pid
    return -1
