"""Three greedy heuristic bots with different priorities: high-pip settlements,
city-rushing, and road/settlement expansion. They're intentionally imperfect so
self-play produces varied, non-degenerate games.
"""
from __future__ import annotations

from ..engine import rules as R
from ..engine.board import PIPS, RESOURCES, TOPO
from ..engine.state import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    GameState,
)

_MAX_ACTIONS = 20  # safety cap on actions per turn


def _spot_value(state: GameState, vid: int) -> float:
    """Score a settlement spot by production pips + resource diversity."""
    b = state.board
    pips = 0
    kinds = set()
    for h in TOPO.vertex_hexes[vid]:
        if b.hex_resource[h] in RESOURCES:
            pips += PIPS.get(b.hex_number[h], 0)
            kinds.add(b.hex_resource[h])
    port_bonus = 1.5 if vid in b.vertex_port else 0.0
    return pips + 0.5 * len(kinds) + port_bonus


def _try_afford(state: GameState, pid: int, cost: dict) -> bool:
    """Attempt to afford ``cost``, using bank/port trades on surplus. Mutating."""
    p = state.players[pid]
    if p.can_afford(cost):
        return True
    for _ in range(10):
        deficit = {r: cost.get(r, 0) - p.resources[r] for r in cost}
        deficit = {r: n for r, n in deficit.items() if n > 0}
        if not deficit:
            return True
        want = next(iter(deficit))
        # find a resource with enough surplus to trade away
        traded = False
        for give in RESOURCES:
            if give in cost:  # don't trade away what we need for this build
                continue
            ratio = R.trade_ratio(state, pid, give)
            if p.resources[give] >= ratio:
                if R.bank_trade(state, pid, give, want):
                    traded = True
                    break
        if not traded:
            return False
    return p.can_afford(cost)


def _best_robber_hex(state: GameState, pid: int) -> int:
    """Hex that hurts opponents most (their pips) and avoids our own tiles."""
    b = state.board
    best_h, best_score = b.robber_hex, -1.0
    for h in range(TOPO.n_hexes):
        if h == b.robber_hex or b.hex_resource[h] not in RESOURCES:
            continue
        opp = own = 0
        for v in TOPO.vertex_hexes[h]:
            o = state.vertex_owner[v]
            if o == -1:
                continue
            mult = 2 if state.vertex_type[v] == R.CITY else 1
            pip = PIPS.get(b.hex_number[h], 0) * mult
            if o == pid:
                own += pip
            else:
                opp += pip
        score = opp - 0.5 * own
        if score > best_score:
            best_score, best_h = score, h
    return best_h


class BaseBot:
    name = "base"

    def setup_settlement(self, state: GameState, pid: int) -> int:
        spots = R.legal_settlement_spots(state, pid, setup=True)
        return max(spots, key=lambda v: _spot_value(state, v))

    def setup_road(self, state: GameState, pid: int, settlement_vid: int) -> int:
        edges = R.legal_road_edges(state, pid, setup_vertex=settlement_vid)
        # extend toward the best neighboring empty spot
        def edge_val(e):
            a, b = TOPO.edges[e]
            other = b if a == settlement_vid else a
            return _spot_value(state, other)
        return max(edges, key=edge_val)

    def choose_robber(self, state: GameState, pid: int) -> int:
        return _best_robber_hex(state, pid)

    def _build_best_settlement(self, state, pid) -> bool:
        spots = R.legal_settlement_spots(state, pid)
        if not spots:
            return False
        if not _try_afford(state, pid, COST_SETTLEMENT):
            return False
        vid = max(spots, key=lambda v: _spot_value(state, v))
        if R.can_build_settlement(state, pid, vid):
            R.place_settlement(state, pid, vid)
            return True
        return False

    def _build_best_city(self, state, pid) -> bool:
        spots = R.legal_city_spots(state, pid)
        if not spots:
            return False
        if not _try_afford(state, pid, COST_CITY):
            return False
        vid = max(spots, key=lambda v: _spot_value(state, v))
        if R.can_build_city(state, pid, vid):
            R.upgrade_city(state, pid, vid)
            return True
        return False

    def _build_road_toward_expansion(self, state, pid) -> bool:
        edges = R.legal_road_edges(state, pid)
        if not edges:
            return False
        if not _try_afford(state, pid, COST_ROAD):
            return False
        # prefer roads whose far endpoint is a legal-ish future settlement
        def edge_val(e):
            a, b = TOPO.edges[e]
            return max(_spot_value(state, a), _spot_value(state, b))
        eid = max(edges, key=edge_val)
        if R.can_build_road(state, pid, eid):
            R.place_road(state, pid, eid)
            return True
        return False

    def _buy_dev(self, state, pid) -> bool:
        if not _try_afford(state, pid, COST_DEV):
            return False
        return R.buy_dev_card(state, pid) is not None

    def _maybe_play_knight(self, state, pid) -> bool:
        p = state.players[pid]
        if p.dev_knight > 0:
            R.play_knight(state, pid, self.choose_robber(state, pid))
            return True
        return False

    def act(self, state: GameState, pid: int) -> None:
        raise NotImplementedError


class PipMaxBot(BaseBot):
    """Expand to high-pip settlement spots; upgrade to city opportunistically."""
    name = "pipmax"

    def act(self, state, pid):
        self._maybe_play_knight(state, pid)
        for _ in range(_MAX_ACTIONS):
            if self._build_best_settlement(state, pid):
                continue
            if self._build_best_city(state, pid):
                continue
            if self._build_road_toward_expansion(state, pid):
                continue
            if self._buy_dev(state, pid):  # resource sink -> VP progress
                continue
            break


class CityRushBot(BaseBot):
    """Cities first, then settlements; buys dev cards for knights/VP."""
    name = "cityrush"

    def act(self, state, pid):
        self._maybe_play_knight(state, pid)
        for _ in range(_MAX_ACTIONS):
            if self._build_best_city(state, pid):
                continue
            if self._build_best_settlement(state, pid):
                continue
            if self._buy_dev(state, pid):
                continue
            if self._build_road_toward_expansion(state, pid):
                continue
            break


class ExpanderBot(BaseBot):
    """Roads + settlements to spread out and chase longest road."""
    name = "expander"

    def act(self, state, pid):
        self._maybe_play_knight(state, pid)
        for _ in range(_MAX_ACTIONS):
            if self._build_best_settlement(state, pid):
                continue
            if self._build_road_toward_expansion(state, pid):
                continue
            if self._build_best_city(state, pid):
                continue
            if self._buy_dev(state, pid):  # resource sink -> VP progress
                continue
            break


BOT_STYLES = [PipMaxBot, CityRushBot, ExpanderBot]


def make_bots(n_players: int, rng) -> list:
    """Random style per seat."""
    return [rng.choice(BOT_STYLES)() for _ in range(n_players)]
