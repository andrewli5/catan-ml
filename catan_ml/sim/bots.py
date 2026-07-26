"""Greedy heuristic bots that choose one Action at a time.
They route through catan_ml.engine.actions so every state mutation stays on the
new action path.
"""
from __future__ import annotations

from ..engine import actions
from ..engine import phases
from ..engine import rules as R
from ..engine.actions import Action
from ..engine.board import PIPS, RESOURCES, TOPO
from ..engine.state import (
    COST_CITY,
    COST_DEV,
    COST_ROAD,
    COST_SETTLEMENT,
    GameState,
)


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


def _edge_value(state: GameState, eid: int) -> float:
    """Prefer roads whose far endpoint is a strong future settlement."""
    a, b = TOPO.edges[eid]
    return max(_spot_value(state, a), _spot_value(state, b))


def _robber_victims(state: GameState, pid: int, hex_id: int) -> list:
    victims = []
    for v in TOPO.hex_vertices[hex_id]:
        o = state.vertex_owner[v]
        if o != -1 and o != pid and state.players[o].hand_size() > 0:
            victims.append(o)
    return victims


def _best_robber_hex(state: GameState, pid: int) -> int:
    """Hex that hurts opponents most (their pips) and avoids our own tiles."""
    b = state.board
    candidates = []
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
        candidates.append((score, h))
    if not candidates:
        # Fallback: should never happen on a normal board.
        return (b.robber_hex + 1) % TOPO.n_hexes
    return max(candidates, key=lambda x: x[0])[1]


class BaseBot:
    name = "base"
    build_order = ["settlement", "city", "road", "dev"]

    def setup_settlement(self, state: GameState, pid: int) -> int:
        spots = R.legal_settlement_spots(state, pid, setup=True)
        return max(spots, key=lambda v: _spot_value(state, v))

    def setup_road(self, state: GameState, pid: int, settlement_vid: int) -> int:
        edges = R.legal_road_edges(state, pid, setup_vertex=settlement_vid)
        return max(edges, key=lambda e: _edge_value(state, e))

    def choose_robber(self, state: GameState, pid: int) -> int:
        return _best_robber_hex(state, pid)

    def _robber_action(self, state: GameState, pid: int, kind: str) -> Action:
        target = self.choose_robber(state, pid)
        victims = _robber_victims(state, pid, target)
        victim = max(victims, key=lambda o: state.players[o].hand_size()) if victims else -1
        return Action(kind, (target, victim))

    def _setup_settlement_action(self, state: GameState, pid: int) -> Action:
        vid = self.setup_settlement(state, pid)
        return Action("setup_settlement", (vid,))

    def _setup_road_action(self, state: GameState, pid: int) -> Action:
        vid = state.setup_last_settlement
        eid = self.setup_road(state, pid, vid)
        return Action("setup_road", (eid,))

    def _discard_action(self, state: GameState, pid: int) -> Action:
        p = state.players[pid]
        n = p.hand_size() // 2
        discard = {r: 0 for r in RESOURCES}
        for r in sorted(RESOURCES, key=lambda r: -p.resources[r]):
            if n <= 0:
                break
            take = min(p.resources[r], n)
            discard[r] = take
            n -= take
        return Action("discard", (discard,))

    def _affording_trade_action(self, state: GameState, pid: int, cost: dict) -> Action | None:
        """Return one bank/port trade that moves toward affording ``cost``."""
        p = state.players[pid]
        deficit = {r: cost.get(r, 0) - p.resources[r] for r in cost
                   if cost.get(r, 0) > p.resources[r]}
        if not deficit:
            return None
        want = next(iter(deficit))
        for give in RESOURCES:
            if give in cost:
                continue
            ratio = R.trade_ratio(state, pid, give)
            if p.resources[give] >= ratio and state.bank[want] > 0:
                return Action("bank_trade", (give, want))
        return None

    def _build_action(self, state: GameState, pid: int, btype: str) -> Action | None:
        p = state.players[pid]
        if btype == "settlement":
            spots = R.legal_settlement_spots(state, pid)
            if not spots:
                return None
            if p.can_afford(COST_SETTLEMENT):
                vid = max(spots, key=lambda v: _spot_value(state, v))
                return Action("build_settlement", (vid,))
            return self._affording_trade_action(state, pid, COST_SETTLEMENT)
        if btype == "city":
            spots = R.legal_city_spots(state, pid)
            if not spots:
                return None
            if p.can_afford(COST_CITY):
                vid = max(spots, key=lambda v: _spot_value(state, v))
                return Action("build_city", (vid,))
            return self._affording_trade_action(state, pid, COST_CITY)
        if btype == "road":
            edges = R.legal_road_edges(state, pid)
            if not edges:
                return None
            if p.can_afford(COST_ROAD):
                eid = max(edges, key=lambda e: _edge_value(state, e))
                return Action("build_road", (eid,))
            return self._affording_trade_action(state, pid, COST_ROAD)
        if btype == "dev":
            if not state.dev_card_bought_this_turn and state.dev_deck and p.can_afford(COST_DEV):
                return Action("buy_dev", ())
        return None

    def _main_action(self, state: GameState, pid: int) -> Action:
        p = state.players[pid]
        if (not state.dev_card_played_this_turn and not state.dev_card_bought_this_turn
                and p.dev_cards["knight"]):
            return self._robber_action(state, pid, "play_knight")
        for btype in self.build_order:
            action = self._build_action(state, pid, btype)
            if action is not None:
                return action
        return Action("end_turn", ())

    def _trade_response_action(self, state: GameState, pid: int) -> Action:
        return Action("reject_trade", ())

    def choose_action(self, state: GameState, pid: int | None = None) -> Action:
        pid = pid if pid is not None else actions.active_player(state)
        phase = state.phase
        if phase == phases.SETUP:
            if state.setup_road_pending:
                return self._setup_road_action(state, pid)
            return self._setup_settlement_action(state, pid)
        if phase == phases.PRE_ROLL:
            p = state.players[pid]
            if (not state.dev_card_played_this_turn and not state.dev_card_bought_this_turn
                    and p.dev_cards["knight"]):
                return self._robber_action(state, pid, "play_knight")
            return Action("roll", ())
        if phase == phases.DISCARD:
            return self._discard_action(state, pid)
        if phase == phases.ROBBER:
            return self._robber_action(state, pid, "robber")
        if phase == phases.MAIN:
            return self._main_action(state, pid)
        if phase == phases.TRADE_RESPONSE:
            return self._trade_response_action(state, pid)
        return Action("end_turn", ())


class PipMaxBot(BaseBot):
    """Expand to high-pip settlement spots; upgrade to city opportunistically."""
    name = "pipmax"
    build_order = ["settlement", "city", "road", "dev"]


class CityRushBot(BaseBot):
    """Cities first, then settlements; buys dev cards for knights/VP."""
    name = "cityrush"
    build_order = ["city", "settlement", "dev", "road"]


class ExpanderBot(BaseBot):
    """Roads + settlements to spread out and chase longest road."""
    name = "expander"
    build_order = ["settlement", "road", "city", "dev"]


BOT_STYLES = [PipMaxBot, CityRushBot, ExpanderBot]


def make_bots(n_players: int, rng) -> list:
    """Random style per seat."""
    return [rng.choice(BOT_STYLES)() for _ in range(n_players)]
