"""Action ADT, legal action enumeration, and apply_action dispatcher.

Every mutation of game state in Phase 1 flows through apply_action.
"""
from __future__ import annotations

from .board import RESOURCES, TOPO
from . import phases
from . import rules as R
from .state import COST_DEV, GameState


def _freeze(obj):
    """Return a hashable snapshot of a nested structure of dicts/tuples/lists."""
    if isinstance(obj, dict):
        return frozenset((k, _freeze(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


class Action:
    """Hashable, serializable action token with kind + positional args."""

    __slots__ = ("kind", "args")

    def __init__(self, kind: str, args: tuple):
        self.kind = kind
        self.args = args

    def __repr__(self) -> str:
        return f"Action({self.kind!r}, {self.args!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Action) and self.kind == other.kind \
            and _freeze(self.args) == _freeze(other.args)

    def __hash__(self) -> int:
        return hash((self.kind, _freeze(self.args)))


def _a(kind: str, *args):
    return Action(kind, args)


def active_player(state: GameState) -> int:
    return phases.active_player(state)


def _robber_victims(state: GameState, pid: int, hex_id: int) -> list:
    victims = set()
    for v in TOPO.hex_vertices[hex_id]:
        o = state.vertex_owner[v]
        if o != -1 and o != pid and state.players[o].hand_size() > 0:
            victims.add(o)
    return sorted(victims)


def _knight_or_robber_actions(state: GameState, pid: int, kind: str) -> list:
    actions = []
    for h in range(TOPO.n_hexes):
        if h == state.board.robber_hex:
            continue
        victims = _robber_victims(state, pid, h)
        if victims:
            for v in victims:
                actions.append(_a(kind, h, v))
        else:
            actions.append(_a(kind, h, -1))
    return actions


def _discard_actions(state: GameState, pid: int) -> list:
    p = state.players[pid]
    n = p.hand_size() // 2
    if n == 0:
        return [_a("discard", {r: 0 for r in RESOURCES})]
    actions = []

    def rec(idx: int, left: int, cur: dict) -> None:
        if idx == len(RESOURCES):
            if left == 0:
                actions.append(_a("discard", dict(cur)))
            return
        r = RESOURCES[idx]
        max_c = min(p.resources[r], left)
        for c in range(max_c + 1):
            cur[r] = c
            rec(idx + 1, left - c, cur)
        cur.pop(r, None)

    rec(0, n, {})
    return actions


def _road_building_actions(state: GameState, pid: int) -> list:
    p = state.players[pid]
    if p.dev_cards["road_building"] == 0:
        return []
    legal1 = [e for e in range(TOPO.n_edges) if R.can_build_road(state, pid, e, free=True)]
    if not legal1:
        return [_a("play_road_building", None, None)]
    actions = []
    for e1 in legal1:
        actions.append(_a("play_road_building", e1, None))
        tmp = state.clone()
        R.place_road(tmp, pid, e1, free=True)
        for e2 in range(TOPO.n_edges):
            if e2 != e1 and R.can_build_road(tmp, pid, e2, free=True):
                actions.append(_a("play_road_building", e1, e2))
    return actions


def _one_one_trade_dict(give_res: str, want_res: str) -> tuple:
    give = {r: 1 if r == give_res else 0 for r in RESOURCES}
    want = {r: 1 if r == want_res else 0 for r in RESOURCES}
    return give, want


def legal_actions(state: GameState) -> list:
    """Exhaustive legal actions for the current phase and active player."""
    pid = active_player(state)
    p = state.players[pid]

    if state.phase == phases.SETUP:
        if state.setup_road_pending:
            return [_a("setup_road", eid)
                    for eid in R.legal_road_edges(state, pid, state.setup_last_settlement)]
        return [_a("setup_settlement", vid)
                for vid in R.legal_settlement_spots(state, pid, setup=True)]

    if state.phase == phases.PRE_ROLL:
        acts = [_a("roll")]
        if not state.dev_card_played_this_turn and not state.dev_card_bought_this_turn:
            acts += _knight_or_robber_actions(state, pid, "play_knight")
        if p.dev_cards["vp"]:
            acts.append(_a("reveal_vp"))
        return acts

    if state.phase == phases.ROLL:
        return [_a("roll")]

    if state.phase == phases.DISCARD:
        return _discard_actions(state, pid)

    if state.phase == phases.ROBBER:
        return _knight_or_robber_actions(state, pid, "robber")

    if state.phase == phases.MAIN:
        acts = []
        for v in R.legal_settlement_spots(state, pid):
            acts.append(_a("build_settlement", v))
        for e in R.legal_road_edges(state, pid):
            acts.append(_a("build_road", e))
        for v in R.legal_city_spots(state, pid):
            acts.append(_a("build_city", v))

        if state.dev_deck and p.can_afford(COST_DEV):
            acts.append(_a("buy_dev"))

        if not state.dev_card_played_this_turn and not state.dev_card_bought_this_turn:
            if p.dev_cards["knight"]:
                acts += _knight_or_robber_actions(state, pid, "play_knight")
            acts += _road_building_actions(state, pid)
            if p.dev_cards["year_of_plenty"]:
                for i, r1 in enumerate(RESOURCES):
                    for r2 in RESOURCES[i:]:
                        if state.bank[r1] > 0 or state.bank[r2] > 0:
                            acts.append(_a("play_year_of_plenty", r1, r2))
            if p.dev_cards["monopoly"]:
                for r in RESOURCES:
                    acts.append(_a("play_monopoly", r))

        if p.dev_cards["vp"]:
            acts.append(_a("reveal_vp"))

        for give in RESOURCES:
            ratio = R.trade_ratio(state, pid, give)
            if p.resources[give] >= ratio:
                for get in RESOURCES:
                    if get == give or state.bank[get] <= 0:
                        continue
                    acts.append(_a("bank_trade", give, get))

        # Player trades: enumerate 1-for-1 offers for now.
        for target in range(state.n_players):
            if target == pid:
                continue
            t = state.players[target]
            for give_res in RESOURCES:
                if p.resources[give_res] == 0:
                    continue
                for want_res in RESOURCES:
                    if want_res == give_res or t.resources[want_res] == 0:
                        continue
                    give, want = _one_one_trade_dict(give_res, want_res)
                    acts.append(_a("offer_trade", give, want, target))

        acts.append(_a("end_turn"))
        return acts

    if state.phase == phases.TRADE_RESPONSE:
        offer = state.trade_offer
        acts = [_a("accept_trade"), _a("reject_trade")]
        responder = offer["responder"]
        r = state.players[responder]
        offerer = offer["offerer"]
        o = state.players[offerer]
        for give_res in RESOURCES:
            if r.resources[give_res] == 0:
                continue
            for want_res in RESOURCES:
                if want_res == give_res or o.resources[want_res] == 0:
                    continue
                give, want = _one_one_trade_dict(give_res, want_res)
                acts.append(_a("counter_trade", give, want))
        return acts

    return []


def _apply(state: GameState, action: Action) -> None:
    pid = active_player(state)
    kind = action.kind
    args = action.args

    if kind == "setup_settlement":
        R.place_settlement(state, pid, args[0], free=True)
    elif kind == "setup_road":
        R.place_road(state, pid, args[0], free=True, setup_vertex=state.setup_last_settlement)
    elif kind == "roll":
        R.roll_dice(state)
    elif kind == "discard":
        R.discard_cards(state, pid, args[0])
    elif kind == "robber":
        R.move_robber(state, pid, args[0], args[1])
    elif kind == "build_settlement":
        R.place_settlement(state, pid, args[0])
    elif kind == "build_road":
        R.place_road(state, pid, args[0])
    elif kind == "build_city":
        R.upgrade_city(state, pid, args[0])
    elif kind == "buy_dev":
        R.buy_dev_card(state, pid)
        state.dev_card_bought_this_turn = True
    elif kind == "play_knight":
        R.play_knight(state, pid, args[0], args[1])
    elif kind == "play_road_building":
        R.play_road_building(state, pid, args)
    elif kind == "play_year_of_plenty":
        R.play_year_of_plenty(state, pid, args[0], args[1])
    elif kind == "play_monopoly":
        R.play_monopoly(state, pid, args[0])
    elif kind == "reveal_vp":
        R.reveal_vp(state, pid)
    elif kind == "bank_trade":
        R.bank_trade(state, pid, args[0], args[1])
    elif kind == "offer_trade":
        pass
    elif kind == "accept_trade":
        offer = state.trade_offer
        R.apply_player_trade(state, offer["offerer"], offer["responder"], offer["give"], offer["want"])
    elif kind == "reject_trade":
        pass
    elif kind == "counter_trade":
        pass
    elif kind == "end_turn":
        pass
    else:
        raise ValueError(f"unknown action kind: {kind}")


def apply_action(state: GameState, action: Action) -> None:
    """Apply one legal action, advance phase, and optionally check invariants."""
    _apply(state, action)

    winner = R.check_winner(state)
    if winner != -1:
        state.phase = phases.END
    else:
        phases.advance(state, action)

    if state.check_conservation:
        from . import invariant
        invariant.check(state)
