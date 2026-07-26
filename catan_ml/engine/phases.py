"""Explicit turn/phase state machine."""
from __future__ import annotations

from .board import RESOURCES, TOPO
from .state import GameState


SETUP = "setup"
PRE_ROLL = "pre-roll"
ROLL = "roll"
DISCARD = "discard"
ROBBER = "robber"
MAIN = "main"
TRADE_RESPONSE = "trade-response"
END = "end"


def active_player(state: GameState) -> int:
    if state.phase == SETUP:
        return state.setup_order[state.setup_index]
    if state.phase == DISCARD:
        return state.discard_queue[0] if state.discard_queue else state.current_player
    if state.phase == TRADE_RESPONSE:
        return state.trade_offer["responder"]
    return state.current_player


def _reset_turn_flags(state: GameState) -> None:
    state.turn_has_rolled = False
    state.dev_card_played_this_turn = False
    state.dev_card_bought_this_turn = False


def advance(state: GameState, action) -> None:
    """Update phase and turn state after an action has been applied."""
    phase = state.phase
    kind = action.kind
    args = action.args

    if phase == SETUP:
        if kind == "setup_settlement":
            state.setup_road_pending = True
            state.setup_last_settlement = args[0]
            # Second settlement in the snake draft yields starting resources.
            if state.setup_index >= state.n_players:
                pid = state.setup_order[state.setup_index]
                for h in TOPO.vertex_hexes[args[0]]:
                    res = state.board.hex_resource[h]
                    if res in RESOURCES:
                        state.players[pid].gain(res, 1)
                        state.bank[res] -= 1
        elif kind == "setup_road":
            state.setup_road_pending = False
            state.setup_last_settlement = -1
            state.setup_index += 1
            if state.setup_index < len(state.setup_order):
                state.current_player = state.setup_order[state.setup_index]
            else:
                # Setup complete; first player starts the real turns.
                state.current_player = state.setup_order[0]
                state.setup_order = []
                state.setup_index = 0
                state.turn_number = 0
                state.phase = PRE_ROLL
                _reset_turn_flags(state)

    elif phase == PRE_ROLL:
        if kind == "roll":
            roll = state.dice_last_roll
            if roll == 7:
                n = state.n_players
                start = state.current_player
                q = []
                for i in range(n):
                    pid = (start + i) % n
                    if state.players[pid].hand_size() > 7:
                        q.append(pid)
                state.discard_queue = q
                state.phase = DISCARD if q else ROBBER
            else:
                from . import rules as R
                R.produce(state, roll)
                state.phase = MAIN

    elif phase == DISCARD:
        if kind == "discard":
            state.discard_queue = state.discard_queue[1:]
            if not state.discard_queue:
                state.phase = ROBBER

    elif phase == ROBBER:
        if kind == "robber":
            state.phase = MAIN

    elif phase == MAIN:
        if kind == "end_turn":
            pid = state.current_player
            state.current_player = (pid + 1) % state.n_players
            state.turn_number += 1
            state.phase = PRE_ROLL
            _reset_turn_flags(state)
        elif kind == "offer_trade":
            give, want, target = args
            state.trade_offer = {
                "offerer": state.current_player,
                "responder": target,
                "give": dict(give),
                "want": dict(want),
            }
            state.phase = TRADE_RESPONSE

    elif phase == TRADE_RESPONSE:
        if kind in ("accept_trade", "reject_trade"):
            state.trade_offer = None
            state.phase = MAIN
        elif kind == "counter_trade":
            offer = state.trade_offer
            state.trade_offer = {
                "offerer": offer["responder"],
                "responder": offer["offerer"],
                "give": dict(args[0]),
                "want": dict(args[1]),
            }
