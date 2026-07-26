"""Card-conservation invariants wired behind GameState.check_conservation."""
from __future__ import annotations

from .board import RESOURCES
from .state import DEV_COUNTS, GameState


def check(state: GameState) -> None:
    """Assert resources and dev cards are conserved after every action."""
    # 19 of each resource in bank + hands
    for r in RESOURCES:
        total = state.bank.get(r, 0) + sum(p.resources.get(r, 0) for p in state.players)
        assert total == 19, f"resource {r}: bank+hands={total}, expected 19"

    # dev cards: deck + held + played == 25 with correct composition
    counts = {}
    for card in state.dev_deck:
        counts[card] = counts.get(card, 0) + 1
    for p in state.players:
        for kind, n in p.dev_cards.items():
            counts[kind] = counts.get(kind, 0) + n
        for kind, n in p.played_dev.items():
            counts[kind] = counts.get(kind, 0) + n

    for kind, expected in DEV_COUNTS.items():
        assert counts.get(kind, 0) == expected, f"dev {kind}: {counts.get(kind, 0)} != {expected}"
    assert sum(counts.values()) == 25, f"dev total {sum(counts.values())} != 25"
