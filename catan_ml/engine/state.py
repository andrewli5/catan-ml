"""Mutable game state + player state + compact serialization."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .board import Board, RESOURCES, TOPO


def _derive_child(rng: random.Random, name: str) -> random.Random:
    """Derive an independent named stream from a master rng without consuming it."""
    data = (name + str(rng.getstate())).encode()
    digest = hashlib.sha256(data).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def _clone_rng(rng: random.Random) -> random.Random:
    """Copy a random.Random object's full internal state."""
    clone = random.Random()
    clone.setstate(rng.getstate())
    return clone


# Vertex building types
EMPTY, SETTLEMENT, CITY = 0, 1, 2

# Build costs as resource->count.
COST_ROAD = {"wood": 1, "brick": 1}
COST_SETTLEMENT = {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}
COST_CITY = {"wheat": 2, "ore": 3}
COST_DEV = {"sheep": 1, "wheat": 1, "ore": 1}

WIN_VP = 10

# Development cards: official 25-card deck.
DEV_CARDS = ["knight", "vp", "road_building", "year_of_plenty", "monopoly"]
DEV_COUNTS = {
    "knight": 14,
    "vp": 5,
    "road_building": 2,
    "year_of_plenty": 2,
    "monopoly": 2,
}


def _empty_dev_dict() -> dict:
    return {k: 0 for k in DEV_CARDS}


@dataclass
class PlayerState:
    pid: int
    resources: dict = field(default_factory=lambda: {r: 0 for r in RESOURCES})
    settlements: int = 0            # count on board
    cities: int = 0                 # count on board
    roads: int = 0                  # count on board
    dev_cards: dict = field(default_factory=_empty_dev_dict)  # unplayed
    played_dev: dict = field(default_factory=_empty_dev_dict)  # revealed/played
    has_longest_road: bool = False
    has_largest_army: bool = False
    longest_road_len: int = 0

    @property
    def dev_vp(self) -> int:
        return self.dev_cards["vp"]

    @property
    def dev_knight(self) -> int:
        return self.dev_cards["knight"]

    @property
    def knights_played(self) -> int:
        return self.played_dev["knight"]

    @property
    def revealed_vp(self) -> int:
        return self.played_dev["vp"]

    def hand_size(self) -> int:
        return sum(self.resources.values())

    def can_afford(self, cost: dict) -> bool:
        return all(self.resources[r] >= n for r, n in cost.items())

    def pay(self, cost: dict) -> None:
        for r, n in cost.items():
            self.resources[r] -= n

    def gain(self, resource: str, n: int = 1) -> None:
        self.resources[resource] += n

    def public_vp(self) -> int:
        """VP visible to opponents (excludes hidden VP dev cards)."""
        vp = self.settlements + 2 * self.cities + self.revealed_vp
        if self.has_longest_road:
            vp += 2
        if self.has_largest_army:
            vp += 2
        return vp

    def total_vp(self) -> int:
        return self.public_vp() + self.dev_vp

    def clone(self) -> "PlayerState":
        return PlayerState(
            pid=self.pid,
            resources=dict(self.resources),
            settlements=self.settlements,
            cities=self.cities,
            roads=self.roads,
            dev_cards=dict(self.dev_cards),
            played_dev=dict(self.played_dev),
            has_longest_road=self.has_longest_road,
            has_largest_army=self.has_largest_army,
            longest_road_len=self.longest_road_len,
        )


@dataclass
class GameState:
    board: Board
    n_players: int
    players: list
    vertex_owner: list          # player id per vertex, -1 = none
    vertex_type: list           # EMPTY/SETTLEMENT/CITY per vertex
    edge_owner: list            # player id per edge, -1 = none
    bank: dict = field(default_factory=lambda: {r: 19 for r in RESOURCES})
    current_player: int = 0
    turn_number: int = 0
    dice_last_roll: int = 0
    dev_deck: list = field(default_factory=list)  # shuffled, drawn without replacement
    winner: int = -1
    phase: str = "setup"
    setup_order: list = field(default_factory=list)
    setup_index: int = 0
    setup_road_pending: bool = False
    setup_last_settlement: int = -1
    turn_has_rolled: bool = False
    dev_card_played_this_turn: bool = False
    dev_card_bought_this_turn: bool = False
    discard_queue: list = field(default_factory=list)
    trade_offer: dict | None = None
    check_conservation: bool = False
    rng: random.Random = field(default_factory=random.Random)           # agent stream
    rng_board: random.Random = field(default_factory=random.Random)     # board layout
    rng_dice: random.Random = field(default_factory=random.Random)      # dice + robber
    rng_deck: random.Random = field(default_factory=random.Random)      # dev deck

    @property
    def dev_deck_remaining(self) -> int:
        return len(self.dev_deck)

    @classmethod
    def new_game(cls, n_players: int, rng: random.Random | None = None) -> "GameState":
        rng = rng or random.Random()
        # Independent streams derived deterministically from the master seed. The
        # master ``rng`` itself is the agent stream; board/dice/deck streams are
        # seeded from a snapshot of its state so search sampling cannot perturb
        # game rolls.
        rng_board = _derive_child(rng, "board")
        rng_dice = _derive_child(rng, "dice")
        rng_deck = _derive_child(rng, "deck")
        board = Board.random(rng_board)
        deck = []
        for kind, cnt in DEV_COUNTS.items():
            deck += [kind] * cnt
        rng_deck.shuffle(deck)
        setup_order = list(range(n_players)) + list(range(n_players - 1, -1, -1))
        return cls(
            board=board,
            n_players=n_players,
            players=[PlayerState(pid=i) for i in range(n_players)],
            vertex_owner=[-1] * TOPO.n_vertices,
            vertex_type=[EMPTY] * TOPO.n_vertices,
            edge_owner=[-1] * TOPO.n_edges,
            bank={r: 19 for r in RESOURCES},
            dev_deck=deck,
            setup_order=setup_order,
            rng=rng,
            rng_board=rng_board,
            rng_dice=rng_dice,
            rng_deck=rng_deck,
        )

    def leader_vp(self) -> int:
        return max(p.total_vp() for p in self.players)

    def clone(self) -> "GameState":
        """Cheap copy: lists/dicts/rngs are fresh; scalars and strings shared."""
        return GameState(
            board=self.board.clone(),
            n_players=self.n_players,
            players=[p.clone() for p in self.players],
            vertex_owner=list(self.vertex_owner),
            vertex_type=list(self.vertex_type),
            edge_owner=list(self.edge_owner),
            bank=dict(self.bank),
            current_player=self.current_player,
            turn_number=self.turn_number,
            dice_last_roll=self.dice_last_roll,
            dev_deck=list(self.dev_deck),
            winner=self.winner,
            phase=self.phase,
            setup_order=list(self.setup_order),
            setup_index=self.setup_index,
            setup_road_pending=self.setup_road_pending,
            setup_last_settlement=self.setup_last_settlement,
            turn_has_rolled=self.turn_has_rolled,
            dev_card_played_this_turn=self.dev_card_played_this_turn,
            dev_card_bought_this_turn=self.dev_card_bought_this_turn,
            discard_queue=list(self.discard_queue),
            trade_offer={k: dict(v) if isinstance(v, dict) else v
                         for k, v in self.trade_offer.items()} if self.trade_offer else None,
            check_conservation=self.check_conservation,
            rng=_clone_rng(self.rng),
            rng_board=_clone_rng(self.rng_board),
            rng_dice=_clone_rng(self.rng_dice),
            rng_deck=_clone_rng(self.rng_deck),
        )

    def to_row(self) -> dict:
        """Compact serializable snapshot (one dict; expand per-player later)."""
        b = self.board
        row = {
            "turn_number": self.turn_number,
            "current_player": self.current_player,
            "dice_last_roll": self.dice_last_roll,
            "robber_hex": b.robber_hex,
            "n_players": self.n_players,
            "phase": self.phase,
            "hex_resource": list(b.hex_resource),
            "hex_number": list(b.hex_number),
            "vertex_owner": list(self.vertex_owner),
            "vertex_type": list(self.vertex_type),
            "edge_owner": list(self.edge_owner),
            "bank": dict(self.bank),
            # str keys: parquet/Arrow map columns require string keys
            "vertex_port": {str(k): v for k, v in b.vertex_port.items()},
        }
        for p in self.players:
            k = f"p{p.pid}_"
            row[k + "res"] = dict(p.resources)
            row[k + "settlements"] = p.settlements
            row[k + "cities"] = p.cities
            row[k + "roads"] = p.roads
            row[k + "knights_played"] = p.knights_played
            row[k + "dev_vp"] = p.dev_vp
            row[k + "dev_knight"] = p.dev_knight
            row[k + "longest_road_len"] = p.longest_road_len
            row[k + "has_longest_road"] = int(p.has_longest_road)
            row[k + "has_largest_army"] = int(p.has_largest_army)
            row[k + "public_vp"] = p.public_vp()
            row[k + "total_vp"] = p.total_vp()
        return row
