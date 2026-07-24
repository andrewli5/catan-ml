"""Mutable game state + player state + compact serialization."""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .board import Board, RESOURCES, TOPO

# Vertex building types
EMPTY, SETTLEMENT, CITY = 0, 1, 2

# Build costs as resource->count.
COST_ROAD = {"wood": 1, "brick": 1}
COST_SETTLEMENT = {"wood": 1, "brick": 1, "sheep": 1, "wheat": 1}
COST_CITY = {"wheat": 2, "ore": 3}
COST_DEV = {"sheep": 1, "wheat": 1, "ore": 1}

WIN_VP = 10


@dataclass
class PlayerState:
    pid: int
    resources: dict = field(default_factory=lambda: {r: 0 for r in RESOURCES})
    settlements: int = 0            # count on board
    cities: int = 0                 # count on board
    roads: int = 0                  # count on board
    knights_played: int = 0
    dev_vp: int = 0                 # hidden victory-point dev cards held
    dev_knight: int = 0             # unplayed knight cards held
    has_longest_road: bool = False
    has_largest_army: bool = False
    longest_road_len: int = 0

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
        vp = self.settlements + 2 * self.cities
        if self.has_longest_road:
            vp += 2
        if self.has_largest_army:
            vp += 2
        return vp

    def total_vp(self) -> int:
        return self.public_vp() + self.dev_vp


@dataclass
class GameState:
    board: Board
    n_players: int
    players: list
    vertex_owner: list          # player id per vertex, -1 = none
    vertex_type: list           # EMPTY/SETTLEMENT/CITY per vertex
    edge_owner: list            # player id per edge, -1 = none
    current_player: int = 0
    turn_number: int = 0
    dice_last_roll: int = 0
    dev_deck: list = field(default_factory=list)  # shuffled, drawn without replacement
    winner: int = -1
    rng: random.Random = field(default_factory=random.Random)

    @property
    def dev_deck_remaining(self) -> int:
        return len(self.dev_deck)

    @classmethod
    def new_game(cls, n_players: int, rng: random.Random | None = None) -> "GameState":
        rng = rng or random.Random()
        board = Board.random(rng)
        # Realistic finite deck: 5 VP + 20 knights (other card types folded into
        # knights since year-of-plenty/monopoly/road-building are out of scope).
        deck = ["vp"] * 5 + ["knight"] * 20
        rng.shuffle(deck)
        return cls(
            board=board,
            n_players=n_players,
            players=[PlayerState(pid=i) for i in range(n_players)],
            vertex_owner=[-1] * TOPO.n_vertices,
            vertex_type=[EMPTY] * TOPO.n_vertices,
            edge_owner=[-1] * TOPO.n_edges,
            dev_deck=deck,
            rng=rng,
        )

    def leader_vp(self) -> int:
        return max(p.total_vp() for p in self.players)

    def to_row(self) -> dict:
        """Compact serializable snapshot (one dict; expand per-player later)."""
        b = self.board
        row = {
            "turn_number": self.turn_number,
            "current_player": self.current_player,
            "dice_last_roll": self.dice_last_roll,
            "robber_hex": b.robber_hex,
            "n_players": self.n_players,
            "hex_resource": list(b.hex_resource),
            "hex_number": list(b.hex_number),
            "vertex_owner": list(self.vertex_owner),
            "vertex_type": list(self.vertex_type),
            "edge_owner": list(self.edge_owner),
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
