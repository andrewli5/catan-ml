"""Static board topology + per-game board layout for standard Catan.

Topology (19 hexes, 54 vertices, 72 edges) is generated from exact integer
axial coordinates and cached as a module-level singleton. A ``Board`` instance
holds the randomized per-game layout (resources, number tokens, robber, ports).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Resource types
WOOD, BRICK, SHEEP, WHEAT, ORE, DESERT = "wood", "brick", "sheep", "wheat", "ore", "desert"
RESOURCES = (WOOD, BRICK, SHEEP, WHEAT, ORE)  # tradeable/producible

# Standard tile distribution (18 resource hexes + 1 desert = 19)
TILE_COUNTS = {WOOD: 4, SHEEP: 4, WHEAT: 4, BRICK: 3, ORE: 3, DESERT: 1}
# Standard number tokens for the 18 non-desert hexes (no 7).
NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# pips = number of ways to roll the number (dots on the token); 7 excluded.
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Rows of the standard hex board (pointy-top), counts per row.
_ROW_SIZES = [3, 4, 5, 4, 3]

# Pointy-top corner offsets in integer axial coordinates (scale factor 3).
_VERTEX_SCALE = 3
_CORNER_OFFSETS = [(1, 1), (-1, 2), (-2, 1), (-1, -1), (1, -2), (2, -1)]


@dataclass
class BoardTopology:
    """Immutable board graph shared across all games."""

    hex_centers: list          # (q, r) per hex
    hex_vertices: list         # list[list[int]] : 6 vertex ids per hex
    vertex_coords: list        # (q, r) per vertex
    vertex_hexes: list         # list[list[int]] : hex ids touching each vertex
    vertex_neighbors: list     # list[list[int]] : adjacent vertex ids
    vertex_edges: list         # list[list[int]] : edge ids touching each vertex
    edges: list                # list[tuple[int,int]] : (v0, v1) per edge
    edge_hex_count: list       # hexes bordering each edge (1 => coast)
    perimeter_edges: list      # edge ids on the coast (border exactly 1 hex)

    @property
    def n_hexes(self) -> int:
        return len(self.hex_centers)

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_coords)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def _build_topology() -> BoardTopology:
    # 1. hex centers in integer axial coords (r index shifted so rows are -2..2)
    hex_centers = []
    for r_idx, n in enumerate(_ROW_SIZES):
        r = r_idx - 2
        q0 = max(-2, -r - 2)
        for i in range(n):
            hex_centers.append((q0 + i, r))

    # 2. vertices per hex; each corner is an integer (q, r) key.
    vert_index = {}          # integer point -> vertex id
    vertex_coords = []
    hex_vertices = []
    for q, r in hex_centers:
        vids = []
        for dq, dr in _CORNER_OFFSETS:
            key = (_VERTEX_SCALE * q + dq, _VERTEX_SCALE * r + dr)
            if key not in vert_index:
                vert_index[key] = len(vertex_coords)
                vertex_coords.append(key)
            vids.append(vert_index[key])
        hex_vertices.append(vids)

    # 3. edges (adjacent corners of each hex, deduped) + edge->hex count
    edge_index = {}          # frozenset{v0,v1} -> edge id
    edges = []
    edge_hex_count = []
    for vids in hex_vertices:
        for k in range(6):
            a, b = vids[k], vids[(k + 1) % 6]
            key = frozenset((a, b))
            if key not in edge_index:
                edge_index[key] = len(edges)
                edges.append((min(a, b), max(a, b)))
                edge_hex_count.append(0)
            edge_hex_count[edge_index[key]] += 1

    # 4. adjacency lists
    n_v = len(vertex_coords)
    vertex_hexes = [[] for _ in range(n_v)]
    for h, vids in enumerate(hex_vertices):
        for v in vids:
            vertex_hexes[v].append(h)

    vertex_edges = [[] for _ in range(n_v)]
    vertex_neighbors = [set() for _ in range(n_v)]
    for eid, (a, b) in enumerate(edges):
        vertex_edges[a].append(eid)
        vertex_edges[b].append(eid)
        vertex_neighbors[a].add(b)
        vertex_neighbors[b].add(a)
    vertex_neighbors = [sorted(s) for s in vertex_neighbors]

    perimeter_edges = [e for e, c in enumerate(edge_hex_count) if c == 1]

    return BoardTopology(
        hex_centers=hex_centers,
        hex_vertices=hex_vertices,
        vertex_coords=vertex_coords,
        vertex_hexes=vertex_hexes,
        vertex_neighbors=vertex_neighbors,
        vertex_edges=vertex_edges,
        edges=edges,
        edge_hex_count=edge_hex_count,
        perimeter_edges=perimeter_edges,
    )


# Singleton topology (generated once at import).
TOPO = _build_topology()
assert TOPO.n_hexes == 19, TOPO.n_hexes
assert TOPO.n_vertices == 54, TOPO.n_vertices
assert TOPO.n_edges == 72, TOPO.n_edges


# Port kinds: generic 3:1 or a specific 2:1 resource port.
PORT_GENERIC = "3:1"
PORT_KINDS = [PORT_GENERIC] * 4 + [f"2:1_{r}" for r in RESOURCES]  # 9 ports

# Fixed harbour edges around the standard board (clockwise, no two share a vertex).
PORT_EDGES = [19, 33, 50, 62, 70, 59, 48, 29, 13]


@dataclass
class Board:
    """Randomized per-game layout on the shared topology."""

    hex_resource: list                # resource string per hex
    hex_number: list                  # number token per hex (0 for desert)
    robber_hex: int                   # hex id where robber sits
    vertex_port: dict = field(default_factory=dict)  # vertex id -> port kind

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "Board":
        rng = rng or random.Random()
        # resources
        resources = []
        for res, cnt in TILE_COUNTS.items():
            resources += [res] * cnt
        rng.shuffle(resources)
        # numbers to non-desert hexes
        numbers = list(NUMBER_TOKENS)
        rng.shuffle(numbers)
        hex_resource = resources
        hex_number = []
        ni = 0
        robber_hex = 0
        for h, res in enumerate(hex_resource):
            if res == DESERT:
                hex_number.append(0)
                robber_hex = h
            else:
                hex_number.append(numbers[ni])
                ni += 1
        # ports: official 9 fixed harbour edges; kinds shuffled per game.
        kinds = list(PORT_KINDS)
        rng.shuffle(kinds)
        vertex_port = {}
        for eid, kind in zip(PORT_EDGES, kinds):
            a, b = TOPO.edges[eid]
            vertex_port[a] = kind
            vertex_port[b] = kind
        return cls(hex_resource, hex_number, robber_hex, vertex_port)

    def clone(self) -> "Board":
        """Shallow copy of all mutable per-game layout fields."""
        return Board(
            hex_resource=list(self.hex_resource),
            hex_number=list(self.hex_number),
            robber_hex=self.robber_hex,
            vertex_port=dict(self.vertex_port),
        )

    def vertex_pips(self, vid: int) -> int:
        """Total production pips for a vertex (sum over adjacent hexes)."""
        return sum(PIPS.get(self.hex_number[h], 0) for h in TOPO.vertex_hexes[vid])
