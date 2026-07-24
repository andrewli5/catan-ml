"""Turn logged game states into numeric per-player feature rows.

Each state expands to one row per player, labeled ``won`` = 1 for the eventual
winner. ``game_id`` stays on every row so training can split by game rather than
by row. ``row_to_player_features`` is also used at inference time.
"""
from __future__ import annotations

from ..engine.board import PIPS, RESOURCES, TOPO
from ..engine.state import CITY

# Model input columns, in order.
FEATURE_COLUMNS = [
    # global / game context
    "turn_number",
    "n_players",
    # target player pieces & VP
    "vp_public",
    "vp_total",
    "n_settlements",
    "n_cities",
    "n_roads",
    "longest_road_len",
    "has_longest_road",
    "has_largest_army",
    "knights_played",
    "dev_vp",
    "dev_knight",
    # resources in hand
    "res_wood",
    "res_brick",
    "res_sheep",
    "res_wheat",
    "res_ore",
    "total_resources",
    # production potential (pips) by resource + summary
    "pips_wood",
    "pips_brick",
    "pips_sheep",
    "pips_wheat",
    "pips_ore",
    "pips_total",
    "distinct_resources_produced",
    # ports
    "n_ports",
    "has_generic_port",
    "n_2to1_ports",
    # robber pressure
    "robber_blocks_me",
    # relational (vs opponents)
    "vp_total_minus_max_opp",
    "vp_rank",
    "pips_total_minus_max_opp",
    "vp_share",
]

# Kept for grouping but not fed to the model. turn_number is deliberately a
# feature (see FEATURE_COLUMNS), not an id.
ID_COLUMNS = ["game_id", "player_id"]
LABEL_COLUMN = "won"


def _player_pips(row: dict, pid: int) -> dict:
    """Production pips per resource for a player's settlements/cities."""
    hex_res = row["hex_resource"]
    hex_num = row["hex_number"]
    owner = row["vertex_owner"]
    vtype = row["vertex_type"]
    pips = {r: 0 for r in RESOURCES}
    for v in range(TOPO.n_vertices):
        if owner[v] != pid:
            continue
        mult = 2 if vtype[v] == CITY else 1
        for h in TOPO.vertex_hexes[v]:
            res = hex_res[h]
            if res in RESOURCES:
                pips[res] += PIPS.get(hex_num[h], 0) * mult
    return pips


def _player_ports(row: dict, pid: int) -> tuple:
    """(n_ports, has_generic, n_2to1) for the player's owned vertices."""
    owner = row["vertex_owner"]
    ports = row["vertex_port"]
    n = generic = two = 0
    for v_str, kind in ports.items():
        v = int(v_str)
        if owner[v] == pid:
            n += 1
            if kind == "3:1":
                generic = 1
            else:
                two += 1
    return n, generic, two


def row_to_player_features(row: dict, pid: int) -> dict:
    """Feature dict for one player in one logged state (no label/ids)."""
    n_players = row["n_players"]
    pk = f"p{pid}_"
    res = row[pk + "res"]

    pips = _player_pips(row, pid)
    n_ports, has_generic, n_2to1 = _player_ports(row, pid)

    vp_total = row[pk + "total_vp"]
    pips_total = sum(pips.values())

    opp_vp = [row[f"p{o}_total_vp"] for o in range(n_players) if o != pid]
    opp_pips = [sum(_player_pips(row, o).values()) for o in range(n_players) if o != pid]
    max_opp_vp = max(opp_vp) if opp_vp else 0
    max_opp_pips = max(opp_pips) if opp_pips else 0
    all_vp_sum = sum(row[f"p{o}_total_vp"] for o in range(n_players))
    vp_rank = 1 + sum(1 for o in range(n_players)
                      if o != pid and row[f"p{o}_total_vp"] > vp_total)

    robber_hex = row["robber_hex"]
    robber_blocks_me = int(any(
        row["vertex_owner"][v] == pid for v in TOPO.hex_vertices[robber_hex]
    ))

    return {
        "turn_number": row["turn_number"],
        "n_players": n_players,
        "vp_public": row[pk + "public_vp"],
        "vp_total": vp_total,
        "n_settlements": row[pk + "settlements"],
        "n_cities": row[pk + "cities"],
        "n_roads": row[pk + "roads"],
        "longest_road_len": row[pk + "longest_road_len"],
        "has_longest_road": row[pk + "has_longest_road"],
        "has_largest_army": row[pk + "has_largest_army"],
        "knights_played": row[pk + "knights_played"],
        "dev_vp": row[pk + "dev_vp"],
        "dev_knight": row[pk + "dev_knight"],
        "res_wood": res["wood"],
        "res_brick": res["brick"],
        "res_sheep": res["sheep"],
        "res_wheat": res["wheat"],
        "res_ore": res["ore"],
        "total_resources": sum(res.values()),
        "pips_wood": pips["wood"],
        "pips_brick": pips["brick"],
        "pips_sheep": pips["sheep"],
        "pips_wheat": pips["wheat"],
        "pips_ore": pips["ore"],
        "pips_total": pips_total,
        "distinct_resources_produced": sum(1 for r in RESOURCES if pips[r] > 0),
        "n_ports": n_ports,
        "has_generic_port": has_generic,
        "n_2to1_ports": n_2to1,
        "robber_blocks_me": robber_blocks_me,
        "vp_total_minus_max_opp": vp_total - max_opp_vp,
        "vp_rank": vp_rank,
        "pips_total_minus_max_opp": pips_total - max_opp_pips,
        "vp_share": vp_total / all_vp_sum if all_vp_sum > 0 else 1.0 / n_players,
    }


def row_to_examples(row: dict) -> list:
    """Expand one logged state into per-player labeled examples."""
    examples = []
    for pid in range(row["n_players"]):
        feats = row_to_player_features(row, pid)
        feats.update({
            "game_id": row["game_id"],
            "player_id": pid,
            LABEL_COLUMN: int(row["winner"] == pid),
        })
        examples.append(feats)
    return examples


def build_feature_frame(raw_df) -> "pd.DataFrame":
    """Convert a raw logged-states DataFrame into a feature DataFrame."""
    import pandas as pd

    rows = []
    for rec in raw_df.to_dict("records"):
        rows.extend(row_to_examples(rec))
    df = pd.DataFrame(rows)
    ordered = ID_COLUMNS + FEATURE_COLUMNS + [LABEL_COLUMN]
    return df[ordered]


if __name__ == "__main__":
    import argparse
    import pandas as pd

    ap = argparse.ArgumentParser(description="Raw logged states -> feature parquet")
    ap.add_argument("--raw", default="data/games.parquet")
    ap.add_argument("--out", default="data/features.parquet")
    ap.add_argument("--sample", type=int, default=5,
                    help="print N sample feature rows for inspection")
    args = ap.parse_args()

    raw = pd.read_parquet(args.raw)
    feats = build_feature_frame(raw)
    feats.to_parquet(args.out, index=False)
    print(f"Wrote {len(feats)} feature rows -> {args.out}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(feats.head(args.sample).T)
