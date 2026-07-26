# Catan Win-Probability Estimator

Estimates each player's probability of winning a game of Settlers of Catan from
the current board + game state, at any point in the game. A heuristic-bot
simulator generates games, states are logged with the eventual winner as the
label, and gradient-boosted trees learn `state -> P(win)`.

This is a learning project: the goal is a working end-to-end pipeline with a
model whose probabilities are **calibrated** (a "70% win chance" happens ~70%
of the time), not a perfect Catan AI.

## Results (3,000 simulated games, split by game)

Held-out set = 538 games / 152k player-rows. Split is **by game** so correlated
turns never leak across train/test.

| model | log loss | accuracy | ROC-AUC | Brier |
|-------|---------:|---------:|--------:|------:|
| base rate (constant) | 0.633 | – | – | – |
| logistic regression  | 0.414 | 0.806 | 0.872 | 0.134 |
| **gradient boosted trees** | **0.396** | **0.816** | **0.886** | **0.128** |

*These numbers are from the original full-information feature pipeline. Phase 3
will retrain on player-perspective observations and refresh them.*

- `reports/calibration.png` — both models track the diagonal closely.
- `reports/vp_monotonicity.png` — predicted win prob rises smoothly with VP
  (0.20 at 2 VP → 0.99 at 10 VP) and matches the observed win rate at each VP.
- `reports/trajectories.png` — per-game win-prob curves converge to the winner.

## Architecture (clearly separated modules)

```
catan_ml/
  engine/     board.py (topology+layout), state.py (GameState),
              rules.py (legality, production, dev cards, trading),
              actions.py (Action ADT + legal_actions/apply),
              phases.py (turn phase machine), longest_road.py (exact incremental),
              invariant.py (card-conservation checks)
  sim/        bots.py (3 heuristic styles), simulate.py (turn loop + parquet logging)
  features/   extract.py (raw state -> numeric per-player feature rows; single source of the feature schema)
  training/   train.py (logreg + GBT, metrics, calibration), sanity.py (monotonicity + trajectory plots)
  inference/  predict.py (load model -> per-player win probs; function + CLI)
tests/        test_engine.py (legality invariants, 50-game smoke, no-hang regression),
              test_golden.py (subtle full-rules scenarios)
```

Board topology (19 hexes / 54 vertices / 72 edges) is generated from exact
integer axial coordinates and cached; each game randomizes resources, number
tokens, and ports.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e ".[dev]"
```

The GBT model is scikit-learn's `HistGradientBoostingClassifier` (no LightGBM,
so no `libomp`/OpenMP system dependency on macOS).

## End-to-end run

```bash
# 1. Simulate games -> raw logged states (parquet)
./.venv/bin/python -m catan_ml.sim.simulate --games 3000 --out data/games.parquet --log-every 2

# 2. Raw states -> numeric feature rows (prints 5 sample rows)
./.venv/bin/python -m catan_ml.features.extract --raw data/games.parquet --out data/features.parquet --sample 5

# 3. Train baseline + GBT, write metrics + calibration plot
./.venv/bin/python -m catan_ml.training.train --features data/features.parquet

# 4. Sanity checks (VP monotonicity + trajectory plots)
./.venv/bin/python -m catan_ml.training.sanity

# 5. Predict on a random simulated state
./.venv/bin/python -m catan_ml.inference.predict --demo --seed 3
```

Tests: `./.venv/bin/python -m pytest`

## Inference

```python
from catan_ml.inference.predict import load_model, predict_from_state
from catan_ml.engine.state import GameState

bundle = load_model("models/gbt.joblib")
probs = predict_from_state(bundle, some_game_state)  # {player_id: win_prob}, sums to 1
```

The CLI also accepts a JSON state row (`GameState.to_row()` format) via
`--state state.json`.

## A training example

One logged state produces one row **per player**: the ~34 numeric features
describe that player (VP, pieces, resources in hand, production pips by
resource, ports, robber pressure) plus relational features (VP lead over best
opponent, VP rank, VP share). Label `won` = 1 if that player eventually won.
`game_id` is kept on every row for the group-aware split.

## Key simplifications and their effect on the win-probability signal

All are intentional (agreed in `PLAN.md`); the ones that could distort the
signal are flagged.

- **Full Catan rules.** The engine now implements all five dev-card types with
  the official 25-card deck (14 knights, 5 VP, 2 road-building, 2 year-of-plenty,
  2 monopoly), a real bank of 19 cards per resource with the official shortfall
  rule, bank/port and player-to-player trading, the standard 9-harbour port
  layout, player-chosen discards and robber destination/victim, exact
  incremental longest-road computation, and immediate win checks after every
  action.
- **The main remaining simplification is the bots.** They are greedy heuristics
  (training-data generators, not strong opponents) and can plateau when the board
  saturates or the dev deck empties. Those unfinished games are dropped via
  `drop_unfinished`, so labels only come from games with a real winner.
- **Piece limits enforced** (15 roads / 5 settlements / 4 cities). This is real
  Catan, and the bounded road graph keeps the longest-road search exact and fast
  (see `tests/test_engine.py::test_no_hang_many_games`).
- **The training pipeline still uses full ground-truth rows.** Phase 3 will
  replace this with an observation layer so the model only sees what a real
  player at the table can see.

## Ideas for v2

- **Observation-based retraining** so the model only sees player-perspective
  information and can be used in live play.
- **Stronger / more varied bots** (e.g. lightweight lookahead or MCTS) so the
  data reflects better play and the model generalizes beyond greedy heuristics.
- **Richer board features** — per-hex number/resource encodings or a graph
  neural net over the board, instead of the current pip summaries.
- **Neural net** once data is larger; compare calibration vs GBT.
- **Real game logs** (e.g. Colonist/online replays) instead of bot self-play to
  remove heuristic-bot bias.
- **Sequence model** using the turn trajectory rather than treating each state
  independently.
