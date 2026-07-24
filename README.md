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

- `reports/calibration.png` — both models track the diagonal closely.
- `reports/vp_monotonicity.png` — predicted win prob rises smoothly with VP
  (0.20 at 2 VP → 0.99 at 10 VP) and matches the observed win rate at each VP.
- `reports/trajectories.png` — per-game win-prob curves converge to the winner.

## Architecture (clearly separated modules)

```
catan_ml/
  engine/     board.py (topology+layout), state.py (GameState), rules.py (legality, production, dev cards, longest road)
  sim/        bots.py (3 heuristic styles), simulate.py (turn loop + parquet logging)
  features/   extract.py (raw state -> numeric per-player feature rows; single source of the feature schema)
  training/   train.py (logreg + GBT, metrics, calibration), sanity.py (monotonicity + trajectory plots)
  inference/  predict.py (load model -> per-player win probs; function + CLI)
tests/        test_engine.py (legality invariants, 50-game smoke, no-hang regression)
```

Board topology (19 hexes / 54 vertices / 72 edges) is generated geometrically
and cached; each game randomizes resources, number tokens, and ports.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
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

Tests: `./.venv/bin/python -m tests.test_engine`

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

- **Dev cards: knights + VP cards only** (no year-of-plenty / monopoly /
  road-building). Deck is the realistic finite `5 VP + 20 knights`, drawn
  without replacement. **Why this matters:** an earlier version drew VP cards
  probabilistically and over-produced them (~2.5 hidden VP per winner vs the
  real max of 5 in the *whole* deck), which inflated hidden information and made
  late-game states look artificially non-deterministic. With the finite deck,
  winner VP breaks down ~53% buildings / 25% hidden dev VP / 22% longest-road +
  largest-army — hidden dev VP is a realistic ~25% source of uncertainty, which
  is *good*: it keeps win probability from collapsing to a trivial VP lookup.
- **Trading: bank/port only** (4:1, 3:1, 2:1), no player-to-player trades.
  Slightly slows resource fluidity; minor signal impact.
- **Robber discard on 7**: players over 7 cards discard a random half. Minor.
- **Weak bots plateau ~10% of games** (board saturates + dev deck empties → no
  legal move). Those games are dropped (`drop_unfinished`), so labels are only
  from games with a real winner. 3,000 attempts → ~2,686 usable games.
- **Piece limits enforced** (15 roads / 5 settlements / 4 cities). This is real
  Catan, and it also bounds the road graph so the longest-road search stays
  fast (an earlier unbounded version hung for tens of minutes on the exponential
  trail search — see `tests/test_engine.py::test_no_hang_many_games`).

## Ideas for v2

- **Full dev cards + player-to-player trading** — the two biggest realism gaps;
  trading especially changes resource dynamics and comeback potential.
- **Stronger / more varied bots** (e.g. lightweight lookahead or MCTS) so the
  data reflects better play and the model generalizes beyond greedy heuristics.
- **Richer board features** — per-hex number/resource encodings or a graph
  neural net over the board, instead of the current pip summaries.
- **Neural net** once data is larger; compare calibration vs GBT.
- **Real game logs** (e.g. Colonist/online replays) instead of bot self-play to
  remove heuristic-bot bias.
- **Sequence model** using the turn trajectory rather than treating each state
  independently.
