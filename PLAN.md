# Plan: production Catan engine + win-probability & move recommendation

## Goal

Two capabilities, from a single model:

1. **Evaluate** — given a position, the probability each player wins.
2. **Recommend** — given a position, the next move that maximizes our win probability.

Two hard constraints that shape everything below:

- **Player-perspective information only.** We know everything about ourselves
  (resource hand, dev cards held, VP). About each opponent we know only: pieces
  on the board, number of resource cards, number of unplayed dev cards, number
  of knights played, and which dev cards they have played. Nothing else.
- **Full Catan rules.** Every dev card, player-to-player trading, real bank
  limits, real port layout, correct longest road / largest army.

End target is advising play against bots on colonist.io, so the core stays
independent of any particular front end.

## Scope decisions

In scope:

- Full-rules engine with an explicit action space and phase machine.
- Player-perspective observation layer.
- Value model retrained on observations (scikit-learn GBT).
- Greedy 1-ply value search, then shallow within-turn expectimax.
- Self-play value iteration, arena-gated.
- colonist.io **advisor** adapter: read-only, prints the recommended move.

Explicitly deferred (all cheap to add later behind the `Observation` seam, none
needed for the above to work):

- Log-based card counting / opponent hand posterior.
- Belief sampling and determinization.
- MO-ISMCTS, policy head, AlphaZero-style training.
- PyTorch, neural nets, GNN board encoders.
- Automating input to colonist.io (advisor mode avoids the ToS/account risk).

## Current state: what must change

Reusable as-is: board topology generation, the sim → features → train → predict
pipeline shape, group-by-game splitting, calibration reporting.

### Rules gaps

- **Dev deck is wrong.** `state.py:86` has `["vp"] * 5 + ["knight"] * 20`. Real
  deck is 14 knight / 5 VP / 2 road-building / 2 year-of-plenty / 2 monopoly.
- **No bank.** `rules.py:142-156` mints resources with no 19-per-type limit and
  no partial-distribution rule.
- **Robber is not player-controlled.** `rules.py:159-173` discards a *random*
  half, picks the victim at *random*, and never enforces that the robber must
  move to a different hex.
- **No turn phases.** Bots call `act()` and may do anything in any order. No
  one-dev-card-per-turn rule, and no restriction on playing a card the turn it
  was bought (`bots.py::_maybe_play_knight` plays immediately).
- **No player-to-player trading.**
- **Ports are non-standard.** `board.py:174-183` samples 9 random perimeter
  edges, so ports can cluster or adjoin, and `vertex_port[a] = kind` silently
  overwrites when two chosen edges share a vertex. Real Catan has 9 fixed
  harbour positions.
- **Longest road is slow *and* silently wrong.** `rules.py:277-315` is an
  exponential DFS with `budget = 200_000` that returns a too-short answer when
  the budget is exhausted, recomputed from scratch for every player on every
  road/settlement placement. The road-break-by-settlement case does not follow
  official rules.
- **Win detected late.** `check_winner` only runs at end of turn
  (`simulate.py:64`), so mid-turn wins stretch games.

### Architecture gaps

- **No action representation and no `clone()`.** Search is impossible today.
  This is the most important missing seam.
- **Full-information leakage.** `state.py::to_row` (102-133) emits every
  player's exact resource dict and hidden VP cards, and
  `features/extract.py::row_to_player_features` consumes them. The current
  model is trained on information a real player does not have and cannot
  transfer to live play.
- **Improper probabilities.** `inference/predict.py:34-38` scores players
  independently then normalizes to sum to 1.

## Target architecture

```
catan_ml/
  engine/        board.py, state.py, rules.py      full-rules ground truth
                 actions.py       Action ADT + legal_actions() + apply()   [NEW]
                 phases.py        turn/phase state machine                [NEW]
                 longest_road.py  correct + incremental                   [NEW]
  observation.py Observation + observe(state, pid)                        [NEW]
  features/      Observation -> vector (versioned schema, single source)
  agents/        Agent protocol; heuristics; value search; expectimax     [NEW]
  training/      value training, self-play loop, arena + Elo         [arena NEW]
  inference/     win_prob(obs) + recommend_move(obs)
  adapters/      colonist_io: their state -> Observation (advisor)        [NEW]
```

Two seams make everything modular. **`Observation`**: any source — simulator,
colonist adapter, replay file — produces one. **`Action`**: any consumer
executes one. Colonist integration then touches nothing outside `adapters/`.

## Key design decisions

### Model shape: one observation in, distribution over seats out

The current approach — score each player separately, normalize to sum to 1 —
**stops working under player-perspective information.** Normalizing needs all
players' observations, and in live play we only ever have our own; we cannot
compute an opponent's observation because we do not know their hidden cards.

So: a single model takes one observation and outputs a distribution over which
seat wins, with classes as *relative* seat offsets (0 = me, 1 = next player, 2,
3), masked for 2- and 3-player games. `P(I win)` is the class-0 probability.
Training rows are one per `(state, observer)`, built strictly from that
observer's observation, labelled with the winner's offset relative to them.

This is a plain multiclass `HistGradientBoostingClassifier`: proper
distribution, single model, no new dependencies, and usable at inference with
only our own view.

### No belief module is needed

A value model trained on partial observations already marginalizes over hidden
information. 1-ply value search evaluates `V(observation)` on resulting
positions, so it needs no opponent hand sampling. Within-turn expectimax uses
chance nodes for dice, dev-card draws and robber steals — none of which require
modelling opponent decisions or hidden hands.

### Bank composition is only partly knowable

Bank *total* remaining is exactly derivable from public hand sizes; per-resource
composition is not, because opponents' hand composition is hidden. The
observation schema must represent that honestly rather than leaking the true
bank contents.

## Phases

Each phase leaves behind one runnable check — the smallest thing that fails if
the logic breaks.

### Phase 0 — Foundation

- `pyproject.toml` with pinned deps; pytest as the runner; one CI workflow.
- Cheap array-based `GameState.clone()`, not `deepcopy`.
- Split RNG streams (board / dice / deck / agent search) so search sampling
  cannot perturb game rolls.
- Hexes to integer axial coordinates, replacing the float-rounded `_round_pt`
  keys in `board.py`. Fragile today, and it is what makes the colonist board
  mapping a pure function in Phase 7.

**Check:** existing `tests/test_engine.py` invariants pass under pytest, plus a
`clone()` round-trip equality test.

### Phase 1 — Full-rules engine

- All 5 dev card types, correct 14/5/2/2/2 deck, one card per turn, no playing
  a card the turn it was bought, VP cards revealable at any time to win.
- Bank with 19-per-resource limits and the official partial-distribution rule.
- Player-to-player trading: offer, accept, counter. Correct bank and port ratios.
- Player-chosen discards, robber destination (must differ from current), and
  steal victim.
- `actions.py` (Action ADT, exhaustive `legal_actions` per phase) and
  `phases.py` (setup → pre-roll → roll → discard → robber → main →
  trade-response → end).
- Correct incremental longest road, including breaking a road with a settlement.
- Immediate mid-turn win check.

**Check:** card conservation asserted after every action across thousands of
fuzzed games — `bank + all hands == 19` for each resource, and
`deck + held + played == 25` for dev cards — plus golden scenarios for
monopoly, road building, bank exhaustion, longest-road transfer on a break, and
largest-army ties.

### Phase 2 — Observation layer

- `observe(state, pid) -> Observation` implementing the information spec exactly.
- Features derived strictly from `Observation`, with a schema version stamped
  into model bundles and verified at load.

**Check:** a leak test. Perturb a hidden field in the full state (swap an
opponent's wheat for ore, change a hidden VP card) and assert
`observe(state, pid)` serializes byte-identically.

### Phase 3 — Retrain

- Multiclass relative-seat model as described above.
- Calibration reported split by game phase, not one aggregate that hides
  early-game failure.
- Keep the existing full-information model as a measured upper bound, so the
  cost of hidden information is a number rather than a guess.

**Check:** held-out calibration and log loss per game phase, split by game.

### Phase 4 — Agent v1: greedy 1-ply value search

- `Agent` protocol: `act(observation) -> Action`.
- Enumerate legal actions, apply on a clone, evaluate, take the argmax, repeat
  until "end turn" wins.
- Arena harness with Elo and confidence intervals, plus a fixed benchmark
  position suite.

**Check:** statistically significant win rate over the heuristic bots at 2, 3
and 4 players.

### Phase 5 — Shallow within-turn expectimax

- Search our own action sequence to the end of the turn, with chance nodes for
  dice, dev-card draws and robber steals. Leaf evaluation is the value model on
  the resulting observation. No opponent model, no belief.

**Check:** arena-gated improvement over the Phase 4 agent.

### Phase 6 — Self-play value iteration

- Loop: play with the current agent, log observations, retrain, arena-gate,
  promote.
- Mix opponents (heuristics plus past checkpoints) to avoid self-play collapse.
- Revisit a neural net only if this measurably plateaus.

**Check:** monotone Elo across promoted checkpoints against a frozen baseline.

### Phase 7 — colonist.io advisor

- `adapters/colonist_io/` parses their state into an `Observation` and prints
  the recommended move for manual execution. Read-only.
- Sub-second recommendation budget; batched inference.

**Check:** replayed colonist states round-trip into `Observation` and produce a
legal recommended action.

## Cross-cutting

- **Performance:** profile before optimizing. Expected hot spots are longest
  road (fixed in Phase 1), `legal_actions` enumeration, and `clone()`. A
  compiled core only if measured numbers demand it.
- **Determinism:** every game reproducible from a seed; every arena result
  reproducible.
- **Model bundles** carry feature schema version, engine rules version, and
  training config. Loading a mismatched bundle fails loudly.

## Risks

- **Rules surface is the bulk of the work.** Trading and the phase machine are
  where subtle bugs hide; the card-conservation invariant is the main defense.
- **Partial observability caps accuracy.** Expect visibly worse log loss than
  the current full-information numbers in the README. That is correct behaviour,
  not a regression, which is why Phase 3 retains the full-info reference bound.
- **Self-play bias.** Strong against itself, unknown against humans, with no
  real-game data in scope. The fixed benchmark suite is the only guard until
  real replays are added.
- **Pure-Python search depth** may limit how deep Phase 5 can go.
