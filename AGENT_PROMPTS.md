# Delegation prompts, one per phase

Phases are sequential: each assumes the previous one is merged. Run them in
order 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7. The only phase that can overlap is 7,
whose colonist parser work needs nothing beyond the Phase 2 `Observation`
schema.

Prepend **Preamble** to every phase prompt. Each phase block below is otherwise
self-contained.

---

## Preamble (prepend to every prompt)

````
You are working in a Python repo at /Users/al879s/personal-codebase/catan-ml —
a Settlers of Catan engine plus an ML model that estimates win probability.

Read PLAN.md at the repo root in full before doing anything. It is the spec.
Your task is exactly one phase of it, named below.

Rules:
- Use the existing venv: ./.venv/bin/python. Do not create another one.
- Match the existing code style: `from __future__ import annotations`,
  dataclasses, concise module docstrings, no type-annotation ceremony.
- Minimal diffs. No abstractions, config systems, or plugin layers that this
  phase does not require. Deletion is better than addition.
- No new third-party dependencies without asking first.
- Do not add or remove comments and docstrings beyond what the change needs.
- Stay inside your phase. If you spot work that belongs to a later phase, write
  it down in your final report and leave it alone.
- Leave behind exactly one runnable check for non-trivial logic: an
  assert-based self-check or a small pytest file. No fixtures, no mocks, no new
  test frameworks.
- Never weaken or delete an existing test to make something pass.
- EVERY command you run must be guaranteed to terminate on its own without me
  intervening: prefix it with `timeout <seconds>` using a budget you chose
  deliberately, close stdin and disable paging and prompts so nothing can ever
  block waiting on input (`< /dev/null`, `PAGER=cat`, assume-yes flags, and
  never a REPL, editor, watch mode, or foreground server), and size any
  simulation, training, or arena run to fit that budget by smoke-testing at a
  tiny game count before scaling up. This codebase has a history of a
  pathological hang in the longest-road search. If a command hits its timeout,
  stop and find the root cause. Do not retry blindly and do not raise the
  timeout to paper over it.
- Do not commit unless asked.

When done, report: files changed, the exact command to run your check, the
result of running it, and anything you deliberately left undone.
````

---

## Phase 0 — Foundation

````
Your task is Phase 0 ("Foundation") from PLAN.md.

Deliverables:

1. Packaging. Add pyproject.toml as the single source of dependency truth
   (numpy, pandas, pyarrow, scikit-learn, matplotlib; pytest as a dev extra),
   pinned to versions compatible with what is installed in .venv. Delete
   requirements.txt and update the Setup section of README.md to match.

2. pytest. Make `./.venv/bin/python -m pytest` discover and pass the existing
   tests in tests/. Add pytest config to pyproject.toml. Keep the existing
   test functions; you may drop the hand-rolled `_run_all` / `__main__` runner
   in tests/test_engine.py if pytest fully replaces it.

3. GameState.clone(). Add a cheap clone to catan_ml/engine/state.py. Must not
   use copy.deepcopy. Must copy everything mutable: the board (hex_resource,
   hex_number, robber_hex, vertex_port), every PlayerState including its
   resources dict, vertex_owner, vertex_type, edge_owner, dev_deck, and all
   scalars. Mutating a clone must never be observable in the original, and
   vice versa. This will be called millions of times by search later, so keep
   it allocation-light.

4. Split RNG streams. Today a single random.Random drives board layout, dice,
   deck shuffling, and every bot decision, so any future search that samples
   randomness would perturb the actual dice. Derive independent named streams
   (board, dice, deck, agent) deterministically from one master seed, so a
   given master seed still reproduces a given game exactly.

5. Integer axial coordinates. catan_ml/engine/board.py currently builds
   topology from floating-point geometry and dedupes vertices by rounding
   coordinates to 3 decimals (`_round_pt`). Replace this with exact integer
   axial/cube hex coordinates and integer-keyed vertex and edge identity. Keep
   the same public surface (TOPO with n_hexes/n_vertices/n_edges,
   hex_vertices, vertex_hexes, vertex_neighbors, vertex_edges, edges,
   edge_hex_count, perimeter_edges).

Constraints:
- Phase 0 changes no game rules and no observable game behaviour. Do not touch
  rules.py logic, dev cards, trading, the robber, or the bots.
- Vertex and edge ids are allowed to change as a result of item 5, but they
  must be deterministic and identical across processes and runs. Existing
  data/*.parquet and models/*.joblib are gitignored and regenerable, so you do
  not need to migrate them.

Checks to leave behind:
- clone independence: mutate every mutable field of a clone, assert the
  original is unchanged, and assert the reverse.
- topology determinism: the 19/54/72 counts still hold, every edge joins two
  distinct valid vertices, adjacency is symmetric, and ids are stable across
  two separate interpreter invocations.
- The pre-existing tests, including test_no_hang_many_games, must still pass
  inside their existing time budget.
````

---

## Phase 1 — Full-rules engine

````
Your task is Phase 1 ("Full-rules engine") from PLAN.md. This is the largest
phase. Work in the order given below — the invariant in step 1 is what will
catch your own mistakes in steps 2 onward, so build it first.

1. Card-conservation invariant, FIRST. Write the check before the features.
   After every state mutation it must hold that, for each of the five
   resources, bank + sum over all player hands == 19; and for dev cards,
   deck remaining + all held + all played == 25. Wire it behind a debug flag
   so the fuzz test can enable it and normal simulation is not slowed.

2. Bank. Add a real bank of 19 cards per resource to the game state.
   catan_ml/engine/rules.py:142-156 currently mints resources from nothing.
   Implement the official shortfall rule: if the bank cannot pay every player
   entitled to a resource on a roll, nobody receives that resource — unless
   exactly one player is entitled, who then receives whatever remains.

3. Dev deck. catan_ml/engine/state.py:86 builds `["vp"] * 5 + ["knight"] * 20`.
   The real deck is 14 knight, 5 victory point, 2 road building, 2 year of
   plenty, 2 monopoly = 25.

4. Dev card mechanics, all five types. Knight moves the robber and counts
   toward largest army. Road building places 2 free roads. Year of plenty
   takes any 2 resources from the bank. Monopoly takes every card of one named
   resource from every opponent. Victory point cards are hidden and may be
   revealed at any time, including to win. Enforce: at most one dev card
   played per turn (revealing VP cards to win is exempt), and a card cannot be
   played on the turn it was bought. catan_ml/sim/bots.py currently plays a
   knight the instant it is drawn.

5. Player-to-player trading. Offer a set of resources for a set of resources,
   with accept, reject, and counter-offer. Only on the offering player's turn.
   Both sides must actually hold what they are giving. Dev cards are not
   tradeable. Keep the existing bank and port ratio logic.

6. Robber and discards become player decisions. rules.py:159-173 currently
   discards a random half, picks the victim at random, and lets the robber
   stay on its current hex. Required: the robber must move to a different hex;
   the moving player chooses the destination and chooses which adjacent
   opponent holding cards to steal from; and on a 7 every player over 7 cards
   discards exactly floor(n/2) cards of their own choosing.

7. catan_ml/engine/actions.py. An Action type (hashable and serializable), an
   exhaustive legal_actions(state) for whatever phase the game is in, and
   apply_action(state, action). Every mutation of game state must flow through
   this one path by the end of the phase.

8. catan_ml/engine/phases.py. An explicit phase machine: setup, pre-roll
   (knight only), roll, discard, robber, main, trade-response, end.

9. catan_ml/engine/longest_road.py. rules.py:277-315 is an exponential DFS
   with a `budget = 200_000` escape hatch that SILENTLY RETURNS A WRONG,
   TOO-SHORT ANSWER when the budget runs out, and it recomputes from scratch
   for all players on every single placement. Replace it with something exact
   and fast, incremental if that is what it takes. Handle the official rule for
   a road broken by an opponent's new settlement, including who holds the card
   after a break and what happens on ties. Delete the budget hack.

10. Real port layout. board.py:174-183 samples 9 random perimeter edges, so
    harbours can cluster or adjoin, and `vertex_port[a] = kind` silently
    overwrites when two chosen edges share a vertex. Use the 9 fixed harbour
    positions of the standard board.

11. Immediate win. check_winner only runs at end of turn
    (catan_ml/sim/simulate.py:64). A player who reaches 10 VP mid-turn wins
    right then.

12. Route catan_ml/sim/bots.py and catan_ml/sim/simulate.py through the new
    action API. Bots may stay greedy and simple — they are training-data
    generators, not the eventual agent — but they must be able to use the new
    rules, and simulate.py must still run end to end and write parquet.

Out of scope, do not touch: the observation layer, feature engineering, model
training, and any agent or search code. Extend GameState.to_row() only as much
as keeping simulate.py working requires; Phase 2 and 3 replace the feature
path entirely, so do not redesign it.

Checks to leave behind:
- A fuzz test running several thousand games with the conservation invariant
  enabled after every action.
- Golden-scenario tests, one per rule that is easy to get subtly wrong:
  monopoly across multiple opponents; road building with only one legal road
  left; bank exhaustion hitting the shortfall rule; longest road transferring
  when an opponent's settlement breaks the holder's road; largest army ties;
  a dev card bought this turn being unplayable; discard of exactly floor(n/2).
- test_no_hang_many_games must still pass in its existing budget. Longest road
  is the known hazard here — if that test slows down, you have reintroduced
  the exponential blowup.
````

---

## Phase 2 — Observation layer

````
Your task is Phase 2 ("Observation layer") from PLAN.md.

The point of this phase: the model must only ever see what a real player at the
table can see. Today GameState.to_row() (catan_ml/engine/state.py:102-133)
dumps every player's exact resource dict and hidden victory point cards, and
catan_ml/features/extract.py consumes them, so the current model is trained on
information nobody actually has.

1. catan_ml/observation.py, with an Observation dataclass and
   observe(state, pid) -> Observation. Exactly this information, no more:

   Fully known: the whole board (hex resources, number tokens, harbours,
   robber position), every settlement, city and road with its owner, and for
   the observer themselves — exact resource hand, exact dev cards held
   including hidden VP cards, and true total VP.

   Per opponent, only: pieces on the board, number of resource cards held,
   number of unplayed dev cards held, number of knights played, which dev
   cards they have played, public VP, and whether they hold longest road or
   largest army.

   Derived and legitimately known: total cards remaining in the bank (exactly
   derivable from public hand sizes) and dev deck cards remaining (exactly
   derivable). Per-resource bank composition is NOT knowable, because opponent
   hand composition is hidden — do not expose it. Dev deck composition is only
   knowable to the extent implied by the observer's own hand plus every card
   played by anyone.

2. Order opponents by turn order relative to the observer, not by absolute
   seat id. Phase 3's model depends on this, so the ordering belongs here.

3. Rewrite catan_ml/features/extract.py to derive its vector strictly from an
   Observation. Nothing else may be an input. Add a FEATURE_SCHEMA_VERSION
   constant that Phase 3 will stamp into model bundles.

4. Keep simulate.py logging full ground-truth states, and derive observations
   in the feature-building step. Full states are cheap and let the observation
   schema change without re-simulating. The leak test in step 5 is what
   guarantees no ground truth reaches the model.

Out of scope: model training, agents, search.

Check to leave behind — a leak test, and make it thorough, because this is the
one invariant the whole project rests on. Build a state, take
observe(state, pid), then perturb only information that pid cannot see: swap an
opponent's wheat for ore while keeping their hand size, change an opponent's
hidden VP card to a knight while keeping their unplayed dev count, reorder the
dev deck. After each perturbation, assert the serialized observation is
byte-identical. Then assert that perturbing something pid CAN see (their own
hand, any board piece) does change it — otherwise the test could pass
vacuously.
````

---

## Phase 3 — Retrain on observations

````
Your task is Phase 3 ("Retrain") from PLAN.md. Read the "Key design decisions"
section of PLAN.md especially carefully before starting.

The current model scores each player independently and normalizes the results
to sum to 1 (catan_ml/inference/predict.py:34-38). That approach is now
impossible: normalizing needs every player's observation, and in live play we
only have our own, because computing an opponent's observation would require
knowing their hidden cards.

1. Reshape the model. One observation in, a probability distribution over which
   seat wins out. Classes are seat offsets RELATIVE to the observer: 0 = the
   observer, 1 = next player in turn order, and so on. P(observer wins) is
   simply the class-0 probability. Use a multiclass
   HistGradientBoostingClassifier. Do not introduce PyTorch or any neural net.

2. Build training rows as one per (logged state, observer) pair, features from
   that observer's Observation only, labelled with the eventual winner's seat
   offset relative to that observer. Keep splitting by game_id so correlated
   turns never straddle train and test.

3. Handle variable player counts (2, 3, 4). Prefer a single model with
   n_players as a feature and invalid classes masked and renormalized at
   predict time, rather than three separate models — but verify calibration
   separately per player count and say so in your report if the single model
   is materially worse for any of them.

4. Metrics. Log loss and Brier as primary. Calibration must be reported split
   by game phase (early / mid / late) and by player count, not as one
   aggregate number that hides early-game failure. Update the reports/ plots
   accordingly.

5. Keep the old full-information model trainable as a reference upper bound, so
   the accuracy cost of hidden information is a measured number. Expect the new
   model to be clearly worse than the numbers currently in README.md. That is
   correct behaviour, not a regression — say so in the README when you update
   the results table.

6. Model bundles must carry FEATURE_SCHEMA_VERSION, an engine rules version,
   the training config, and how player counts are handled. Loading a bundle
   whose schema version does not match the code must fail loudly, not silently
   mispredict.

7. Update catan_ml/inference/predict.py for the new single-observation
   interface, and delete the normalize-across-players path.

Out of scope: agents, search, move recommendation.

Checks to leave behind: a round-trip test asserting predicted probabilities are
a valid distribution over the valid seats for 2, 3 and 4 players; and a test
that loading a bundle with a mismatched schema version raises.
````

---

## Phase 4 — Greedy 1-ply agent and arena

````
Your task is Phase 4 ("Agent v1: greedy 1-ply value search") from PLAN.md.

1. An Agent protocol in catan_ml/agents/: act(observation) -> Action. Adapt the
   existing heuristic bots to it so they can be arena opponents.

2. ValueSearchAgent. For each legal action: clone the state, apply the action,
   take the resulting observation, evaluate P(I win) with the Phase 3 model,
   and keep the argmax. Repeat until "end turn" is the best action. Guard
   against loops with an action cap per turn and repeated-state detection —
   trade and bank-trade actions can cycle.

3. Resolve this subtlety explicitly, it is the crux of the phase. Actions apply
   to a state, but at inference we only possess an Observation. So implement
   reconstruct_state(observation, rng) -> GameState, which fills opponents'
   hidden hands with a plausible assignment consistent with their known hand
   sizes and the known bank total. Every evaluation must then go through
   observe() on the resulting state, so nothing you invented during
   reconstruction can leak into the value estimate. Reconstruction details only
   affect the simulated outcome of actions that genuinely depend on hidden
   cards — monopoly, robber steals, and player trades. Document that
   approximation in the module docstring and do not build a belief sampler or
   particle filter to fix it; that is deliberately out of scope.

4. An arena harness in catan_ml/training/: play N games between named agents,
   rotate seats so seat order cannot confound the result, and report win rates
   with confidence intervals plus Elo. Also add a small fixed suite of
   benchmark positions with expected-move or expected-evaluation assertions, so
   strength regressions are visible without running a full tournament.

Out of scope: deeper search, chance-node expectimax, self-play retraining,
belief modelling.

Check to leave behind: an arena run showing ValueSearchAgent beating the
heuristic bots at 2, 3 and 4 players with the result outside its confidence
interval. Report the actual numbers. If it does not beat them, do not tune
until it does — report that, with your diagnosis of why, because the likely
cause is a Phase 3 model problem rather than the search.
````

---

## Phase 5 — Within-turn expectimax

````
Your task is Phase 5 ("Shallow within-turn expectimax") from PLAN.md.

Extend the Phase 4 agent to search our own action sequence to the end of our
turn, with chance nodes, evaluating the Phase 3 model at the leaves.

Chance nodes to model:
- Dice: the 11 outcomes with their true probabilities.
- Dev card draw: the distribution over the remaining deck, derived only from
  public information — the observer's own cards plus every card anyone has
  played. Do not peek at the real deck order.
- Robber steal: uniform over the victim's unknown cards.

Rules:
- No opponent decision modelling. No belief sampling or determinization. The
  search covers our own turn only; the leaf evaluation is the value model on
  the observation that results.
- Enforce a wall-clock budget per move, with iterative deepening or a node cap,
  so this stays usable in live play. Make the budget a parameter.
- Watch for combinatorial explosion in the action sequence, especially trades
  and multi-build turns. If the branching factor is the problem, prune by
  evaluating and keeping the top-k actions per node rather than raising the
  time limit.

Out of scope: searching past our own turn, opponent policies, MCTS, policy
networks.

Check to leave behind: an arena run showing this agent beating the Phase 4
agent, seats rotated, with confidence intervals, plus a test asserting the
per-move time budget is respected. Report the measured branching factor and
median nodes per move.
````

---

## Phase 6 — Self-play value iteration

````
Your task is Phase 6 ("Self-play value iteration") from PLAN.md.

Build the improvement loop: generate self-play games with the current best
agent, build features, retrain the value model, run the arena against the
current best, and promote the challenger only if it wins outside the confidence
interval.

Requirements:
- Mix the opponent pool: the heuristic bots plus previously promoted
  checkpoints, not only the current best against itself. Pure self-play here
  invites collapse into a narrow style.
- A checkpoint registry with Elo history, so progress across iterations is
  visible and a regression can be rolled back.
- A plateau stop condition, so the loop terminates on its own rather than
  burning compute for noise.
- One entry point that runs a full iteration end to end, resumable if
  interrupted.
- Stay on scikit-learn GBT. If and only if you measure a clear plateau, report
  it with the numbers and recommend whether a neural net is warranted — but do
  not implement one in this phase.

Out of scope: neural nets, policy heads, MCTS, new dependencies.

Check to leave behind: a short end-to-end run with small game counts proving
the loop works — generate, retrain, arena, promote or reject, registry updated.
Then report Elo across at least three real iterations.
````

---

## Phase 7 — colonist.io advisor

````
Your task is Phase 7 ("colonist.io advisor") from PLAN.md.

Build catan_ml/adapters/colonist_io/: read a colonist.io game position, turn it
into an Observation, and print the recommended move.

This is READ-ONLY and advisory. Do not automate input, drive a browser, inject
scripts, or send anything to colonist.io. The user executes the suggested move
by hand. This constraint is deliberate — it avoids terms-of-service and
account-ban exposure — so do not "improve" on it.

1. Investigate the input format first and report what you find before building
   the full parser. Likely candidates are the in-game log text or an exported
   or captured state payload. Pin whatever you choose behind a small documented
   parser interface, and commit real captured samples as fixture files so the
   parser is testable without a live game.

2. Board coordinate mapping: colonist's hex, vertex and edge identifiers to
   ours. Phase 0 moved the engine to integer axial coordinates specifically so
   this can be a pure function. Write it as one, and test it against a fixture
   rather than a live game.

3. A CLI that takes a parsed position and prints the top-N recommended actions
   with their win probabilities, plus the current win probability for every
   player. Human-readable output — this is the actual user interface of the
   project.

4. Respect a sub-second budget for the recommendation, and batch model
   inference. Report the measured latency.

Out of scope: any automation of play, browser control, network calls to
colonist.io, and account handling.

Check to leave behind: a fixture-based round-trip test — a saved colonist
position parses into an Observation, the board mapping matches the expected
axial ids, and the advisor returns a legal action for that position.
````
