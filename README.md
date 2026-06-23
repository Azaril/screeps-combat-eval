# screeps-combat-eval

> The combat policy layer: a runnable, metric-producing experiment register over the combat sim — the tactics-tuning loop.

`screeps-combat-eval` is the top layer of the Screeps combat family. It turns the ADR 0008a `EXP-*` experiment
register into runnable Rust: each experiment sets up a scenario, runs it through the authoritative combat engine
against a scripted or managed opponent, extracts **measured metrics**, and gates each metric to pass/fail. This is
the **tactics-tuning loop** — change a tunable in `screeps-combat-decision`, run the register, watch the numbers
move, and let the gates flag regressions. It is a component extracted from the
[screeps-ibex](https://github.com/Azaril/screeps-ibex) workspace.

On top of the register it ships three reusable evaluation tools: a per-side **metrics** computer over a combat
recording, a self-play **scoring** adjudicator (including stalemate verdicts), a position-search **CPU bench**, and a
self-play **tournament** with an exploitability ship-gate.

## Installation

Add it as a git dependency:

```toml
[dependencies]
screeps-combat-eval = { git = "https://github.com/Azaril/screeps-combat-eval" }
```

It depends on the rest of the combat family (`screeps-combat-engine`, `screeps-combat-decision`,
`screeps-combat-agent`) plus `screeps-game-api`. It builds host-only — there is no wasm/runtime dependency, so it
runs as a normal `cargo test` / `cargo run --example` target.

## Usage

Run the whole register and render the report — the core tuning-loop call:

```rust
use screeps_combat_eval::{register, report};

let results = register();
print!("{}", report(&results));

// Programmatic gate check: did everything pass?
let all_passed = results.iter().all(|r| r.pass);
assert!(all_passed, "a tactics tunable regressed an EXP-* gate");
```

`register()` returns one [`ExperimentResult`] per `EXP-*` entry. Each result carries its `id`, the `hypothesis`
under test, a `Vec<Metric>` (each with a measured `value`, the `gate` text it was checked against, and a `pass`
bool), and an overall `pass` (all metrics passed). `report()` renders the lot as a readable text dashboard.

Compute per-side metrics directly from any [`CombatRecording`] (the five families: healing, DPS, positioning,
survivability, efficiency):

```rust
use screeps_combat_eval::metrics::{SideMetrics, worst_cohesion};

let m = SideMetrics::from_recording(&recording, /* side: PlayerId */ 0);
println!("creep DPS {} (tower {} attributed separately)", m.creep_damage_dealt, m.tower_damage_dealt);
println!("damage taken {}, survival rate {:.2}", m.damage_taken, m.survival_rate);
println!("mean nearest-enemy range {:.2}, melee exposure {:.2}", m.mean_nearest_enemy_range, m.melee_exposure_rate);

// Worst cohesion spread the side ever reached (matches the live seg-57 canary measure).
let spread = worst_cohesion(&recording, 0);
```

Adjudicate a self-play engagement, including the stalemate case:

```rust
use screeps_combat_eval::scoring::{score, Verdict};

let s = score(&recording, /* a_owner */ 0, /* b_owner */ 1);
match s.verdict {
    Verdict::SideA => println!("A wins (decisive: {})", s.decisive),
    Verdict::SideB => println!("B wins (decisive: {})", s.decisive),
    Verdict::Draw  => println!("stalemate — residual HP-slope advantage {}", s.residual_advantage_a),
}
```

Run the self-play tournament + exploitability ship-gate:

```rust
use screeps_combat_eval::tournament::{run_tournament, exploitability, strategy_population, report, TournamentBudget};
use screeps_combat_decision::kite::SquadTacticParams;

let pop = strategy_population();
let result = run_tournament(&pop, TournamentBudget::Quick);
print!("{}", report(&result)); // mean-payoff ranking + meta-Nash weights

// Is the shipped default exploitable by anything in the population?
let exploit = exploitability(SquadTacticParams::default(), &pop, TournamentBudget::Quick);
println!("default exploitability = {exploit} net HP (<= 0 ⇒ unexploitable)");
```

Run the position-search CPU bench (the ADR 0019 Stage 3b budget gate):

```rust
use screeps_combat_eval::bench::run_compound_worst_case;

let r = run_compound_worst_case(25);
println!("{} blocks x {} ticks: {:.1} us/block-tick ({} plans)", r.blocks, r.ticks, r.per_block_tick_us, r.plans);
```

## Running the register

The crate ships an example that runs the register and prints the report — the tactics-tuning dashboard:

```bash
cargo run --example run_register -p screeps-combat-eval
```

It prints, per experiment, the pass/fail mark, the hypothesis, and each gated metric with its measured value and the
gate it was checked against, e.g.:

```text
EXP-* register — 10/10 experiments passed
[ok] EXP-FOUND-1 — two-phase kill inequality predicts kill-or-not (damage-then-heal netting)
    [ok] 70 dps vs 36 heal → target dies (gate dies)
    [ok] 30 dps vs 60 heal → target survives (gate survives)
...
```

The gates also run under the test suite, so `cargo test -p screeps-combat-eval` enforces the register (all
experiments pass), the position-search budget, and the tournament ship-gate as standing regression guards:

```bash
cargo test -p screeps-combat-eval
```

## The experiment register

`register()` runs these `EXP-*` experiments (each gated to pass/fail):

| ID | Hypothesis (gist) |
| --- | --- |
| `EXP-FOUND-1` | The two-phase kill inequality predicts kill-or-not (DPS > heal ⇒ dies; DPS < heal ⇒ survives). |
| `EXP-KITE-1` | A range-3 kiter at MOVE parity takes 0 melee damage and chips the chaser to death. |
| `EXP-FOCUS-1` | Focus-fire out-DPSes the aggregate heal: 3×ranged clears a self-healing turtle fast, unscathed. |
| `EXP-TOWER-1` | An edge drain sustains via self-heal and bleeds the tower's energy to zero. |
| `EXP-COMP-1` | A higher-DPS composition clears a heal-wall strictly faster (duo vs quad ticks-to-clear). |
| `EXP-BREACH-1` | A ranged siege breaks a hostile rampart **shield** before the spawn it covers (shield-over-spawn apply layer). |
| `EXP-NEST-1` | A 3-tower defender nest deals attributed tower damage and bleeds attackers (no defender creeps). |
| `EXP-COHESION-1` | A managed ranged squad threads a wall corridor cohesively (one shared kite goal), focus-fires, and survives. |
| `EXP-POS-SELFPLAY-1` | Two managed ranged squads close and trade fire (the utility drives both sides to contact) and stay cohesive. |
| `EXP-POS-KITE-1` | A managed ranged squad kites a melee squad — out-survives it and kills it without being caught. |

The first group (`FOUND` / `KITE` / `FOCUS` / `TOWER` / `COMP`) are the foundational sim-runnable experiments; the
`BREACH` / `NEST` group exercises the room-variety scenario builder (walls / ramparts / towers); the `COHESION` /
`POS-*` group drives the full ADR 0019 managed-squad positioning utility. The harder register items (BREACH-2 / DEF-2
/ CTRL / PARITY) and the sim-vs-server parity oracle are follow-on increments.

## Module reference

- **`(crate root)`** — the `EXP-*` register itself: [`register`], [`report`], and the framework types [`Metric`]
  and [`ExperimentResult`].
- **`metrics`** — [`SideMetrics::from_recording`] computes one side's performance over a [`CombatRecording`] across
  five families (healing, DPS, positioning, survivability, efficiency). Tower damage is attributed separately from
  creep DPS, and cohesion reuses the shared `screeps_combat_decision::cohesion::measure` (the same measure the live
  seg-57 canary emits). Also exposes [`cohesion_series`] and [`worst_cohesion`].
- **`scoring`** — self-play / stalemate adjudication: [`score`] returns an [`EngagementScore`] with a [`Verdict`]
  (SideA / SideB / Draw). Non-decisive engagements are scored on **residual HP slope** (who is winning the recent war
  of attrition), not HP level — so a passive turtle can't game it.
- **`bench`** — the position-search CPU bench ([`run_compound_worst_case`], [`run_compound_worst_case_shared`]) over a
  compound worst case (open room, 6 towers, mobile chasers, 4 converging blocks). Establishes and guards the per-tick
  budget for the unified position utility.
- **`tournament`** — self-play tournament + exploitability ship-gate: [`run_tournament`] builds an antisymmetric
  payoff matrix (the `matrix: Vec<Vec<i64>>` field of [`TournamentResult`]) over a [`BASKET`] of beds (open field,
  corridor, tower crossfire), ranks strategies by mean payoff, and solves a meta-Nash mix via fictitious play
  ([`meta_nash`]). [`exploitability`] is the robustness gate ("is there a hard counter to how we fight?"). Budget is a
  [`TournamentBudget`] tier (`Quick` for CI, `Thorough` for final eval).

## How it works

Every experiment is run through the same authoritative path the rest of the family uses: scenarios are assembled
(directly, or via `screeps_combat_agent::scenario::ScenarioBuilder` for walls/ramparts/towers), then resolved tick by
tick by `screeps_combat_engine`. Per-creep scenarios use `screeps_combat_agent::opponents::run_engagement` (the real
`IbexAgent` vs a scripted opponent — `RushAgent`, `TurtleAgent`, `DrainAgent`, …); managed-squad scenarios drive
`ManagedSimSquad` through `decide_squad_with_pathing` (the ADR 0019 positioning utility), with two managed squads
fighting head-to-head via the crate's `run_managed` runner that merges every squad's intents each tick.

Because metrics, scoring, and gates all read the resulting `CombatRecording` (a per-frame replay) rather than any
agent's internal state, the evaluation is **body-free where it can be** (positioning reads geometry; stalemate
scoring reads per-frame HP) and uncontaminated (tower output is subtracted out of creep DPS). The whole thing is
deterministic — same code, same seed, same numbers — which is what makes it a usable regression gate and a
reproducible tuning loop.

## Related crates

- [screeps-combat-engine](https://github.com/Azaril/screeps-combat-engine) — the authoritative tick-resolution combat
  simulator (bodies, damage, towers, recordings) this layer runs scenarios through.
- [screeps-combat-decision](https://github.com/Azaril/screeps-combat-decision) — the tactics/position-scoring layer
  (kite presets, cohesion, the tunables this register exists to tune).
- [screeps-combat-agent](https://github.com/Azaril/screeps-combat-agent) — the agents, scenario builder, and
  `run_engagement` harness that drive `IbexAgent` and scripted opponents through the engine.
