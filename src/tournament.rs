//! Self-play tournament + exploitability ship-gate (ADR 0020 §4.3 step 4 / §5).
//!
//! Generalizes the single-bed `sweep_kite_weights` into a **population tournament over a bed basket**:
//! every strategy (a `SquadTacticParams`) plays every other in symmetric self-play across a basket of
//! beds (open field, a wall corridor, mutual tower crossfire), each match scored by the net-HP
//! exchange, into an antisymmetric [`PayoffMatrix`]. From the matrix:
//! - a zero-sum **mean-payoff ranking** (who beats the field);
//! - the **exploitability** of a candidate — the largest margin any population strategy beats it by —
//!   the robustness **ship-gate** (is there a hard counter to how we fight? — the "adaptive, not
//!   counterable" test);
//! - a **meta-Nash mixed strategy** (fictitious play) — the robust mix to randomize over, which is the
//!   bridge to the step-6 adaptivity layer.
//!
//! The **bed basket** is what gives the gate teeth: a single open-field bed is low-decisiveness
//! (strategies tie), but terrain + tower pressure make positioning/engage choices actually diverge.
//! Budget is a tunable tier (operator §8.4): `Quick` (CI) vs `Thorough` (final eval). All matches run
//! on the deterministic Rust sim — reproducible, no GPU/ML.
//!
//! Residual (next): asymmetric attacker-vs-defender beds with the §8.6 objective-aware turtle scorer,
//! scripted archetypes vs the managed squad, PFSP opponent mixing + behavioral de-dup, and formal Elo
//! (equivalent to the mean-payoff ranking for a complete round-robin, so omitted here).

use screeps::{Part, Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_decision::kernel::KernelParams;
use screeps_combat_decision::kite::{KiteScoreParams, SquadTacticParams};
use screeps_combat_engine::{CombatWorld, PlayerId, SimBody, SimCreep, SimTower};

use rayon::prelude::*;

use crate::harness::generate::Rng;
use crate::harness::roster::random_squad;
use crate::harness::terrain_import::{decode_terrain, fixtures};
use crate::harness::validate::assault_score;
use crate::{ranged_file, run_managed};

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos(x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room())
}

/// A named tactical strategy = the position-scoring weights the managed squad fights with.
#[derive(Clone, Copy, Debug)]
pub struct Strategy {
    pub name: &'static str,
    pub tactics: SquadTacticParams,
}

/// Tournament compute budget (operator §8.4): same code, different depth.
#[derive(Clone, Copy, Debug)]
pub enum TournamentBudget {
    /// CI / iteration — short matches.
    Quick,
    /// Final evaluation — longer matches (more of each fight resolves).
    Thorough,
}

impl TournamentBudget {
    fn ticks(self) -> usize {
        match self {
            TournamentBudget::Quick => 50,
            TournamentBudget::Thorough => 100,
        }
    }
}

/// A symmetric self-play bed. Each is mirror-symmetric (both sides identical, opposite ends) so the
/// antisymmetric payoff is meaningful; the basket spans the regimes where strategies diverge.
#[derive(Clone, Copy, Debug)]
pub enum Bed {
    /// Open room — a straight ranged brawl (low-decisiveness, the baseline).
    OpenField,
    /// A central wall with a 3-wide gap — both squads must thread the corridor to engage (cohesion +
    /// advance choices matter).
    Corridor,
    /// Each side has a tower covering the centre — fighting under mutual crossfire (the safety term +
    /// Lanchester heal/tower calc bite).
    TowerCrossfire,
    /// REAL imported terrain (fixture `idx`), **mirror-symmetrized** (left half reflected to the right) so
    /// the self-play payoff stays meaningful — both sides fight across identical real walls/swamps (ADR
    /// 0025 §12 Stage 4). Deployment zones are forced clear so squads always field.
    Imported(usize),
}

/// The standard synthetic basket the tournament averages each match over.
pub const BASKET: [Bed; 3] = [Bed::OpenField, Bed::Corridor, Bed::TowerCrossfire];

impl Bed {
    /// A distinct per-bed seed key (the enum carries data now, so it is not `as u32`-castable).
    fn key(&self) -> u32 {
        match self {
            Bed::OpenField => 0,
            Bed::Corridor => 1,
            Bed::TowerCrossfire => 2,
            Bed::Imported(i) => 100 + *i as u32,
        }
    }
}

/// Apply a `bed`'s terrain + mirrored towers to a world (the symmetric battlefield, creeps aside).
fn apply_bed_terrain(world: &mut CombatWorld, bed: Bed) {
    match bed {
        Bed::OpenField => {}
        Bed::Imported(idx) => {
            let fx = fixtures();
            let real = decode_terrain(&fx[idx % fx.len()].terrain);
            // Mirror the left half (x<25) onto the right (49-x) → a symmetric battlefield from real terrain.
            for x in 0..25u8 {
                for y in 0..50u8 {
                    let (wall, swamp) = (real.is_wall(x, y), real.swamps.contains(&(x, y)));
                    for tx in [x, 49 - x] {
                        if wall {
                            world.terrain.walls.insert((tx, y));
                        } else if swamp {
                            world.terrain.swamps.insert((tx, y));
                        }
                    }
                }
            }
            // Clear the two deployment zones (start files + a move-out margin) so both squads always field.
            for x in 0..12u8 {
                for y in 18..32u8 {
                    for tx in [x, 49 - x] {
                        world.terrain.walls.remove(&(tx, y));
                        world.terrain.swamps.remove(&(tx, y));
                    }
                }
            }
        }
        Bed::Corridor => {
            for y in 0..=49u8 {
                if !(24..=26).contains(&y) {
                    world.terrain.walls.insert((25, y));
                }
            }
        }
        Bed::TowerCrossfire => {
            world.towers.push(SimTower { id: 100, owner: 0, pos: pos(14, 25), energy: 1000, hits: 3000, hits_max: 3000 });
            world.towers.push(SimTower { id: 101, owner: 1, pos: pos(35, 25), energy: 1000, hits: 3000, hits_max: 3000 });
        }
    }
}

/// Build a fresh symmetric world for `bed`: two identical 3×ranged squads at opposite ends.
fn build_bed(bed: Bed) -> CombatWorld {
    let mut creeps = ranged_file(0, 1, 8, 24, 3);
    creeps.extend(ranged_file(1, 11, 41, 24, 3));
    let mut world = CombatWorld { creeps, ..Default::default() };
    apply_bed_terrain(&mut world, bed);
    world
}

/// Build a symmetric world where BOTH sides field the same `bodies` (a random composition), at opposite
/// ends of `bed` — so a self-play match isolates the KernelParams difference while the *composition* is
/// varied across the basket (ADR 0025 basket enrichment: tune against diverse comps, not mirror-of-ranged).
fn build_bed_comp(bed: Bed, bodies: &[Vec<Part>]) -> CombatWorld {
    let file = |owner: PlayerId, first: u32, x: u8| -> Vec<SimCreep> {
        bodies
            .iter()
            .enumerate()
            .map(|(i, b)| SimCreep { id: first + i as u32, owner, pos: pos(x, 22 + i as u8), body: SimBody::unboosted(b), fatigue: 0 })
            .collect()
    };
    let mut creeps = file(0, 1, 8);
    creeps.extend(file(1, 1000, 41));
    let mut world = CombatWorld { creeps, ..Default::default() };
    apply_bed_terrain(&mut world, bed);
    world
}

/// One match on `bed`: side-0 fights with `side0`, side-1 with `side1`. Returns side-0's net-HP
/// advantage (HP it retained − HP side-1 retained); a wipe shows as the full margin (decisive).
fn play_bed(bed: Bed, side0: SquadTacticParams, side1: SquadTacticParams, ticks: usize) -> i64 {
    let mut world = build_bed(bed);
    let a_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 0).map(|c| c.id).collect();
    let b_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 1).map(|c| c.id).collect();
    let mut squads = [
        ManagedSimSquad::new(0, a_ids, pos(41, 25)).with_tactics(side0),
        ManagedSimSquad::new(1, b_ids, pos(8, 25)).with_tactics(side1),
    ];
    run_managed(&mut world, &mut squads, ticks);
    let kept = |owner| -> i64 { world.creeps.iter().filter(|c| c.owner == owner && c.is_alive()).map(|c| c.body.hits as i64).sum() };
    kept(0) - kept(1)
}

/// Like [`play_bed`] but both sides field the given (random) composition — the comp-varied match.
fn play_bed_comp(bed: Bed, bodies: &[Vec<Part>], side0: SquadTacticParams, side1: SquadTacticParams, ticks: usize) -> i64 {
    let mut world = build_bed_comp(bed, bodies);
    let a_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 0).map(|c| c.id).collect();
    let b_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 1).map(|c| c.id).collect();
    let mut squads = [
        ManagedSimSquad::new(0, a_ids, pos(41, 25)).with_tactics(side0),
        ManagedSimSquad::new(1, b_ids, pos(8, 25)).with_tactics(side1),
    ];
    run_managed(&mut world, &mut squads, ticks);
    let kept = |owner| -> i64 { world.creeps.iter().filter(|c| c.owner == owner && c.is_alive()).map(|c| c.body.hits as i64).sum() };
    kept(0) - kept(1)
}

/// A **diverse, comp-varied basket**: each `Bed` × `n_comps` seeded random squad compositions (ADR 0025
/// — tune the kernel against a population of compositions, not just mirror-of-ranged). Both sides field
/// the same comp per entry, so a match isolates the KernelParams while the basket spans comps + terrain.
pub fn comp_basket(n_comps: u32, energy: u32) -> Vec<(Bed, Vec<Vec<Part>>)> {
    let mut out = Vec::new();
    for &bed in &BASKET {
        for s in 0..n_comps {
            // Distinct seed per (bed, comp) so the population is varied + reproducible; 2–5 creeps a side.
            let mut rng = Rng::seeded(s * BASKET.len() as u32 + bed.key() + 1);
            let n = rng.range(2, 5) as u8;
            out.push((bed, random_squad(&mut rng, energy, n)));
        }
    }
    out
}

/// The §12 Stage 4 **realistic open-combat basket**: the synthetic [`comp_basket`] PLUS a few imported
/// real-terrain beds (mirror-symmetrized), each with seeded random comps — so the kernel tournament tunes
/// over real walls/swamps, not only hand-authored terrain.
pub fn realistic_comp_basket(n_comps: u32, energy: u32) -> Vec<(Bed, Vec<Vec<Part>>)> {
    let mut out = comp_basket(n_comps, energy);
    let n_fix = fixtures().len().min(4); // a bounded handful of real beds (keeps `Thorough` in minutes)
    for i in 0..n_fix {
        for s in 0..n_comps {
            let mut rng = Rng::seeded(s.wrapping_mul(97).wrapping_add(i as u32).wrapping_add(500));
            let n = rng.range(2, 5) as u8;
            out.push((Bed::Imported(i), random_squad(&mut rng, energy, n)));
        }
    }
    out
}

/// The §12 Stage 4 **realistic base-attack set**: the `Raze` scenarios from the foreman-planned bases +
/// the imported rooms (the "destroy the base" lens over real terrain + real foreman layouts). `Raze` is
/// the breach-relevant objective; the other kinds exercise plumbing, not positioning under fire.
pub fn realistic_base_scenarios() -> Vec<crate::harness::scenario::Scenario> {
    use crate::harness::generate::{ForemanGenerator, Generator, ImportedRoom};
    use crate::harness::scenario::ObjectiveKind;
    // Raze (destroy the core) + Breach (crack the rampart ring) — the two breach-relevant objectives that
    // put positioning under fire; Farm/Secure/Declaim exercise plumbing, not assault positioning.
    let attack = |s: &crate::harness::scenario::Scenario| matches!(s.objectives[0].kind, ObjectiveKind::Raze | ObjectiveKind::Breach);
    let fg = ForemanGenerator { n_comps: 1 };
    let ir = ImportedRoom { multi_room: false, n_comps: 1 };
    let mut out: Vec<_> = (0..fg.count()).map(|i| fg.generate(i)).filter(attack).collect();
    out.extend((0..ir.count()).map(|i| ir.generate(i)).filter(attack));
    out
}

/// Antisymmetric payoff of `a` vs `b` over a **comp-varied basket** (both side assignments, cancelling
/// start-side bias). The diverse-opponent analogue of [`payoff`].
pub fn payoff_over_comps(basket: &[(Bed, Vec<Vec<Part>>)], a: SquadTacticParams, b: SquadTacticParams, ticks: usize) -> i64 {
    if basket.is_empty() {
        return 0;
    }
    let sum: i64 = basket.iter().map(|(bed, bodies)| (play_bed_comp(*bed, bodies, a, b, ticks) - play_bed_comp(*bed, bodies, b, a, ticks)) / 2).sum();
    sum / basket.len() as i64
}

/// Round-robin `strategies` over a comp-varied basket (the kernel-tuning analogue of [`run_tournament`]).
pub fn run_tournament_over_comps(strategies: &[Strategy], basket: &[(Bed, Vec<Vec<Part>>)], ticks: usize) -> TournamentResult {
    let n = strategies.len();
    let mut matrix = vec![vec![0i64; n]; n];
    // Each upper-triangle cell is an independent round-robin sum — run them in PARALLEL (rayon). Matches
    // are pure (fresh world per call), so this is deterministic regardless of completion order.
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
    let cells: Vec<(usize, usize, i64)> = pairs
        .par_iter()
        .map(|&(i, j)| (i, j, payoff_over_comps(basket, strategies[i].tactics, strategies[j].tactics, ticks)))
        .collect();
    for (i, j, p) in cells {
        matrix[i][j] = p;
        matrix[j][i] = -p;
    }
    let mut ranking: Vec<(usize, f64)> = (0..n).map(|i| (i, matrix[i].iter().sum::<i64>() as f64 / n.max(1) as f64)).collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let nash = meta_nash(&matrix, 2000);
    TournamentResult { names: strategies.iter().map(|s| s.name).collect(), matrix, ranking, nash }
}

/// Antisymmetric payoff of `a` vs `b`, **averaged over the bed basket** and over both side
/// assignments (to cancel the start-side bias the deterministic tie-break introduces).
/// `payoff(a,b) == -payoff(b,a)`.
pub fn payoff(a: SquadTacticParams, b: SquadTacticParams, ticks: usize) -> i64 {
    let per_bed: i64 = BASKET.iter().map(|&bed| (play_bed(bed, a, b, ticks) - play_bed(bed, b, a, ticks)) / 2).sum();
    per_bed / BASKET.len() as i64
}

/// The shipped-default population the gate runs against — the default plus deliberate archetypes a
/// real opponent might field. A candidate any of these beats decisively is exploitable.
pub fn strategy_population() -> Vec<Strategy> {
    let base = SquadTacticParams::default();
    let with_engage = |f: fn(&mut KiteScoreParams)| {
        let mut e = base.engage;
        f(&mut e);
        SquadTacticParams { kite: base.kite, engage: e, healer: base.healer, kernel: base.kernel }
    };
    let with_kite = |f: fn(&mut KiteScoreParams)| {
        let mut k = base.kite;
        f(&mut k);
        SquadTacticParams { kite: k, engage: base.engage, healer: base.healer, kernel: base.kernel }
    };
    vec![
        Strategy { name: "default", tactics: base },
        Strategy { name: "aggressive", tactics: with_engage(|e| { e.w_dmg = 3.0; e.w_taken = 0.3; }) },
        Strategy { name: "cautious", tactics: with_engage(|e| { e.w_taken = 1.5; e.w_dmg = 1.0; }) },
        Strategy { name: "kite-heavy", tactics: with_kite(|k| { k.w_future = 2.0; k.w_prox = 1.0; }) },
        Strategy { name: "advance-heavy", tactics: with_engage(|e| { e.w_prox = 3.0; }) },
    ]
}

/// The ADR-0025 EV-**kernel** tuning population: the shipped default plus deliberate variations of the
/// kernel's position-shaping seam ([`KernelParams`]) — the tournament's verdict on which positioning
/// constants win the self-play field (and whether the shipped seed is exploitable). This is what "tune
/// the kernel" means now that engaged positioning is the kernel, not the kite weights.
pub fn kernel_population() -> Vec<Strategy> {
    let base = SquadTacticParams::default();
    let with_kernel = |name: &'static str, f: fn(&mut KernelParams)| {
        let mut k = base.kernel;
        f(&mut k);
        Strategy { name, tactics: SquadTacticParams { kernel: k, ..base } }
    };
    vec![
        Strategy { name: "k-default", tactics: base }, // approach 2 / incumbency 3 / discoh 10 / K 3 / spacing 1
        with_kernel("k-approach-hot", |k| k.approach_coef = 4), // close harder
        with_kernel("k-sticky", |k| k.incumbency_coef = 6),     // stronger hold (less jitter, less responsive)
        with_kernel("k-loose-coh", |k| { k.cohesion_k = 5; k.discohesion_coef = 4; }), // spread more (cover more tiles)
        with_kernel("k-tight-coh", |k| { k.cohesion_k = 2; k.discohesion_coef = 20; }), // ball up tight
        with_kernel("k-spread", |k| k.spacing_coef = 4),        // anti-stack harder
    ]
}

/// A FINE grid sweep of the kernel's position-shaping seam for the §12 Stage 4 **thorough** re-tune
/// (`approach × incumbency × cohesion` = 48 configs) — the many-minutes population the rayon-parallel
/// tournament explores to map the open-combat ↔ base-attack tradeoff surface. Names are leaked
/// (`a{approach}-i{incumbency}-{cohesion}`); acceptable for an on-demand dashboard.
pub fn kernel_population_grid() -> Vec<Strategy> {
    let base = SquadTacticParams::default();
    let mut out = Vec::new();
    for approach in [1i64, 2, 3, 4] {
        for incumbency in [2i64, 3, 4, 6] {
            for (ck, dc, tag) in [(2u32, 20i64, "tight"), (3, 10, "def"), (5, 4, "loose")] {
                let mut k = base.kernel;
                k.approach_coef = approach;
                k.incumbency_coef = incumbency;
                k.cohesion_k = ck;
                k.discohesion_coef = dc;
                let name: &'static str = Box::leak(format!("a{approach}-i{incumbency}-{tag}").into_boxed_str());
                out.push(Strategy { name, tactics: SquadTacticParams { kernel: k, ..base } });
            }
        }
    }
    out
}

/// The result of a tournament: the antisymmetric payoff matrix, each strategy's mean payoff (the
/// zero-sum ranking score), and the meta-Nash mixed strategy (the robust mix to randomize over).
#[derive(Clone, Debug)]
pub struct TournamentResult {
    pub names: Vec<&'static str>,
    pub matrix: Vec<Vec<i64>>,
    /// `(strategy index, mean payoff over the field)`, best first.
    pub ranking: Vec<(usize, f64)>,
    /// Meta-Nash mixed strategy (probabilities over `names`) — the step-6 adaptivity mixing distribution.
    pub nash: Vec<f64>,
}

/// Meta-Nash mixed strategy of a symmetric zero-sum payoff matrix via **fictitious play**: each round
/// best-responds to the opponent's empirical mix; the empirical play frequencies converge to a Nash
/// equilibrium. The result is the robust randomization weight per strategy (dominated strategies → ~0).
pub fn meta_nash(matrix: &[Vec<i64>], iters: usize) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let mut counts = vec![1.0f64; n]; // empirical play counts, start uniform
    for _ in 0..iters {
        let total: f64 = counts.iter().sum();
        let (mut best, mut best_v) = (0usize, f64::NEG_INFINITY);
        for (i, row) in matrix.iter().enumerate() {
            let v: f64 = row.iter().zip(&counts).map(|(&p, &c)| p as f64 * c / total).sum();
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        counts[best] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    counts.iter().map(|c| c / total).collect()
}

/// Run the full round-robin over `strategies` (each pair over the bed basket) and rank + solve Nash.
pub fn run_tournament(strategies: &[Strategy], budget: TournamentBudget) -> TournamentResult {
    let ticks = budget.ticks();
    let n = strategies.len();
    let mut matrix = vec![vec![0i64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let p = payoff(strategies[i].tactics, strategies[j].tactics, ticks);
            matrix[i][j] = p;
            matrix[j][i] = -p;
        }
    }
    let mut ranking: Vec<(usize, f64)> = (0..n).map(|i| (i, matrix[i].iter().sum::<i64>() as f64 / n.max(1) as f64)).collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let nash = meta_nash(&matrix, 2000);
    TournamentResult { names: strategies.iter().map(|s| s.name).collect(), matrix, ranking, nash }
}

/// **Exploitability** of `candidate` against `population`: the largest margin (net HP) any population
/// strategy beats it by. ≤ 0 ⇒ unexploitable by the field (a robust strategy). The ship-gate.
pub fn exploitability(candidate: SquadTacticParams, population: &[Strategy], budget: TournamentBudget) -> i64 {
    let ticks = budget.ticks();
    population.par_iter().map(|opp| payoff(opp.tactics, candidate, ticks)).max().unwrap_or(0)
}

/// **Base attack/defend tuning** (ADR 0025 — the asymmetric lens, vs the symmetric open-combat
/// tournament). Each strategy's managed attacker squad assaults every defended base in `scenarios`; we
/// rank by total objective-progress [`assault_score`] (raze the base + survive). No payoff matrix — the
/// "opponent" is the base, so it's an absolute-score ranking, not self-play. Returns `(index, total
/// score)`, best first.
pub fn base_attack_ranking(strategies: &[Strategy], scenarios: &[crate::harness::scenario::Scenario]) -> Vec<(usize, i64)> {
    // Score every (strategy × base) assault in PARALLEL (rayon) — each is an independent managed sim — then
    // reduce per strategy. This is the heaviest Stage-4 computation (winnable siege forces over real bases).
    let pairs: Vec<(usize, &crate::harness::scenario::Scenario)> =
        strategies.iter().enumerate().flat_map(|(i, _)| scenarios.iter().map(move |sc| (i, sc))).collect();
    let scored: Vec<(usize, i64)> = pairs
        .par_iter()
        .filter_map(|&(i, sc)| assault_score(sc, strategies[i].tactics).map(|a| (i, a.score)))
        .collect();
    let mut totals = vec![0i64; strategies.len()];
    for (i, s) in scored {
        totals[i] += s;
    }
    let mut ranking: Vec<(usize, i64)> = totals.into_iter().enumerate().collect();
    ranking.sort_by_key(|&(_, s)| std::cmp::Reverse(s));
    ranking
}

/// Render a base-attack ranking as a readable table.
pub fn base_attack_report(strategies: &[Strategy], ranking: &[(usize, i64)]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "Base attack/defend ranking — total objective-progress over the base set:");
    for &(i, score) in ranking {
        let _ = writeln!(s, "  {:>14}  {:+}", strategies[i].name, score);
    }
    s
}

/// Render a tournament result as a readable table (the tuning-loop dashboard).
pub fn report(result: &TournamentResult) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "Self-play tournament — {} strategies over {} beds (mean payoff | Nash weight):", result.names.len(), BASKET.len());
    for &(i, score) in &result.ranking {
        let _ = writeln!(s, "  {:>14}  {:+6.0} | {:.2}", result.names[i], score, result.nash[i]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn tournament_matrix_is_antisymmetric_and_zero_sum() {
        let pop = strategy_population();
        let r = run_tournament(&pop, TournamentBudget::Quick);
        for i in 0..pop.len() {
            assert_eq!(r.matrix[i][i], 0, "a strategy ties itself");
            for j in 0..pop.len() {
                assert_eq!(r.matrix[i][j], -r.matrix[j][i], "payoff is antisymmetric");
            }
        }
        assert!(r.ranking.iter().map(|&(_, s)| s).sum::<f64>().abs() < 1.0, "zero-sum: ranking sums to ~0");
        // Nash is a valid distribution.
        assert!((r.nash.iter().sum::<f64>() - 1.0).abs() < 1e-6, "Nash mix sums to 1");
        assert!(r.nash.iter().all(|&w| w >= 0.0), "Nash weights are non-negative");
    }

    #[test]
    fn shipped_default_is_not_grossly_exploitable() {
        // The robustness ship-gate (ADR 0020): no population archetype beats the shipped default by
        // more than a gross margin across the bed basket — our default fighting style has no hard
        // counter in the field. (A tighter Nash/exploitability bound + asymmetric objective beds land
        // with the adaptivity layer; this is the standing regression guard.)
        let pop = strategy_population();
        let exploit = exploitability(SquadTacticParams::default(), &pop, TournamentBudget::Quick);
        println!("[ADR0020 tournament] default exploitability = {exploit} net HP\n{}", report(&run_tournament(&pop, TournamentBudget::Quick)));
        const GROSS: i64 = 1500; // ~1.5 creeps' HP; a real hard-counter exceeds this
        assert!(exploit <= GROSS, "the shipped default has a hard counter in the population ({exploit} net HP) — needs adaptivity or a retune");
    }

    /// ADR 0025 — kick off self-play tuning of the EV kernel: round-robin the [`kernel_population`]
    /// (variations of the kernel's position-shaping seam) over the bed basket, rank by mean payoff, solve
    /// the meta-Nash mix, and report the shipped seed's exploitability. Run on demand (it is the tuning
    /// dashboard, not a CI gate): `cargo test -p screeps-combat-eval --lib kernel_tournament -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kernel_tournament() {
        let pop = kernel_population();
        // Comp-varied basket (ADR 0025): 4 random comps per bed × 3 beds = 12 diverse battlefields, so the
        // ranking reflects which KernelParams wins across *compositions + terrain*, not one fixed comp.
        let basket = comp_basket(4, 5600);
        let r = run_tournament_over_comps(&pop, &basket, TournamentBudget::Thorough.ticks());
        println!("{}", report(&r));
        let best = r.ranking[0];
        println!("[ADR0025 kernel tournament] {} beds × comps; field winner = {} ({:+.0} mean payoff, Nash {:.2})", basket.len(), r.names[best.0], best.1, r.nash[best.0]);
    }

    /// ADR 0025 — the BASE attack/defend tuning lens (vs the symmetric open-combat `kernel_tournament`):
    /// rank the kernel population by how well each assaults the realistic defended-base set (objective
    /// progress + survival). Run:
    /// `cargo test -p screeps-combat-eval --lib base_attack_tournament -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn base_attack_tournament() {
        let pop = kernel_population();
        let bases = crate::harness::generate::realistic_bases();
        let ranking = base_attack_ranking(&pop, &bases);
        println!("{}", base_attack_report(&pop, &ranking));
        let (best, score) = ranking[0];
        println!("[ADR0025 base-attack] {} bases; best assaulter = {} ({:+})", bases.len(), pop[best].name, score);
        // Sanity: the assault makes SOME objective progress across the set (not a total wall).
        assert!(score > 0, "no kernel config made any base progress — investigate breach/siege");
    }

    /// ADR 0025 §12 Stage 4 — the **realistic** open-combat re-tune: round-robin the kernel population over
    /// the synthetic basket PLUS imported real-terrain beds (rayon-parallel). Reports the field winner, the
    /// Nash-heaviest (robust) config, and the shipped default's exploitability. Run:
    /// `cargo test -p screeps-combat-eval --lib realistic_kernel_tournament -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn realistic_kernel_tournament() {
        let pop = kernel_population();
        let basket = realistic_comp_basket(3, 5600);
        let r = run_tournament_over_comps(&pop, &basket, TournamentBudget::Thorough.ticks());
        println!("{}", report(&r));
        let best = r.ranking[0];
        let nash_best = (0..r.nash.len()).max_by(|&a, &b| r.nash[a].partial_cmp(&r.nash[b]).unwrap()).unwrap();
        let exploit = exploitability(SquadTacticParams::default(), &pop, TournamentBudget::Thorough);
        println!(
            "[ADR0025 §12 realistic kernel tournament] {} beds (synthetic + imported real terrain)\n  field winner = {} ({:+.0} mean payoff)\n  Nash-heaviest (robust) = {} ({:.2})\n  shipped-default exploitability = {} net HP",
            basket.len(), r.names[best.0], best.1, r.names[nash_best], r.nash[nash_best], exploit
        );
    }

    /// ADR 0025 §12 Stage 4 — the **realistic** base-attack re-tune: rank the kernel population by how well
    /// each razes the foreman-planned + imported real bases (rayon-parallel managed sims). Run:
    /// `cargo test -p screeps-combat-eval --lib realistic_base_attack -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn realistic_base_attack() {
        let pop = kernel_population();
        let bases = realistic_base_scenarios();
        let ranking = base_attack_ranking(&pop, &bases);
        println!("{}", base_attack_report(&pop, &ranking));
        let (best, score) = ranking[0];
        println!("[ADR0025 §12 realistic base-attack] {} real bases (foreman + imported, Raze/Breach); best assaulter = {} ({:+})", bases.len(), pop[best].name, score);
    }

    /// ADR 0025 §12 Stage 4 — the **THOROUGH** re-tune (operator-requested many-minutes run): sweep the
    /// 48-config [`kernel_population_grid`] over BOTH lenses — the realistic open-combat tournament (large
    /// comp-varied basket + imported real terrain) AND the realistic base-attack set — fully rayon-
    /// parallel, then report the open↔base tradeoff surface + the best-open / best-base / best-BALANCED
    /// (rank-sum) configs. Run:
    /// `cargo test --release -p screeps-combat-eval --lib realistic_full_retune -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn realistic_full_retune() {
        let grid = kernel_population_grid();
        let basket = realistic_comp_basket(8, 5600);
        let bases = realistic_base_scenarios();
        let n = grid.len();
        println!("[realistic full re-tune] {n} kernel configs × {} self-play beds + {} base scenarios (rayon-parallel)…", basket.len(), bases.len());

        let t = run_tournament_over_comps(&grid, &basket, TournamentBudget::Thorough.ticks());
        let base = base_attack_ranking(&grid, &bases);

        // Per-config metrics. exploitability_i = the largest margin any opponent beats i by = -min(row i).
        let mean: Vec<f64> = (0..n).map(|i| t.matrix[i].iter().sum::<i64>() as f64 / n as f64).collect();
        let exploit: Vec<i64> = (0..n).map(|i| -t.matrix[i].iter().copied().min().unwrap_or(0)).collect();
        let mut base_score = vec![0i64; n];
        for &(i, s) in &base {
            base_score[i] = s;
        }
        // Rank each lens (0 = best); the balanced pick minimizes the rank-sum.
        let mut open_order: Vec<usize> = (0..n).collect();
        open_order.sort_by(|&a, &b| mean[b].partial_cmp(&mean[a]).unwrap());
        let mut base_order: Vec<usize> = (0..n).collect();
        base_order.sort_by_key(|&i| std::cmp::Reverse(base_score[i]));
        let mut open_rank = vec![0usize; n];
        let mut base_rank = vec![0usize; n];
        for (r, &i) in open_order.iter().enumerate() {
            open_rank[i] = r;
        }
        for (r, &i) in base_order.iter().enumerate() {
            base_rank[i] = r;
        }
        let mut balanced: Vec<usize> = (0..n).collect();
        balanced.sort_by_key(|&i| open_rank[i] + base_rank[i]);

        println!("\n{:>14} | {:>9} | {:>7} | {:>9} | open#  base#", "config", "open-mean", "exploit", "base");
        for &i in &balanced {
            println!("{:>14} | {:>+9.0} | {:>7} | {:>+9} |  {:>3}   {:>3}", grid[i].name, mean[i], exploit[i], base_score[i], open_rank[i], base_rank[i]);
        }
        let (bo, bb, bal) = (open_order[0], base_order[0], balanced[0]);
        println!(
            "\n[ADR0025 §12 THOROUGH re-tune] {n} configs\n  best OPEN-combat = {} ({:+.0} mean, exploit {})\n  best BASE-attack = {} ({:+})\n  best BALANCED    = {} (open#{} base#{}, mean {:+.0}, exploit {}, base {:+})",
            grid[bo].name, mean[bo], exploit[bo], grid[bb].name, base_score[bb],
            grid[bal].name, open_rank[bal], base_rank[bal], mean[bal], exploit[bal], base_score[bal],
        );
    }

    #[test]
    fn ev_per_cpu_at_large_n_is_bounded() {
        // ADR 0020 §5/§7: a design that wins on HP but blows the per-tick CPU budget at large N must
        // FAIL the gate. Time a 10-v-10 managed self-play (the blob regime, step 5) and bound the
        // per-squad-tick cost. LOOSE (native-host proxy, like bench.rs) — a death-spiral guard, not a
        // tight Screeps-ms threshold.
        let mut world = build_bed(Bed::OpenField);
        // Scale up to 10 creeps a side.
        world.creeps = ranged_file(0, 1, 8, 20, 10);
        world.creeps.extend(ranged_file(1, 21, 41, 20, 10));
        let a_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 0).map(|c| c.id).collect();
        let b_ids: Vec<_> = world.creeps.iter().filter(|c| c.owner == 1).map(|c| c.id).collect();
        let mut squads = [
            ManagedSimSquad::new(0, a_ids, pos(41, 25)),
            ManagedSimSquad::new(1, b_ids, pos(8, 25)),
        ];
        let ticks = 30usize;
        let start = Instant::now();
        run_managed(&mut world, &mut squads, ticks);
        let per_squad_tick_us = start.elapsed().as_secs_f64() * 1e6 / (ticks * 2) as f64;
        println!("[ADR0020 tournament] 10v10 EV/CPU = {per_squad_tick_us:.1} us/squad-tick");
        assert!(per_squad_tick_us < 20_000.0, "large-N managed combat blew the CPU budget: {per_squad_tick_us:.0} us/squad-tick");
    }
}
