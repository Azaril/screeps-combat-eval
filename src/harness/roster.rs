//! Random squad-composition population + **Lanchester validation** (ADR 0025 basket enrichment,
//! operator 2026-06-25). Two jobs:
//!
//! 1. **Random opponents for tuning** — sample a large, varied population of squad compositions
//!    (free-form body mixes within an energy budget, not just the bot's archetype slots), so the
//!    self-play tournament tunes the kernel against *diverse* foes instead of mirror-of-itself.
//! 2. **Lanchester validation** — for many random A-vs-B matchups, compare the engage gate's PREDICTED
//!    balance / win-probability ([`predict_engage`]) against the ACTUAL sim outcome (both sides driven by
//!    the EV kernel). The sign-agreement rate measures how well the force-strength model predicts real
//!    fights; the **outliers** (predicted-favoured but lost, or a flat-prediction blowout) surface
//!    mispredicted or degenerate ("broken") compositions worth a closer look or a model fix.

use crate::harness::generate::Rng;
use crate::run_managed;
use screeps::{Part, Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_agent::SimView;
use screeps_combat_decision::{predict_engage, CombatCreepDto, EnginePrediction, EngageObjective, SquadMemberView, SquadOrderState, SquadView};
use screeps_combat_engine::constants::{ATTACK_POWER, RANGED_ATTACK_POWER};
use screeps_combat_engine::{CombatWorld, PlayerId, SimBody, SimCreep};

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos(x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x.min(49)).unwrap(), RoomCoordinate::new(y.min(49)).unwrap(), room())
}

/// Energy cost of a body part (the standard Screeps costs) — bounds the random body to a budget.
fn part_cost(p: Part) -> u32 {
    match p {
        Part::Tough => 10,
        Part::Move | Part::Carry => 50,
        Part::Attack => 80,
        Part::Work => 100,
        Part::RangedAttack => 150,
        Part::Heal => 250,
        Part::Claim => 600,
        _ => 50,
    }
}

/// Body-degrade order (engine degrades `body[0]` first): TOUGH soaks first, MOVE/HEAL survive longest so
/// a hurt creep stays mobile + can still sustain. Realistic ordering for the sampled bodies.
fn degrade_order(p: Part) -> u8 {
    match p {
        Part::Tough => 0,
        Part::Work => 1,
        Part::Attack => 2,
        Part::RangedAttack => 3,
        Part::Carry => 4,
        Part::Heal => 5,
        Part::Move => 6,
        Part::Claim => 7,
        _ => 4,
    }
}

/// Sample ONE random creep body within `energy`: a random weight per combat part (biased toward a couple
/// of dominant roles) + always some MOVE for mobility, filled round-robin to the budget / 50-part cap,
/// then ordered for degradation. Always has at least one offensive or heal part (no inert creeps).
pub(crate) fn random_body(rng: &mut Rng, energy: u32) -> Vec<Part> {
    let mut weights = [
        (Part::Tough, rng.range(0, 3)),
        (Part::Attack, rng.range(0, 5)),
        (Part::RangedAttack, rng.range(0, 5)),
        (Part::Heal, rng.range(0, 4)),
        (Part::Work, rng.range(0, 2)),
        (Part::Move, rng.range(2, 5)), // always mobile-ish (fatigue is ~free with enough MOVE)
    ];
    if weights[1].1 + weights[2].1 + weights[3].1 + weights[4].1 == 0 {
        weights[1].1 = 2; // guarantee a weapon if the sample rolled all-zero offense/heal
    }
    let mut body: Vec<Part> = Vec::new();
    let mut spent = 0u32;
    let mut counts = [0u32; 6];
    // Fill by proportional under-representation: each step add the affordable, positive-weight part that
    // is most "behind" its weight (counts[i]/w[i] smallest, compared by integer cross-multiply — no
    // float, no starvation, so MOVE always gets its share regardless of part order).
    loop {
        let mut best: Option<usize> = None;
        for (i, &(p, w)) in weights.iter().enumerate() {
            if w == 0 || body.len() >= 50 || spent + part_cost(p) > energy {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    // i is more behind than b iff counts[i]*w[b] < counts[b]*w[i].
                    if counts[i] * weights[b].1 < counts[b] * w {
                        best = Some(i);
                    }
                }
            }
        }
        match best {
            Some(i) => {
                let p = weights[i].0;
                body.push(p);
                counts[i] += 1;
                spent += part_cost(p);
            }
            None => break,
        }
    }
    body.sort_by_key(|&p| degrade_order(p));
    body
}

/// Sample a squad of `n` random creeps, each within `energy`. Crate-internal (takes the `pub(crate)`
/// `Rng`); out-of-crate callers use [`sample_squad`] (seed-only).
pub(crate) fn random_squad(rng: &mut Rng, energy: u32, n: u8) -> Vec<Vec<Part>> {
    (0..n).map(|_| random_body(rng, energy)).collect()
}

/// Deterministic, seed-only squad sampler (the public entry point for out-of-crate drivers, e.g. the
/// render corpus example): construct the SplitMix64 RNG from `seed` internally and sample `n` free-form
/// bodies within `energy`. Same seed → identical squad (respects the sim-determinism fence: no
/// wall-clock, no ambient RNG). Prefer this over exposing `Rng` when a caller only needs a reproducible
/// composition.
pub fn sample_squad(seed: u32, energy: u32, n: u8) -> Vec<Vec<Part>> {
    random_squad(&mut Rng::seeded(seed), energy, n)
}

/// Place a squad's bodies as a vertical file of `owner` creeps at column `x`, ids from `first_id`.
fn place(world: &mut CombatWorld, owner: PlayerId, first_id: u32, bodies: &[Vec<Part>], x: u8, y0: u8) -> Vec<u32> {
    let mut ids = Vec::new();
    for (i, b) in bodies.iter().enumerate() {
        let id = first_id + i as u32;
        world.creeps.push(SimCreep { id, owner, pos: pos(x, y0 + i as u8), body: SimBody::unboosted(b), fatigue: 0 });
        ids.push(id);
    }
    ids
}

/// The engage prediction for `owner`'s force vs the rest, on an open `world` (creep-vs-creep, no
/// structures) — builds the `SquadView` from `owner`'s perspective and asks the Lanchester gate.
fn predict_for(world: &CombatWorld, owner: PlayerId) -> EnginePrediction {
    let centre = pos(25, 25);
    let sim = SimView::from_world(world, owner, centre, room());
    let to_member = |c: &CombatCreepDto| SquadMemberView {
        hits: c.hits,
        hits_max: c.hits_max,
        heal_power: c.working_parts(Part::Heal) as u32,
        pos: Some(c.pos),
        has_ranged: c.working_parts(Part::RangedAttack) > 0,
        melee_power: c.working_parts(Part::Attack) as u32 * ATTACK_POWER,
        ranged_power: c.working_parts(Part::RangedAttack) as u32 * RANGED_ATTACK_POWER,
        ..Default::default()
    };
    let members: Vec<SquadMemberView> = sim.friends().iter().map(to_member).collect();
    let centroid = members.iter().filter_map(|m| m.pos).next();
    let view = SquadView {
        members: &members,
        hostiles: sim.hostiles(),
        structures: &[],
        retreat_threshold: 0.3,
        current_state: SquadOrderState::Engaged,
        enemy_safe_mode: false,
        engage_objective: EngageObjective::Destroy,
        enemy_stalled: false,
        drain_stance: false,
    };
    predict_engage(&view, centroid)
}

/// Total living HP of `owner` in `world`.
fn living_hp(world: &CombatWorld, owner: PlayerId) -> i64 {
    world.creeps.iter().filter(|c| c.owner == owner && c.is_alive()).map(|c| c.body.hits as i64).sum()
}

/// One validated matchup: the side-0 prediction vs what actually happened.
#[derive(Clone, Copy, Debug)]
pub struct Matchup {
    pub seed: u64,
    /// Predicted balance μ (>0 ⇒ side 0 favoured) and win-probability permille for side 0.
    pub predicted_balance: i64,
    pub predicted_win_permille: i64,
    pub predicted_unwinnable: bool,
    /// Actual net HP (side0 − side1) at the end; >0 ⇒ side 0 won the exchange.
    pub actual_net_hp: i64,
    /// True when prediction sign agrees with outcome sign (or both ~even).
    pub agree: bool,
}

/// The validation summary over a population of random matchups.
#[derive(Clone, Debug)]
pub struct ValidationReport {
    pub trials: usize,
    /// Fraction of decisive matchups where the predicted favourite matched the actual winner.
    pub sign_accuracy: f64,
    /// Matchups where the model was confidently wrong (predicted a clear favourite that LOST) — the
    /// mispredicted / "broken" compositions worth inspecting.
    pub outliers: Vec<Matchup>,
}

/// Run `trials` random A-vs-B matchups (each a fresh seed): sample two squads, predict side-0's balance,
/// then play both sides under the EV kernel and compare. Decisive matchups (clear predicted favourite +
/// clear outcome) feed the sign-accuracy; confidently-wrong ones are collected as outliers. `energy` and
/// the squad-size range bound the population; `ticks` is the match length.
pub fn lanchester_validation(trials: usize, energy: u32, ticks: usize) -> ValidationReport {
    // "Clear" thresholds: a predicted favourite is one past this balance; a decisive outcome is one past
    // this net-HP margin. Below them the matchup is too even to score a sign.
    const CLEAR_BALANCE: i64 = 150; // ±15% Lanchester margin
    const DECISIVE_HP: i64 = 400; // ~half a creep
    let mut decisive = 0usize;
    let mut correct = 0usize;
    let mut outliers = Vec::new();
    let mut matchups = Vec::new();
    for t in 0..trials {
        let mut rng = Rng::seeded(t as u32);
        let n_a = rng.range(2, 6) as u8;
        let n_b = rng.range(2, 6) as u8;
        let a = random_squad(&mut rng, energy, n_a);
        let b = random_squad(&mut rng, energy, n_b);
        let mut world = CombatWorld::default();
        let a_ids = place(&mut world, 0, 1, &a, 8, 22);
        let b_ids = place(&mut world, 1, 1000, &b, 41, 22);
        let pred = predict_for(&world, 0);
        let mut squads = [
            ManagedSimSquad::new(0, a_ids, pos(41, 25)),
            ManagedSimSquad::new(1, b_ids, pos(8, 25)),
        ];
        run_managed(&mut world, &mut squads, ticks);
        let net = living_hp(&world, 0) - living_hp(&world, 1);
        let m = Matchup {
            seed: t as u64,
            predicted_balance: pred.balance,
            predicted_win_permille: pred.win_permille,
            predicted_unwinnable: pred.unwinnable,
            actual_net_hp: net,
            agree: pred.balance.signum() == net.signum() || (pred.balance.abs() < CLEAR_BALANCE && net.abs() < DECISIVE_HP),
        };
        matchups.push(m);
        if pred.balance.abs() >= CLEAR_BALANCE && net.abs() >= DECISIVE_HP {
            decisive += 1;
            if pred.balance.signum() == net.signum() {
                correct += 1;
            } else {
                outliers.push(m); // confidently predicted a winner that lost
            }
        }
    }
    ValidationReport {
        trials,
        sign_accuracy: if decisive == 0 { 1.0 } else { correct as f64 / decisive as f64 },
        outliers,
    }
}

/// Render a [`ValidationReport`] (the validation dashboard).
pub fn report(r: &ValidationReport) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "Lanchester validation — {} random matchups:", r.trials);
    let _ = writeln!(s, "  predicted-favourite sign accuracy (decisive matchups): {:.1}%", r.sign_accuracy * 100.0);
    let _ = writeln!(s, "  outliers (predicted a clear winner that LOST): {}", r.outliers.len());
    for o in r.outliers.iter().take(10) {
        let _ = writeln!(s, "    seed {:>4}: predicted balance {:+5} ({}‰ win) → actual net HP {:+6}", o.seed, o.predicted_balance, o.predicted_win_permille, o.actual_net_hp);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_body_is_within_budget_and_not_inert() {
        for seed in 0..50u32 {
            let mut rng = Rng::seeded(seed);
            let energy = [1300, 2300, 5600, 12_900][seed as usize % 4];
            let body = random_body(&mut rng, energy);
            let cost: u32 = body.iter().map(|&p| part_cost(p)).sum();
            assert!(cost <= energy, "body within budget (cost {cost} <= {energy})");
            assert!(body.len() <= 50, "body within the 50-part cap ({} parts)", body.len());
            assert!(
                body.iter().any(|&p| matches!(p, Part::Attack | Part::RangedAttack | Part::Heal | Part::Work)),
                "body has a useful part (not inert): {body:?}"
            );
            assert!(body.contains(&Part::Move), "body has MOVE for mobility");
        }
    }

    #[test]
    fn random_squads_are_deterministic_per_seed() {
        let s1 = random_squad(&mut Rng::seeded(7), 5600, 4);
        let s2 = random_squad(&mut Rng::seeded(7), 5600, 4);
        assert_eq!(s1, s2, "same seed → identical squad (reproducible)");
    }

    /// On-demand Lanchester validation over a random population — the "does the force model predict real
    /// fights, and where does it break?" dashboard. Run:
    /// `cargo test -p screeps-combat-eval --lib lanchester_validation_dashboard -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn lanchester_validation_dashboard() {
        let r = lanchester_validation(200, 5600, 80);
        println!("{}", report(&r));
        // A sanity floor only (this is a tuning/diagnostic dashboard, not a tight gate): the Lanchester
        // sign should beat a coin flip by a clear margin, or the force model is mis-specified.
        assert!(r.sign_accuracy >= 0.6, "force-strength prediction is barely better than chance ({:.0}%) — investigate the outliers", r.sign_accuracy * 100.0);
    }
}
