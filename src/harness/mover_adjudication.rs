//! The COMBAT-corpus movement adjudication (ADR 0033 M5 tail, operator-directed 2026-07-01) — the
//! evidence gate for rover tunables tuned on the HAUL corpus before they ship as global defaults.
//! Standing case: rover-eval's tournament (`screeps-rover-eval/src/tuning.rs`) found `ladder(8)` —
//! `StuckThresholds` scaled 4× slower — adds +0.08 H on hauling, but it slows every escalation tier
//! (friendly-avoid 2→8, shove 7→28, `report_failure` 12→48, a job-layer contract) and immobility
//! under fire is a combat-specific cost the haul corpus cannot see. This module runs the SAME
//! seeded matchup population under candidate [`MoverConfig`]s and compares REAL combat outcomes.
//!
//! Bed choice: the [`roster`](super::roster) matchup substrate (random squads → `ManagedSimSquad`
//! self-play, the `lanchester_validation` shape) is the cheapest bed that yields outcome metrics —
//! wins (net-HP sign), ticks-to-decision, first-blood (approach speed), damage traded — and it is
//! run-until-WIPE here (not fixed-tick) so ticks-to-outcome is a real signal. Half the corpus adds
//! a one-gap PINCH wall (the rover-eval `pinch` scenario transplanted): both squads funnel through
//! a 3-wide gap, so the stuck-escalation ladder actually fires (open-field kiting barely queues).
//! Everything is integer/seeded (`sim-core::rng`, no floats ordered) per the determinism fence.
//!
//! A second adjudication axis (§D5.4 decision 9's recorded gate) runs the SAME corpus with each
//! side's BINDING member bidding `R_O` on the NUMERIC priority lane instead of its enum tier
//! ([`run_matchup_with_bid_mode`]): the FIXTURE-constant arm (the slice-4 gate, retained as the
//! control — verdict on `adjudicate_w_priority_bids_on_the_combat_corpus`) and the ADR 0033
//! slice-7 REAL-annotation arm ([`binding_bids_real`]: `assess`/`win_probability`/`value_e`, the
//! same kernels the live `squad_objective_bid` runs — verdict on
//! `adjudicate_real_annotation_bids_on_the_combat_corpus`, the shipped live wiring's evidence).

use crate::harness::roster::{living_hp, place, random_squad};
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_decision::force_sizing::{assess, tower_intel_from, win_probability, DefenseProfile, ForceBudget, RequiredForce};
use screeps_combat_decision::objective_value::{value_e, ObjectiveIntel, ObjectiveValueKind};
use screeps_combat_engine::{resolve_tick, CombatWorld, CreepId, Intents, SimBodyCombat};
use screeps_rover::{MovementPriority, StuckThresholds};
use screeps_sim_core::rng::Rng;
use screeps_sim_core::MoverConfig;
use std::collections::HashMap;

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos(x: u8, y: u8) -> Position {
    Position::new(
        RoomCoordinate::new(x).unwrap(),
        RoomCoordinate::new(y).unwrap(),
        room(),
    )
}

/// Squad energy budget per creep (the roster population's standard tier).
const ENERGY: u32 = 5_600;

/// A [`StuckThresholds`] ladder built from its tier-1 base with the default tier SPACING ratios —
/// a local REBUILD of rover-eval's `tuning.rs::ladder` (rover-eval depends ON this crate, so the
/// helper cannot be imported back without a dependency cycle; keep the two byte-equivalent).
/// `ladder(8)` = the haul-tournament candidate: avoid 8 / all 16 / ops 20 / shove 28 / report 48 /
/// no-progress 60 (each default ×4, `div_ceil` over the tier-1 default of 2).
pub fn ladder(avoid_friendly: u16) -> StuckThresholds {
    let d = StuckThresholds::default();
    let scale = |v: u16| {
        ((v as u32 * avoid_friendly as u32).div_ceil(d.avoid_friendly_creeps as u32)) as u16
    };
    StuckThresholds {
        avoid_friendly_creeps: avoid_friendly.max(1),
        avoid_all_friendly_creeps: scale(d.avoid_all_friendly_creeps).max(avoid_friendly + 1),
        increase_ops: scale(d.increase_ops),
        enable_shoving: scale(d.enable_shoving),
        report_failure: scale(d.report_failure),
        no_progress_repath: scale(d.no_progress_repath),
    }
}

/// One matchup's outcome under one config — all integers so `Eq` is exact (the determinism pin
/// compares whole outcome vectors; no float is ever ordered, per the fence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchupOutcome {
    /// A side was WIPED inside the cap (the decisive outcome; undecided = timeout standoff).
    pub decided: bool,
    /// `signum(net_hp)` at the end: +1 side 0 won the exchange, -1 side 1, 0 dead even.
    pub winner: i8,
    /// Ticks to the wipe, or the cap when undecided (so slower closing is visible either way).
    pub ticks: u32,
    /// First tick any HP was traded — the APPROACH-speed signal (a slower escalation ladder that
    /// wedges the funnel shows up here before it shows in wins).
    pub first_blood: Option<u32>,
    /// Living HP side 0 − side 1 at the end.
    pub net_hp: i64,
    /// Total HP removed across both sides (start − end) — "the fight actually happened".
    pub damage_traded: u32,
}

/// The §D5.4 decision-9 W-PRIORITY fixture: the squad's objective rate `R_O = value_e /
/// est_ticks`, INTEGER milli-e/t (rover-eval `value.rs::quantize_w`'s lane; no float is ever
/// built, per the fence). The constants are a representative economic-unlock objective — the
/// energy a cleared room yields over one creep lifetime — chosen so the bid lands mid-band:
/// 240_000 e / 600 t = 400 e/t ⇒ 400_000 milli-e/t.
const BID_VALUE_E: i64 = 240_000;
const BID_EST_TICKS: i64 = 600;

/// The binding member's numeric bid: `Normal`-anchor + quantized `R_O` — the Normal-band slot
/// (1_400_000, strictly between `Normal` = 1M and `High` = 2M on the shared i64 lane), per the
/// §D5.4 binding-member-bids-full-R_O contention rail. Non-binding members keep their enum
/// anchors ("others anchor") — the axis under adjudication is exactly the recorded live-adoption
/// step: the mission-binding creep's tier replaced by its objective-derived value.
fn binding_member_bid() -> i64 {
    MovementPriority::Normal.anchor_value() + (BID_VALUE_E * 1000) / BID_EST_TICKS
}

/// The squad's BINDING member (§D5.4: the member whose progress binds the objective rate — for a
/// kill objective, the max damage-output member). Deterministic: max
/// `attack_power + ranged_attack_power`, ties to the LOWEST creep id (stable, no map iteration).
/// An all-support squad (zero damage everywhere) still binds through its lowest id — some member
/// carries the objective claim.
fn binding_member(world: &CombatWorld, ids: &[CreepId]) -> Option<CreepId> {
    let mut best: Option<(u32, CreepId)> = None;
    for &id in ids {
        if let Some(c) = world
            .movement
            .creeps
            .iter()
            .find(|c| c.id == id && c.is_alive())
        {
            let power = c.body.attack_power() + c.body.ranged_attack_power();
            best = match best {
                Some((bp, bi)) if power > bp || (power == bp && id < bi) => Some((power, id)),
                None => Some((power, id)),
                keep => keep,
            };
        }
    }
    best.map(|(_, id)| id)
}

/// The FIXTURE-constant control arm: binding member → `Normal_anchor + 400_000` milli-e/t.
fn binding_bids(world: &CombatWorld, ids: &[CreepId]) -> HashMap<CreepId, i64> {
    binding_member(world, ids)
        .map(|id| (id, binding_member_bid()))
        .into_iter()
        .collect()
}

/// The REAL-ANNOTATION value_e fixture (§D5.4 decision (5) parity): the same 240_000-e economic-unlock
/// magnitude as the fixture arm, but PRICED THROUGH the live kernel (`objective_value::value_e`'s
/// FarmCore arm — income · horizon), so the arm exercises the exact pricing rail the bot's
/// `squad_objective_bid` runs (income 160 e/t × 1500 t = 240_000 e).
const REAL_BID_INCOME_E_T: f32 = 160.0;
const REAL_BID_HORIZON_T: f32 = 1500.0;
/// The real arm's on-site budget (ticks) — the fixture `BID_EST_TICKS` magnitude, fed to the oracle as
/// `ForceBudget::onsite_budget_ticks` (and the est-ticks fallback for an unwinnable matchup).
const REAL_BID_ONSITE_TICKS: u32 = 600;

/// The REAL-ANNOTATION arm (ADR 0033 slice 7): derive each side's binding-member bid from the SAME
/// kernels the live path uses — `assess` (the force-sizing oracle) sized against the OPPOSING roster
/// (objective_hits = enemy living HP, incoming = enemy creep dps; `RequiredForce::from_assessment` is
/// the sized force, its est_ticks the §D5.4 denominator), `win_probability` over the squad's FIELDED
/// heal vs the incoming dps, and the `value_e` fixture above. `R_O = p_win · value_e / est_ticks`,
/// quantized ONCE to integer milli-e/t and anchored `Normal + clamp(·, 1, 999_999)` — the identical
/// band/shape as the live `military_priority_bid` (squad_manager.rs). Pure scalar f32 kernels over
/// id-ordered slices; the single float product is quantized before it reaches the resolver (the fence).
fn binding_bids_real(world: &CombatWorld, own_ids: &[CreepId], enemy_ids: &[CreepId]) -> HashMap<CreepId, i64> {
    let alive = |ids: &[CreepId]| -> Vec<&screeps_sim_core::SimCreep> {
        ids.iter()
            .filter_map(|id| world.movement.creeps.iter().find(|c| c.id == *id && c.is_alive()))
            .collect()
    };
    let own = alive(own_ids);
    let enemy = alive(enemy_ids);
    let fielded_heal: u32 = own.iter().map(|c| c.body.heal_power()).sum();
    let own_dps: u32 = own.iter().map(|c| c.body.attack_power() + c.body.ranged_attack_power()).sum();
    let tank_hp: u32 = own.iter().map(|c| c.body.hits).max().unwrap_or(0);
    let enemy_dps: u32 = enemy.iter().map(|c| c.body.attack_power() + c.body.ranged_attack_power()).sum();
    let enemy_hp: u32 = enemy.iter().map(|c| c.body.hits).sum();

    // The oracle's inputs for a creep-force kill objective: no towers (this bed has none), the enemy's
    // living HP as the objective to remove, its dps as the incoming fire.
    let profile = DefenseProfile {
        towers: Vec::new(),
        breach_hits: 0,
        objective_hits: enemy_hp,
        repair_per_tick: 0.0,
        safe_mode: false,
        tower_intel: tower_intel_from(true, true), // scouted-empty: we SEE the (towerless) field
    };
    let budget = ForceBudget {
        max_heal_per_tick: fielded_heal as f32,
        max_dismantle_dps: own_dps as f32,
        tank_effective_hp: tank_hp as f32,
        onsite_budget_ticks: REAL_BID_ONSITE_TICKS,
    };
    let a = assess(&profile, enemy_dps as f32, &budget);
    // The sized force is the annotation's RequiredForce (annotate.rs's decision-(5) shape); the binding
    // member bids the FULL R_O, so only the assessment's est_ticks enters the bid — the from_assessment
    // call pins that this arm runs the same R2 mapping the live annotations feed (dead-code-free: an
    // all-zero force only occurs for a wiped enemy, when the matchup is already decided).
    let _required = RequiredForce::from_assessment(&a);
    // Unwinnable-inside-the-window ⇒ conversion takes at least the whole on-site budget (est_ticks is 0
    // on the unwinnable arm — never a divide toward the band ceiling).
    let est_ticks = if a.winnable { a.est_ticks.max(1) } else { REAL_BID_ONSITE_TICKS };
    let p_win = win_probability(fielded_heal as f32, enemy_dps as f32);
    let v_e = value_e(
        ObjectiveValueKind::FarmCore,
        &ObjectiveIntel { income_per_tick: REAL_BID_INCOME_E_T, horizon: REAL_BID_HORIZON_T, ..Default::default() },
    );
    let r_o_milli = (f64::from(p_win) * f64::from(v_e) * 1000.0 / f64::from(est_ticks)).round() as i64;
    let bid = MovementPriority::Normal.anchor_value() + r_o_milli.clamp(1, 999_999);
    binding_member(world, own_ids).map(|id| (id, bid)).into_iter().collect()
}

/// Which priority treatment a matchup runs under (the adjudication's axis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BidMode {
    /// Enum tiers only — the shipped default (the control).
    EnumOnly,
    /// Binding member bids the fixture-constant `R_O` (400_000 milli-e/t) — the slice-4 gate's arm,
    /// retained as the fixture control.
    FixtureRO,
    /// Binding member bids the REAL-kernel `R_O` ([`binding_bids_real`]) — the slice-7 live-adoption
    /// evidence arm.
    RealRO,
}

/// Run ONE seeded matchup under `config`: sample two random squads (the roster population),
/// optionally wall the midfield into a one-gap pinch (odd seeds), and self-play both sides with
/// `config` threaded into their movers (`ManagedSimSquad::with_mover_config`) until a side is
/// wiped or `tick_cap`. Same seed + config ⇒ identical outcome (everything is seeded/ordered).
pub fn run_matchup(seed: u32, config: &MoverConfig, tick_cap: u32) -> MatchupOutcome {
    run_matchup_with_bid_mode(seed, config, tick_cap, BidMode::EnumOnly)
}

/// [`run_matchup`] with the w-priority axis as a bool (the historical slice-4 gate entry): `true` =
/// the fixture-`R_O` arm ([`BidMode::FixtureRO`]), `false` = byte-identical to the plain fn.
pub fn run_matchup_with_bids(
    seed: u32,
    config: &MoverConfig,
    tick_cap: u32,
    value_bids: bool,
) -> MatchupOutcome {
    let mode = if value_bids { BidMode::FixtureRO } else { BidMode::EnumOnly };
    run_matchup_with_bid_mode(seed, config, tick_cap, mode)
}

/// [`run_matchup`] under an explicit [`BidMode`]: each side's BINDING member bids `R_O` on the
/// numeric priority lane (fixture-constant or real-kernel — [`binding_bids`]/[`binding_bids_real`])
/// while every other member keeps enum-anchor ordering. `EnumOnly` is byte-identical to the plain fn
/// (empty bid map ⇒ enum-only ordering).
pub fn run_matchup_with_bid_mode(
    seed: u32,
    config: &MoverConfig,
    tick_cap: u32,
    mode: BidMode,
) -> MatchupOutcome {
    let mut rng = Rng::seeded(seed);
    let n_a = rng.range(2, 6) as u8;
    let n_b = rng.range(2, 6) as u8;
    let a = random_squad(&mut rng, ENERGY, n_a);
    let b = random_squad(&mut rng, ENERGY, n_b);

    let mut world = CombatWorld::default();
    // Odd seeds: a wall column at x=25 with a 3-wide gap (y=24..=26) — the rover-eval `pinch`
    // transplanted between the two spawn files, so BOTH squads funnel and the escalation tiers
    // (friendly-avoid → ops → shove) genuinely fire. Even seeds stay open-field (the control).
    if seed % 2 == 1 {
        for y in 0..=49u8 {
            if !(24..=26).contains(&y) {
                world.movement.terrain.walls.insert((25, y));
            }
        }
    }
    let a_ids = place(&mut world, 0, 1, &a, 8, 22);
    let b_ids = place(&mut world, 1, 1000, &b, 41, 22);
    let start_hp = (living_hp(&world, 0) + living_hp(&world, 1)) as u32;

    let (bids_a, bids_b) = match mode {
        BidMode::EnumOnly => (HashMap::new(), HashMap::new()),
        BidMode::FixtureRO => (binding_bids(&world, &a_ids), binding_bids(&world, &b_ids)),
        // Real kernels, symmetric: each side priced against the OTHER's living roster at placement.
        BidMode::RealRO => (
            binding_bids_real(&world, &a_ids, &b_ids),
            binding_bids_real(&world, &b_ids, &a_ids),
        ),
    };
    let mut squads = [
        ManagedSimSquad::new(0, a_ids, pos(41, 25))
            .with_mover_config(config.clone())
            .with_priority_bids(bids_a),
        ManagedSimSquad::new(1, b_ids, pos(8, 25))
            .with_mover_config(config.clone())
            .with_priority_bids(bids_b),
    ];

    let mut first_blood = None;
    let mut decided = false;
    let mut ticks = tick_cap;
    for tick in 0..tick_cap {
        // Merge both squads' intents into one engine tick (the `run_managed` shape, no towers here).
        let mut all = Intents::new();
        for sq in squads.iter_mut() {
            let i = sq.step(&world);
            all.creeps.extend(i.creeps);
            all.moves.extend(i.moves);
            all.pulls.extend(i.pulls);
            all.reasons.extend(i.reasons);
        }
        resolve_tick(&mut world, &all);
        let (hp0, hp1) = (living_hp(&world, 0), living_hp(&world, 1));
        if first_blood.is_none() && (hp0 + hp1) < start_hp as i64 {
            first_blood = Some(tick);
        }
        if hp0 == 0 || hp1 == 0 {
            decided = true;
            ticks = tick + 1;
            break;
        }
    }
    let net_hp = living_hp(&world, 0) - living_hp(&world, 1);
    MatchupOutcome {
        decided,
        winner: net_hp.signum() as i8,
        ticks,
        first_blood,
        net_hp,
        damage_traded: start_hp.saturating_sub((living_hp(&world, 0) + living_hp(&world, 1)) as u32),
    }
}

/// Run the whole seeded corpus under one config.
pub fn run_corpus(seeds: std::ops::Range<u32>, config: &MoverConfig, tick_cap: u32) -> Vec<MatchupOutcome> {
    seeds.map(|s| run_matchup(s, config, tick_cap)).collect()
}

/// [`run_corpus`] under the FIXTURE w-priority treatment arm ([`BidMode::FixtureRO`]).
pub fn run_corpus_with_bids(
    seeds: std::ops::Range<u32>,
    config: &MoverConfig,
    tick_cap: u32,
) -> Vec<MatchupOutcome> {
    seeds
        .map(|s| run_matchup_with_bid_mode(s, config, tick_cap, BidMode::FixtureRO))
        .collect()
}

/// [`run_corpus`] under the REAL-annotation w-priority arm ([`BidMode::RealRO`]) — the slice-7
/// re-adjudication's treatment corpus.
pub fn run_corpus_with_real_bids(
    seeds: std::ops::Range<u32>,
    config: &MoverConfig,
    tick_cap: u32,
) -> Vec<MatchupOutcome> {
    seeds
        .map(|s| run_matchup_with_bid_mode(s, config, tick_cap, BidMode::RealRO))
        .collect()
}

/// Paired A-vs-B comparison over the same seed set — the adjudication's number sheet. Integer
/// aggregates only; `ticks`/`first_blood` deltas are summed over the seeds where BOTH configs
/// produced the signal (paired, so an undecided outlier can't skew a mean).
#[derive(Clone, Debug, Default)]
pub struct Comparison {
    pub seeds: u32,
    pub decided: (u32, u32),
    /// Side-0 wins by net-HP sign among each config's own outcomes.
    pub side0_wins: (u32, u32),
    /// Seeds whose winner SIGN differs between the configs (the outcome-flip count).
    pub flipped: u32,
    /// Σ ticks-to-decision over seeds decided under BOTH configs (paired makespan).
    pub paired_ticks: (u64, u64),
    /// How many seeds were decided under both (the paired_ticks denominator).
    pub both_decided: u32,
    /// Σ first-blood tick over seeds where both drew blood (paired approach speed).
    pub paired_first_blood: (u64, u64),
    pub both_bled: u32,
    /// Σ damage traded over ALL seeds (fight intensity; a wedged mover starves this).
    pub damage: (u64, u64),
}

/// Build the paired comparison (panics on length mismatch — same seed set required).
pub fn compare(a: &[MatchupOutcome], b: &[MatchupOutcome]) -> Comparison {
    assert_eq!(a.len(), b.len(), "paired comparison needs the same seed set");
    let mut c = Comparison {
        seeds: a.len() as u32,
        ..Default::default()
    };
    for (oa, ob) in a.iter().zip(b) {
        c.decided.0 += u32::from(oa.decided);
        c.decided.1 += u32::from(ob.decided);
        c.side0_wins.0 += u32::from(oa.winner > 0);
        c.side0_wins.1 += u32::from(ob.winner > 0);
        c.flipped += u32::from(oa.winner != ob.winner);
        if oa.decided && ob.decided {
            c.both_decided += 1;
            c.paired_ticks.0 += oa.ticks as u64;
            c.paired_ticks.1 += ob.ticks as u64;
        }
        if let (Some(fa), Some(fb)) = (oa.first_blood, ob.first_blood) {
            c.both_bled += 1;
            c.paired_first_blood.0 += fa as u64;
            c.paired_first_blood.1 += fb as u64;
        }
        c.damage.0 += oa.damage_traded as u64;
        c.damage.1 += ob.damage_traded as u64;
    }
    c
}

/// Render a comparison (labels are the two config names; the dashboard line format).
pub fn report(label_a: &str, label_b: &str, c: &Comparison) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(s, "combat mover adjudication — {} vs {} over {} seeded matchups:", label_a, label_b, c.seeds);
    let _ = writeln!(s, "  decided (wipe inside cap): {} vs {}", c.decided.0, c.decided.1);
    let _ = writeln!(s, "  side-0 wins (net-HP sign): {} vs {}", c.side0_wins.0, c.side0_wins.1);
    let _ = writeln!(s, "  outcome flips (winner sign differs): {}", c.flipped);
    let _ = writeln!(
        s,
        "  Σ ticks-to-decision over the {} both-decided seeds: {} vs {}",
        c.both_decided, c.paired_ticks.0, c.paired_ticks.1
    );
    let _ = writeln!(
        s,
        "  Σ first-blood tick over the {} both-bled seeds: {} vs {}",
        c.both_bled, c.paired_first_blood.0, c.paired_first_blood.1
    );
    let _ = writeln!(s, "  Σ damage traded: {} vs {}", c.damage.0, c.damage.1);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    // Test-only import (the non-test fns take the config as a parameter): scoping it here keeps
    // the lib target warning-free.
    use screeps_combat_agent::pathing::combat_mover_config;

    /// The candidate under adjudication: `ladder(8)` + the (already-default) reuse 20 — exactly the
    /// haul tournament's H=0.851 point (rover-eval `tuning.rs`, ADR 0033 M5).
    fn ladder8() -> MoverConfig {
        // Rebased on the COMBAT default (currently ≡ the kernel default — holding-as-a-request
        // closed the registration split, see combat_mover_config) so the candidate axis stays
        // the ladder ALONE even if the combat default ever diverges again.
        MoverConfig {
            stuck_thresholds: ladder(8),
            ..combat_mover_config()
        }
    }

    /// FAST smoke (non-ignored): the config seam reaches the combat bed end-to-end and the bed is
    /// deterministic — same seed + config ⇒ identical whole-outcome structs (the fence's spread-0
    /// shape); and the two probe configs both complete a matchup on each terrain family.
    #[test]
    fn mover_config_reaches_the_combat_bed_and_is_deterministic() {
        let seeds = 0..2u32; // seed 0 = open field, seed 1 = pinch
        let a1 = run_corpus(seeds.clone(), &combat_mover_config(), 80);
        let a2 = run_corpus(seeds.clone(), &combat_mover_config(), 80);
        assert_eq!(a1, a2, "same seeds + config ⇒ bit-identical outcomes");
        let b = run_corpus(seeds.clone(), &ladder8(), 80);
        assert_eq!(b.len(), 2);
        // Sanity, not adjudication (that's the ignored corpus run): the bed produced signal —
        // squads actually fought on at least one family under both configs.
        assert!(
            a1.iter().any(|o| o.first_blood.is_some()) && b.iter().any(|o| o.first_blood.is_some()),
            "the matchup bed drew blood under both configs: {a1:?} / {b:?}"
        );
        // The w-priority treatment arm (§D5.4 decision 9) rides the same fence: deterministic,
        // and the bid plumbing reaches the bed (it must at least complete both families).
        let w1 = run_corpus_with_bids(seeds.clone(), &combat_mover_config(), 80);
        let w2 = run_corpus_with_bids(seeds.clone(), &combat_mover_config(), 80);
        assert_eq!(w1, w2, "same seeds + bids ⇒ bit-identical outcomes");
        assert!(
            w1.iter().any(|o| o.first_blood.is_some()),
            "the matchup bed drew blood under value bids: {w1:?}"
        );
        // The slice-7 REAL-annotation arm rides the same fence: the kernel-derived bids are pure
        // scalar math over id-ordered slices, so the corpus is bit-deterministic and completes both
        // terrain families.
        let r1 = run_corpus_with_real_bids(seeds.clone(), &combat_mover_config(), 80);
        let r2 = run_corpus_with_real_bids(seeds, &combat_mover_config(), 80);
        assert_eq!(r1, r2, "same seeds + REAL bids ⇒ bit-identical outcomes");
        assert!(
            r1.iter().any(|o| o.first_blood.is_some()),
            "the matchup bed drew blood under real-kernel bids: {r1:?}"
        );
    }

    /// The REAL-kernel bid lands in the same (Normal, High) band as the fixture arm and the live
    /// `military_priority_bid` — pinned over a real seeded placement so the band invariant can't
    /// silently drift when the kernels change.
    #[test]
    fn real_kernel_bids_stay_inside_the_normal_high_band() {
        let mut rng = Rng::seeded(3);
        let a = random_squad(&mut rng, ENERGY, 4);
        let b = random_squad(&mut rng, ENERGY, 4);
        let mut world = CombatWorld::default();
        let a_ids = place(&mut world, 0, 1, &a, 8, 22);
        let b_ids = place(&mut world, 1, 1000, &b, 41, 22);
        for (own, enemy) in [(&a_ids, &b_ids), (&b_ids, &a_ids)] {
            let bids = binding_bids_real(&world, own, enemy);
            assert_eq!(bids.len(), 1, "exactly one binding member per side");
            for &bid in bids.values() {
                assert!(
                    bid > MovementPriority::Normal.anchor_value() && bid < MovementPriority::High.anchor_value(),
                    "real bid {bid} must sit strictly inside (Normal, High)"
                );
            }
        }
    }

    /// THE ADJUDICATION (on demand):
    /// `cargo test -p screeps-combat-eval mover_adjudication --release -- --ignored --nocapture`
    /// 60 seeded matchups (30 open + 30 pinch), run-until-wipe cap 300, under
    ///   A = `combat_mover_config()` (reuse 20, ladder(2) — the shipped default),
    ///   B = `ladder(8)` + reuse 20   (the haul-tournament candidate),
    ///   C = default but `report_failure` 12→48 ONLY (the job-layer-contract axis isolated).
    /// Prints the paired number sheet; the asserts are the adjudication's ratchets (loose bounds
    /// derived from the recorded runs). RECORDED VERDICT (re-run 2026-07-01 under
    /// HOLDING-AS-A-REQUEST — the bed changed, the verdict HELD): combat-near-NEUTRAL on
    /// OUTCOMES — 3/60 winner flips (all at undecided stalemate margins), decisiveness 13 vs 12,
    /// Σ ticks-to-decision 2269 vs 2240 over the 12 both-decided seeds — but directionally WORSE
    /// contact quality in CONGESTION: pinch-family damage traded −12% (214,558 → 188,648) and
    /// first blood later (Σ 1751 → 1786 over 28 seeds) under ladder(8) — the
    /// immobility-under-fire cost the haul corpus cannot see (the pre-holds run showed the same
    /// pinch −12%, so the signal is robust to the bed change). Most matchups stalemate (kiting
    /// standoffs, STALL_LIMIT disengage), so wipes are rare by design; first-blood/damage/net-HP
    /// are the primary paired signals. Recommendation recorded in the ADR: keep split defaults
    /// (do NOT ship ladder(8) globally); deliver its haul win via per-request `StuckThresholds`.
    #[test]
    #[ignore]
    fn adjudicate_ladder8_on_the_combat_corpus() {
        const CAP: u32 = 300;
        let seeds = 0..60u32;
        let a = run_corpus(seeds.clone(), &combat_mover_config(), CAP);
        let b = run_corpus(seeds.clone(), &ladder8(), CAP);
        let c_cfg = MoverConfig {
            stuck_thresholds: StuckThresholds {
                report_failure: 48,
                ..Default::default()
            },
            // Rebased on the COMBAT default like every arm here. (Historical: when the combat
            // default carried registration OFF, `..Default::default()` here silently compared
            // registration modes instead of the report_failure axis. The defaults re-converged
            // when holding-as-a-request landed, but the rebase discipline stays.)
            ..combat_mover_config()
        };
        let c = run_corpus(seeds.clone(), &c_cfg, CAP);

        let ab = compare(&a, &b);
        println!("{}", report("default", "ladder(8)", &ab));
        // Per-family split (even = open, odd = pinch): the pinch is where escalation speed bites.
        let split = |v: &[MatchupOutcome], parity: u32| -> Vec<MatchupOutcome> {
            v.iter().enumerate().filter(|(i, _)| *i as u32 % 2 == parity).map(|(_, o)| *o).collect()
        };
        let open = compare(&split(&a, 0), &split(&b, 0));
        let pinch = compare(&split(&a, 1), &split(&b, 1));
        println!("open-field only:\n{}", report("default", "ladder(8)", &open));
        println!("pinch only:\n{}", report("default", "ladder(8)", &pinch));
        // Divergent seeds — where the mover config visibly changed the fight (evidence lines).
        for (i, (oa, ob)) in a.iter().zip(&b).enumerate() {
            if oa.winner != ob.winner || (oa.net_hp - ob.net_hp).abs() > 400 {
                println!(
                    "  seed {i:>3} ({}): default {:?} | ladder(8) {:?}",
                    if i % 2 == 1 { "pinch" } else { "open" },
                    oa,
                    ob
                );
            }
        }

        // The report_failure AXIS PROBE: in this bed the driver discards `MovementResult`s (the sim
        // has no job layer), and rover's tier-4 does nothing else in-process (the sole consult is
        // pass-3's `should_report_failure_with`, which inserts Failed-instead-of-Stuck and continues
        // — no state change either way), so 12→48 alone must be outcome-IDENTICAL. A failure here
        // means tier 4 grew a physics side effect.
        assert_eq!(
            a, c,
            "report_failure 12→48 alone changed sim outcomes — tier 4 is no longer contract-only"
        );

        // Bed sanity: the corpus produces signal to adjudicate on. Wipes are structurally rare
        // (random-comp self-play kites to the STALL_LIMIT standoff), so the floor is blood drawn
        // nearly everywhere + a handful of decisive wipes — the recorded run: 38/40 bled, 3 wipes.
        assert!(ab.both_bled * 10 >= ab.seeds * 8, "too few matchups drew blood ({}/{})", ab.both_bled, ab.seeds);
        assert!(ab.decided.0 >= 2, "default decided too few matchups ({})", ab.decided.0);
        // The adjudication ratchets (loose, from the recorded 2026-07-01 run — verdict: outcome-
        // NEUTRAL, movement-quality-negative in congestion): ladder(8) must not collapse
        // decisiveness and must not mass-flip winners (< a quarter of the corpus).
        assert!(
            2 * ab.decided.1 >= ab.decided.0,
            "ladder(8) collapsed decisiveness: {} vs {}",
            ab.decided.1,
            ab.decided.0
        );
        assert!(
            ab.flipped * 4 < ab.seeds,
            "ladder(8) flipped a large share of outcomes ({}/{})",
            ab.flipped,
            ab.seeds
        );
    }

    /// THE W-PRIORITY COMBAT GATE (§D5.4 decision 9's recorded gate; on demand):
    /// `cargo test -p screeps-combat-eval mover_adjudication --release -- --ignored --nocapture`
    /// The SAME 60 seeded matchups (30 open + 30 pinch), run-until-wipe cap 300, under
    ///   A = enum-only priorities (the shipped combat default), vs
    ///   W = A + each side's BINDING member bidding the fixture `R_O` on the numeric lane
    ///       ([`binding_bids`]: max-damage member, Normal-anchor + 400_000 milli-e/t; every
    ///       other member keeps its enum anchor).
    /// This was the evidence that unblocked live military w-adoption; the LIVE wiring has since
    /// SHIPPED (ADR 0033 slice 7: squad_manager.rs `squad_objective_bid` → `TickOrders::priority_bid`
    /// → squad_combat.rs `apply_squad_move_priority`), and the REAL-annotation re-run lives in
    /// `adjudicate_real_annotation_bids_on_the_combat_corpus` below — this fixture arm is retained
    /// as the constant-bid control.
    ///
    /// RECORDED VERDICT (2026-07-01, under holding-as-a-request): **outcomes IDENTICAL in
    /// aggregate, movement quality neutral-to-POSITIVE in congestion — the gate PASSES.**
    /// Decisiveness 13 vs 13, side-0 wins 32 vs 32 (zero directional bias — both sides bid
    /// symmetrically), 6/60 winner-sign flips (2 open / 4 pinch, all in UNDECIDED stalemate
    /// territory — chaotic re-rolls of margin fights, e.g. seed 22 +3,672 → dead-even; the two
    /// larger net-HP swings, seeds 31/53, go one each way). Σ ticks-to-decision 1180 vs 1285
    /// over only 7 both-decided seeds (too few to read); Σ first blood 3424 → 3380 (EARLIER
    /// contact under bids); Σ damage traded +5% (406,826 → 428,634), pinch family +8%
    /// (214,558 → 231,685) — the OPPOSITE sign of ladder(8)'s congestion cost: a Normal-band
    /// binding-member bid slots BELOW the other members' High anchors, so the squad's damage
    /// carrier stops out-bidding its own escorts for the forward tile and the pack packs the
    /// funnel tighter. No combat harm anywhere the sheet measures; family split shows no
    /// congestion penalty. Live military w-adoption is UNBLOCKED from the mover's side — the
    /// remaining prerequisite is the war-layer objective EV feed (frozen war.rs).
    ///
    /// RE-RUN 2026-07-02 (the slice-6 bed — pool reserve + damper + shove chains changed the enum
    /// BASELINE): decisiveness 16 vs 11, side-0 wins 32 vs 32, flips 8/60, Σ first blood 3,255 vs
    /// 3,255 (identical), Σ damage 413,838 vs 407,624 (open +2%, pinch −5%) — and this fixture arm
    /// is OUTCOME-IDENTICAL, seed-for-seed, to the REAL-annotation arm
    /// (`adjudicate_real_annotation_bids_on_the_combat_corpus`, where the verdict is recorded in
    /// full): within the (Normal, High) band the bid MAGNITUDE never decided a tile on this corpus.
    /// The gate still HOLDS (ratchets pass); this arm is now the constant-bid control.
    #[test]
    #[ignore]
    fn adjudicate_w_priority_bids_on_the_combat_corpus() {
        const CAP: u32 = 300;
        let seeds = 0..60u32;
        let a = run_corpus(seeds.clone(), &combat_mover_config(), CAP);
        let w = run_corpus_with_bids(seeds.clone(), &combat_mover_config(), CAP);

        let aw = compare(&a, &w);
        println!("{}", report("enum", "w-bids", &aw));
        let split = |v: &[MatchupOutcome], parity: u32| -> Vec<MatchupOutcome> {
            v.iter().enumerate().filter(|(i, _)| *i as u32 % 2 == parity).map(|(_, o)| *o).collect()
        };
        let open = compare(&split(&a, 0), &split(&w, 0));
        let pinch = compare(&split(&a, 1), &split(&w, 1));
        println!("open-field only:\n{}", report("enum", "w-bids", &open));
        println!("pinch only:\n{}", report("enum", "w-bids", &pinch));
        for (i, (oa, ow)) in a.iter().zip(&w).enumerate() {
            if oa.winner != ow.winner || (oa.net_hp - ow.net_hp).abs() > 400 {
                println!(
                    "  seed {i:>3} ({}): enum {:?} | w-bids {:?}",
                    if i % 2 == 1 { "pinch" } else { "open" },
                    oa,
                    ow
                );
            }
        }

        // Bed sanity (same floor as the ladder adjudication): signal exists to adjudicate on.
        assert!(aw.both_bled * 10 >= aw.seeds * 8, "too few matchups drew blood ({}/{})", aw.both_bled, aw.seeds);
        assert!(aw.decided.0 >= 2, "enum arm decided too few matchups ({})", aw.decided.0);
        // The gate's ratchets: value bids must not collapse decisiveness and must not mass-flip
        // winners — the "no combat damage" bar the live adoption cites.
        assert!(
            2 * aw.decided.1 >= aw.decided.0,
            "w-bids collapsed decisiveness: {} vs {}",
            aw.decided.1,
            aw.decided.0
        );
        assert!(
            aw.flipped * 4 < aw.seeds,
            "w-bids flipped a large share of outcomes ({}/{})",
            aw.flipped,
            aw.seeds
        );
    }

    /// THE REAL-ANNOTATION RE-ADJUDICATION (ADR 0033 slice 7 — the live-wiring evidence; on demand):
    /// `cargo test -p screeps-combat-eval mover_adjudication --release -- --ignored --nocapture`
    /// The SAME 60 seeded matchups (30 open + 30 pinch), run-until-wipe cap 300, under
    ///   A = enum-only priorities (the shipped combat default), vs
    ///   R = A + each side's BINDING member bidding the REAL-KERNEL `R_O` ([`binding_bids_real`]:
    ///       `assess`-sized vs the opposing roster + `win_probability` + the `value_e` fixture —
    ///       the same kernels the live `squad_objective_bid` runs, same Normal-band anchoring),
    /// with the fixture-constant arm (`adjudicate_w_priority_bids_on_the_combat_corpus`) retained
    /// as the control. This is the §D5.4 decision-9 gate RE-RUN with real annotations — the evidence
    /// the shipped live wiring (squad_manager.rs `squad_objective_bid` → `TickOrders::priority_bid`
    /// → squad_combat.rs `apply_squad_move_priority`) cites.
    ///
    /// RECORDED VERDICT (2026-07-02, this slice — the slice-6 bed: pool reserve + windowed damper +
    /// shove chains): **the gate HOLDS with REAL bids, and the real arm is OUTCOME-IDENTICAL to the
    /// adjudicated fixture arm, seed-for-seed** (both arms re-run on this bed the same day; every
    /// per-seed `MatchupOutcome` matches bit-for-bit). The treatment effect is BAND-SHAPE-driven —
    /// the binding member slotting into (Normal, High) under its escorts' enum-High anchors — not
    /// magnitude-driven: on this corpus no contested tile's resolution ever depended on the bid's
    /// value within the band, so swapping the 400_000 fixture constant for the kernel-derived `R_O`
    /// changes nothing the sheet measures. vs enum: decisiveness 16 vs 11 (fewer wipes inside the
    /// cap; both-decided fights resolve FASTER, Σ 1,664 → 1,594 over 8 seeds), side-0 wins 32 vs 32
    /// (zero directional bias — symmetric bids), 8/60 winner-sign flips (4 open / 4 pinch, almost
    /// all at undecided-stalemate margins, e.g. seed 13 +1,262 → −200; the largest, seed 35, swings
    /// −11,544 → +10,584 in a still-undecided standoff). Σ first blood 3,255 vs 3,255 (IDENTICAL
    /// contact timing over the 56 both-bled seeds); Σ damage traded −1.5% (413,838 → 407,624; open
    /// +2%, pinch −5%). No decisiveness collapse, no mass flips, no directional bias — live
    /// military w-as-priority ships on this evidence. (The enum BASELINE itself moved vs the
    /// 2026-07-01 recording because slices 5–6 changed the bed; the paired comparison is what the
    /// gate reads.)
    #[test]
    #[ignore]
    fn adjudicate_real_annotation_bids_on_the_combat_corpus() {
        const CAP: u32 = 300;
        let seeds = 0..60u32;
        let a = run_corpus(seeds.clone(), &combat_mover_config(), CAP);
        let r = run_corpus_with_real_bids(seeds.clone(), &combat_mover_config(), CAP);

        let ar = compare(&a, &r);
        println!("{}", report("enum", "real-w-bids", &ar));
        let split = |v: &[MatchupOutcome], parity: u32| -> Vec<MatchupOutcome> {
            v.iter().enumerate().filter(|(i, _)| *i as u32 % 2 == parity).map(|(_, o)| *o).collect()
        };
        let open = compare(&split(&a, 0), &split(&r, 0));
        let pinch = compare(&split(&a, 1), &split(&r, 1));
        println!("open-field only:\n{}", report("enum", "real-w-bids", &open));
        println!("pinch only:\n{}", report("enum", "real-w-bids", &pinch));
        for (i, (oa, or)) in a.iter().zip(&r).enumerate() {
            if oa.winner != or.winner || (oa.net_hp - or.net_hp).abs() > 400 {
                println!(
                    "  seed {i:>3} ({}): enum {:?} | real-w-bids {:?}",
                    if i % 2 == 1 { "pinch" } else { "open" },
                    oa,
                    or
                );
            }
        }

        // Bed sanity + the same gate ratchets as the fixture arm: real bids must not collapse
        // decisiveness and must not mass-flip winners — the "no combat damage" bar the live wiring cites.
        assert!(ar.both_bled * 10 >= ar.seeds * 8, "too few matchups drew blood ({}/{})", ar.both_bled, ar.seeds);
        assert!(ar.decided.0 >= 2, "enum arm decided too few matchups ({})", ar.decided.0);
        assert!(
            2 * ar.decided.1 >= ar.decided.0,
            "real w-bids collapsed decisiveness: {} vs {}",
            ar.decided.1,
            ar.decided.0
        );
        assert!(
            ar.flipped * 4 < ar.seeds,
            "real w-bids flipped a large share of outcomes ({}/{})",
            ar.flipped,
            ar.seeds
        );
    }
}
