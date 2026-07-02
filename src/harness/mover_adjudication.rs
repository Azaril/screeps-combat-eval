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
//! side's BINDING member bidding a fixture `R_O` on the NUMERIC priority lane instead of its enum
//! tier ([`run_matchup_with_bids`]) — the offline evidence live military w-as-priority adoption
//! cites (verdict recorded on `adjudicate_w_priority_bids_on_the_combat_corpus`).

use crate::harness::roster::{living_hp, place, random_squad};
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
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
/// kill objective, the max damage-output member) → its numeric bid. Deterministic: max
/// `attack_power + ranged_attack_power`, ties to the LOWEST creep id (stable, no map iteration).
/// An all-support squad (zero damage everywhere) still binds through its lowest id — some member
/// carries the objective claim.
fn binding_bids(world: &CombatWorld, ids: &[CreepId]) -> HashMap<CreepId, i64> {
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
    best.map(|(_, id)| (id, binding_member_bid()))
        .into_iter()
        .collect()
}

/// Run ONE seeded matchup under `config`: sample two random squads (the roster population),
/// optionally wall the midfield into a one-gap pinch (odd seeds), and self-play both sides with
/// `config` threaded into their movers (`ManagedSimSquad::with_mover_config`) until a side is
/// wiped or `tick_cap`. Same seed + config ⇒ identical outcome (everything is seeded/ordered).
pub fn run_matchup(seed: u32, config: &MoverConfig, tick_cap: u32) -> MatchupOutcome {
    run_matchup_with_bids(seed, config, tick_cap, false)
}

/// [`run_matchup`] with the w-priority axis: `value_bids` = each side's BINDING member bids the
/// fixture `R_O` on the numeric priority lane ([`binding_bids`]) while every other member keeps
/// enum-anchor ordering — the §D5.4 decision-9 offline combat gate's treatment arm. `false` is
/// byte-identical to the plain fn (empty bid map ⇒ enum-only ordering).
pub fn run_matchup_with_bids(
    seed: u32,
    config: &MoverConfig,
    tick_cap: u32,
    value_bids: bool,
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

    let (bids_a, bids_b) = if value_bids {
        (binding_bids(&world, &a_ids), binding_bids(&world, &b_ids))
    } else {
        (HashMap::new(), HashMap::new())
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

/// [`run_corpus`] under the w-priority treatment arm ([`run_matchup_with_bids`], bids ON).
pub fn run_corpus_with_bids(
    seeds: std::ops::Range<u32>,
    config: &MoverConfig,
    tick_cap: u32,
) -> Vec<MatchupOutcome> {
    seeds
        .map(|s| run_matchup_with_bids(s, config, tick_cap, true))
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
        let w2 = run_corpus_with_bids(seeds, &combat_mover_config(), 80);
        assert_eq!(w1, w2, "same seeds + bids ⇒ bit-identical outcomes");
        assert!(
            w1.iter().any(|o| o.first_blood.is_some()),
            "the matchup bed drew blood under value bids: {w1:?}"
        );
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
    /// This is the evidence live military w-adoption will cite (blocked today on war-layer
    /// objective EV — operations/war.rs is another agent's paused work; the plumbing itself —
    /// the numeric lane + `with_priority_bids` — is proven here).
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
}
