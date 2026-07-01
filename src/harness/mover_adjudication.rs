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

use crate::harness::roster::{living_hp, place, random_squad};
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_engine::{resolve_tick, CombatWorld, Intents};
use screeps_rover::StuckThresholds;
use screeps_sim_core::rng::Rng;
use screeps_sim_core::MoverConfig;

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

/// Run ONE seeded matchup under `config`: sample two random squads (the roster population),
/// optionally wall the midfield into a one-gap pinch (odd seeds), and self-play both sides with
/// `config` threaded into their movers (`ManagedSimSquad::with_mover_config`) until a side is
/// wiped or `tick_cap`. Same seed + config ⇒ identical outcome (everything is seeded/ordered).
pub fn run_matchup(seed: u32, config: &MoverConfig, tick_cap: u32) -> MatchupOutcome {
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

    let mut squads = [
        ManagedSimSquad::new(0, a_ids, pos(41, 25)).with_mover_config(config.clone()),
        ManagedSimSquad::new(1, b_ids, pos(8, 25)).with_mover_config(config.clone()),
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

    /// The candidate under adjudication: `ladder(8)` + the (already-default) reuse 20 — exactly the
    /// haul tournament's H=0.851 point (rover-eval `tuning.rs`, ADR 0033 M5).
    fn ladder8() -> MoverConfig {
        MoverConfig {
            stuck_thresholds: ladder(8),
            ..Default::default()
        }
    }

    /// FAST smoke (non-ignored): the config seam reaches the combat bed end-to-end and the bed is
    /// deterministic — same seed + config ⇒ identical whole-outcome structs (the fence's spread-0
    /// shape); and the two probe configs both complete a matchup on each terrain family.
    #[test]
    fn mover_config_reaches_the_combat_bed_and_is_deterministic() {
        let seeds = 0..2u32; // seed 0 = open field, seed 1 = pinch
        let a1 = run_corpus(seeds.clone(), &MoverConfig::default(), 80);
        let a2 = run_corpus(seeds.clone(), &MoverConfig::default(), 80);
        assert_eq!(a1, a2, "same seeds + config ⇒ bit-identical outcomes");
        let b = run_corpus(seeds, &ladder8(), 80);
        assert_eq!(b.len(), 2);
        // Sanity, not adjudication (that's the ignored corpus run): the bed produced signal —
        // squads actually fought on at least one family under both configs.
        assert!(
            a1.iter().any(|o| o.first_blood.is_some()) && b.iter().any(|o| o.first_blood.is_some()),
            "the matchup bed drew blood under both configs: {a1:?} / {b:?}"
        );
    }

    /// THE ADJUDICATION (on demand):
    /// `cargo test -p screeps-combat-eval mover_adjudication --release -- --ignored --nocapture`
    /// 60 seeded matchups (30 open + 30 pinch), run-until-wipe cap 300, under
    ///   A = `MoverConfig::default()` (reuse 20, ladder(2) — the shipped default),
    ///   B = `ladder(8)` + reuse 20   (the haul-tournament candidate),
    ///   C = default but `report_failure` 12→48 ONLY (the job-layer-contract axis isolated).
    /// Prints the paired number sheet; the asserts are the adjudication's ratchets (loose bounds
    /// derived from the recorded 2026-07-01 run). RECORDED VERDICT: combat-NEUTRAL on OUTCOMES —
    /// 3/60 winner flips (all at stalemate margins), decisiveness 11 vs 13, and Σ ticks-to-decision
    /// IDENTICAL on every both-decided seed (2057 vs 2057: decisive fights are force-imbalance
    /// blowouts the mover config cannot flip) — but directionally WORSE contact quality in
    /// CONGESTION: pinch-family damage traded −12% (241,937 → 212,981) and first blood later
    /// (Σ 1738 → 1754 over 28 seeds) under ladder(8) — the immobility-under-fire cost the haul
    /// corpus cannot see (the open family moved oppositely, +9% damage, so the pinch delta is the
    /// escalation ladder, not a global shift). Most matchups stalemate (kiting standoffs,
    /// STALL_LIMIT disengage), so wipes are rare by design; first-blood/damage/net-HP are the
    /// primary paired signals. Recommendation recorded in the ADR: keep split defaults (do NOT
    /// ship ladder(8) globally); deliver its haul win via per-request `StuckThresholds` instead.
    #[test]
    #[ignore]
    fn adjudicate_ladder8_on_the_combat_corpus() {
        const CAP: u32 = 300;
        let seeds = 0..60u32;
        let a = run_corpus(seeds.clone(), &MoverConfig::default(), CAP);
        let b = run_corpus(seeds.clone(), &ladder8(), CAP);
        let c_cfg = MoverConfig {
            stuck_thresholds: StuckThresholds {
                report_failure: 48,
                ..Default::default()
            },
            ..Default::default()
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
}
