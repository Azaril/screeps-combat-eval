//! CompositionParams TUNING HARNESS (ADR 0031 §2c/D16/D17, 0031a §4) — drive the EV-optimizer weighting
//! from a tournament sweep.
//!
//! The pieces:
//! - [`evaluate_params`]: a PURE scorer that runs the calibration + sizing + creep-clear + the
//!   defended-regime acceptance beds with a GIVEN [`CompositionParams`] (threaded through
//!   `siege_doctrine_plan_with` / the validators / `run_defended_lifecycle_with_params`, which already
//!   route the knobs into `optimize_composition`/`emit_requirement`/`clear_force`), and folds them into a
//!   [`ParamScore`].
//! - [`ParamScore`] + [`ParamScore::winning_efficient_key`]: the WINNING-BUT-EFFICIENT ranking — among
//!   gates-held points, win-rate DESC then mean-spawn-cost-per-win ASC (the cheapest force that still wins);
//!   points failing the gates rank last.
//! - The ENV-DRIVEN grid sweep ([`tests::sweep_composition_params`], `#[ignore]`): reads the param ranges +
//!   a regime filter + an output path from env vars, iterates the cross-product, scores each point, writes
//!   the ranked table to the output path + stdout, and asserts the top point holds the gates. This is what
//!   the explore agents RUN — read-only, parallel, different env, no code edits.
//!
//! Determinism: the sweep is over the bit-deterministic sim (the determinism fences elsewhere assert the
//! per-case stability); [`tests::sweep_point_is_deterministic`] pins a sample point run-twice-equal.

use crate::harness::generate::{ForceSpec, Generator, Layout, RandomDefendedBase};
use crate::harness::lifecycle::{run_defended_lifecycle_with_params, ColonyFormingScenario, EconomyPressure, Home, LifecycleOutcome};
use crate::harness::validate::{CreepClearWins, OracleCalibration, SizingWins, Validator};
use screeps_combat_decision::composition::{assemble_force, CompositionParams};
use screeps_combat_decision::force_sizing::RequiredForce;

/// The acceptance-bed gate thresholds (ADR 0031a §4 / 0031 D16): FP ≤ 1.0%, FN ≤ 20.0%, and the canonical
/// defended-core acceptance bed must be Killed.
pub const FP_GATE: f64 = 0.010;
pub const FN_GATE: f64 = 0.200;

/// How many seeded `RandomDefendedBase` scenarios the calibration + sizing beds run over. Smaller than the
/// shipped 200-scenario CI gate so a wide grid sweep stays in minutes; large enough that the FP/FN rates are
/// meaningful (the CI gate's `oracle_is_calibrated_against_the_engine` keeps the authoritative 200-bed proof).
pub const SWEEP_CALIBRATION_SCENARIOS: u32 = 80;

/// The score of ONE [`CompositionParams`] point over the eval beds.
#[derive(Clone, Copy, Debug)]
pub struct ParamScore {
    /// Scenarios the sizing oracle FIELDED a winnable+breach force on (the FP denominator).
    pub fielded: u32,
    /// Combined WIN RATE across the sizing + creep-clear beds (won / attempted).
    pub win_rate: f64,
    /// FALSE-POSITIVE rate on the calibration bed (fielded-but-did-not-breach / fielded). Gate ≤ FP_GATE.
    pub fp_rate: f64,
    /// FALSE-NEGATIVE rate on the calibration bed (deferred-but-the-ceiling-breached / deferred). Gate ≤ FN_GATE.
    pub fn_rate: f64,
    /// Mean spawn cost per WIN across the fielded+won forces (efficiency: cheapest force that still wins).
    pub mean_spawn_cost_per_win: f64,
    /// Did the calibration FP/FN gates hold AND the acceptance defended-core bed get Killed?
    pub gates_held: bool,
}

impl ParamScore {
    /// The WINNING-BUT-EFFICIENT sort key (descending preference): gates-held points first, then by win-rate
    /// DESC, then by mean-spawn-cost-per-win ASC (the cheapest force that still wins). Points failing the
    /// gates rank LAST regardless of their win/efficiency. Returns a tuple ordered so that `sort_by` with
    /// `.partial_cmp` and a final `.reverse()` puts the BEST first — see [`rank`].
    ///
    /// Encoded as `(gates_held, win_rate, -cost)` — larger is better on every field, so a plain DESC sort on
    /// the tuple yields the winning-but-efficient order.
    pub fn winning_efficient_key(&self) -> (u8, f64, f64) {
        (self.gates_held as u8, self.win_rate, -self.mean_spawn_cost_per_win)
    }
}

/// Which defended-regime bed family the sweep grades a point over (the `SWEEP_REGIME` env filter):
/// `Structure` = creep-free structure breach (calibration + sizing); `Creep` = creep-clear; `Defended` =
/// the acceptance defended-core regime; `All` = the union (the default — the full ADR 0031a §4 grade).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    Structure,
    Creep,
    Defended,
    All,
}

impl Regime {
    pub fn parse(s: &str) -> Option<Regime> {
        match s.trim().to_ascii_lowercase().as_str() {
            "structure" => Some(Regime::Structure),
            "creep" => Some(Regime::Creep),
            "defended" => Some(Regime::Defended),
            "all" => Some(Regime::All),
            _ => None,
        }
    }
    fn runs_structure(self) -> bool {
        matches!(self, Regime::Structure | Regime::All)
    }
    fn runs_creep(self) -> bool {
        matches!(self, Regime::Creep | Regime::All)
    }
    fn runs_defended(self) -> bool {
        matches!(self, Regime::Defended | Regime::All)
    }
}

/// The high-energy forming bed the acceptance defended-core regime runs on (mirrors lifecycle's
/// `defended_forming`): 4 RCL8 homes so the breach force can be sized + formed under economy contention.
fn defended_forming() -> ColonyFormingScenario {
    ColonyFormingScenario {
        composition: assemble_force(&RequiredForce { heal_parts: 40, immune_struct_parts: 30, ..Default::default() }, 12_900)
            .expect("placeholder comp assembles at RCL8"),
        homes: (0..4).map(|_| Home { energy_capacity: 12_900, income: 1000, start_energy: 12_900 }).collect(),
        economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
        combat_priority: 87.5,
        per_member_cap: 12_900,
        budget_ticks: 4000,
        member_ttl: 1500,
        renew: false,
    }
}

/// One acceptance defended-core regime: `(label, rampart_hits, towers, approach layout, defender force)`.
type AcceptanceRegime = (&'static str, u32, Vec<((u8, u8), u32)>, Layout, ForceSpec);

/// The acceptance defended-core regimes the gate requires a Kill on (the canonical bed + the ADR 0031 P3
/// graded sweep regimes — rampart thickness / tower presence / approach / guard strength).
fn acceptance_regimes() -> Vec<AcceptanceRegime> {
    vec![
        ("canonical: rampart + tower + guard", 30_000, vec![((24, 16), 100_000)], Layout::Open, ForceSpec::Guard(2)),
        ("rampart-only + light guard", 50_000, vec![], Layout::Open, ForceSpec::Guard(1)),
        ("tower-only + guard", 0, vec![((24, 16), 100_000)], Layout::Open, ForceSpec::Guard(2)),
        ("corridor choke + guard", 20_000, vec![((24, 16), 100_000)], Layout::Corridor, ForceSpec::Guard(2)),
    ]
}

/// THE PURE SCORER (ADR 0031 §2c/D16/D17): run the calibration + sizing + creep-clear + the defended-regime
/// acceptance beds with `params` and fold them into a [`ParamScore`]. `regime` selects which bed families to
/// run (the `SWEEP_REGIME` filter); families it skips contribute neutrally (the gate still requires the
/// calibration FP/FN + acceptance Kill when their families run).
///
/// `gates_held` ⇔ (FP ≤ [`FP_GATE`]) AND (FN ≤ [`FN_GATE`]) AND every run acceptance defended bed Killed.
/// When a family is filtered out, its gate component is vacuously satisfied (so a `Creep`-only sweep is
/// graded purely on the creep-clear win-rate + the calibration FP/FN, which always run on the structure bed).
///
/// Pure over the bit-deterministic sim — same `params` ⇒ same `ParamScore` (the determinism contract; see
/// [`tests::sweep_point_is_deterministic`]).
pub fn evaluate_params(params: &CompositionParams, regime: Regime) -> ParamScore {
    // ── Calibration FP/FN (always runs — the structure bed is the FP/FN substrate) ──
    // The FP/FN gates are the universal honesty gate: a knob set that over-commits (FP) or over-defers (FN)
    // is rejected regardless of the regime filter, so the calibration bed runs unconditionally.
    let calib = {
        let gen = RandomDefendedBase { n: SWEEP_CALIBRATION_SCENARIOS };
        let mut v = OracleCalibration::with_params(*params);
        for i in 0..gen.count() {
            v.validate(&gen.generate(i));
        }
        *v.tally()
    };

    // ── Sizing wins + efficiency (structure regime) ──
    let mut won = 0u64;
    let mut attempted = 0u64;
    let mut winning_cost = 0u64;
    let mut fielded = 0u32;
    if regime.runs_structure() {
        let gen = RandomDefendedBase { n: SWEEP_CALIBRATION_SCENARIOS };
        let mut v = SizingWins::with_params(*params);
        for i in 0..gen.count() {
            v.validate(&gen.generate(i));
        }
        won += v.won as u64;
        attempted += v.attempted as u64;
        winning_cost += v.winning_spawn_cost;
        fielded = calib.fielded;
    }

    // ── Creep-clear wins + efficiency (creep regime) ──
    if regime.runs_creep() {
        use crate::harness::generate::CreepClearBed;
        let gen = CreepClearBed;
        let mut v = CreepClearWins::with_params(*params);
        for i in 0..gen.count() {
            v.validate(&gen.generate(i));
        }
        won += v.won as u64;
        attempted += v.attempted as u64;
        winning_cost += v.winning_spawn_cost;
    }

    // ── Defended-regime acceptance beds (defended regime): every run bed must be Killed ──
    let mut acceptance_killed = true;
    if regime.runs_defended() {
        let bed = defended_forming();
        for (_name, rampart, towers, layout, force) in acceptance_regimes() {
            let out = run_defended_lifecycle_with_params(&bed, rampart, &towers, layout, force, params);
            acceptance_killed &= matches!(out, LifecycleOutcome::Killed { .. });
        }
    }

    let win_rate = if attempted == 0 { 0.0 } else { won as f64 / attempted as f64 };
    let mean_spawn_cost_per_win = if won == 0 { 0.0 } else { winning_cost as f64 / won as f64 };
    let gates_held = calib.fp_rate() <= FP_GATE && calib.fn_rate() <= FN_GATE && acceptance_killed;

    ParamScore {
        fielded,
        win_rate,
        fp_rate: calib.fp_rate(),
        fn_rate: calib.fn_rate(),
        mean_spawn_cost_per_win,
        gates_held,
    }
}

/// Rank `(params, score)` points winning-but-efficiently (BEST first): gates-held points, then win-rate DESC,
/// then mean-spawn-cost-per-win ASC; gates-failing points last. The single source of truth for the sweep
/// ordering + the "did anything beat Default" comparison.
pub fn rank(mut points: Vec<(CompositionParams, ParamScore)>) -> Vec<(CompositionParams, ParamScore)> {
    points.sort_by(|a, b| {
        b.1.winning_efficient_key()
            .partial_cmp(&a.1.winning_efficient_key())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    points
}

/// Format a `(params, score)` row for the output file + stdout (one line per point).
pub fn format_row(rank: usize, p: &CompositionParams, s: &ParamScore) -> String {
    format!(
        "#{:<3} hold={:.3} over={:.3} mem_energy={:<5} commit={:.4} dyn={:.2} w_e={:.4} w_c={:.2} | gates={} win={:.3} fp={:.4} fn={:.4} cost/win={:.0} fielded={}",
        rank + 1,
        p.hold_margin,
        p.over_power_margin,
        p.member_energy,
        p.commit_ev_threshold,
        p.dynamic_margin,
        p.w_energy,
        p.w_creep,
        if s.gates_held { "HELD" } else { "FAIL" },
        s.win_rate,
        s.fp_rate,
        s.fn_rate,
        s.mean_spawn_cost_per_win,
        s.fielded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    /// Parse a comma-separated `f32` list from an env var (e.g. `"1.15,1.3,1.45"`); `default` when unset/empty.
    fn env_f32_list(key: &str, default: &[f32]) -> Vec<f32> {
        match std::env::var(key) {
            Ok(s) if !s.trim().is_empty() => s.split(',').filter_map(|t| t.trim().parse::<f32>().ok()).collect(),
            _ => default.to_vec(),
        }
    }
    /// Parse a comma-separated `u32` list from an env var; `default` when unset/empty.
    fn env_u32_list(key: &str, default: &[u32]) -> Vec<u32> {
        match std::env::var(key) {
            Ok(s) if !s.trim().is_empty() => s.split(',').filter_map(|t| t.trim().parse::<u32>().ok()).collect(),
            _ => default.to_vec(),
        }
    }

    /// THE ENV-DRIVEN GRID SWEEP (ADR 0031a §4 Tier 1) — what the explore agents RUN (read-only, parallel,
    /// different env, NO code edits). Reads the param ranges + a regime filter + an output path from env vars,
    /// iterates the cross-product, scores each point via [`evaluate_params`], writes the winning-but-efficient
    /// ranked table to `SWEEP_OUT` (+ stdout), and ASSERTS the top point holds the gates.
    ///
    /// Env vars (all optional — unset ⇒ the 0031a Tier-1 ranges):
    /// - `SWEEP_HOLD`         hold_margin list, default `1.15,1.3,1.45,1.6`
    /// - `SWEEP_OVERPOWER`    over_power_margin list, default `1.3,1.5,1.8`
    /// - `SWEEP_MEMBER_ENERGY` member_energy list, default `1300,2000,3000,5400`
    /// - `SWEEP_COMMIT`       commit_ev_threshold list, default `0` (Tier-1 leaves the EV floor at the seed;
    ///                        the 0.1·V / 0.2·V rungs need a target_value, swept in a later tier)
    /// - `SWEEP_DYNAMIC`      dynamic_margin list, default `1.0`
    /// - `SWEEP_W_ENERGY`     w_energy list, default `0.001` (the seed)
    /// - `SWEEP_TOUGH`        reserved (the TOUGH ladder is internal to `optimize_composition`; accepted +
    ///                        ignored so the explore env stays forward-compatible)
    /// - `SWEEP_REGIME`       `structure|creep|defended|all`, default `all`
    /// - `SWEEP_OUT`          output path, default `<scratch>/sweep_composition_params.txt`
    ///
    /// Run:
    ///   `cargo test --release -p screeps-combat-eval --lib sweep_composition_params -- --ignored --nocapture`
    /// with the env vars set, e.g.:
    ///   `SWEEP_HOLD=1.15,1.3,1.45,1.6 SWEEP_OVERPOWER=1.3,1.5,1.8 SWEEP_MEMBER_ENERGY=1300,2000,3000,5400 \`
    ///   `SWEEP_REGIME=all SWEEP_OUT=/tmp/sweep.txt cargo test --release -p screeps-combat-eval --lib \`
    ///   `sweep_composition_params -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn sweep_composition_params() {
        let holds = env_f32_list("SWEEP_HOLD", &[1.15, 1.3, 1.45, 1.6]);
        let overs = env_f32_list("SWEEP_OVERPOWER", &[1.3, 1.5, 1.8]);
        let mems = env_u32_list("SWEEP_MEMBER_ENERGY", &[1300, 2000, 3000, 5400]);
        let commits = env_f32_list("SWEEP_COMMIT", &[0.0]);
        let dynamics = env_f32_list("SWEEP_DYNAMIC", &[1.0]);
        let w_energies = env_f32_list("SWEEP_W_ENERGY", &[0.001]);
        let regime = std::env::var("SWEEP_REGIME").ok().and_then(|s| Regime::parse(&s)).unwrap_or(Regime::All);
        let out_path = std::env::var("SWEEP_OUT").unwrap_or_else(|_| {
            let dir = std::env::temp_dir();
            dir.join("sweep_composition_params.txt").to_string_lossy().into_owned()
        });

        // Cross-product → the candidate param set (Vec-ordered, deterministic).
        let mut candidates: Vec<CompositionParams> = Vec::new();
        for &hold in &holds {
            for &over in &overs {
                for &mem in &mems {
                    for &commit in &commits {
                        for &dynamic in &dynamics {
                            for &w_e in &w_energies {
                                candidates.push(CompositionParams {
                                    hold_margin: hold,
                                    over_power_margin: over,
                                    member_energy: mem,
                                    commit_ev_threshold: commit,
                                    dynamic_margin: dynamic,
                                    w_energy: w_e,
                                    ..Default::default()
                                });
                            }
                        }
                    }
                }
            }
        }

        let n = candidates.len();
        eprintln!(
            "[sweep] {n} points (hold×over×mem×commit×dyn×w_e = {}×{}×{}×{}×{}×{}), regime={regime:?}, out={out_path}",
            holds.len(), overs.len(), mems.len(), commits.len(), dynamics.len(), w_energies.len()
        );

        // Score every candidate in PARALLEL (each is an independent pure sim run — deterministic regardless
        // of completion order). The Default baseline is scored too (appended) so the table shows it inline.
        let default = CompositionParams::default();
        let mut all = candidates.clone();
        if !all.iter().any(|c| *c == default) {
            all.push(default);
        }
        let scored: Vec<(CompositionParams, ParamScore)> = all.par_iter().map(|p| (*p, evaluate_params(p, regime))).collect();
        let default_score = scored.iter().find(|(p, _)| *p == default).map(|(_, s)| *s).unwrap();

        let ranked = rank(scored);

        // Did any SWEPT point strictly beat Default? (gates held + a better winning-but-efficient key.)
        let default_key = default_score.winning_efficient_key();
        let beat_default = ranked.iter().any(|(p, s)| {
            *p != default && s.gates_held && s.winning_efficient_key() > default_key
        });

        // Build the report.
        use std::fmt::Write;
        let mut report = String::new();
        let _ = writeln!(report, "# CompositionParams sweep — {n} points, regime={regime:?}");
        let _ = writeln!(report, "# ranked winning-but-efficient: gates-held, then win_rate DESC, then cost/win ASC");
        let _ = writeln!(report, "# gates: FP<={FP_GATE} AND FN<={FN_GATE} AND acceptance defended bed Killed");
        let _ = writeln!(report, "# Default baseline: {}", format_row(0, &default, &default_score));
        let _ = writeln!(report, "# any swept point strictly beats Default: {beat_default}");
        let _ = writeln!(report, "#");
        for (i, (p, s)) in ranked.iter().enumerate() {
            let marker = if *p == default { "  <== Default" } else { "" };
            let _ = writeln!(report, "{}{marker}", format_row(i, p, s));
        }

        if let Err(e) = std::fs::write(&out_path, &report) {
            eprintln!("[sweep] WARNING: could not write {out_path}: {e}");
        }
        println!("{report}");
        println!("[sweep] wrote {} ranked points to {out_path}", ranked.len());
        let (top_p, top_s) = ranked.first().expect("at least one point");
        println!("[sweep] TOP: {}", format_row(0, top_p, top_s));
        println!("[sweep] any swept point strictly beats Default: {beat_default}");

        // The harness ASSERTS the top point holds the gates (a sweep that can't produce ANY gated-held point
        // is a misconfigured range / a regression, not a tuning result).
        assert!(top_s.gates_held, "the sweep's top point must hold the gates (FP/FN + acceptance); got {top_s:?}");
    }

    /// DETERMINISM: a sample point scores bit-identically run-twice (the sweep is over the bit-deterministic
    /// sim). A spread here means a result-affecting nondeterminism regressed (cf. the engine determinism
    /// fence `sim_is_deterministic_over_rounds`). Cheap (one point, Structure regime) so it can run in CI.
    #[test]
    fn sweep_point_is_deterministic() {
        let p = CompositionParams { member_energy: 3000, ..Default::default() };
        let a = evaluate_params(&p, Regime::Structure);
        let b = evaluate_params(&p, Regime::Structure);
        assert_eq!(a.fielded, b.fielded, "fielded is deterministic");
        assert_eq!(a.fp_rate.to_bits(), b.fp_rate.to_bits(), "fp_rate is bit-deterministic");
        assert_eq!(a.fn_rate.to_bits(), b.fn_rate.to_bits(), "fn_rate is bit-deterministic");
        assert_eq!(a.win_rate.to_bits(), b.win_rate.to_bits(), "win_rate is bit-deterministic");
        assert_eq!(
            a.mean_spawn_cost_per_win.to_bits(),
            b.mean_spawn_cost_per_win.to_bits(),
            "mean_spawn_cost_per_win is bit-deterministic"
        );
        assert_eq!(a.gates_held, b.gates_held, "gates_held is deterministic");
    }

    /// The Default knob set must HOLD the gates over the structure regime (the calibration FP/FN seed). This
    /// is the behavior-preserving floor: if Default ever fails its own gates, the sizing seed regressed.
    #[test]
    fn default_params_hold_the_structure_gates() {
        let s = evaluate_params(&CompositionParams::default(), Regime::Structure);
        println!("[default] {}", format_row(0, &CompositionParams::default(), &s));
        assert!(s.fp_rate <= FP_GATE, "Default FP rate {} exceeds the gate {FP_GATE}", s.fp_rate);
        assert!(s.fn_rate <= FN_GATE, "Default FN rate {} exceeds the gate {FN_GATE}", s.fn_rate);
        assert!(s.gates_held, "Default holds the structure gates");
    }

    /// The winning-but-efficient ranking PUTS gates-held points first, then win-rate DESC, then cost ASC. A
    /// pure unit test of [`rank`] (no sim) — the ordering contract the sweep depends on.
    #[test]
    fn ranking_is_winning_but_efficient() {
        let mk = |gates: bool, win: f64, cost: f64| ParamScore {
            fielded: 10,
            win_rate: win,
            fp_rate: 0.0,
            fn_rate: 0.0,
            mean_spawn_cost_per_win: cost,
            gates_held: gates,
        };
        let p = CompositionParams::default();
        let pts = vec![
            (p, mk(true, 0.9, 5000.0)),  // gated, high win, expensive
            (p, mk(true, 0.9, 3000.0)),  // gated, high win, CHEAPER → should outrank the above
            (p, mk(true, 0.8, 1000.0)),  // gated, lower win → ranks below both 0.9s despite cheapest
            (p, mk(false, 1.0, 100.0)),  // gates FAIL → ranks LAST despite perfect win + cheapest
        ];
        let ranked = rank(pts);
        assert_eq!(ranked[0].1.mean_spawn_cost_per_win, 3000.0, "cheapest of the top win-rate ranks first");
        assert_eq!(ranked[1].1.mean_spawn_cost_per_win, 5000.0, "pricier same-win ranks second");
        assert_eq!(ranked[2].1.win_rate, 0.8, "lower win-rate ranks third");
        assert!(!ranked[3].1.gates_held, "gates-failing ranks last");
    }

    /// ADR 0031 #39 DRAIN — the finite-energy multi-tower drain BED: the oracle PICKS Drain + the lifecycle
    /// now FIELDS a drain comp (P2/P3), but the ASSEMBLED multi-member soak needs tank-forward coordination.
    ///
    /// A core behind a thin rampart, guarded by FOUR finite-energy towers at MODERATE energy (1500 each —
    /// drainable: ~150 shots to dry, and a tank can soak that long) clustered at point-blank. The four
    /// towers at the breach standoff deal far more than a single squad can out-heal, so a BREACH force is
    /// NOT winnable head-on. But the towers are FINITE, so the configuration IS winnable by a DRAIN — and
    /// `assess` now PICKS `AssaultMode::Drain` (asserted below) and the lifecycle FIELDS the oracle-sized
    /// drain comp through the drain stance + `breach_drain` tactics (P2/P3 — no longer the breach-only path).
    ///
    /// HONEST STATUS (reported, not faked): the END-TO-END drain+breach is PROVEN ORACLE-DRIVEN at the
    /// tactic layer by `screeps-combat-agent`'s
    /// `the_oracle_decides_drain_then_a_sized_squad_bleeds_the_towers_and_breaches` (a SOLO oracle-sized tank
    /// drains the finite towers dry then dismantles the dead base — the make-or-break, GREEN). At THIS
    /// lifecycle the oracle-sized comp is a MULTI-MEMBER assembled force; the towers focus-fire the nearest
    /// member and the heal is SPREAD across healers (no single member solo-out-heals the focused falloff), so
    /// the assembled blob is wiped before the soak completes (verified: `RosterWiped`). Closing this needs the
    /// tank-forward soak coordination (TOUGH front presents the armor + healers heal-the-tank — the formation
    /// soak) that is the documented P2-efficiency follow-on, NOT part of the drain DECISION+SIZING this task
    /// builds. So this bed is still NOT `Killed` at the multi-member lifecycle — for the new, documented
    /// reason (assembled-soak coordination), not the old "breach-only pipeline can't field a drain at all".
    ///
    /// NOTE: deliberately NOT added to `acceptance_regimes()` — that set is the gate every point must KILL,
    /// and the multi-member assembled soak is not yet there.
    #[test]
    fn finite_multi_tower_drain_bed_oracle_picks_drain_but_assembled_soak_needs_tank_forward_coord() {
        use crate::harness::generate::{ForceSpec, Layout};
        use screeps_combat_decision::composition::CompositionParams;
        use screeps_combat_decision::force_sizing::{assess, AssaultMode, DefenseProfile, ForceBudget, TowerThreat};

        // The finite-energy multi-tower drain bed: 4 towers @ 1500 energy each, point-blank to the breach
        // standoff, behind a thin rampart, with a small guard.
        let bed = defended_forming();
        let drain_towers: Vec<((u8, u8), u32)> = vec![((24, 24), 1500), ((26, 24), 1500), ((24, 26), 1500), ((26, 26), 1500)];
        let rampart = 8_000u32;
        let params = CompositionParams::default();

        // (1) The drain MATH agrees this bed is drainable-but-not-breachable for a representative single
        //     squad budget: at the FALLOFF STANDOFF (range 20, where the runtime drain tactic holds — the P1
        //     tactic stands off rather than soaking point-blank), `assess` returns `AssaultMode::Drain`. A
        //     direct BREACH cannot out-heal even the falloff fire with the hold margin, but the FINITE drain
        //     is feasible (the tank's HP + heal over the drain ≥ the falloff damage over the drain ticks).
        //     This is the existence proof the runtime tactic targets. (The oracle's drain model soaks at the
        //     supplied `range_to_assault`; refining it to SELECT the standoff is P2 — here we evaluate it at
        //     the standoff the P1 tactic already implements.)
        let profile = DefenseProfile {
            objective_hits: 30_000,
            breach_hits: rampart,
            repair_per_tick: 0.0,
            // Four energized finite towers, evaluated at the falloff standoff (range 20 → 150/tower).
            towers: drain_towers.iter().map(|&(_, e)| TowerThreat { range_to_assault: 20, energy: e }).collect(),
            safe_mode: false,
        };
        // A single squad budget whose heal beats 4×150 falloff (drain) but not 4×150×1.3 with the hold
        // margin (breach), and whose tank HP + heal over the drain ticks ≥ the falloff damage over them.
        let budget = ForceBudget {
            max_heal_per_tick: 600.0,
            max_dismantle_dps: 600.0,
            tank_effective_hp: 20_000.0,
            onsite_budget_ticks: 1500,
        };
        let a = assess(&profile, 0.0, &budget); // drain bed has no defender creeps (structure-only)
        assert!(a.winnable, "the bed IS winnable for a single squad — via drain ({})", a.reason);
        assert_eq!(a.mode, AssaultMode::Drain, "and the winning mode is DRAIN, not breach ({})", a.reason);

        // (2) The lifecycle now PICKS Drain (the oracle, P2) and FIELDS a drain comp through the drain stance +
        //     `breach_drain` tactics (P3) — NOT the old breach-only path. The ASSEMBLED multi-member soak,
        //     though, gets focus-fired before it bleeds the towers dry (no single member solo-out-heals the
        //     falloff; the tank-forward heal-the-tank coordination is the documented P2-efficiency follow-on).
        //     So this bed is still NOT Killed at the multi-member lifecycle — the END-TO-END oracle-driven
        //     drain+breach is GREEN at the SOLO-tank tactic layer (combat-agent
        //     `the_oracle_decides_drain_then_a_sized_squad_bleeds_the_towers_and_breaches`).
        let out = run_defended_lifecycle_with_params(&bed, rampart, &drain_towers, Layout::Open, ForceSpec::Guard(1), &params);
        assert!(
            !matches!(out, LifecycleOutcome::Killed { .. }),
            "the ASSEMBLED multi-member drain soak is not Killed yet (got {out:?}) — it needs tank-forward soak \
             coordination (the P2-efficiency follow-on); the oracle-driven drain+breach is proven end-to-end at \
             the solo-tank tactic layer (combat-agent the_oracle_decides_drain_then_a_sized_squad_bleeds...)"
        );
    }
}
