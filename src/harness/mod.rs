//! The staged combat harness (ADR 0023a): four swappable stages — **generation** ([`generate`]) →
//! **evaluation** ([`evaluate`], a generic run-until loop) → **validation** ([`validate`]) →
//! **visualization** (the replay player, Phase V) — so a generator, a validator, and a stop condition
//! compose freely. [`run_suite`] crosses a generator with a validator. The P-FORCE oracle-calibration
//! WIN is one `(generator, validator)` pair; [`calibrate`] is the convenience that runs it.

pub mod evaluate;
pub mod foreman_capture;
pub mod generate;
pub mod lifecycle;
pub mod param_sweep;
pub mod report;
pub mod roster;
pub mod scenario;
pub mod terrain_import;
pub mod validate;
pub mod visualize;

use generate::{Generator, RandomDefendedBase};
use validate::{Calibration, OracleCalibration, Validator, Verdict};

/// The aggregate of running a generator's scenarios through a validator.
#[derive(Clone, Debug)]
pub struct SuiteReport {
    pub generator: String,
    pub validator: String,
    pub scenarios: u32,
    pub passed: u32,
    pub verdicts: Vec<Verdict>,
}

/// Cross every scenario a generator offers with a validator (stage 1 → stage 3). Generation ⊥
/// validation: any generator pairs with any validator.
pub fn run_suite(generator: &dyn Generator, validator: &mut dyn Validator) -> SuiteReport {
    let verdicts: Vec<Verdict> = (0..generator.count()).map(|i| validator.validate(&generator.generate(i))).collect();
    let passed = verdicts.iter().filter(|v| v.pass).count() as u32;
    SuiteReport {
        generator: generator.label().into(),
        validator: validator.label().into(),
        scenarios: generator.count(),
        passed,
        verdicts,
    }
}

/// Convenience: run the oracle-calibration over `n` seeded `RandomDefendedBase` scenarios and return
/// the FP/FN tally (the P-FORCE WIN gate substrate).
pub fn calibrate(n: u32) -> Calibration {
    let generator = RandomDefendedBase { n };
    let mut validator = OracleCalibration::new();
    run_suite(&generator, &mut validator);
    *validator.tally()
}

/// Convenience: render an interactive HTML replay of the oracle's decision on seeded scenario `index`
/// (Generation → Evaluation+record → Visualization). Write it to a `.html` and open it.
pub fn calibration_replay(index: u32) -> String {
    let scenario = RandomDefendedBase { n: index + 1 }.generate(index);
    validate::render_calibration_replay(&scenario)
}

#[cfg(test)]
mod tests {
    use super::*;
    use generate::{CreepClearBed, Designed, ForemanGenerator, ImportedRoom, Permutations};
    use scenario::ObjectiveKind;
    use validate::{clear_outcome_at, CreepClearWins, ManagedSquadIntegration, SelfPlay, SizingWins};

    /// The WIN gate (ADR 0022 P-FORCE / ADR 0023a stages 1–3): over 200 seeded defended-base scenarios,
    /// the force-sizing oracle is calibrated against the engine — winnable verdicts breach (fp ≤ 1%) and
    /// defers are real (fn ≤ 20%). Run with `-- --nocapture` to see the dashboard.
    #[test]
    fn oracle_is_calibrated_against_the_engine() {
        let c = calibrate(200);
        println!("{}", c.report());
        assert!(c.fielded >= 20, "too few fielded scenarios to calibrate FP ({})", c.fielded);
        assert!(c.deferred >= 20, "too few deferred scenarios to calibrate FN ({})", c.deferred);
        assert!(
            c.fp_rate() <= 0.01,
            "FALSE POSITIVES {}/{} (fp_rate {:.3} > 0.010)\n{}",
            c.false_positives,
            c.fielded,
            c.fp_rate(),
            c.report()
        );
        assert!(
            c.fn_rate() <= 0.20,
            "FALSE NEGATIVES {}/{} (fn_rate {:.3} > 0.200)\n{}",
            c.false_negatives,
            c.deferred,
            c.fn_rate(),
            c.report()
        );
    }

    /// Determinism: the same seed count yields the same tally (SplitMix64 over the index).
    #[test]
    fn calibration_is_deterministic() {
        assert_eq!(format!("{:?}", calibrate(64)), format!("{:?}", calibrate(64)));
    }

    /// Stage wiring smoke test: `run_suite` produces one verdict per generated scenario.
    #[test]
    fn run_suite_visits_every_scenario() {
        let generator = RandomDefendedBase { n: 16 };
        let mut validator = OracleCalibration::new();
        let report = run_suite(&generator, &mut validator);
        assert_eq!(report.verdicts.len(), 16);
        assert_eq!(report.scenarios, 16);
        assert_eq!(validator.tally().scenarios, 16);
    }

    /// ADR 0026 §9.10 L6 — the CREEP-CLEAR sizing gate: a `force_sizing::clear_force`-sized squad CLEARS
    /// the graded defender forces it is sized against (the keystone validation before the
    /// `PlayerDefend`/`PlayerRaid` rungs wire it live). Run with `-- --ignored --nocapture` for the
    /// per-bed dashboard. `#[ignore]` — it runs the moving brain over 4 beds (slower) + the win-rate
    /// threshold is still being calibrated for the L6 sweep.
    #[test]
    #[ignore]
    fn creep_clear_sizing_clears_the_bed() {
        let g = CreepClearBed;
        let mut v = CreepClearWins::default();
        for i in 0..g.count() {
            let s = g.generate(i);
            let verdict = v.validate(&s);
            println!("{:48} -> {}", s.label, verdict.detail);
        }
        println!("creep-clear win rate: {:.0}% ({}/{} fielded)", v.win_rate() * 100.0, v.won, v.attempted);
        assert!(v.attempted >= 2, "at least some beds fielded a sized squad ({})", v.attempted);
        assert!(v.win_rate() >= 0.75, "clear_force-sized squads should clear most winnable beds (got {:.0}%)", v.win_rate() * 100.0);
    }

    /// ADR 0026 §9.10 L6b — the `COORDINATED_DPS_MARGIN` sweep on the creep-clear bed. For a range of
    /// margins, field a `clear_force`-sized squad + score the payoff: winning DOMINATES (1M each), then
    /// minimize spawn cost (the tiebreak) → the LEANEST margin that reliably clears the whole bed (the
    /// minimum-favorable-force principle). Prints the curve + the best margin vs the shipped seed. Run with
    /// `-- --ignored --nocapture`. `#[ignore]` — exploratory tuning, not a CI gate.
    #[test]
    #[ignore]
    fn creep_clear_margin_sweep() {
        let g = CreepClearBed;
        let margins = [1.0_f32, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0];
        let mut best: Option<(f32, i64)> = None;
        for &m in &margins {
            let (mut wins, mut fielded, mut cost, mut ticks) = (0u32, 0u32, 0u64, 0u64);
            for i in 0..g.count() {
                if let Some(o) = clear_outcome_at(&g.generate(i), m) {
                    fielded += 1;
                    if o.cleared {
                        wins += 1;
                    }
                    cost += o.spawn_cost as u64;
                    ticks += o.ticks as u64;
                }
            }
            // Winning dominates; among margins that clear the same count, lowest spawn cost (then ticks).
            let payoff = wins as i64 * 1_000_000 - cost as i64 - ticks as i64;
            println!("margin {m:.2}: {wins}/{fielded} cleared | cost {cost} | ticks {ticks} | payoff {payoff}");
            if best.map(|(_, p)| payoff > p).unwrap_or(true) {
                best = Some((m, payoff));
            }
        }
        let (bm, _) = best.expect("swept at least one margin");
        println!("BEST margin: {bm:.2} (shipped COORDINATED_DPS_MARGIN seed = {:.2})", screeps_combat_decision::force_sizing::COORDINATED_DPS_MARGIN);
        assert!(bm >= 1.0);
    }

    /// Full chain smoke test: Generation → Evaluation(record) → Visualization yields a self-contained
    /// HTML replay with the verdict + frames embedded.
    #[test]
    fn calibration_replay_renders_html() {
        let html = calibration_replay(7);
        assert!(html.starts_with("<!doctype html>") && html.trim_end().ends_with("</html>"));
        assert!(html.contains("window.REPLAY=") && html.contains("IbexReplay.start"));
        assert!(html.contains("\"frames\":[{"), "embeds a non-empty frame array");
        assert!(html.contains("\"verdict\":"));
    }

    /// Phase B: the terrain-rich generators produce assessable scenarios (every scenario has an
    /// objective + valid staging; the oracle runs without panicking over the whole grid).
    #[test]
    fn terrain_generators_produce_assessable_scenarios() {
        let cases: [(&dyn Generator, &str); 2] = [(&Permutations, "perm"), (&Designed, "designed")];
        for (g, label) in cases {
            assert!(g.count() > 0, "{label} offers scenarios");
            let mut v = OracleCalibration::new();
            let report = run_suite(g, &mut v);
            assert_eq!(report.verdicts.len(), g.count() as usize, "{label}: a verdict per scenario");
            // Every scenario has at least one objective with the staging tiles populated.
            for i in 0..g.count() {
                let s = g.generate(i);
                assert!(!s.objectives.is_empty(), "{label}#{i} has an objective");
                let o = &s.objectives[0];
                assert!(!o.front_tiles.is_empty() && !o.support_tiles.is_empty(), "{label}#{i} staged");
            }
        }
    }

    /// The traversal lens works end-to-end: a moving managed squad navigates the open-field designed
    /// fixture to the objective + engages (the movement the operator validates). `Designed#0` is
    /// terrain-free so pathing is unobstructed — the squad must reach the objective vicinity.
    #[test]
    fn managed_squad_navigates_to_the_objective() {
        // Permutations#0 is open / no rampart / no towers / no force — pure navigation, so the moving
        // squad MUST reach + engage the objective (isolates the pathing from the engage/retreat gate).
        let scenario = Permutations.generate(0);
        let mut v = ManagedSquadIntegration;
        let verdict = v.validate(&scenario);
        assert!(verdict.pass, "the managed assault did not reach/engage the undefended objective: {}", verdict.detail);
    }

    /// Validator-swap (ADR 0023a): the SAME generator feeds a different validator (`SizingWins`) — the
    /// win-rate lens runs over the calibration substrate and reports a sane rate.
    #[test]
    fn sizing_wins_is_a_swappable_lens_over_the_same_generator() {
        let mut v = SizingWins::default();
        let report = run_suite(&RandomDefendedBase { n: 64 }, &mut v);
        assert_eq!(report.verdicts.len(), 64);
        assert!(v.attempted > 0, "some scenarios are winnable+fielded sizing attempts");
        assert!(v.win_rate() >= 0.99, "the fielded sized force wins its attempts (win_rate {:.3})", v.win_rate());
    }

    /// Self-play realism + the kiting-stalemate fix (operator-requested): in the open no-tower skirmish
    /// (Designed#5) BOTH sides run the squad brain AND the close-to-kill gradient now CLOSES the fight to
    /// a decisive result instead of freezing at range until timeout. Guards both the "opponent is a
    /// static dummy" and the "ranged standoff never resolves" regressions.
    #[test]
    fn self_play_resolves_decisively_not_a_standoff() {
        let scenario = Designed.generate(5);
        let mut v = SelfPlay;
        let verdict = v.validate(&scenario);
        assert!(verdict.pass, "self-play did not resolve (frozen standoff?): {}", verdict.detail);
        assert!(!verdict.detail.contains("Timeout"), "self-play froze at range instead of closing to a result: {}", verdict.detail);
    }

    /// Empty-recording regression (operator-flagged): self-play over a scenario with NO defender creeps
    /// (Permutations#0 = open / no towers / no force) must still record frames — not terminate at tick 0
    /// on a vacuous "defender wiped". Guards the conditional `SideWiped(defender)` stop.
    #[test]
    fn self_play_records_frames_even_without_defenders() {
        let html = validate::render_self_play_replay(&Permutations.generate(0));
        assert!(!html.contains("\"frames\":[]"), "self-play recorded zero frames (the empty-recording bug)");
        assert!(html.contains("\"frames\":[{"), "non-empty frame array embedded");
    }

    /// Multi-room cross-room movement (operator-flagged): the twin-room assault (Designed#4) must
    /// actually CROSS W1N1→W2N1 and reach/engage the objective — guards the `ManagedSimSquad` travel
    /// mode that fixes the room-scoped-view "no cross-room movement" bug.
    #[test]
    fn multi_room_assault_crosses_the_border() {
        let scenario = Designed.generate(4);
        let mut v = ManagedSquadIntegration;
        let verdict = v.validate(&scenario);
        assert!(verdict.pass, "the assault did not cross into the objective room + engage: {}", verdict.detail);
    }

    // ── ADR 0025 §12 Stage 2: imported-terrain scenarios (single + multi-room × objectives × comps) ──

    /// Every (real-terrain fixture × objective kind × comp seed) produces an ASSESSABLE scenario — valid
    /// staging + the sizing oracle runs over the whole enumeration without panicking (mirrors
    /// `terrain_generators_produce_assessable_scenarios` for the real-terrain generator).
    #[test]
    fn imported_room_every_kind_is_assessable() {
        let g = ImportedRoom { multi_room: false, n_comps: 2 };
        assert!(g.count() > 0, "imported-room offers scenarios");
        let mut v = OracleCalibration::new();
        let report = run_suite(&g, &mut v);
        assert_eq!(report.verdicts.len(), g.count() as usize, "a verdict per scenario");
        let mut kinds_seen = std::collections::HashSet::new();
        for i in 0..g.count() {
            let s = g.generate(i);
            assert!(!s.objectives.is_empty(), "imported#{i} has an objective");
            let o = &s.objectives[0];
            assert!(!o.front_tiles.is_empty() && !o.support_tiles.is_empty(), "imported#{i} ({:?}) is staged", o.kind);
            // The objective + staging sit on the navigable interior (not in a wall).
            let t = s.world.terrain_for(o.room);
            assert!(!t.is_wall(o.pos.x().u8(), o.pos.y().u8()), "imported#{i} objective is on a clear tile");
            kinds_seen.insert(o.kind);
        }
        assert_eq!(kinds_seen.len(), 5, "all five objective kinds are exercised, got {kinds_seen:?}");
    }

    /// A `Declaim` scenario carries a defender-owned controller at the objective (the world plumbing the
    /// declaim stop-condition reads — even though the attacker's kernel does not yet target controllers,
    /// ADR 0025 §11 #11 / §12 fallback #4).
    #[test]
    fn imported_declaim_has_a_controller() {
        let g = ImportedRoom { multi_room: false, n_comps: 1 };
        let declaim = (0..g.count()).map(|i| g.generate(i)).find(|s| s.objectives[0].kind == ObjectiveKind::Declaim).expect("a declaim scenario exists");
        let ctrl = &declaim.world.controllers;
        assert!(!ctrl.is_empty(), "declaim scenario has a controller");
        assert!(ctrl.iter().any(|c| c.owner == Some(generate::DEFENDER) && c.pos == declaim.objectives[0].pos), "the controller is defender-owned at the objective tile");
    }

    /// The traversal lens over REAL terrain: a moving managed squad navigates an imported single-room
    /// `Raze` base + reaches/engages the objective (or is wiped trying). Gates on REACH across the real
    /// fixtures — the operator's "terrain renders + a squad navigates it" Stage 1/2 smoke test.
    #[test]
    fn imported_room_navigable() {
        let g = ImportedRoom { multi_room: false, n_comps: 1 };
        // The Raze fixtures are indices 0..fixtures (kind index 0). Field the moving squad on each.
        let raze: Vec<_> = (0..g.count()).map(|i| g.generate(i)).filter(|s| s.objectives[0].kind == ObjectiveKind::Raze).collect();
        assert!(!raze.is_empty(), "there are single-room Raze fixtures");
        let mut passed = 0;
        for s in &raze {
            let mut v = ManagedSquadIntegration;
            if v.validate(s).pass {
                passed += 1;
            }
        }
        assert!(passed * 5 >= raze.len() * 3, "the managed squad navigates most real-terrain bases ({passed}/{} reached/engaged)", raze.len());
    }

    /// The multi-room imported variant is well-formed: it stages the assault in a DIFFERENT room than the
    /// objective and stays assessable (the cross-border REACH on real terrain is the standing §11 #10
    /// caveat, so this gates on well-formedness + oracle-assessability, not strict crossing).
    #[test]
    fn multi_room_imported_is_assessable() {
        let g = ImportedRoom { multi_room: true, n_comps: 1 };
        assert!(g.count() > 0, "multi-room imported offers scenarios");
        let mut v = OracleCalibration::new();
        let report = run_suite(&g, &mut v);
        assert_eq!(report.verdicts.len(), g.count() as usize, "a verdict per scenario");
        for i in 0..g.count() {
            let s = g.generate(i);
            let o = &s.objectives[0];
            assert_ne!(o.entry.room_name(), o.room, "multi-room#{i} stages in a different room than the objective");
        }
    }

    // ── ADR 0025 §12 Stage 3: realistic FOREMAN-PLANNED bases over real terrain ──

    /// A cached foreman base realizes into a populated world: real terrain + the planner's spawn(s),
    /// energized towers, and a rampart ring (a real base shape, not empty).
    #[test]
    fn foreman_cache_realizes() {
        use screeps_combat_engine::StructureKind;
        let g = ForemanGenerator { n_comps: 1 };
        assert!(g.count() > 0, "the committed foreman cache is non-empty");
        let s = g.generate(0); // base 0, kind Raze
        let o = &s.objectives[0];
        let spawns = s.world.structures.iter().filter(|st| st.kind == StructureKind::Spawn).count();
        let ramparts = s.world.structures.iter().filter(|st| st.kind == StructureKind::Rampart).count();
        assert!(spawns >= 1, "realized base has a spawn ({spawns})");
        assert!(!s.world.towers.is_empty(), "realized base has energized towers ({})", s.world.towers.len());
        assert!(ramparts >= 1, "realized base has a rampart ring ({ramparts})");
        let walls = s.world.terrain_for(o.room).walls.len();
        assert!(walls > 50, "real terrain decoded ({walls} walls)");
    }

    /// Every (foreman base × objective kind × comp) is assessable: valid staging on clear tiles, the
    /// sizing oracle runs over the whole enumeration without panic, and all five kinds are exercised.
    #[test]
    fn foreman_base_is_assessable() {
        let g = ForemanGenerator { n_comps: 1 };
        let mut v = OracleCalibration::new();
        let report = run_suite(&g, &mut v);
        assert_eq!(report.verdicts.len(), g.count() as usize, "a verdict per scenario");
        let mut kinds = std::collections::HashSet::new();
        for i in 0..g.count() {
            let s = g.generate(i);
            let o = &s.objectives[0];
            assert!(!o.front_tiles.is_empty() && !o.support_tiles.is_empty(), "foreman#{i} ({:?}) staged", o.kind);
            let t = s.world.terrain_for(o.room);
            for fp in &o.front_tiles {
                assert!(!t.is_wall(fp.x().u8(), fp.y().u8()), "foreman#{i} front tile is on a clear tile");
            }
            kinds.insert(o.kind);
        }
        assert_eq!(kinds.len(), 5, "all five kinds over foreman bases, got {kinds:?}");
    }

    /// The breach objective targets a RAMPART (the real ring's breach point), proving the adaptive
    /// breach geometry derived from the foreman ring (not the synthetic west gate).
    #[test]
    fn foreman_breach_targets_a_rampart() {
        use screeps_combat_engine::StructureKind;
        let g = ForemanGenerator { n_comps: 1 };
        let breach = (0..g.count()).map(|i| g.generate(i)).find(|s| s.objectives[0].kind == ObjectiveKind::Breach).expect("a breach scenario exists");
        let o = &breach.objectives[0];
        let target = breach.world.structures.iter().find(|st| st.id == o.id).expect("breach objective id is a real structure");
        assert_eq!(target.kind, StructureKind::Rampart, "breach targets a rampart gate from the real ring");
    }

    /// Eyeball hook (operator visual validation): render managed assaults on the real foreman bases.
    #[test]
    #[ignore]
    fn write_foreman_replays() {
        use crate::harness::validate::render_managed_replay;
        let g = ForemanGenerator { n_comps: 1 };
        for i in 0..g.count().min(5) {
            let s = g.generate(i);
            let html = render_managed_replay(&s);
            std::fs::write(format!("foreman-replay-{}.html", s.label.replace(['#', ':'], "_")), html).unwrap();
        }
    }

    /// ADR 0024 regression gate: the hierarchical positioning fix drove period-2 ("A-B-A") movement
    /// oscillation in the **single-room** assaults to near-zero (measured ≤1.6% across Designed#0-3,5).
    /// This locks that in. The **cross-room** twin-room siege (Designed#4) is excluded from the strict
    /// bound — it still oscillates heavily (~93%) in the engaged breach phase because the strategic path
    /// isn't stitched across the room seam (the flagged "cross-room edge/flee / multi-room strategic
    /// path" follow-up, ADR 0024 Open Questions + §Future-work); it is reported here as the tracked
    /// baseline so a future fix is measurable and a catastrophic regression is still caught.
    #[test]
    fn positioning_oscillation_stays_low_across_designed() {
        use crate::harness::validate::managed_oscillation_rate;
        let mut single_room = Vec::new();
        for i in 0..Designed.count() {
            let s = Designed.generate(i);
            let cross_room = s.objectives.iter().any(|o| o.entry.room_name() != o.room);
            if let Some(r) = managed_oscillation_rate(&s) {
                println!("designed#{i} oscillation {:.1}%{}", r * 100.0, if cross_room { " (cross-room, excluded)" } else { "" });
                if cross_room {
                    assert!(r <= 0.97, "cross-room oscillation must not fully regress ({:.1}%)", r * 100.0);
                } else {
                    single_room.push(r);
                }
            }
        }
        assert!(!single_room.is_empty(), "at least one single-room Designed assault fielded a squad");
        let mean = single_room.iter().sum::<f64>() / single_room.len() as f64;
        println!("mean single-room period-2 oscillation {:.2}%", mean * 100.0);
        assert!(single_room.iter().all(|&r| r <= 0.10), "every single-room scenario stays ≤10% period-2 oscillation");
        assert!(mean <= 0.05, "mean single-room period-2 oscillation stays low ({:.2}%)", mean * 100.0);
    }

    /// The dashboard writes per-scenario replays + a contact-sheet index (smoke).
    #[test]
    fn dashboard_writes_an_index() {
        let dir = std::env::temp_dir().join("ibex-harness-dashboard-test");
        let n = report::write_dashboard(dir.to_str().unwrap()).unwrap();
        assert!(n > 0);
        assert!(dir.join("index.html").exists(), "index.html written");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// On-demand: write sample replays to `target/replays/` for eyeballing — the movement-rich MANAGED
    /// assaults over the terrain-rich Designed fixtures (the depth the operator wants) + a calibration
    /// breach/defer pair. `cargo test -p screeps-combat-eval --lib -- --ignored write_sample_replays`.
    #[test]
    #[ignore]
    fn write_sample_replays() {
        use crate::harness::validate::{render_calibration_replay, render_managed_replay};
        let _ = std::fs::create_dir_all("target/replays");
        // Movement-rich managed assaults over every terrain-rich Designed fixture (incl. multi-room).
        for i in 0..Designed.count() {
            let s = Designed.generate(i);
            let name = s.label.replace([' ', '#'], "_");
            std::fs::write(format!("target/replays/managed-{name}.html"), render_managed_replay(&s)).unwrap();
        }
        // A calibration breach + defer (the sizing-pure lens).
        let gen = RandomDefendedBase { n: 200 };
        let (mut breach, mut defer) = (false, false);
        for i in 0..200 {
            let html = render_calibration_replay(&gen.generate(i));
            if html.contains("→ BREACHED") && !breach {
                std::fs::write(format!("target/replays/calib-breach-seed{i}.html"), &html).unwrap();
                breach = true;
            } else if html.contains("deferred (") && !defer {
                std::fs::write(format!("target/replays/calib-defer-seed{i}.html"), &html).unwrap();
                defer = true;
            }
            if breach && defer {
                break;
            }
        }
        println!("wrote managed (x{}) + calibration replays to target/replays/", Designed.count());
    }
}
