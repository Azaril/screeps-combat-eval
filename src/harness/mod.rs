//! The staged combat harness (ADR 0023a): four swappable stages — **generation** ([`generate`]) →
//! **evaluation** ([`evaluate`], a generic run-until loop) → **validation** ([`validate`]) →
//! **visualization** (the replay player, Phase V) — so a generator, a validator, and a stop condition
//! compose freely. [`run_suite`] crosses a generator with a validator. The P-FORCE oracle-calibration
//! WIN is one `(generator, validator)` pair; [`calibrate`] is the convenience that runs it.

pub mod evaluate;
pub mod generate;
pub mod report;
pub mod scenario;
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
    use generate::{Designed, Permutations};
    use validate::{ManagedSquadIntegration, SelfPlay, SizingWins};

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
