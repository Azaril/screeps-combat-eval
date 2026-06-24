//! The staged combat harness (ADR 0023a): four swappable stages — **generation** ([`generate`]) →
//! **evaluation** ([`evaluate`], a generic run-until loop) → **validation** ([`validate`]) →
//! **visualization** (the replay player, Phase V) — so a generator, a validator, and a stop condition
//! compose freely. [`run_suite`] crosses a generator with a validator. The P-FORCE oracle-calibration
//! WIN is one `(generator, validator)` pair; [`calibrate`] is the convenience that runs it.

pub mod evaluate;
pub mod generate;
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
        assert!(html.contains("const FRAMES=["));
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

    /// On-demand: write a few sample replays (a winnable-breach + a deferred) to `target/replays/` for
    /// eyeballing. `cargo test -p screeps-combat-eval --lib -- --ignored write_sample_replays`.
    #[test]
    #[ignore]
    fn write_sample_replays() {
        use crate::harness::validate::render_calibration_replay;
        let _ = std::fs::create_dir_all("target/replays");
        let gen = RandomDefendedBase { n: 200 };
        let (mut wrote_breach, mut wrote_defer) = (false, false);
        for i in 0..200 {
            let s = gen.generate(i);
            let html = render_calibration_replay(&s);
            let breached = html.contains("→ BREACHED");
            let deferred = html.contains("deferred (");
            if breached && !wrote_breach {
                std::fs::write(format!("target/replays/breach-seed{i}.html"), &html).unwrap();
                wrote_breach = true;
            } else if deferred && !wrote_defer {
                std::fs::write(format!("target/replays/defer-seed{i}.html"), &html).unwrap();
                wrote_defer = true;
            }
            if wrote_breach && wrote_defer {
                break;
            }
        }
        println!("wrote samples to target/replays/ (breach={wrote_breach}, defer={wrote_defer})");
    }
}
