//! Stage 4 dashboard (ADR 0023a) — write a run's per-scenario replays + a contact-sheet `index.html`
//! linking them, so the operator browses the variety (layouts, forces, movement) and each scenario's
//! validator verdict in one place. Host-only; writes to a directory the operator opens.

use crate::harness::generate::{Designed, Generator, Permutations, RandomDefendedBase};
use crate::harness::validate::{calibration_replay_data, self_play_replay_data, OracleCalibration, SelfPlay, Validator};
use crate::harness::visualize::write_replay;
use std::fmt::Write as _;

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

struct Entry {
    file: String,
    title: String,
    lens: &'static str,
    verdict: String,
    pass: bool,
}

/// Write replays + a contact-sheet `index.html` under `dir`. Covers the terrain-rich Designed fixtures
/// (movement-rich MANAGED assaults + their traversal verdict), a `Permutations` sample (managed), and a
/// `RandomDefendedBase` sample (sizing-pure CALIBRATION replays + the oracle verdict) — so the index
/// spans both lenses + the layout/force variety. Returns the number of replays written.
pub fn write_dashboard(dir: &str) -> std::io::Result<usize> {
    std::fs::create_dir_all(dir)?;
    let mut entries: Vec<Entry> = Vec::new();

    // Designed fixtures — SELF-PLAY (both sides run the squad brain → both move + fight) over terrain + forces.
    {
        let g = Designed;
        let mut v = SelfPlay;
        for i in 0..g.count() {
            let s = g.generate(i);
            let name = format!("selfplay-designed-{i}");
            let (rec, meta) = self_play_replay_data(&s);
            write_replay(dir, &name, &rec, &meta)?;
            let verdict = v.validate(&s);
            entries.push(Entry { file: format!("{name}.html"), title: s.label, lens: "self-play", verdict: verdict.detail, pass: verdict.pass });
        }
    }
    // A few permutations — self-play over the enumerated layout grid.
    {
        let g = Permutations;
        let mut v = SelfPlay;
        for i in [0u32, 13, 40, 75] {
            let s = g.generate(i);
            let name = format!("selfplay-perm-{i}");
            let (rec, meta) = self_play_replay_data(&s);
            write_replay(dir, &name, &rec, &meta)?;
            let verdict = v.validate(&s);
            entries.push(Entry { file: format!("{name}.html"), title: s.label, lens: "self-play", verdict: verdict.detail, pass: verdict.pass });
        }
    }
    // RandomDefendedBase — the sizing-pure calibration lens.
    {
        let g = RandomDefendedBase { n: 8 };
        let mut v = OracleCalibration::new();
        for i in 0..8 {
            let s = g.generate(i);
            let name = format!("calib-rdb-{i}");
            let (rec, meta) = calibration_replay_data(&s);
            write_replay(dir, &name, &rec, &meta)?;
            let verdict = v.validate(&s);
            entries.push(Entry { file: format!("{name}.html"), title: s.label, lens: "calibration", verdict: verdict.detail, pass: verdict.pass });
        }
    }

    let mut idx = String::new();
    idx.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Combat harness replays</title><style>");
    idx.push_str("body{background:#0b1020;color:#e5e7eb;font:14px/1.6 ui-monospace,Menlo,Consolas,monospace;margin:24px;}");
    idx.push_str("h1{font-size:18px;} table{border-collapse:collapse;width:100%;} td,th{border-bottom:1px solid #334155;padding:6px 10px;text-align:left;}");
    idx.push_str("a{color:#3b82f6;} .ok{color:#22c55e;} .no{color:#ef4444;} .lens{color:#94a3b8;}</style></head><body>");
    idx.push_str("<h1>Combat harness replays — open any scenario to scrub the engagement</h1>");
    idx.push_str("<table><tr><th>scenario</th><th>lens</th><th>verdict</th></tr>");
    for e in &entries {
        let mark = if e.pass { "<span class=\"ok\">●</span>" } else { "<span class=\"no\">●</span>" };
        let _ = write!(
            idx,
            "<tr><td>{mark} <a href=\"{}\">{}</a></td><td class=\"lens\">{}</td><td>{}</td></tr>",
            esc(&e.file),
            esc(&e.title),
            e.lens,
            esc(&e.verdict)
        );
    }
    idx.push_str("</table></body></html>");
    std::fs::write(format!("{dir}/index.html"), idx)?;
    Ok(entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On-demand: write the full replay dashboard to `target/replays-dashboard/` (open `index.html`).
    /// `cargo test -p screeps-combat-eval --lib -- --ignored write_replay_dashboard --nocapture`.
    #[test]
    #[ignore]
    fn write_replay_dashboard() {
        let n = write_dashboard("target/replays-dashboard").unwrap();
        println!("wrote {n} replays + index.html to target/replays-dashboard/");
    }
}
