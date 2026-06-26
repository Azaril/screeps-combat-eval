//! ADR 0025 §12 Stage 3a — the **offline foreman base-capture tool**. Runs the foreman room planner on
//! the committed real-terrain fixtures and writes the combat-realizable structure placements to
//! `resources/captured-bases.json` (the cache the fast `ForemanGenerator` loads). SLOW (the planner
//! escalates anchor beams, ~3.6–55s/room) — run ONCE, manually, never in CI:
//!
//!   cargo run --release -p screeps-combat-eval --bin capture_base -- [N]
//!
//! `N` (optional) caps how many fixtures to plan (default: all). Existing cached rooms are kept and only
//! missing ones are planned, so re-runs are incremental.

use screeps_combat_eval::harness::foreman_capture::{capture, CapturedBase};
use screeps_combat_eval::harness::terrain_import::fixtures;
use std::time::Instant;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Cache {
    bases: Vec<CapturedBase>,
}

fn main() {
    let limit: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(usize::MAX);
    // Anchor to the crate dir so the cache lands in resources/ regardless of CWD (cargo run uses the
    // workspace root as CWD).
    let out_path = concat!(env!("CARGO_MANIFEST_DIR"), "/resources/captured-bases.json");

    // Incremental: keep already-cached rooms, only plan the rest.
    let mut cache: Cache = std::fs::read_to_string(out_path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    let have: std::collections::HashSet<String> = cache.bases.iter().map(|b| b.room.clone()).collect();

    let fx = fixtures();
    let todo: Vec<_> = fx.iter().filter(|f| !have.contains(&f.room)).take(limit).collect();
    println!("capture_base: {} fixtures, {} already cached, planning {} (foreman is SLOW)…", fx.len(), have.len(), todo.len());

    for (i, fixture) in todo.iter().enumerate() {
        let t = Instant::now();
        match capture(fixture) {
            Ok(base) => {
                let (spawns, towers, ramparts, walls) = base.structures.iter().fold((0, 0, 0, 0), |(s, t, r, w), st| match st.kind.as_str() {
                    "spawn" => (s + 1, t, r, w),
                    "tower" => (s, t + 1, r, w),
                    "rampart" => (s, t, r + 1, w),
                    _ => (s, t, r, w + 1),
                });
                println!("  [{}/{}] {} planned in {:.1}s — {spawns} spawn, {towers} tower, {ramparts} rampart, {walls} wall", i + 1, todo.len(), base.room, t.elapsed().as_secs_f64());
                cache.bases.push(base);
                // Persist after each room so a long run is resumable / partial results survive.
                std::fs::write(out_path, serde_json::to_string(&cache).expect("serialize cache")).expect("write cache");
            }
            Err(e) => println!("  [{}/{}] {} SKIPPED: {e}", i + 1, todo.len(), fixture.room),
        }
    }
    println!("capture_base: done — {} bases cached in {out_path}", cache.bases.len());
}
