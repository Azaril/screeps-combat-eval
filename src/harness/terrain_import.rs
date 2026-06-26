//! ADR 0025 §12 Stage 1 — **real terrain import** into the host combat sim. Brings a handful of REAL
//! mmo:shard3 room terrains (committed in `resources/real-terrain.json`, extracted from the
//! `screeps-foreman-bench` map dump) into a [`CombatTerrain`], so the EV-kernel tuning + Lanchester
//! validation can run on realistic walls/swamps instead of only hand-authored synthetic beds.
//!
//! Why committed fixtures, not a live fetch: a room's terrain is a 2500-char digit string (`screeps-
//! rest-api` `types.rs`), so a handful is ~13 KB — fully deterministic, no credentials, no async, no CI
//! network. The live rest-api path (rate-capped, credential-gated) is reserved for the offline capture
//! tool (§12 Stage 3a). The `FastRoomTerrain` ↔ `CombatTerrain` bridge the foreman planner needs lands
//! with Stage 3 (it pulls the `screeps-foreman` dep); Stage 1 is the decoder + fixtures only.

use screeps_combat_engine::CombatTerrain;

/// A committed real-terrain fixture: room name + the 2500-char encoded terrain + the source-object
/// positions the foreman planner needs in Stage 3 (controller / sources / mineral). Owned (parsed from
/// the embedded JSON), not `&'static`, so the fixture data stays a single source of truth in the JSON.
///
/// The object positions are **terrain-aligned** (on clear tiles): the raw dump coords are systematically
/// nudged one tile into wall-edges (a dump-tool bug — every object is exactly Chebyshev-distance 1 from an
/// open tile under the verified row-major `y*50+x` terrain decode, confirmed across 3000 rooms vs the
/// `screeps-game-api` `LocalRoomTerrain` convention), so [`fixtures`] snaps each to its nearest open tile
/// at load (the coordinate fix). This preserves the room's real source/controller LAYOUT (the snapped
/// tile is adjacent to the dump tile) while guaranteeing the foreman planner gets valid, non-wall inputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TerrainFixture {
    pub room: String,
    pub terrain: String,
    pub controller: (u8, u8),
    pub sources: Vec<(u8, u8)>,
    pub mineral: Option<(u8, u8)>,
}

#[derive(serde::Deserialize)]
struct FixtureFile {
    rooms: Vec<RawFixture>,
}
#[derive(serde::Deserialize)]
struct RawFixture {
    room: String,
    terrain: String,
    controller: [u8; 2],
    sources: Vec<[u8; 2]>,
    mineral: Option<[u8; 2]>,
}

/// The committed real-terrain fixtures (parsed from the embedded `resources/real-terrain.json`). Varied
/// real rooms (mixed wall/swamp density, controller + sources) — the substrate for the Stage 2
/// `ImportedRoom` generator + the Stage 4 realistic tournament basket. Object coords are snapped to clear
/// tiles at load ([`snap_to_open`]; see [`TerrainFixture`] — the coordinate fix).
pub fn fixtures() -> Vec<TerrainFixture> {
    let raw: FixtureFile = serde_json::from_str(include_str!("../../resources/real-terrain.json"))
        .expect("embedded resources/real-terrain.json parses");
    raw.rooms
        .into_iter()
        .map(|r| {
            let terrain = decode_terrain(&r.terrain);
            // Snap each object to its nearest open tile (the dump-coord fix), keeping them distinct.
            let mut taken = std::collections::HashSet::new();
            let controller = snap_to_open(&terrain, r.controller[0], r.controller[1], &taken);
            taken.insert(controller);
            let mut sources = Vec::with_capacity(r.sources.len());
            for s in &r.sources {
                let snapped = snap_to_open(&terrain, s[0], s[1], &taken);
                taken.insert(snapped);
                sources.push(snapped);
            }
            let mineral = r.mineral.map(|m| {
                let snapped = snap_to_open(&terrain, m[0], m[1], &taken);
                taken.insert(snapped);
                snapped
            });
            TerrainFixture { room: r.room, terrain: r.terrain, controller, sources, mineral }
        })
        .collect()
}

/// The nearest open (non-wall) tile to `(x,y)` not already in `taken`, by 8-connected BFS — the
/// coordinate fix for the dump's wall-nudged object coords (every object is exactly distance 1 from open,
/// so this returns the true adjacent tile). `taken` keeps co-snapped objects distinct (two sources never
/// collapse onto one tile). Falls back to `(x,y)` if the room is fully walled (never, for real rooms).
///
/// This is a WORKAROUND for a suspect, not-yet-root-caused dump offset — see
/// `docs/design/0025a-coordinate-offset-anomaly.md` (terrain decode is verified correct; the dump's
/// object coords are the bug; revisit when the dump provenance / a live-API cross-check is available).
pub fn snap_to_open(terrain: &CombatTerrain, x: u8, y: u8, taken: &std::collections::HashSet<(u8, u8)>) -> (u8, u8) {
    use std::collections::{HashSet, VecDeque};
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    let mut q = VecDeque::from([(x as i32, y as i32)]);
    while let Some((cx, cy)) = q.pop_front() {
        if !(0..50).contains(&cx) || !(0..50).contains(&cy) || !seen.insert((cx, cy)) {
            continue;
        }
        let t = (cx as u8, cy as u8);
        if !terrain.is_wall(t.0, t.1) && !taken.contains(&t) {
            return t;
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)] {
            q.push_back((cx + dx, cy + dy));
        }
    }
    (x, y)
}

/// Decode a 2500-char shard terrain string (row-major `index = y*50 + x`; each char a hex digit whose
/// bits are `1`=wall, `2`=swamp — `3` is wall+swamp ⇒ wall wins) into the engine's sparse
/// [`CombatTerrain`]. The inverse of the `screeps-foreman-bench` terrain visitor. Out-of-spec chars
/// decode as plain. A non-2500 string decodes what it has (callers should pass full rooms).
pub fn decode_terrain(encoded: &str) -> CombatTerrain {
    let mut t = CombatTerrain::default();
    for (i, ch) in encoded.chars().enumerate().take(2500) {
        let v = ch.to_digit(16).unwrap_or(0) as u8;
        let (x, y) = ((i % 50) as u8, (i / 50) as u8);
        if v & 1 != 0 {
            t.walls.insert((x, y));
        } else if v & 2 != 0 {
            t.swamps.insert((x, y));
        }
    }
    t
}

/// Decode a 2500-char terrain string into the foreman planner's dense [`FastRoomTerrain`] buffer (the
/// 2500-byte `TerrainFlags` form: bit 0 = wall, bit 1 = swamp — the same per-char hex value). The bridge
/// the §12 Stage 3 capture tool feeds the planner. Row-major `y*50+x`, matching `FastRoomTerrain`'s
/// `Location::to_index` (verified: both are `y*50+x`).
pub fn decode_fast(encoded: &str) -> screeps_foreman::terrain::FastRoomTerrain {
    let buffer: Vec<u8> = encoded.chars().take(2500).map(|c| c.to_digit(16).unwrap_or(0) as u8).collect();
    screeps_foreman::terrain::FastRoomTerrain::new(buffer)
}

/// Bridge the foreman dense [`FastRoomTerrain`] to the engine sparse [`CombatTerrain`] (the inverse of
/// what the planner consumes). Used where a plan was built over a `FastRoomTerrain` but the sim needs the
/// `CombatTerrain` form. Walls dominate swamps (matching [`decode_terrain`]).
pub fn fast_to_combat(fast: &screeps_foreman::terrain::FastRoomTerrain) -> CombatTerrain {
    let mut t = CombatTerrain::default();
    for y in 0..50u8 {
        for x in 0..50u8 {
            if fast.is_wall(x, y) {
                t.walls.insert((x, y));
            } else if fast.is_swamp(x, y) {
                t.swamps.insert((x, y));
            }
        }
    }
    t
}

/// Encode a [`CombatTerrain`] back to the 2500-char form (walls→`'1'`, swamps→`'2'`, plain→`'0'`). The
/// inverse of [`decode_terrain`] — used only by the round-trip test (and handy for capturing fixtures).
pub fn encode_terrain(t: &CombatTerrain) -> String {
    (0..2500)
        .map(|i| {
            let (x, y) = ((i % 50) as u8, (i / 50) as u8);
            if t.is_wall(x, y) {
                '1'
            } else if t.swamps.contains(&(x, y)) {
                '2'
            } else {
                '0'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_roundtrips_a_known_pattern() {
        let mut t = CombatTerrain::default();
        t.walls.insert((0, 0));
        t.walls.insert((49, 49));
        t.walls.insert((10, 3)); // index 3*50+10 = 160
        t.swamps.insert((5, 5));
        t.swamps.insert((25, 25));
        let round = decode_terrain(&encode_terrain(&t));
        assert_eq!(round.walls, t.walls, "walls survive encode→decode");
        assert_eq!(round.swamps, t.swamps, "swamps survive encode→decode");
    }

    #[test]
    fn decode_places_tiles_at_the_right_coords() {
        // A single wall at index 160 = (x=10, y=3); a single swamp at index 2 = (x=2, y=0).
        let mut s = vec!['0'; 2500];
        s[160] = '1';
        s[2] = '2';
        let t = decode_terrain(&s.into_iter().collect::<String>());
        assert!(t.is_wall(10, 3) && t.walls.len() == 1, "wall at (10,3)");
        assert!(t.swamps.contains(&(2, 0)) && t.swamps.len() == 1, "swamp at (2,0)");
        // '3' (wall+swamp bits) is a WALL, not a swamp.
        let mut s2 = vec!['0'; 2500];
        s2[0] = '3';
        let t2 = decode_terrain(&s2.into_iter().collect::<String>());
        assert!(t2.is_wall(0, 0) && t2.swamps.is_empty(), "'3' decodes as wall");
    }

    /// Largest 4-connected open (non-wall) component as a fraction of all open tiles — a navigability
    /// measure (a real room's interior is one big connected space, not fragmented pockets).
    fn largest_open_fraction(t: &CombatTerrain) -> f64 {
        let open = |x: i32, y: i32| (0..50).contains(&x) && (0..50).contains(&y) && !t.is_wall(x as u8, y as u8);
        let total = (0..50).flat_map(|x| (0..50).map(move |y| (x, y))).filter(|&(x, y)| open(x, y)).count();
        if total == 0 {
            return 0.0;
        }
        let mut seen = std::collections::HashSet::new();
        let mut best = 0usize;
        for sx in 0..50i32 {
            for sy in 0..50i32 {
                if !open(sx, sy) || seen.contains(&(sx, sy)) {
                    continue;
                }
                let mut stack = vec![(sx, sy)];
                let mut size = 0;
                while let Some((x, y)) = stack.pop() {
                    if !open(x, y) || !seen.insert((x, y)) {
                        continue;
                    }
                    size += 1;
                    stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]);
                }
                best = best.max(size);
            }
        }
        best as f64 / total as f64
    }

    #[test]
    fn fixtures_are_real_navigable_rooms() {
        let fx = fixtures();
        assert!(fx.len() >= 3, "a handful of real-terrain fixtures ({} found)", fx.len());
        for f in &fx {
            assert_eq!(f.terrain.chars().count(), 2500, "{} terrain is a full 50x50 room", f.room);
            assert!(!f.sources.is_empty(), "{} carries source metadata", f.room);
            let terrain = decode_terrain(&f.terrain);
            // A REAL room: meaningful walls, but not sealed.
            assert!(terrain.walls.len() > 50, "{} has real terrain ({} walls)", f.room, terrain.walls.len());
            assert!(terrain.walls.len() < 2400, "{} is not fully walled ({} walls)", f.room, terrain.walls.len());
            // Navigable: the open interior is one big connected component (a squad can traverse it).
            let frac = largest_open_fraction(&terrain);
            assert!(frac > 0.6, "{} interior is navigable (largest open component {:.0}% of open tiles)", f.room, frac * 100.0);
            // The COORDINATE FIX: snapped object positions are on clear tiles + distinct.
            assert!(!terrain.is_wall(f.controller.0, f.controller.1), "{} controller snapped to a clear tile", f.room);
            for &(sx, sy) in &f.sources {
                assert!(!terrain.is_wall(sx, sy), "{} source ({sx},{sy}) snapped to a clear tile", f.room);
            }
            let mut all: Vec<(u8, u8)> = vec![f.controller];
            all.extend(&f.sources);
            if let Some(m) = f.mineral {
                assert!(!terrain.is_wall(m.0, m.1), "{} mineral snapped to a clear tile", f.room);
                all.push(m);
            }
            let distinct: std::collections::HashSet<_> = all.iter().collect();
            assert_eq!(distinct.len(), all.len(), "{} objects snap to distinct tiles", f.room);
        }
    }

    #[test]
    fn fast_to_combat_matches_decode() {
        // The two terrain bridges agree: decode_fast → fast_to_combat == decode_terrain (the planner
        // path and the sim path see the same walls/swamps).
        let fx = fixtures();
        let f = &fx[0];
        let via_fast = fast_to_combat(&decode_fast(&f.terrain));
        let direct = decode_terrain(&f.terrain);
        assert_eq!(via_fast.walls, direct.walls, "{}: fast/direct walls agree", f.room);
        assert_eq!(via_fast.swamps, direct.swamps, "{}: fast/direct swamps agree", f.room);
    }

    #[test]
    fn snap_recovers_objects_within_one_tile() {
        // The dump's wall-nudged coords are exactly distance 1 from open, so the snapped tile is adjacent
        // to (i.e., recovers) the real object position — not a far-off guess.
        let raw: FixtureFile = serde_json::from_str(include_str!("../../resources/real-terrain.json")).unwrap();
        for r in &raw.rooms {
            let terrain = decode_terrain(&r.terrain);
            let taken = std::collections::HashSet::new();
            let snapped = snap_to_open(&terrain, r.controller[0], r.controller[1], &taken);
            let cheby = (snapped.0 as i32 - r.controller[0] as i32).abs().max((snapped.1 as i32 - r.controller[1] as i32).abs());
            assert!(cheby <= 1, "{} controller snap moved {cheby} tiles (expected <=1)", r.room);
        }
    }
}
