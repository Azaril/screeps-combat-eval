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

/// The committed real-terrain fixtures (parsed from the embedded `resources/real-terrain.json`). A few
/// varied real rooms (mixed wall/swamp density, controller + sources) — the substrate for the Stage 2
/// `ImportedRoom` generator + the Stage 4 realistic tournament basket.
pub fn fixtures() -> Vec<TerrainFixture> {
    let raw: FixtureFile = serde_json::from_str(include_str!("../../resources/real-terrain.json"))
        .expect("embedded resources/real-terrain.json parses");
    raw.rooms
        .into_iter()
        .map(|r| TerrainFixture {
            room: r.room,
            terrain: r.terrain,
            controller: (r.controller[0], r.controller[1]),
            sources: r.sources.into_iter().map(|s| (s[0], s[1])).collect(),
            mineral: r.mineral.map(|m| (m[0], m[1])),
        })
        .collect()
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
        // NOTE (Stage 3 caveat): the bench map dump's OBJECT coords (controller/sources) are not cleanly
        // alignable with its terrain-string index convention (no single transform puts all objects on
        // clear tiles), so Stage 1 validates only the TERRAIN (the real wall/swamp map a squad navigates).
        // The controller/source ALIGNMENT must be resolved when Stage 3 feeds them to the foreman planner
        // (re-derive from the terrain, or snap to the nearest clear tile).
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
        }
    }
}
