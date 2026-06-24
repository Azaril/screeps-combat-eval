//! Stage 1 — scenario generation (ADR 0023a). A [`Generator`] produces [`Scenario`]s; it owns the
//! layout + opponent placement and is oracle-agnostic (a validator derives any oracle profile from the
//! world). [`RandomDefendedBase`] is the seeded single-room defended-base generator the oracle
//! calibration runs on (the Move B draws, now behind the seam). Phases B/C add permutation, designed,
//! and multi-room generators + opponent force specs.

use crate::harness::scenario::{Objective, Scenario};
use screeps::{Position, RoomCoordinate, RoomName};
use screeps_combat_engine::{CombatWorld, PlayerId, StructureKind};
use screeps_combat_agent::scenario::ScenarioBuilder;

pub const ATTACKER: PlayerId = 0;
pub const DEFENDER: PlayerId = 1;

// ── Bed geometry (the Move B single-room layout; Phase B generalizes) ──────────────────────────────
// Core (spawn) at (25,25), the breach rampart at (24,25). The focus dismantler sits at the corner
// (24,24): range 1 to BOTH rampart (orthogonal) and core (diagonal). All front tiles are range 1 to
// both (whole squad dismantles both phases); the focus is front[0] (unique core-range-1 by order).
// Support (healer) tiles are the focus's neighbours at core-range 2 (full adjacent HEAL, never focus).
const CORE: (u8, u8) = (25, 25);
const RAMPART: (u8, u8) = (24, 25);
const ASSAULT: (u8, u8) = (24, 24);
const FRONT_TILES: [(u8, u8); 4] = [(24, 24), (25, 24), (24, 26), (25, 26)];
const SUPPORT_TILES: [(u8, u8); 5] = [(23, 23), (24, 23), (25, 23), (23, 24), (23, 25)];
const TOWER_TILES: [(u8, u8); 6] = [(24, 8), (24, 12), (24, 16), (20, 24), (28, 24), (24, 32)];

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos_in(room: RoomName, x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
}

/// A scenario source. Seeded by index → fully reproducible.
pub trait Generator {
    fn label(&self) -> &str;
    /// How many distinct scenarios this generator offers.
    fn count(&self) -> u32;
    fn generate(&self, index: u32) -> Scenario;
}

// ── Seeded RNG (SplitMix64 — per-index reproducible; no `rand`/`Date`/`Math.random`) ───────────────
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn seeded(index: u32) -> Self {
        Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1))
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive range `[lo, hi]`.
    pub(crate) fn range(&mut self, lo: u32, hi: u32) -> u32 {
        debug_assert!(hi >= lo);
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
    pub(crate) fn chance(&mut self, pct: u32) -> bool {
        self.range(0, 99) < pct
    }
    pub(crate) fn pick(&mut self, xs: &[u32]) -> u32 {
        xs[(self.next_u64() % xs.len() as u64) as usize]
    }
}

/// The seeded single-room defended-base generator (the oracle-calibration substrate): a core behind a
/// (usually present) rampart, 0–6 towers of varied energy/range, a small safe-mode chance, and RCL-ish
/// member energy + on-site budget. No opponent CREEPS yet (`enemy_dps` is the validator's concern; the
/// force-spec hook lands in Phase C).
pub struct RandomDefendedBase {
    pub n: u32,
}

impl Generator for RandomDefendedBase {
    fn label(&self) -> &str {
        "random-defended-base"
    }
    fn count(&self) -> u32 {
        self.n
    }
    fn generate(&self, index: u32) -> Scenario {
        let mut rng = Rng::seeded(index);
        let member_energy = rng.pick(&[1300, 1800, 2300, 3300, 5600, 12_900]);
        let onsite_budget = rng.range(600, 1400);
        let core_hits = rng.range(20_000, 100_000);
        let rampart_hits = if rng.chance(70) { rng.range(1, 80_000) } else { 0 };
        let n_towers = rng.range(0, 6);
        let safe_mode = rng.chance(5);

        let rm = room();
        let mut b = ScenarioBuilder::empty(rm);
        let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), CORE.0, CORE.1, core_hits, core_hits);
        for &(tx, ty) in TOWER_TILES.iter().take(n_towers as usize) {
            let energy = if rng.chance(25) { rng.range(0, 9) } else { rng.range(100, 100_000) };
            b.tower(DEFENDER, tx, ty, energy);
        }
        let mut world: CombatWorld = if rampart_hits > 0 {
            b.rampart(DEFENDER, RAMPART.0, RAMPART.1, rampart_hits).build()
        } else {
            b.build()
        };
        if safe_mode {
            world.safe_mode_owner = Some(DEFENDER);
        }

        let objective = Objective {
            id: core_id,
            room: rm,
            pos: pos_in(rm, CORE.0, CORE.1),
            assault_pos: pos_in(rm, ASSAULT.0, ASSAULT.1),
            front_tiles: FRONT_TILES.iter().map(|&(x, y)| pos_in(rm, x, y)).collect(),
            support_tiles: SUPPORT_TILES.iter().map(|&(x, y)| pos_in(rm, x, y)).collect(),
        };

        Scenario {
            world,
            objectives: vec![objective],
            attacker_owner: ATTACKER,
            defender_owner: DEFENDER,
            member_energy,
            onsite_budget,
            label: format!("random-defended-base#{index}"),
            seed: index as u64,
        }
    }
}
