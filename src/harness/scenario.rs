//! The harness scenario model (ADR 0023a, stage 1 output) — a generated world + objectives, with NO
//! knowledge of how it's evaluated or validated. A [`Generator`](crate::harness::generate::Generator)
//! produces these; a [`Validator`](crate::harness::validate::Validator) consumes them. Single- or
//! multi-room.

use screeps_combat_engine::{CombatWorld, PlayerId, StructureId};
use screeps::{Position, RoomName};

/// What the attacker must achieve at one target. A scenario may carry several (multi-room sieges).
/// Carries the **staging geometry** the generator computed from the layout (where to stand to engage),
/// so a validator can field a force WITHOUT re-deriving the approach — keeping validation layout-
/// agnostic and generation the sole owner of "where the breach/approach is".
#[derive(Clone, Debug)]
pub struct Objective {
    /// The structure to destroy (a core / spawn).
    pub id: StructureId,
    pub room: RoomName,
    pub pos: Position,
    /// The focus tile an attacker assaults from — `front_tiles[0]`. Tower ranges to the assault are
    /// measured here (it's the closest-to-objective attacker tile, so the defense focus-fires it).
    pub assault_pos: Position,
    /// Tiles at range 1 to BOTH the breach gate and the objective (where dismantlers stand and hit
    /// both phases without moving). `[0]` == `assault_pos` == the unique focus.
    pub front_tiles: Vec<Position>,
    /// Tiles adjacent (range 1) to `assault_pos` and NOT the focus (where healers stand → full
    /// adjacent `HEAL_POWER`, never mistaken for the focus by the defense AI).
    pub support_tiles: Vec<Position>,
}

/// A generated scenario: a world (terrain + structures + towers + DEFENDER creeps = the opponent force,
/// already placed) + the objective(s) + the two sides' owners + per-engagement budgets. Oracle-/
/// validator-agnostic — a validator that needs an oracle `DefenseProfile` derives it from `world`.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub world: CombatWorld,
    pub objectives: Vec<Objective>,
    pub attacker_owner: PlayerId,
    pub defender_owner: PlayerId,
    /// Spawn-energy capacity the attacker's home affords (sizes the fielded bodies). RCL-ish.
    pub member_energy: u32,
    /// Ticks the attacker has on-site (`CREEP_LIFE_TIME − spawn − travel`); the evaluation tick cap.
    pub onsite_budget: u32,
    pub label: String,
    pub seed: u64,
}
