//! The harness scenario model (ADR 0023a, stage 1 output) — a generated world + objectives, with NO
//! knowledge of how it's evaluated or validated. A [`Generator`](crate::harness::generate::Generator)
//! produces these; a [`Validator`](crate::harness::validate::Validator) consumes them. Single- or
//! multi-room.

use screeps::{Position, RoomName};
use screeps_combat_engine::{CombatWorld, PlayerId, StructureId};

/// What KIND of objective the attacker pursues (ADR 0025 §12 Stage 2). Drives the run-until stop
/// condition + (Declaim) the world population; the staging geometry is shared. (`EngageObjective` in the
/// decision crate is only Destroy/Hold; the kernel's per-kind targeting is the §11 #11 follow-on — for
/// now `Farm`/`Declaim` exercise the stop-condition + world plumbing.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ObjectiveKind {
    /// Hold a tile under fire (mine/keep) — never destroy; runs to the on-site budget, scored by survival.
    Farm,
    /// Clear the room's defenders, then hold — stop when the defender side is wiped.
    Secure,
    /// Break the rampart shielding the core — stop when the breach rampart (`Objective.id`) falls.
    Breach,
    /// Destroy the core/spawn — the default (today's behaviour); stop when `Objective.id` is destroyed.
    #[default]
    Raze,
    /// Neutralize the enemy controller (`attackController`) — stop when the controller at `Objective.pos`
    /// becomes unowned. Needs a `SimController` in the world.
    Declaim,
}

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
    /// Where a MOVING assault stages from — a clear tile on the approach (room-edge-ish), distinct from
    /// the in-range `front_tiles`. The `ManagedSquadIntegration` validator spawns the squad here and
    /// lets it path to the objective (so the replay shows real movement); `OracleCalibration` ignores it
    /// (it stages in-range for sizing-purity).
    pub entry: Position,
    /// What achieving this objective MEANS (drives the run-until stop condition + world population). The
    /// synthetic generators default to [`ObjectiveKind::Raze`] (today's destroy-the-spawn behaviour).
    pub kind: ObjectiveKind,
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
