//! Stage 2 — the generic evaluator (ADR 0023a). Drives an attacker + a defender through the
//! authoritative engine `resolve_tick` until a swappable [`RunUntil`] predicate fires. Knows nothing
//! about objectives, oracles, or sizing — it just steps the world. Multi-room (the engine is N-room).
//!
//! The attacker produces fresh [`Intents`] each tick; the defender ADDS to them (the `defense_intents`
//! idiom — towers + defender creeps layered over the attacker's intents), so the two compose into one
//! resolved tick. [`evaluate_recorded`] is the same loop but captures a [`CombatRecording`] for the
//! visualizer.

use screeps_combat_engine::{record_tick, resolve_tick, CombatRecording, CombatWorld, Intents, PlayerId, StructureId};

/// Why an evaluation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// All target structures were destroyed (the attacker achieved the objective).
    ObjectivesComplete,
    /// The given side has no living creeps left.
    SideWiped(PlayerId),
    /// The tick budget elapsed with no terminal condition.
    Timeout,
}

/// The result of an evaluation: the final world, ticks elapsed, and why it stopped.
#[derive(Clone, Debug)]
pub struct EvalOutcome {
    pub world: CombatWorld,
    pub ticks: u32,
    pub stop: StopReason,
}

/// A swappable stop condition, checked at the start of each tick. `Some(reason)` ⇒ stop now.
pub trait RunUntil {
    fn check(&self, world: &CombatWorld, tick: u32) -> Option<StopReason>;
}

/// Stop when every listed structure (a core/spawn, in `structures` OR `towers`) is destroyed.
pub struct ObjectivesDestroyed(pub Vec<StructureId>);
impl RunUntil for ObjectivesDestroyed {
    fn check(&self, world: &CombatWorld, _tick: u32) -> Option<StopReason> {
        let alive = |id: StructureId| {
            world.structures.iter().any(|s| s.id == id && s.is_alive()) || world.towers.iter().any(|t| t.id == id && t.is_alive())
        };
        if self.0.iter().all(|&id| !alive(id)) {
            Some(StopReason::ObjectivesComplete)
        } else {
            None
        }
    }
}

/// Stop when `owner` has no living creeps.
pub struct SideWiped(pub PlayerId);
impl RunUntil for SideWiped {
    fn check(&self, world: &CombatWorld, _tick: u32) -> Option<StopReason> {
        if world.creeps.iter().any(|c| c.owner == self.0 && c.is_alive()) {
            None
        } else {
            Some(StopReason::SideWiped(self.0))
        }
    }
}

/// Stop when ANY sub-condition fires (first match wins, in order).
pub struct AnyOf(pub Vec<Box<dyn RunUntil>>);
impl RunUntil for AnyOf {
    fn check(&self, world: &CombatWorld, tick: u32) -> Option<StopReason> {
        self.0.iter().find_map(|c| c.check(world, tick))
    }
}

fn step(
    world: &mut CombatWorld,
    attacker: &mut dyn FnMut(&CombatWorld) -> Intents,
    defender: &mut dyn FnMut(&CombatWorld, &mut Intents),
    rec: Option<&mut CombatRecording>,
) {
    let mut intents = attacker(world);
    defender(world, &mut intents);
    match rec {
        Some(rec) => {
            record_tick(rec, world, &intents);
        }
        None => {
            resolve_tick(world, &intents);
        }
    }
}

fn run(
    mut world: CombatWorld,
    attacker: &mut dyn FnMut(&CombatWorld) -> Intents,
    defender: &mut dyn FnMut(&CombatWorld, &mut Intents),
    run_until: &dyn RunUntil,
    max_ticks: u32,
    mut rec: Option<&mut CombatRecording>,
) -> EvalOutcome {
    let mut ticks = 0;
    loop {
        if let Some(stop) = run_until.check(&world, ticks) {
            return EvalOutcome { world, ticks, stop };
        }
        if ticks >= max_ticks {
            return EvalOutcome { world, ticks, stop: StopReason::Timeout };
        }
        step(&mut world, attacker, defender, rec.as_deref_mut());
        ticks += 1;
    }
}

/// Run an engagement to the `run_until` predicate (or `max_ticks`), no recording.
pub fn evaluate(
    world: CombatWorld,
    attacker: &mut dyn FnMut(&CombatWorld) -> Intents,
    defender: &mut dyn FnMut(&CombatWorld, &mut Intents),
    run_until: &dyn RunUntil,
    max_ticks: u32,
) -> EvalOutcome {
    run(world, attacker, defender, run_until, max_ticks, None)
}

/// Same as [`evaluate`] but captures a per-tick [`CombatRecording`] for the replay visualizer.
pub fn evaluate_recorded(
    world: CombatWorld,
    attacker: &mut dyn FnMut(&CombatWorld) -> Intents,
    defender: &mut dyn FnMut(&CombatWorld, &mut Intents),
    run_until: &dyn RunUntil,
    max_ticks: u32,
) -> (EvalOutcome, CombatRecording) {
    let mut rec = CombatRecording::new();
    let outcome = run(world, attacker, defender, run_until, max_ticks, Some(&mut rec));
    (outcome, rec)
}
