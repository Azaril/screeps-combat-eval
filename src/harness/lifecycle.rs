//! Engine-backed lifecycle harness — the FORMING phase (ADR 0028). Drives the REAL forming kernels
//! (`fielding` K3 → `spawn_throughput` K1 → `rally` K0) over a deterministic colony model + tick loop, so
//! the live "roster stuck at 3/5" failure and the combat-vs-economy spawn-priority lever are reproduced
//! and TUNED offline instead of guessed on Docker. Pure (no `game::*`, no engine) — the engage handoff
//! (place the formed roster → `ManagedSimSquad` → `resolve_tick`) is the next harness phase.
//!
//! Model: each home has one spawn, banks `income`/tick (capped at capacity), and the economy queues a
//! constant HIGH hauler (+ optional CRITICAL miner) competing for the lane each tick. A combat slot's body
//! is built once at `min(best_capacity, per_member_cap)` (K3) and broadcast to every home (the shared
//! token), de-duped across homes within a tick. A spawn occupies its home for `part_count * 3` ticks; the
//! slot fills when it completes. The roster departs when `rally::squad_ready_to_depart` holds.

use screeps::{Position, RoomCoordinate};
use screeps_combat_decision::bodies::MoveProfile;
use screeps_combat_decision::composition::SquadComposition;
use screeps_combat_decision::spawn_throughput::{spawn_step, HomeLanes, QueuedSpawn};
use screeps_combat_decision::{fielding, rally};
use std::collections::BTreeSet;

/// A spawn home in the deterministic colony model.
#[derive(Clone, Copy, Debug)]
pub struct Home {
    pub energy_capacity: u32,
    pub income: u32,
    pub start_energy: u32,
}

/// The economy's per-tick spawn demand at EACH home — the lane contention combat competes against.
#[derive(Clone, Copy, Debug, Default)]
pub struct EconomyPressure {
    /// (priority, body_cost) for a HIGH hauler queued EVERY tick (logistics never sleeps).
    pub hauler: Option<(f32, u32)>,
    /// (priority, body_cost) for a CRITICAL miner queued every `miner_period` ticks.
    pub miner: Option<(f32, u32)>,
    pub miner_period: u32,
}

/// A colony forming scenario: who is being fielded, against what economy, at what priority.
pub struct ColonyFormingScenario {
    pub composition: SquadComposition,
    pub homes: Vec<Home>,
    pub economy: EconomyPressure,
    pub combat_priority: f32,
    pub per_member_cap: u32,
    pub budget_ticks: u32,
}

/// The result of running a forming scenario to completion or the tick budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormingOutcome {
    /// The full roster spawned + is present (`rally::squad_ready_to_depart`) at this tick.
    Completed { ticks: u32 },
    /// The budget elapsed with `filled` of `of` slots — the live "stuck at N/M" stall.
    Stalled { filled: usize, of: usize },
}

const ECON_MINER_ID_BASE: u64 = 1_000_000_000;
const ECON_HAULER_ID_BASE: u64 = 2_000_000_000;

fn dummy_home_pos() -> Position {
    Position::new(RoomCoordinate::new(25).unwrap(), RoomCoordinate::new(25).unwrap(), "W1N1".parse().unwrap())
}

/// Simulate the colony forming the squad. Deterministic: same scenario → same outcome.
pub fn run_forming(s: &ColonyFormingScenario) -> FormingOutcome {
    let n_slots = s.composition.slots.len();
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);
    let mut filled = vec![false; n_slots];
    let mut avail: Vec<u32> = s.homes.iter().map(|h| h.start_energy).collect();
    let mut busy_until: Vec<u32> = vec![0; s.homes.len()];
    // Combat slots currently spawning: (slot_id, completes_at_tick).
    let mut completing: Vec<(u64, u32)> = Vec::new();

    for tick in 0..s.budget_ticks {
        // 1. Complete spawns due this tick → mark their slot filled.
        completing.retain(|&(id, at)| {
            if at <= tick {
                if (id as usize) < n_slots {
                    filled[id as usize] = true;
                }
                false
            } else {
                true
            }
        });

        // 2. Ready to depart? (the K0 rally gate over the present roster.)
        let present = filled.iter().filter(|f| **f).count();
        let member_positions: Vec<Option<Position>> = vec![Some(dummy_home_pos()); present];
        if rally::squad_ready_to_depart(&member_positions, n_slots) {
            return FormingOutcome::Completed { ticks: tick };
        }

        // 3. Field the unfilled combat slots once (K3) — same body broadcast to every home.
        let combat = fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains);

        // Cross-home de-dup within this tick: a slot already in flight (or spawned this tick) is excluded.
        let mut in_flight: BTreeSet<u64> = completing.iter().map(|&(id, _)| id).collect();

        // 4. Each home banks income, then runs one spawn step (K1) over economy + the unfilled combat slots.
        for h in 0..s.homes.len() {
            avail[h] = (avail[h] + s.homes[h].income).min(s.homes[h].energy_capacity);
            if tick < busy_until[h] {
                continue; // this home's spawn is still busy
            }
            let mut queue: Vec<QueuedSpawn> = Vec::new();
            if let Some((p, c)) = s.economy.miner {
                if s.economy.miner_period > 0 && tick % s.economy.miner_period == 0 {
                    queue.push(QueuedSpawn { priority: p, body_cost: c, part_count: (c / 100).max(1), id: ECON_MINER_ID_BASE + (tick as u64) * 100 + h as u64 });
                }
            }
            if let Some((p, c)) = s.economy.hauler {
                queue.push(QueuedSpawn { priority: p, body_cost: c, part_count: (c / 100).max(1), id: ECON_HAULER_ID_BASE + (tick as u64) * 100 + h as u64 });
            }
            for cs in &combat {
                if !in_flight.contains(&cs.id) {
                    queue.push(*cs);
                }
            }

            let mut lane = HomeLanes { idle_spawns: 1, available_energy: avail[h], energy_capacity: s.homes[h].energy_capacity };
            for spawned in spawn_step(&mut lane, &queue) {
                avail[h] = lane.available_energy;
                busy_until[h] = tick + spawned.completes_in;
                if spawned.id < ECON_MINER_ID_BASE {
                    // A combat slot started → it fills when the spawn completes.
                    completing.push((spawned.id, tick + spawned.completes_in));
                    in_flight.insert(spawned.id);
                }
            }
        }
    }

    FormingOutcome::Stalled { filled: filled.iter().filter(|f| **f).count(), of: n_slots }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(combat_priority: f32) -> ColonyFormingScenario {
        ColonyFormingScenario {
            composition: SquadComposition::quad_ranged(),
            homes: vec![
                Home { energy_capacity: 5300, income: 300, start_energy: 2000 },
                Home { energy_capacity: 5300, income: 300, start_energy: 2000 },
            ],
            // Constant HIGH hauler pressure (75), no miner — isolates the combat-vs-economy lane contest.
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority,
            per_member_cap: 3000,
            budget_ticks: 1500,
        }
    }

    #[test]
    fn medium_priority_combat_stalls_below_economy() {
        // Combat below the hauler (50 < 75) → the hauler takes every lane → the roster never completes.
        match run_forming(&scenario(50.0)) {
            FormingOutcome::Stalled { filled, of } => assert!(filled < of, "MEDIUM combat stalls below economy ({filled}/{of})"),
            FormingOutcome::Completed { ticks } => panic!("MEDIUM combat should NOT complete (did at tick {ticks})"),
        }
    }

    #[test]
    fn above_economy_combat_completes_the_roster() {
        // Combat above the hauler (87.5 > 75) → wins lanes → the roster completes within budget.
        match run_forming(&scenario(87.5)) {
            FormingOutcome::Completed { .. } => {}
            FormingOutcome::Stalled { filled, of } => panic!("above-economy combat should complete ({filled}/{of})"),
        }
    }

    #[test]
    fn forming_is_deterministic() {
        assert_eq!(run_forming(&scenario(87.5)), run_forming(&scenario(87.5)));
    }
}
