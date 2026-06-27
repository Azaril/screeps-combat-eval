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
    /// Ticks a spawned member lives before dying of old age (CREEP_LIFE_TIME ≈ 1500). A member that ages
    /// out while the squad is still rallying drops back to unfilled → re-spawn → churn. This is the live
    /// failure when forming is STUCK for longer than a member's life (the bot has no renew today —
    /// `request_renew` has zero callers).
    pub member_ttl: u32,
    /// Whether the colony RENEWS aging present members while rallying (keeps the early roster alive until
    /// the full squad forms) — the missing live behavior. Renewing costs a home's spawn lane for the tick.
    pub renew: bool,
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

/// TTL a single renew action adds to a member (≈ `600 / body_size` per tick in the engine; ~20 for a
/// 30-part member). A renew occupies the home's spawn lane for that tick (no new member spawns).
const RENEW_PER_TICK: u32 = 20;
/// Renew a present member once its remaining TTL drops below this (don't waste lanes renewing fresh ones).
const RENEW_THRESHOLD: u32 = 100;

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
    // Per-filled-slot death tick (set on completion = tick + member_ttl). Members age out unless renewed.
    let mut dies_at: Vec<u32> = vec![0; n_slots];

    for tick in 0..s.budget_ticks {
        // 1. Complete spawns due this tick → mark their slot filled + stamp its death tick.
        completing.retain(|&(id, at)| {
            if at <= tick {
                if (id as usize) < n_slots {
                    filled[id as usize] = true;
                    dies_at[id as usize] = tick + s.member_ttl;
                }
                false
            } else {
                true
            }
        });

        // 1b. Age out members that died of old age while rallying (no renew kept them alive) — they drop
        // back to unfilled and must re-spawn (the live churn when forming outlasts a member's life).
        for i in 0..n_slots {
            if filled[i] && dies_at[i] <= tick {
                filled[i] = false;
            }
        }

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
        // Each aging present member is renewed by at most one home per tick.
        let mut renewed_this_tick: BTreeSet<usize> = BTreeSet::new();

        // 4. Each home banks income, then runs one spawn step (K1) over economy + the unfilled combat slots.
        for h in 0..s.homes.len() {
            avail[h] = (avail[h] + s.homes[h].income).min(s.homes[h].energy_capacity);
            if tick < busy_until[h] {
                continue; // this home's spawn is still busy
            }
            // RENEW: keep the early roster alive while rallying. An aging present member is renewed instead
            // of spawning a new one (the renew occupies this home's lane this tick). Only ONE home renews a
            // given member per tick; a home with no aging member to renew falls through to spawning.
            if s.renew {
                if let Some(slot) = (0..n_slots)
                    .filter(|&i| filled[i] && !renewed_this_tick.contains(&i) && dies_at[i].saturating_sub(tick) < RENEW_THRESHOLD)
                    .min_by_key(|&i| dies_at[i])
                {
                    dies_at[slot] = (dies_at[slot] + RENEW_PER_TICK).min(tick + s.member_ttl);
                    renewed_this_tick.insert(slot);
                    busy_until[h] = tick + 1; // the renew action occupies this home this tick
                    continue;
                }
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

/// The end-to-end lifecycle outcome: did the colony FORM the roster, and did that roster KILL the core
/// (vs stall / wipe / never-form)? Surfaces the form-vs-fight gap the live whack-a-mole could not isolate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifecycleOutcome {
    /// Formed AND the roster destroyed the core.
    Killed { form_ticks: u32, engage_ticks: u32 },
    /// Formed + engaged, but the core survived to the engage budget (under-DPS / disengage).
    Stalled { form_ticks: u32, engage_ticks: u32 },
    /// Formed, but the roster was wiped engaging (under-sized / retreat-into-fire).
    RosterWiped { form_ticks: u32, engage_ticks: u32 },
    /// Forming never completed — nothing departs (the "stuck at N/M" stall).
    NeverFormed { filled: usize, of: usize },
    /// Formed but couldn't be placed at the entry (body wouldn't build / no free tiles).
    CouldNotField { form_ticks: u32 },
}

/// Chain the forming phase into the engine engage: form the roster under economy contention
/// (`run_forming`), then drive that SAME roster against an UNDEFENDED L0 invader core (a 50k-hit spawn,
/// no towers/ramparts/defenders) through the authoritative engine and report whether it actually kills.
/// Reuses the existing engage machinery (`assemble_single_room` + `run_managed_assault_with`), so the
/// engaged roster is the same composition the forming consumed. Deterministic: same scenario → same
/// outcome. The undefended fixture isolates form→travel→raze from defender fire + the retreat gate (the
/// FIRST end-to-end fixture; graded defenders are the same `assemble_single_room` with towers/force/ramparts).
pub fn run_lifecycle(s: &ColonyFormingScenario) -> LifecycleOutcome {
    use crate::harness::evaluate::StopReason;
    use crate::harness::generate::{assemble_single_room, ForceSpec, Layout};
    use crate::harness::validate::run_managed_assault_with;
    use screeps_combat_decision::kite::SquadTacticParams;

    // 1. Forming. If the roster never completes, nothing departs — there is nothing to engage.
    let form_ticks = match run_forming(s) {
        FormingOutcome::Completed { ticks } => ticks,
        FormingOutcome::Stalled { filled, of } => return LifecycleOutcome::NeverFormed { filled, of },
    };

    // 2. The engaged roster == the formed roster: both build from (composition, build_energy).
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);
    let build_energy = best_capacity.min(s.per_member_cap);

    // 3. An undefended L0 core: a 50k-hit spawn at (25,25), no rampart/towers/defenders.
    let engage = assemble_single_room(
        "lifecycle L0 core".into(),
        1, // fixed seed (open/undefended → nothing random to vary)
        build_energy,
        1500, // engage tick budget
        (25, 25),
        0,   // no rampart
        &[], // no towers
        Layout::Open,
        ForceSpec::None, // no defenders
        false,           // no safe mode
    );

    // 4. Engage via the existing managed-assault driver (clone world → place roster at entry →
    //    ManagedSimSquad → resolve_tick to ObjectivesDestroyed | SideWiped(attacker) | Timeout).
    match run_managed_assault_with(&engage, &engage.objectives[0], &s.composition, SquadTacticParams::default()) {
        None => LifecycleOutcome::CouldNotField { form_ticks },
        Some((out, _rec)) => match out.stop {
            StopReason::ObjectivesComplete => LifecycleOutcome::Killed { form_ticks, engage_ticks: out.ticks },
            StopReason::SideWiped(_) => LifecycleOutcome::RosterWiped { form_ticks, engage_ticks: out.ticks },
            _ => LifecycleOutcome::Stalled { form_ticks, engage_ticks: out.ticks },
        },
    }
}

/// Like [`run_lifecycle`], but the roster forms under economy contention AND THEN engages a DEFENDED core —
/// a rampart breach-gate, one energized tower, and a melee guard force — with the composition the ORACLE
/// sizes for that defense. This closes the seam between `SizingWins` (the eval's oracle-sized force, but
/// PRE-PLACED on the staging tiles → ~99% win) and [`run_lifecycle`] (a FORMED roster, but against an
/// UNDEFENDED core): here the SAME oracle-sized force is FORMED under contention AND must TRAVEL in under
/// fire. A `Killed` proves form + travel do NOT degrade a correctly-sized force; a miss isolates the
/// form/travel cost from live UNDER-sizing (which `SizingWins`, being pre-placed + correctly sized, can't
/// see). ADR 0028 + ADR 0029 §10 #1.
///
/// The comp is sized via the EXACT path `SizingWins` uses — `derive_profile` → `siege_ceiling(member_energy)
/// .force_budget(..)` → `siege_doctrine_plan` (validate.rs) — against the defended world, then PUT INTO the
/// forming scenario (replacing its template), so the FORMED roster IS the oracle's force. The defended
/// fixture is deterministic: a fixed seed, `safe_mode = false`, and a fixed `ForceSpec::Guard`. `s`'s
/// economy / homes / priority / ttl / renew drive the forming contention; its `composition` is overridden.
pub fn run_defended_lifecycle(s: &ColonyFormingScenario) -> LifecycleOutcome {
    // Canonical fixture (the acceptance bed): a rampart breach-gate, one energized tower, a melee guard force.
    run_defended_lifecycle_with(s, 30_000, &[((24, 16), 100_000)], crate::harness::generate::Layout::Open, crate::harness::generate::ForceSpec::Guard(2))
}

/// Parameterized defended lifecycle (ADR 0031 P3 — the graded regime sweep): emit_requirement → assemble_force
/// → FORM under economy contention → MOVE in → engage, against a defended core whose rampart / towers / layout
/// / defender force are the regime knobs. Same determinism contract as the canonical bed (fixed seed, no safe
/// mode). Proves the assembler kills-when-winnable / defers-cleanly across defense shapes.
pub fn run_defended_lifecycle_with(
    s: &ColonyFormingScenario,
    rampart_hits: u32,
    towers: &[((u8, u8), u32)],
    layout: crate::harness::generate::Layout,
    force: crate::harness::generate::ForceSpec,
) -> LifecycleOutcome {
    use crate::harness::evaluate::StopReason;
    use crate::harness::generate::assemble_single_room;
    use crate::harness::validate::{derive_profile, run_managed_assault_with, siege_ceiling};
    use screeps_combat_decision::composition::assemble_force;
    use screeps_combat_decision::doctrine::{emit_requirement, DoctrineObjective, EnemyCoordination};
    use screeps_combat_decision::force_sizing::AssaultMode;
    use screeps_combat_decision::kite::SquadTacticParams;

    // The roster's members are built at this energy (K3's per-member cap); the oracle must size at the SAME
    // energy so the FORMED bodies and the sized force agree.
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);
    let build_energy = best_capacity.min(s.per_member_cap);
    const ENGAGE_BUDGET: u32 = 1500; // engage tick budget

    // 1. Build the DEFENDED core ONCE: a rampart breach-gate, one energized tower, a melee guard force.
    //    Deterministic — fixed seed, no safe mode, a fixed guard count.
    let engage = assemble_single_room(
        "defended lifecycle core".into(),
        1, // fixed seed (deterministic fixture)
        build_energy,
        ENGAGE_BUDGET,
        (25, 25),
        rampart_hits, // rampart breach-gate (regime knob)
        towers,       // energized towers (regime knob)
        layout,       // approach layout (regime knob)
        force,        // defender force (regime knob)
        false,        // no safe mode (deterministic)
    );
    let obj = &engage.objectives[0];

    // 2. Size the breach force via the UNIFIED EMITTER + the ASSEMBLER (ADR 0031 P3) against THIS defended
    //    world — emit_requirement folds assess + the anti-creep overlay (the observed guards), then
    //    assemble_force fields the capability vector directly (no template, no sized_for). This is the path
    //    the bot will run at P4; the lifecycle proves it end-to-end now (emit → assemble → form → move → kill).
    let profile = derive_profile(&engage.world, engage.defender_owner, obj);
    let budget = siege_ceiling(engage.member_energy).force_budget(engage.member_energy, engage.onsite_budget);
    let defenders = crate::harness::validate::defender_force(&engage);
    // Coordination from the OBSERVED guards (grouped / self-healing → over-match), matching the doctrine path.
    let coordination = match defenders {
        Some(ef) if ef.count > 1 || ef.heal > 0.0 => EnemyCoordination::Coordinated,
        _ => EnemyCoordination::Individual,
    };
    let (assessment, required) = emit_requirement(
        DoctrineObjective::DismantleStructure,
        &profile,
        defenders,
        Some(&budget),
        coordination,
        0.0,
        screeps_combat_decision::force_sizing::HOLD_MARGIN,
        screeps_combat_decision::force_sizing::COORDINATED_DPS_MARGIN,
    );
    let comp = match (assessment.winnable && assessment.mode == AssaultMode::Breach, assemble_force(&required, engage.member_energy)) {
        (true, Some(assembled)) => assembled,
        // The oracle deferred / drained / the assembler couldn't field the required force at this energy —
        // field the ceiling so the chain still runs (the test then surfaces whether even the ceiling kills).
        _ => siege_ceiling(engage.member_energy),
    };

    // 3. FORM the oracle-sized roster under economy contention (the sized comp replaces `s`'s template).
    let forming_scenario = ColonyFormingScenario {
        composition: comp.clone(),
        homes: s.homes.clone(),
        economy: s.economy,
        combat_priority: s.combat_priority,
        per_member_cap: s.per_member_cap,
        budget_ticks: s.budget_ticks,
        member_ttl: s.member_ttl,
        renew: s.renew,
    };
    let form_ticks = match run_forming(&forming_scenario) {
        FormingOutcome::Completed { ticks } => ticks,
        FormingOutcome::Stalled { filled, of } => return LifecycleOutcome::NeverFormed { filled, of },
    };

    // 4. Engage the FORMED + MOVING roster against the defended core (breach tactics: dismantle through the
    //    gate while out-healing the tower + guards). The engaged comp == the formed comp.
    match run_managed_assault_with(&engage, obj, &comp, SquadTacticParams::breach()) {
        None => LifecycleOutcome::CouldNotField { form_ticks },
        Some((out, _rec)) => match out.stop {
            StopReason::ObjectivesComplete => LifecycleOutcome::Killed { form_ticks, engage_ticks: out.ticks },
            StopReason::SideWiped(_) => LifecycleOutcome::RosterWiped { form_ticks, engage_ticks: out.ticks },
            _ => LifecycleOutcome::Stalled { form_ticks, engage_ticks: out.ticks },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps_combat_decision::composition::assemble_force;
    use screeps_combat_decision::force_sizing::RequiredForce;

    /// A multi-slot placeholder composition (assembled template-free, ADR 0031 D16) the forming tests
    /// override with the oracle-sized one; several RANGED + several HEAL members at the home cap.
    fn placeholder_comp() -> SquadComposition {
        assemble_force(&RequiredForce { heal_parts: 40, immune_struct_parts: 30, ..Default::default() }, 12_900)
            .expect("assembles a multi-slot placeholder")
    }

    /// Build a forming scenario: `homes` spawn homes at `income`/tick, combat at `combat_priority` vs a
    /// constant HIGH hauler (75), members live `member_ttl` ticks, `renew` keeps the rallying roster alive.
    fn forming(homes: usize, income: u32, combat_priority: f32, member_ttl: u32, renew: bool, budget: u32) -> ColonyFormingScenario {
        ColonyFormingScenario {
            composition: placeholder_comp(),
            homes: (0..homes).map(|_| Home { energy_capacity: 5300, income, start_energy: 2000 }).collect(),
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority,
            per_member_cap: 3000,
            budget_ticks: budget,
            member_ttl,
            renew,
        }
    }

    /// The baseline 2-home, fresh-member (ttl 1500), no-renew scenario the original forming/lifecycle tests use.
    fn scenario(combat_priority: f32) -> ColonyFormingScenario {
        forming(2, 300, combat_priority, 1500, false, 1500)
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

    // ── Single- vs multi-room spawning + rally/renew (operator-requested) ──

    #[test]
    fn single_room_forms_the_roster() {
        // One home, fresh members (ttl 1500) → forms serially within budget.
        match run_forming(&forming(1, 400, 87.5, 1500, false, 3000)) {
            FormingOutcome::Completed { .. } => {}
            o => panic!("single-room above-economy should form, got {o:?}"),
        }
    }

    #[test]
    fn multi_room_forms_faster_than_single_room() {
        // Parallel spawning across homes forms the same roster in fewer ticks than one serial home.
        let single = run_forming(&forming(1, 400, 87.5, 1500, false, 3000));
        let multi = run_forming(&forming(4, 400, 87.5, 1500, false, 3000));
        match (single, multi) {
            (FormingOutcome::Completed { ticks: s }, FormingOutcome::Completed { ticks: m }) => {
                assert!(m < s, "multi-room parallel spawning forms faster than single-room serial ({m} < {s})");
            }
            other => panic!("both single + multi room should complete, got {other:?}"),
        }
    }

    #[test]
    fn stuck_forming_loses_early_members_without_renew() {
        // A slow single home where forming OUTLASTS a member's life (ttl scaled to 200 for a fast
        // deterministic test; the live equivalent is a form stalled >1500t by spawn contention). Early
        // members age out → drop back to unfilled → the roster never has the full set present at once.
        match run_forming(&forming(1, 200, 87.5, 200, false, 4000)) {
            FormingOutcome::Stalled { filled, of } => assert!(filled < of, "early members die → stuck ({filled}/{of})"),
            FormingOutcome::Completed { ticks } => panic!("a too-slow form must NOT complete without renew (did at {ticks})"),
        }
    }

    #[test]
    fn renew_completes_the_stuck_form() {
        // The SAME stuck scenario, but the colony RENEWS the rallying roster (the missing live behavior) →
        // early members stay alive until the full squad forms → it completes.
        match run_forming(&forming(1, 200, 87.5, 200, true, 4000)) {
            FormingOutcome::Completed { .. } => {}
            FormingOutcome::Stalled { filled, of } => panic!("renew should keep the roster alive + complete ({filled}/{of})"),
        }
    }

    #[test]
    fn forming_is_deterministic() {
        assert_eq!(run_forming(&scenario(87.5)), run_forming(&scenario(87.5)));
    }

    // ── End-to-end: form → engine engage → kill (ADR 0028 engage handoff) ──

    #[test]
    fn above_economy_roster_forms_and_kills_an_undefended_core() {
        // The full chain: form above economy (completes) → travel → raze the 50k-hit core.
        match run_lifecycle(&scenario(87.5)) {
            LifecycleOutcome::Killed { .. } => {}
            other => panic!("expected the formed roster to kill the undefended core, got {other:?}"),
        }
    }

    #[test]
    fn medium_priority_never_forms_so_never_engages() {
        // The form gate prevents a doomed engage: MEDIUM stalls forming → NeverFormed (no engage attempt).
        match run_lifecycle(&scenario(50.0)) {
            LifecycleOutcome::NeverFormed { .. } => {}
            other => panic!("MEDIUM should never form, got {other:?}"),
        }
    }

    #[test]
    fn lifecycle_is_deterministic() {
        assert_eq!(run_lifecycle(&scenario(87.5)), run_lifecycle(&scenario(87.5)));
    }

    // ── Defended end-to-end: oracle-sized force, FORMED + MOVING, kills a defended core (ADR 0029 §10 #1) ──

    /// A high-energy forming scenario (4 RCL8 homes, per-member cap == capacity) so the build energy is the
    /// home's 12_900 and the oracle can size its FULL breach force. `run_defended_lifecycle` overrides the
    /// placeholder composition with the oracle-sized one; this only supplies the homes + economy contention.
    fn defended_forming() -> ColonyFormingScenario {
        ColonyFormingScenario {
            composition: placeholder_comp(), // placeholder — replaced by the oracle-sized comp
            homes: (0..4).map(|_| Home { energy_capacity: 12_900, income: 1000, start_energy: 12_900 }).collect(),
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 87.5, // above the hauler (75) → combat wins the lane
            per_member_cap: 12_900,
            budget_ticks: 4000,
            member_ttl: 1500,
            renew: false,
        }
    }

    // The acceptance gate (ADR 0029 §10 #1 / 0031): an oracle-sized force, FORMED + MOVING, must KILL a
    // Guard-defended core. Was KNOWN-FAILING — the oracle's siege comp (dismantler + healer) had no
    // anti-creep weapon, so the MOVING brain fixated on the unkillable melee guard and disengaged at 0
    // damage. NOW PASSES (un-ignored 2026-06-27): ADR 0031 P0a (dismantle counts toward fighting strength →
    // no retreat at t0) + P1b (SiegeBreach anti-creep fusion → `siege_assault_quad` with a RangedDPS slot →
    // the squad clears the guard, then breaches). Do NOT soften the assertion; it must keep passing as the
    // assembler (P3) replaces the fusion.
    #[test]
    fn oracle_sized_force_forms_and_kills_a_defended_core() {
        // The seam-closer (ADR 0029 §10 #1): the oracle sizes the breach force for a DEFENDED core (rampart
        // + tower + a melee guard force), that SAME force is FORMED under economy contention, then TRAVELS in
        // and engages. A Killed proves form + travel do NOT degrade a correctly-sized force — discriminating
        // "form/travel degrades a sized force" from "live UNDER-sizing was the whole story" (the gap between
        // `SizingWins` — oracle-sized but PRE-PLACED, ~99% — and `run_lifecycle` — formed but UNDEFENDED).
        match run_defended_lifecycle(&defended_forming()) {
            LifecycleOutcome::Killed { .. } => {}
            other => panic!("an oracle-sized force, FORMED + MOVING, should kill the defended core, got {other:?}"),
        }
    }

    #[test]
    fn defended_lifecycle_is_deterministic() {
        // Fixed seed + safe_mode=false + a fixed ForceSpec → the defended chain is reproducible (it stalls
        // identically each run today; this still holds once the redesign flips the outcome to Killed).
        assert_eq!(run_defended_lifecycle(&defended_forming()), run_defended_lifecycle(&defended_forming()));
    }

    /// ADR 0031 P3 — the GRADED REGIME SWEEP: an emit→assemble force, FORMED + MOVING, must KILL a defended
    /// core across rampart thickness / tower presence / approach layout / guard strength. Melee guards do
    /// not evade, so a correctly-assembled force reliably clears them then breaches — the discriminating
    /// proof that the assembler fields a WINNING force across defense shapes, not just the canonical bed.
    /// Reuses the generous forming bed (`defended_forming`), so the ENGAGE outcome (the assembler's kill
    /// quality) is what is under test, not spawn contention. Determinism is checked alongside.
    #[test]
    fn assembler_kills_across_defended_regimes() {
        use crate::harness::generate::{ForceSpec, Layout};
        let regimes: &[(&str, u32, &[((u8, u8), u32)], Layout, ForceSpec)] = &[
            ("rampart-only + light guard", 50_000, &[], Layout::Open, ForceSpec::Guard(1)),
            ("tower-only + guard", 0, &[((24, 16), 100_000)], Layout::Open, ForceSpec::Guard(2)),
            ("tower + rampart + guard", 30_000, &[((24, 16), 100_000)], Layout::Open, ForceSpec::Guard(2)),
            ("corridor choke + guard", 20_000, &[((24, 16), 100_000)], Layout::Corridor, ForceSpec::Guard(2)),
        ];
        for (name, rampart, towers, layout, force) in regimes {
            let out = run_defended_lifecycle_with(&defended_forming(), *rampart, towers, *layout, *force);
            let out2 = run_defended_lifecycle_with(&defended_forming(), *rampart, towers, *layout, *force);
            assert_eq!(out, out2, "{name}: the regime is deterministic");
            assert!(matches!(out, LifecycleOutcome::Killed { .. }), "{name}: the assembled force should KILL the defended core, got {out:?}");
        }
    }
}
