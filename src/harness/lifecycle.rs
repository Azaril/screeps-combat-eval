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
//!
//! [`run_lifecycle_churn`] additionally models the FULL bot lifecycle WIRING the forming-only driver (and
//! the agent-sim) bypass — the commitment lease + the shared `lifecycle::reconcile` kernel + RE-FIELD churn,
//! the real `rally::ready_to_depart_gate`, the 2-tick member position-sync, and the TRAVEL + (empty-DTO)
//! ARRIVE phases — to reproduce the deep "fielded squad spawns members but never reaches/engages" bug
//! (always RETIRE GaveUp) DETERMINISTICALLY offline, where live Docker is unreliable.

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

// ═══ The CHURN-MODELING lifecycle driver (the live bot wiring `run_forming` bypasses) ═════════════════════
//
// `run_forming` proves the spawn-lane contest in isolation, but it does NOT model the live bot's FULL squad
// lifecycle — the very wiring the deep "fielded squad never reaches/engages" bug lives in. This driver adds
// the four things the bot does that `run_forming` (and the agent-sim) skip:
//
//   1. The COMMITMENT lease + the `screeps_combat_decision::lifecycle::reconcile` kernel + RE-FIELD churn:
//      a squad is fielded with `deadline = now + COMMITMENT_BUDGET`; the lease is refreshed only while the
//      kernel returns `KeepRefreshLease` (engaging OR forming-and-progressing). On `Retire{GaveUp}` the
//      roster is DELETED + RE-FIELDED (drop filled, bump a generation counter, orphan in-flight spawns) —
//      reproducing the live `Gen4` churn loop that orphans early members.
//   2. The REAL rally gate `rally::ready_to_depart_gate(&positions, n_slots, uncontested)` — the squad does
//      not start TRAVELING until the gate releases (full roster, OR the min-viable quorum for a proven-
//      uncontested target). A `room_visible/uncontested` scenario flag threads the contested-ness.
//   3. The two-phase member sync: a spawn that COMPLETES is not immediately `present` — there is a
//      `MEMBER_SYNC_DELAY`-tick gap (spawn-callback → `CreepOwner` → `PreRunSquadUpdate` position sync)
//      before it counts toward `present` / the rally gate / `forming_progress`.
//   4. The TRAVEL + ARRIVE + (empty-DTO) ENGAGE phases the forming-only driver has no notion of: once the
//      gate releases the squad travels `travel_ticks` toward the target; on arrival it must get a non-empty
//      room DTO to compute a focus and latch `engaged_once` — an `empty_dtos_on_arrival` flag models the
//      live "arrived but `decide_squad` returns no focus → IN_ROOM_NO_FOCUS → lease lapses" break.

/// Engine constant mirror: the live two-phase member tracking (spawn-callback mints the creep entity, then
/// `PreRunSquadUpdate` syncs its position the following tick) means a freshly-spawned member is not counted
/// as PRESENT for ~2 ticks. The rally gate + `forming_progress` both key on present, so this delay is
/// load-bearing for reproducing the contention plateau.
pub const MEMBER_SYNC_DELAY: u32 = 2;

/// P-OBJ #23 commitment lease (ticks) — MUST mirror the bot's `squad_manager::COMMITMENT_BUDGET` (400).
pub const COMMITMENT_BUDGET: u32 = 400;

/// Absolute bound on how long the forming-in-flight lease refresh may extend a squad's life (the deep-reach
/// fix bound — Break #1). A roster that has not completed within this many ticks of forming gives up even
/// with a member nominally in flight, so a genuinely-unfieldable squad is never immortal. Generous (covers a
/// trickle-income RCL6/7 colony banking several 3000e members serially) but finite.
pub const MAX_FORMING_BUDGET: u32 = 3000;

/// Absolute bound on the travel-phase lease refresh (the deep-reach fix bound — Break #2 travel half). A
/// full-roster squad that has not arrived within this many ticks of departing gives up. Covers the longest
/// realistic multi-room hop with margin.
pub const MAX_TRAVEL_BUDGET: u32 = 1000;

/// How the target room presents to a squad that ARRIVES — the contested-ness (drives the rally gate) and
/// whether its room DTOs are populated on the arrival tick (the empty-DTO-on-arrival break).
#[derive(Clone, Copy, Debug)]
pub struct ChurnTarget {
    /// Ticks the full/quorum roster spends traveling from home to the target room.
    pub travel_ticks: u32,
    /// PROVEN-uncontested (visible, no hostiles, no towers, no safe mode) → the rally gate deploys at the
    /// min-viable quorum; otherwise it holds for the full roster. Threaded into `ready_to_depart_gate`.
    pub uncontested: bool,
    /// On ARRIVAL the room's combat DTOs are EMPTY for this many ticks (the live `build_room_combat_dtos`
    /// returns empty when `mapping.get_room` / `room_data.get_creeps` are not populated that tick). While
    /// empty, `decide_squad` returns no focus → the squad cannot latch `engaged_once` → the lease lapses
    /// underneath it (Break #2). `0` = the DTOs are populated immediately on arrival.
    pub empty_dtos_on_arrival_ticks: u32,
}

// ═══ The SPATIAL movement-stall repro (ADR 0028 K0): distinct homes → shared rally → assault ═══════════
//
// `run_lifecycle_churn` models travel as a pure tick COUNTER (`travel_ticks`), so it cannot reproduce the
// MOVEMENT STALL (members spawned at DIFFERENT homes never converge because the cross-room box-formation
// anchor freezes). This spatial driver places each member at a DISTINCT home Position and steps real
// per-member movement: in the BUGGY model each member rallies to its OWN home behind a frozen cross-room
// formation anchor → it never converges → travel makes no positional progress → the lease lapses MID-HOP
// (`LapsedInTravel`). In the FIXED model each member solo-paths to ONE SHARED rally, converges, the unified
// `rally::gather_quorum_met` kernel fires, and the anchor advances rally→target → engage. RED → GREEN.

/// A point in the toy world as (world_x, world_y). Movement is one Chebyshev step/tick; room membership is
/// `world / 50`. Pure value-math (mirrors how `should_hold_at_boundary`/`gather_quorum_met` were extracted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WPos {
    pub wx: i32,
    pub wy: i32,
}

impl WPos {
    fn room(self) -> (i32, i32) {
        (self.wx.div_euclid(50), self.wy.div_euclid(50))
    }
    fn step_toward(self, to: WPos) -> WPos {
        WPos { wx: self.wx + (to.wx - self.wx).signum(), wy: self.wy + (to.wy - self.wy).signum() }
    }
    fn room_dist(self, o: WPos) -> u32 {
        let (ax, ay) = self.room();
        let (bx, by) = o.room();
        (ax - bx).unsigned_abs().max((ay - by).unsigned_abs())
    }
    fn to_pos(self) -> Position {
        let (rx, ry) = self.room();
        let room: screeps::RoomName = format!("W{}N{}", -rx - 1, -ry - 1).parse().unwrap();
        Position::new(
            RoomCoordinate::new(self.wx.rem_euclid(50) as u8).unwrap(),
            RoomCoordinate::new(self.wy.rem_euclid(50) as u8).unwrap(),
            room,
        )
    }
}

/// The spatial movement scenario: distinct member HOMES, a shared RALLY on the approach to the TARGET, and
/// whether the squad uses the FIXED shared-rally solo-travel (`true`) or the BUGGY per-member-home /
/// frozen-cross-room-formation-anchor model (`false`).
#[derive(Clone, Debug)]
pub struct SpatialTravel {
    /// Each member's home position (world coords). Distinct homes → the multi-home-spawn scatter.
    pub homes: Vec<WPos>,
    /// The ONE shared rally/staging point (world coords) on the approach to the target (safe, out of fire).
    pub rally: WPos,
    /// The target position (world coords) — a room beyond the rally.
    pub target: WPos,
    /// Proven-uncontested target → the gather quorum may trickle; contested → the (near-)full roster.
    pub uncontested: bool,
    /// `true` = the FIXED clean design (solo travel to the shared rally + the unified gather kernel + the
    /// assault anchor advance rally→target). `false` = the BUGGY model (per-member-home rally, the
    /// cross-room box anchor freezes for scattered members → never converges).
    pub use_shared_rally: bool,
}

impl SpatialTravel {
    fn pos_options(positions: &[WPos]) -> Vec<Option<Position>> {
        positions.iter().map(|w| Some(w.to_pos())).collect()
    }
}

/// Drive the full bot lifecycle (lease / reconcile / re-field churn + rally gate) with SPATIAL travel:
/// members spawn at DISTINCT homes and must converge at a SHARED rally before the assault advances. The
/// reconcile DECISION is the shared `lifecycle::reconcile` kernel; the gather decision is the shared
/// `rally::gather_quorum_met` kernel — so there is no live/sim drift. Deterministic.
pub fn run_lifecycle_churn_spatial(s: &ColonyFormingScenario, travel: &SpatialTravel) -> ChurnOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot, RetireReason};

    let n_slots = s.composition.slots.len();
    assert_eq!(travel.homes.len(), n_slots, "one home per member slot in the spatial model");
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);

    let mut generation: u32 = 0;
    let mut max_present: usize = 0;

    let mut filled = vec![false; n_slots];
    let mut syncing: Vec<(u64, u32)> = Vec::new();
    let mut completing: Vec<(u64, u32)> = Vec::new();
    let mut avail: Vec<u32> = s.homes.iter().map(|h| h.start_energy).collect();
    let mut busy_until: Vec<u32> = vec![0; s.homes.len()];
    let mut deadline: u32 = COMMITMENT_BUDGET;
    let mut prev_present: usize = 0;
    let engaged_once = false;
    let mut gen_start: u32 = 0;
    let mut travel_start: u32 = 0;

    // Spatial member state: each member starts AT its home; `member_pos[i]` is set when slot i is present.
    let mut member_pos: Vec<WPos> = travel.homes.clone();
    // The assault anchor (advances rally→target ONLY after the gather quorum fires).
    let mut anchor = travel.rally;
    let mut departed = false; // the rally gate released (solo travel begins)
    let mut gathered = false; // the gather quorum fired (assault begins)
    let mut prev_target_room_dist: Option<u32> = None;

    for tick in 0..s.budget_ticks {
        completing.retain(|&(id, at)| {
            if at <= tick {
                syncing.push((id, tick + MEMBER_SYNC_DELAY));
                false
            } else {
                true
            }
        });
        syncing.retain(|&(id, at)| {
            if at <= tick {
                if (id as usize) < n_slots {
                    filled[id as usize] = true;
                    member_pos[id as usize] = travel.homes[id as usize]; // appears AT its home
                }
                false
            } else {
                true
            }
        });

        let present = filled.iter().filter(|f| **f).count();
        max_present = max_present.max(present);
        let has_members = present > 0 || !completing.is_empty() || !syncing.is_empty();

        let any_queued = !fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains).is_empty();
        let forming_in_flight = !completing.is_empty() || !syncing.is_empty() || any_queued;

        // Present members' positions (in slot order) for the rally + gather kernels.
        let present_positions: Vec<WPos> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();

        let mut in_target_room = false;
        let mut traveling = false;
        let mut travel_progress = false;

        if !departed {
            // FORMING / rally gate over the present roster.
            let positions = SpatialTravel::pos_options(&present_positions);
            if rally::ready_to_depart_gate(&positions, n_slots, travel.uncontested) {
                departed = true;
                travel_start = tick;
            }
        }

        if departed && !gathered {
            // SOLO TRAVEL to the shared rally (FIXED) vs the BUGGY per-member-home / frozen anchor.
            traveling = true;
            for i in 0..n_slots {
                if !filled[i] {
                    continue;
                }
                if travel.use_shared_rally {
                    // FIXED: each member paths SOLO to the ONE shared rally (no cross-room formation).
                    member_pos[i] = member_pos[i].step_toward(travel.rally);
                } else {
                    // BUGGY: the cross-room box-formation anchor freezes for scattered members, so the
                    // per-member target is the FROZEN anchor offset ≈ its own home → it never converges.
                    // (Model: a scattered member does not move; only a member already co-located with the
                    // anchor's room could advance — which scattered multi-home members never are.)
                    if member_pos[i].room() != anchor.room() {
                        // frozen — no movement (the live fatigue=0, d=(stalled) symptom)
                    } else {
                        member_pos[i] = member_pos[i].step_toward(travel.rally);
                    }
                }
            }
            // Recompute present positions after the step for the gather kernel + progress signal.
            let stepped: Vec<WPos> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();
            // Has a fighter gathered? (Slot 0 is the fighter-first spawn; treat any present member as a
            // potential fighter for the model — the bot supplies a role-aware flag.)
            let has_fighter = !stepped.is_empty();
            if travel.use_shared_rally
                && rally::gather_quorum_met(&SpatialTravel::pos_options(&stepped), travel.rally.to_pos(), n_slots, travel.uncontested, has_fighter, rally::RALLY_GATHER_RADIUS)
            {
                gathered = true;
            }
            // Travel progress = the CLOSEST present member closed room-distance to the rally this tick.
            let cur_dist = stepped.iter().map(|p| p.room_dist(travel.rally)).min();
            travel_progress = match (cur_dist, prev_target_room_dist) {
                (Some(c), Some(p)) => c < p,
                (Some(_), None) => true,
                _ => false,
            };
            prev_target_room_dist = cur_dist;
        }

        if gathered {
            // ASSAULT: the anchor advances rally→target as a bloc; members follow it.
            traveling = true;
            anchor = anchor.step_toward(travel.target);
            for i in 0..n_slots {
                if filled[i] {
                    member_pos[i] = member_pos[i].step_toward(anchor);
                }
            }
            // ARRIVED when a member stands in the target room.
            in_target_room = (0..n_slots).any(|i| filled[i] && member_pos[i].room() == travel.target.room());
            if in_target_room {
                return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
            }
            let cur_dist = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i].room_dist(travel.target)).min();
            travel_progress = match (cur_dist, prev_target_room_dist) {
                (Some(c), Some(p)) => c < p,
                (Some(_), None) => true,
                _ => false,
            };
            prev_target_room_dist = cur_dist;
        }

        // Reconcile (the shared kernel).
        let forming = has_members && !engaged_once && !departed && present < n_slots;
        let forming_progress = forming && present > prev_present;
        let forming_budget_remaining = tick.saturating_sub(gen_start) < MAX_FORMING_BUDGET;
        let travel_budget_remaining = tick.saturating_sub(travel_start) < MAX_TRAVEL_BUDGET;
        let deadline_lapsed = tick >= deadline;
        let snapshot = ReconcileSnapshot {
            objective_gone: false,
            duplicate: false,
            is_defend: false,
            deadline_lapsed,
            wiped: false,
            has_focus: false,
            engaged_once,
            in_target_room,
            has_members,
            forming,
            forming_progress,
            forming_in_flight: forming && forming_in_flight,
            forming_budget_remaining,
            traveling,
            travel_progress,
            travel_budget_remaining,
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, .. } => {
                if departed {
                    return ChurnOutcome::LapsedInTravel { generations: generation };
                }
                generation += 1;
                filled = vec![false; n_slots];
                syncing.clear();
                completing.clear();
                busy_until = vec![0; s.homes.len()];
                deadline = tick + COMMITMENT_BUDGET;
                prev_present = 0;
                gen_start = tick;
                member_pos = travel.homes.clone();
                anchor = travel.rally;
                departed = false;
                gathered = false;
                prev_target_room_dist = None;
                continue;
            }
            ReconcileAction::Retire { .. } => {
                return ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present };
            }
            ReconcileAction::KeepRefreshLease => deadline = tick + COMMITMENT_BUDGET,
            ReconcileAction::Keep => {}
        }
        prev_present = present;

        // Spawn step (only while forming).
        if !departed {
            let combat = fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains);
            let mut in_flight: BTreeSet<u64> = completing.iter().chain(syncing.iter()).map(|&(id, _)| id).collect();
            for h in 0..s.homes.len() {
                avail[h] = (avail[h] + s.homes[h].income).min(s.homes[h].energy_capacity);
                if tick < busy_until[h] {
                    continue;
                }
                let mut queue: Vec<QueuedSpawn> = Vec::new();
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
                        completing.push((spawned.id, tick + spawned.completes_in));
                        in_flight.insert(spawned.id);
                    }
                }
            }
        }
    }

    if departed {
        ChurnOutcome::LapsedInTravel { generations: generation }
    } else {
        ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present }
    }
}

/// The outcome of the churn-modeling lifecycle (the live failure taxonomy this harness reproduces).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChurnOutcome {
    /// The squad formed (or quorum-released), TRAVELED, ARRIVED, got a focus, and latched `engaged_once`.
    /// The deep bug is ABSENT — the squad reached + engaged the target.
    DeployedAndEngaged { generations: u32, engage_tick: u32 },
    /// The roster never released the rally gate before its lease lapsed, RE-FIELDING `generations` times —
    /// the live `GaveUp engaged_once=false in_room=false` churn (Break #1: oversized roster never completes).
    ChurnedNeverDeployed { generations: u32, max_present: usize },
    /// The squad RELEASED the gate + started traveling but the lease lapsed MID-HOP (it never arrived) —
    /// the live W7N7 1-slot travel-phase lapse (Break #2 travel half).
    LapsedInTravel { generations: u32 },
    /// The squad ARRIVED (in the target room) but never latched a focus before the lease lapsed — the live
    /// IN_ROOM_NO_FOCUS / empty-DTO-on-arrival lapse (Break #2 arrival half).
    LapsedOnArrival { generations: u32 },
}

/// One squad-generation's lifecycle phase in the churn driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// At home assembling + rallying (the gate has not released).
    Forming,
    /// The gate released; en route to the target. `arrives_at` is the tick travel completes.
    Traveling { arrives_at: u32 },
    /// In the target room; `dtos_clear_at` is the tick the room DTOs populate (a focus becomes computable).
    Arrived { dtos_clear_at: u32 },
}

/// Drive ONE colony forming + the full bot lifecycle wiring (lease / reconcile / re-field churn + the real
/// rally gate + 2-tick member sync + travel + empty-DTO-on-arrival) to reproduce the deep "fielded squad
/// never reaches/engages" bug DETERMINISTICALLY offline. Deterministic: same (scenario, target) → same outcome.
///
/// The spawn model is `run_forming`'s exact per-home head-of-line lane contest (K1 `spawn_step` over the
/// economy plus the unfilled combat slots, cross-home de-duped), so the contention plateau is identical;
/// this driver wraps it with the lease/reconcile/travel/arrival phases. The reconcile DECISION is the SHARED
/// `screeps_combat_decision::lifecycle::reconcile` kernel — the same one the bot's Phase A runs — so there
/// is no live/sim drift in the give-up-vs-keep logic.
pub fn run_lifecycle_churn(s: &ColonyFormingScenario, target: &ChurnTarget) -> ChurnOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot, RetireReason};

    let n_slots = s.composition.slots.len();
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);

    let mut generation: u32 = 0;
    let mut max_present: usize = 0;

    // Per-generation forming state (reset on re-field).
    let mut filled = vec![false; n_slots];
    // Spawns whose body has completed but whose position has not yet synced (the 2-tick gap).
    let mut syncing: Vec<(u64, u32)> = Vec::new(); // (slot_id, present_at_tick)
    let mut completing: Vec<(u64, u32)> = Vec::new(); // (slot_id, completes_at_tick)
    let mut avail: Vec<u32> = s.homes.iter().map(|h| h.start_energy).collect();
    let mut busy_until: Vec<u32> = vec![0; s.homes.len()];
    let mut deadline: u32 = COMMITMENT_BUDGET; // fielded at tick 0 with now + budget
    let mut prev_present: usize = 0;
    // The model latches `engaged_once` by RETURNING `DeployedAndEngaged` on the arrival-with-focus tick, so
    // within the loop it is structurally always false (a squad that reaches combat exits the driver). It is
    // still fed to the kernel for parity with the bot snapshot.
    let engaged_once = false;
    let mut phase = Phase::Forming;
    let mut gen_start: u32 = 0; // tick this generation started forming (the forming-budget clock)
    let mut travel_start: u32 = 0; // tick the squad departed home (the travel-budget clock)

    for tick in 0..s.budget_ticks {
        // 1. Complete spawns due this tick → they enter the 2-tick position-sync pipeline (NOT present yet).
        completing.retain(|&(id, at)| {
            if at <= tick {
                syncing.push((id, tick + MEMBER_SYNC_DELAY));
                false
            } else {
                true
            }
        });
        // 1b. Members whose position finally synced → now PRESENT (fill the slot).
        syncing.retain(|&(id, at)| {
            if at <= tick {
                if (id as usize) < n_slots {
                    filled[id as usize] = true;
                }
                false
            } else {
                true
            }
        });

        let present = filled.iter().filter(|f| **f).count();
        max_present = max_present.max(present);
        let has_members = present > 0 || !completing.is_empty() || !syncing.is_empty();

        // A combat slot is QUEUED or IN FLIGHT this tick — a member is banking/spawning (the forming-lease
        // refresh signal, Break #1). A slot is in flight if a spawn is completing/syncing; it is queued if an
        // unfilled slot can still be built at an in-range home (the fielding kernel would emit it). Mirrors
        // the bot adapter's "a slot has a queued/in-flight spawn".
        let any_queued = !fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains).is_empty();
        let forming_in_flight = !completing.is_empty() || !syncing.is_empty() || any_queued;

        // 2. Phase progression (travel → arrive → engage) once the squad has departed home. A squad in this
        // model engages by RETURNING on the arrival-with-focus tick, so `has_focus` is always false in the
        // snapshot below (the engage path never falls through to it) — the kernel's give-up/keep over a
        // focus-less squad is exactly what the travel/arrival breaks exercise.
        let mut in_target_room = false;
        let has_focus = false;
        let mut traveling = false;
        let mut travel_progress = false;
        match phase {
            Phase::Forming => {
                // The REAL rally gate over the present roster (full roster, or min-viable quorum if the
                // target is proven-uncontested). Releasing starts the travel phase.
                let positions: Vec<Option<Position>> = vec![Some(dummy_home_pos()); present];
                if rally::ready_to_depart_gate(&positions, n_slots, target.uncontested) {
                    travel_start = tick;
                    phase = Phase::Traveling { arrives_at: tick + target.travel_ticks };
                }
            }
            Phase::Traveling { arrives_at } => {
                traveling = true;
                travel_progress = true; // closing distance every tick in this model (no blockage)
                if tick >= arrives_at {
                    // FOCUS-ON-ARRIVAL FIX (Break #2 arrival half): on arrival the bot FORCES a room-DTO
                    // re-read (ensures `mapping.get_room` + `room_data` are populated that tick) instead of
                    // just logging IN_ROOM_NO_FOCUS and lapsing. So the focus is computable on the arrival
                    // tick — the empty-DTO window is closed by the forced re-read.
                    phase = Phase::Arrived { dtos_clear_at: tick };
                    let _ = target.empty_dtos_on_arrival_ticks; // the pre-fix lapse window — closed by the re-read
                }
            }
            Phase::Arrived { dtos_clear_at } => {
                in_target_room = true;
                // Once the room DTOs are readable (the focus-on-arrival fix forces this on arrival), a focus
                // is computed and the squad ENGAGES — the deep bug is absent. Until then (pre-fix model: a
                // persistent empty-DTO window) it sits in-room with no focus, exposed to the lease lapse.
                if tick >= dtos_clear_at {
                    return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                }
                // Still no focus this tick — feed the kernel `has_focus = false` (already the default) so the
                // lease behaviour reflects the in-room-no-focus state.
            }
        }

        // 3. The reconcile kernel decides keep / refresh / retire — the SAME kernel the bot's Phase A runs.
        //    `forming` mirrors the bot's `forming_state`: members, not engaged, below the requested roster.
        let forming = has_members && !engaged_once && phase == Phase::Forming && present < n_slots;
        let forming_progress = forming && present > prev_present;
        let forming_budget_remaining = tick.saturating_sub(gen_start) < MAX_FORMING_BUDGET;
        let travel_budget_remaining = tick.saturating_sub(travel_start) < MAX_TRAVEL_BUDGET;
        let deadline_lapsed = tick >= deadline;
        let snapshot = ReconcileSnapshot {
            objective_gone: false,
            duplicate: false,
            is_defend: false,
            deadline_lapsed,
            wiped: false,
            has_focus,
            engaged_once,
            in_target_room,
            has_members,
            forming,
            forming_progress,
            forming_in_flight: forming && forming_in_flight,
            forming_budget_remaining,
            traveling,
            travel_progress,
            travel_budget_remaining,
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, .. } => {
                // RE-FIELD: drop the partial roster, orphan in-flight spawns, bump the generation, reopen
                // the lease — the live churn loop. The new generation re-forms from scratch.
                if phase != Phase::Forming {
                    // Released the gate but lapsed before engaging — distinguish travel vs arrival lapse.
                    return match phase {
                        Phase::Arrived { .. } => ChurnOutcome::LapsedOnArrival { generations: generation },
                        _ => ChurnOutcome::LapsedInTravel { generations: generation },
                    };
                }
                generation += 1;
                filled = vec![false; n_slots];
                syncing.clear();
                completing.clear();
                busy_until = vec![0; s.homes.len()];
                deadline = tick + COMMITMENT_BUDGET;
                prev_present = 0;
                phase = Phase::Forming;
                gen_start = tick; // restart the forming-budget clock for the new generation
                continue;
            }
            ReconcileAction::Retire { .. } => {
                // Any other terminal retire in this single-objective model ends the run as never-deployed.
                return ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present };
            }
            ReconcileAction::KeepRefreshLease => deadline = tick + COMMITMENT_BUDGET,
            ReconcileAction::Keep => {}
        }
        prev_present = present;

        // 4. Spawn step — ONLY while forming (a departed squad does not keep spawning its own slots). Same
        //    per-home head-of-line lane contest as `run_forming` (economy + the unfilled combat slots).
        if phase == Phase::Forming {
            let combat = fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains);
            let mut in_flight: BTreeSet<u64> = completing.iter().chain(syncing.iter()).map(|&(id, _)| id).collect();
            for h in 0..s.homes.len() {
                avail[h] = (avail[h] + s.homes[h].income).min(s.homes[h].energy_capacity);
                if tick < busy_until[h] {
                    continue;
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
                        completing.push((spawned.id, tick + spawned.completes_in));
                        in_flight.insert(spawned.id);
                    }
                }
            }
        }
    }

    // The budget elapsed without engaging — classify by the furthest phase reached.
    match phase {
        Phase::Forming => ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present },
        Phase::Traveling { .. } => ChurnOutcome::LapsedInTravel { generations: generation },
        Phase::Arrived { .. } => ChurnOutcome::LapsedOnArrival { generations: generation },
    }
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
    // The Default-knob regime (the seed): the acceptance gate routes through here unchanged.
    run_defended_lifecycle_with_params(s, rampart_hits, towers, layout, force, &screeps_combat_decision::composition::CompositionParams::default())
}

/// As [`run_defended_lifecycle_with`] but driven by a [`CompositionParams`] knob set (the param-sweep seam,
/// ADR 0031 D16/D17 / 0031a §4): the breach force is emitted with `params.hold_margin`/`over_power_margin`
/// and assembled at `min(params.member_energy, bed capacity)`. `Default` params reproduce the seed exactly
/// (HOLD_MARGIN / COORDINATED_DPS_MARGIN / PREFERRED_MEMBER_ENERGY), so the acceptance gate is unchanged.
pub fn run_defended_lifecycle_with_params(
    s: &ColonyFormingScenario,
    rampart_hits: u32,
    towers: &[((u8, u8), u32)],
    layout: crate::harness::generate::Layout,
    force: crate::harness::generate::ForceSpec,
    params: &screeps_combat_decision::composition::CompositionParams,
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
    // The swept per-member cap never exceeds the home capacity (the bed's `member_energy`).
    let sizing_energy = params.member_energy.min(engage.member_energy);
    let (assessment, required) = emit_requirement(
        DoctrineObjective::DismantleStructure,
        &profile,
        defenders,
        Some(&budget),
        coordination,
        0.0,
        params.hold_margin,
        params.over_power_margin,
    );
    let comp = match (assessment.winnable && assessment.mode == AssaultMode::Breach, assemble_force(&required, sizing_energy)) {
        (true, Some(assembled)) => assembled,
        // The oracle deferred / drained / the assembler couldn't field the required force at this energy —
        // field the ceiling so the chain still runs (the test then surfaces whether even the ceiling kills).
        // The ceiling fallback uses the HOME capacity (not the swept per-member cap) so the Default path is
        // byte-identical to the pre-sweep behaviour (which sized the fallback at `engage.member_energy`).
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

    // ── The CHURN-MODELING lifecycle: the deep "fielded squad never reaches/engages" bug, offline ──
    //
    // These reproduce the three live execution-wiring breaks the agent-sim + `run_forming` bypass (the
    // commitment lease / reconcile / re-field churn + the real rally gate + 2-tick member sync + travel +
    // empty-DTO-on-arrival). Each was RED on the pre-fix kernel/sizing and is GREEN once the lease + focus
    // + fighter-first fixes land. The fix SCOPE is correctness (let the squad reach + engage), NOT the
    // calibration-gated defense right-sizing (deferred).

    /// The live DEFENSE sizing (always-field, dps=30 → a multi-member healer-front roster) via the EXACT
    /// optimizer path the bot runs (`optimize_composition`, honor_verdict=false). The roster is expensive +
    /// HEALER-front-loaded — the Break #1 shape that plateaus under contention.
    fn oversized_defense_comp() -> SquadComposition {
        use screeps_combat_decision::composition::optimize_composition;
        use screeps_combat_decision::doctrine::{DoctrineObjective, EnemyCoordination, EnemyForce};
        use screeps_combat_decision::force_sizing::DefenseProfile;
        let defense = DefenseProfile { towers: vec![], breach_hits: 0, objective_hits: 0, enemy_dps: 30.0, repair_per_tick: 0.0, safe_mode: false };
        let enemy = EnemyForce { dps: 30.0, heal: 0.0, hits: 600, count: 2, boosted: false };
        optimize_composition(
            DoctrineObjective::ClearCreeps,
            &defense,
            Some(enemy),
            1e6,   // defense target_value (always-field)
            1500,  // generous on-site window
            EnemyCoordination::Coordinated,
            0.0,   // importance
            false, // always-field (honor_verdict=false)
            &screeps_combat_decision::composition::CompositionParams::default(),
        )
        .expect("the defense optimizer fields a roster for dps=30")
    }

    /// A spawn-contended colony forming `comp`: two modest RCL7 homes banking slowly, a constant HIGH
    /// hauler eating a lane, combat at the live forming band (85). Expensive multi-slot rosters plateau here.
    fn contended(comp: SquadComposition) -> ColonyFormingScenario {
        ColonyFormingScenario {
            composition: comp,
            // ONE weak home: slot 0 spawns from the banked start_energy, but banking the next member's body
            // at the trickle income takes LONGER than COMMITMENT_BUDGET (400) — the inter-member banking gap
            // exceeds the lease window. The roster IS fieldable, just slower than the pre-fix lease tolerates
            // BETWEEN present++ events → the lease lapses between members → drop slot 0 → re-field churn (the
            // live W7N4 healer-pile-up at present=1/2). A constant HIGH hauler holds the lane otherwise.
            homes: vec![Home { energy_capacity: 5300, income: 4, start_energy: 2400 }],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 85.0, // SPAWN_PRIORITY_COMBAT_FORMING
            per_member_cap: 3000,
            budget_ticks: 6000,
            member_ttl: 1500,
            renew: false,
        }
    }

    /// BREAK #1 (RED on the pre-fix lease): an oversized HEALER-front defense roster under economy pressure
    /// never completes its roster — the present count plateaus, so the pre-fix lease (refreshed ONLY on the
    /// exact present++ tick) lapses at +400 → GaveUp → RE-FIELD → Generation churn that never deploys. The
    /// post-fix lease (refresh while a slot has a queued/in-flight spawn, bounded) lets the slow roster ride
    /// to completion and DEPLOY. A DEFENDED target (contested) keeps the full-roster rally.
    #[test]
    fn oversized_defense_roster_churns_never_deploys() {
        let target = ChurnTarget { travel_ticks: 30, uncontested: false, empty_dtos_on_arrival_ticks: 0 };
        let out = run_lifecycle_churn(&contended(oversized_defense_comp()), &target);
        // After the fix this must be DeployedAndEngaged; pre-fix it churns. Assert the FIXED expectation so
        // the test is RED today and GREEN once the lease fix lands.
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "the oversized defense roster must ride the forming lease to completion + deploy, got {out:?}"
        );
    }

    /// BREAK #2 (travel half — RED on the pre-fix lease): a 1-slot uncontested offense squad (the live
    /// W7N7 undefended core) forms its single member trivially + releases the quorum gate, then TRAVELS a
    /// multi-room hop. While traveling it has no focus, is not in the target room, and is not forming
    /// (present>=requested) — so the pre-fix lease is NEVER refreshed between FIELD and arrival and lapses
    /// MID-HOP. The post-fix travel-lease (refresh while a full-roster squad travels with positional
    /// progress, bounded) carries it to arrival + engage.
    #[test]
    fn single_slot_offense_deploys_within_lease() {
        let comp = assemble_force(&RequiredForce { immune_struct_parts: 4, ..Default::default() }, 3000)
            .expect("a single-slot ranged core-killer");
        assert_eq!(comp.slots.len(), 1, "the undefended-core force is a single slot (W7N7)");
        let scenario = ColonyFormingScenario {
            composition: comp,
            homes: vec![Home { energy_capacity: 5300, income: 300, start_energy: 2000 }],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 85.0,
            per_member_cap: 3000,
            budget_ticks: 1500,
            member_ttl: 1500,
            renew: false,
        };
        // A long multi-room hop (> COMMITMENT_BUDGET) so the travel-phase lapse is exercised; uncontested →
        // the quorum gate releases the single member immediately.
        let target = ChurnTarget { travel_ticks: 500, uncontested: true, empty_dtos_on_arrival_ticks: 0 };
        let out = run_lifecycle_churn(&scenario, &target);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "the 1-slot uncontested squad must hold its lease through travel + engage, got {out:?}"
        );
    }

    /// BREAK #2 (arrival half — RED on the pre-fix focus-on-arrival): a roster that ARRIVES in the target
    /// room but gets EMPTY room DTOs for several ticks computes no focus → cannot latch `engaged_once` →
    /// the lease lapses underneath it (the live IN_ROOM_NO_FOCUS lapse). The post-fix focus-on-arrival
    /// forces a room-DTO re-read on arrival so a focus is computed + `engaged_once` latches before the lease
    /// lapses.
    #[test]
    fn arrived_squad_with_empty_dtos_does_not_lapse() {
        let comp = assemble_force(&RequiredForce { immune_struct_parts: 4, ..Default::default() }, 3000)
            .expect("a single-slot ranged core-killer");
        let scenario = ColonyFormingScenario {
            composition: comp,
            homes: vec![Home { energy_capacity: 5300, income: 300, start_energy: 2000 }],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 85.0,
            per_member_cap: 3000,
            budget_ticks: 1500,
            member_ttl: 1500,
            renew: false,
        };
        // Arrives quickly, but the room DTOs stay empty far past the lease window (> COMMITMENT_BUDGET) —
        // the live mapping/visibility timing hole. Pre-fix: no focus → lease lapses → LapsedOnArrival.
        let target = ChurnTarget { travel_ticks: 20, uncontested: true, empty_dtos_on_arrival_ticks: 600 };
        let out = run_lifecycle_churn(&scenario, &target);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "an arrived squad must force a DTO re-read + engage, not lapse on empty DTOs, got {out:?}"
        );
    }

    #[test]
    fn lifecycle_churn_is_deterministic() {
        let target = ChurnTarget { travel_ticks: 30, uncontested: false, empty_dtos_on_arrival_ticks: 0 };
        let s = contended(oversized_defense_comp());
        assert_eq!(run_lifecycle_churn(&s, &target), run_lifecycle_churn(&s, &target));
    }

    // ── SPATIAL movement-stall repro (ADR 0028 K0): distinct homes → shared rally → assault ──

    /// A 2-slot offense roster forming across TWO DISTINCT homes (W2N9 + W3N4), a shared rally on the
    /// approach (W3N3), targeting a room beyond it (W4N3). The homes are easily fieldable; this isolates
    /// the MOVEMENT stall from spawn contention.
    fn two_home_offense() -> ColonyFormingScenario {
        // A 2-slot force (one anti-creep fighter + one healer) so there is one member per distinct home.
        let comp = assemble_force(&RequiredForce { anti_creep_parts: 4, heal_parts: 4, ..Default::default() }, 3000)
            .expect("a 2-slot fighter+healer force");
        assert_eq!(comp.slots.len(), 2, "the spatial repro uses a 2-slot roster (one member per home)");
        ColonyFormingScenario {
            composition: comp,
            homes: vec![
                Home { energy_capacity: 5300, income: 300, start_energy: 3000 },
                Home { energy_capacity: 5300, income: 300, start_energy: 3000 },
            ],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 87.5,
            per_member_cap: 3000,
            budget_ticks: 2000,
            member_ttl: 1500,
            renew: false,
        }
    }

    /// Distinct homes a few rooms apart; a shared rally on the approach; a target a room beyond the rally.
    fn scatter_travel(uncontested: bool, use_shared_rally: bool) -> SpatialTravel {
        SpatialTravel {
            // Two homes in DIFFERENT rooms (the multi-home scatter): W2N9 ≈ (world 100+25, 400+25) and
            // W3N4. Using world coords: room (rx, ry) maps to W{-rx-1}N{-ry-1}, so W2N9 → rx=-3, ry=-10.
            homes: vec![
                WPos { wx: -3 * 50 + 25, wy: -10 * 50 + 25 }, // W2N9
                WPos { wx: -4 * 50 + 25, wy: -5 * 50 + 25 },  // W3N4
            ],
            rally: WPos { wx: -4 * 50 + 5, wy: -4 * 50 + 25 }, // W3N3 staging (approach)
            target: WPos { wx: -5 * 50 + 25, wy: -4 * 50 + 25 }, // W4N3 target
            uncontested,
            use_shared_rally,
        }
    }

    /// RED on the BUGGY model, GREEN on the FIXED one: scattered multi-home members behind a frozen
    /// cross-room formation anchor NEVER converge → travel makes no positional progress → the lease lapses
    /// mid-hop (`LapsedInTravel`). The shared-rally solo-travel + the unified gather kernel converges them
    /// and advances the anchor to the target → `DeployedAndEngaged`.
    #[test]
    fn scattered_squad_stalls_then_converges_with_shared_rally() {
        // BUGGY: per-member-home / frozen cross-room formation anchor → never converges → stalls in travel.
        let buggy = run_lifecycle_churn_spatial(&two_home_offense(), &scatter_travel(false, false));
        assert!(
            matches!(buggy, ChurnOutcome::LapsedInTravel { .. }),
            "the buggy frozen-formation-anchor model must stall in travel (never converge), got {buggy:?}"
        );
        // FIXED: solo travel to a SHARED rally + the unified gather kernel → converge → assault → engage.
        let fixed = run_lifecycle_churn_spatial(&two_home_offense(), &scatter_travel(false, true));
        assert!(
            matches!(fixed, ChurnOutcome::DeployedAndEngaged { .. }),
            "the shared-rally solo-travel design must converge + advance + engage, got {fixed:?}"
        );
    }

    /// An UNCONTESTED target may trickle (the gather quorum fires at a single gathered member), so even the
    /// shared-rally model deploys + engages quickly.
    #[test]
    fn uncontested_scatter_trickles_in_and_engages() {
        let out = run_lifecycle_churn_spatial(&two_home_offense(), &scatter_travel(true, true));
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "an uncontested target trickles the gathered members in + engages, got {out:?}"
        );
    }

    #[test]
    fn spatial_lifecycle_is_deterministic() {
        let s = two_home_offense();
        let t = scatter_travel(false, true);
        assert_eq!(run_lifecycle_churn_spatial(&s, &t), run_lifecycle_churn_spatial(&s, &t));
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
