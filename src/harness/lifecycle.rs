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
    /// This squad is a `Defend` garrison for an OWNED room (war.rs `ObjectiveKind::Defend`). The owned
    /// target room is itself CLEAR (the threat roams a NEIGHBOUR), so on arrival there is NEVER an in-room
    /// focus — the defender stands in the empty owned-room centre. BUG B2: pre-fix the focus-less in-room
    /// defender past its lease `GaveUp` → Phase C re-fielded the SAME defender → Gen churn. Post-fix the
    /// `holding_station` signal garrisons it (lease refreshed) while the Defend objective persists.
    pub is_defend: bool,
    /// BUG B1 (engaged-en-route latch): the TARGET room is kept visible via a HIGH OBSERVE request and has a
    /// hostile in it, so while the squad is still TRAVELING (dist>0, in_room=false) the proximity-free
    /// `select_focus_target` returns a focus → `decide_squad` sets `state=Engaged` → the bot's
    /// `apply_squad_decision` latches `engaged_once=true` with NO in-room gate. The PERMANENT latch kills the
    /// travel lease (`traveling` requires `!engaged_once`) → the squad freezes mid-travel. The FIX gates the
    /// latch on in-room presence (`latch_engaged_in_room_only`).
    pub target_visible_with_hostile_en_route: bool,
    /// FIX B1 toggle: latch `engaged_once` ONLY when a member is actually IN the target room (`in_room_any`).
    /// `false` reproduces the pre-fix bug (latch from `focus.is_some()` regardless of distance); `true` is the
    /// fixed bot (`apply_squad_decision` gates the latch on in-room presence). A far defender with a visible
    /// target-room hostile then does NOT latch + keeps its travel lease.
    pub latch_engaged_in_room_only: bool,
    /// FIX B2 toggle: does the bot adapter SUPPLY the `holding_station` signal to the (shared) reconcile
    /// kernel? `true` is the fixed bot (a Defend garrison's `is_defend && in_target_room && !has_focus`
    /// refreshes its lease → it garrisons); `false` reproduces the pre-fix bot (no signal → the focus-less
    /// in-room defender past its lease GaveUp → Phase C re-fields the SAME defender → Gen churn). The KERNEL
    /// is unchanged either way — this only controls whether the adapter feeds it the signal (no drift).
    pub garrison_holds: bool,
}

impl Default for ChurnTarget {
    fn default() -> Self {
        ChurnTarget {
            travel_ticks: 30,
            uncontested: false,
            empty_dtos_on_arrival_ticks: 0,
            is_defend: false,
            target_visible_with_hostile_en_route: false,
            latch_engaged_in_room_only: true, // default to the FIXED bot behaviour
            garrison_holds: true,             // default to the FIXED bot behaviour
        }
    }
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
    /// BUG A (contested boundary oscillation, the W4N7 defender). Rooms (world room-coords `(rx, ry)`) the
    /// enemy HOLDS — a member that steps INTO one of these in-transit DIES (the multi-home defender's members
    /// die crossing the enemy-held neighbours between their scattered homes and the rally). A dead member
    /// drops back to UNFILLED → `present` falls → re-spawn. Combined with the NON-latched per-tick gather
    /// re-eval (`latch_assault == false`), the squad never reaches a stable quorum → it oscillates
    /// in_room<->travel and never commits the assault. Empty ⇒ no in-transit attrition (the clean path).
    pub enemy_held_rooms: Vec<(i32, i32)>,
    /// FIX A toggle: once `gather_quorum_met` first returns true, LATCH the assault and thereafter take the
    /// assault branch WITHOUT re-evaluating the quorum (so members dying/lagging can't un-commit it), AND
    /// count members already IN the target room as gathered. `false` reproduces the pre-fix bot (re-evaluate
    /// the quorum every tick over all positions, never latch — squad_manager.rs 1255-1262) → oscillation;
    /// `true` is the fixed bot (latch the assault on the first quorum). Default `true` (the fixed behaviour).
    pub latch_assault: bool,
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
    // BUG A: the peak count of members ever gathered at the rally (the oscillation diagnostic — a buggy
    // contested defender never reaches the full quorum, so this stays below the requested roster).
    let mut max_gathered: usize = 0;
    // BUG A: how many times the gather state FLIPPED true→false (un-committed the assault) — the
    // in_room<->travel oscillation. Non-zero ⇒ the buggy non-latched re-eval is thrashing (it never commits
    // a stable assault; the FIXED latch returns `DeployedAndEngaged` before the budget elapses).
    let mut oscillations: u32 = 0;
    let mut prev_gathered = false;

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

        if departed {
            traveling = true;
            // ── GATHER DECISION (evaluated EVERY tick so the buggy non-latched model can OSCILLATE). The
            // gather quorum over the CURRENT positions; FIX A also counts in-target-room members as gathered.
            let pre_step: Vec<WPos> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();
            let has_fighter = !pre_step.is_empty();
            let in_room_count = (0..n_slots).filter(|&i| filled[i] && member_pos[i].room() == travel.target.room()).count();
            let quorum_now = (travel.use_shared_rally
                && rally::gather_quorum_met(&SpatialTravel::pos_options(&pre_step), travel.rally.to_pos(), n_slots, travel.uncontested, has_fighter, rally::RALLY_GATHER_RADIUS))
                // FIX A: members already IN the target room count as gathered (arrived members can't fail it).
                || (travel.latch_assault && in_room_count > 0 && has_fighter);
            max_gathered = max_gathered.max(rally::members_gathered_at(&SpatialTravel::pos_options(&pre_step), travel.rally.to_pos(), rally::RALLY_GATHER_RADIUS));
            if travel.latch_assault {
                // FIXED: LATCH the assault on the FIRST quorum — members dying/lagging can't un-commit it
                // (the bot stops re-evaluating `gather_quorum_met` once it latches the assault state).
                gathered |= quorum_now;
            } else {
                // BUGGY (the live per-tick non-latched re-eval, squad_manager.rs 1255-1262): `gathered`
                // tracks the CURRENT quorum — if a member dies in transit and the quorum drops, the squad
                // REVERTS from assault to solo travel (the in_room<->travel oscillation that never commits).
                gathered = quorum_now;
            }
            // Count a true→false flip — the assault un-committing (the oscillation diagnostic).
            if prev_gathered && !gathered {
                oscillations += 1;
            }
            prev_gathered = gathered;

            if gathered {
                // ASSAULT: the anchor advances rally→target as a bloc; members follow it.
                anchor = anchor.step_toward(travel.target);
                for i in 0..n_slots {
                    if filled[i] {
                        member_pos[i] = member_pos[i].step_toward(anchor);
                        // BUG A attrition during the assault crossing: a SUPPORT member (slot >= 1, the
                        // lagging healer) stepping into an enemy-held room DIES; the lead fighter (slot 0)
                        // tanks the crossing. Pre-fix (non-latched) the support death drops `present` below
                        // the contested quorum → the next tick's re-eval REVERTS assault→travel (oscillation).
                        // The latch keeps the assault committed so the surviving fighter reaches the target.
                        if i >= 1 && travel.enemy_held_rooms.contains(&member_pos[i].room()) {
                            filled[i] = false;
                            member_pos[i] = travel.homes[i];
                        }
                    }
                }
                in_target_room = (0..n_slots).any(|i| filled[i] && member_pos[i].room() == travel.target.room());
                if in_target_room && travel.latch_assault {
                    return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                }
                let cur_dist = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i].room_dist(travel.target)).min();
                travel_progress = match (cur_dist, prev_target_room_dist) {
                    (Some(c), Some(p)) => c < p,
                    (Some(_), None) => true,
                    _ => false,
                };
                prev_target_room_dist = cur_dist;
            } else {
                // SOLO TRAVEL to the shared rally (FIXED) vs the BUGGY per-member-home / frozen anchor.
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
                        if member_pos[i].room() != anchor.room() {
                            // frozen — no movement (the live fatigue=0, d=(stalled) symptom)
                        } else {
                            member_pos[i] = member_pos[i].step_toward(travel.rally);
                        }
                    }
                    // BUG A: in-transit ATTRITION. A member that stepped INTO an enemy-held room DIES — it
                    // drops back to UNFILLED + must re-spawn (the multi-home defender's members dying while
                    // crossing the enemy-held neighbours between their scattered homes and the rally →
                    // `present` falls → the non-latched quorum can never stabilise).
                    if travel.enemy_held_rooms.contains(&member_pos[i].room()) {
                        filled[i] = false;
                        member_pos[i] = travel.homes[i]; // the slot reopens; a fresh member re-spawns at home
                    }
                }
                let stepped: Vec<WPos> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();
                let cur_dist = stepped.iter().map(|p| p.room_dist(travel.rally)).min();
                travel_progress = match (cur_dist, prev_target_room_dist) {
                    (Some(c), Some(p)) => c < p,
                    (Some(_), None) => true,
                    _ => false,
                };
                prev_target_room_dist = cur_dist;
            }
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
            holding_station: false, // offense spatial repro — never a Defend garrison
            reassign_available: false, // ADR 0027 v1 reassign is exercised by `run_v1_flow`, not here
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, .. } => {
                if departed {
                    // BUG A: an oscillating contested squad that never committed the assault → classify the
                    // thrash distinctly from a clean mid-hop travel lapse.
                    if oscillations > 0 {
                        return ChurnOutcome::OscillatedNeverGathered { generations: generation, max_gathered };
                    }
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
            // This driver never feeds `reassign_available=true` (reassignment is exercised by `run_v1_flow`).
            ReconcileAction::Reassign { .. } => unreachable!("reassign_available is always false here"),
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
        // BUG A: the budget elapsed while still departed without ever committing the assault, having
        // oscillated (the buggy non-latched re-eval thrashing) → `OscillatedNeverGathered`. A clean
        // non-oscillating travel lapse stays `LapsedInTravel`.
        if oscillations > 0 {
            ChurnOutcome::OscillatedNeverGathered { generations: generation, max_gathered }
        } else {
            ChurnOutcome::LapsedInTravel { generations: generation }
        }
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
    /// BUG B1: the squad latched `engaged_once` EN ROUTE (a proximity-free focus on the visible target room
    /// while still traveling, dist>0, in_room=false). The permanent latch killed the travel lease
    /// (`traveling` requires `!engaged_once`) → it FROZE mid-hop, never arriving. Fixed by gating the latch
    /// on in-room presence.
    LatchedEnRoute { generations: u32 },
    /// BUG A: a CONTESTED multi-home defender never reached `gather_quorum_met` (members die crossing
    /// enemy-held neighbours → `present` oscillates, and the gather is re-evaluated every tick, never
    /// latched) → it oscillated in_room<->travel and never committed the assault within the budget. Fixed by
    /// LATCHING the assault once the quorum first fires + counting in-room members as gathered.
    OscillatedNeverGathered { generations: u32, max_gathered: usize },
    /// BUG B2 (fixed state): a Defend squad ARRIVED in its clear owned room, found no in-room focus, and
    /// GARRISONED it (lease held) for the whole budget without churning — a single stable generation. The
    /// pre-fix outcome here was repeated GaveUp+refield (`generations` climbing).
    Garrisoned { generations: u32 },
    /// ADR 0027 v1 (whole-squad REASSIGN): the squad reached a non-loss terminal (Resolved/ObjectiveGone)
    /// with a compatible SIBLING objective available, and REBOUND IN PLACE to it — bodies reused, NO
    /// Generation churn (`from_gen == to_gen`). `reassignments` counts how many in-place rebinds happened;
    /// `engage_tick` is when the squad finally engaged the LAST (reassigned) objective. The whole point: a
    /// freed squad reuses its invested bodies instead of retire→re-field (which would climb `generations`).
    Reassigned { from_gen: u32, reassignments: u32, engage_tick: u32 },
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
    // `engaged_once` is the bot's latch (set by `apply_squad_decision` when `state==Engaged`). The model
    // normally latches by RETURNING `DeployedAndEngaged` on the arrival-with-focus tick. BUG B1 makes it
    // mutable: a squad whose VISIBLE target room has a hostile while still TRAVELING gets a proximity-free
    // focus → the pre-fix bot latches `engaged_once=true` en route (no in-room gate) → the travel lease dies.
    let mut engaged_once = false;
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
        let mut has_focus = false;
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
                // ── BUG B1 (engaged-en-route latch). The target room is kept VISIBLE (a HIGH OBSERVE) and has
                // a hostile, so the proximity-free `select_focus_target` returns a focus while we are STILL
                // traveling (in_room=false, dist>0). The bot's `decide_squad` sets `state=Engaged` from
                // `focus.is_some()` with NO proximity gate; `apply_squad_decision` then latches
                // `engaged_once=true`. PRE-FIX (`latch_engaged_in_room_only == false`) it latches HERE,
                // en route, with no member in the room → the travel lease dies (`traveling` needs
                // `!engaged_once`) → the squad FREEZES mid-hop. The FIX gates the latch on in-room presence,
                // so a far defender with a visible target-room hostile does NOT latch + keeps its lease.
                if target.target_visible_with_hostile_en_route && !target.latch_engaged_in_room_only {
                    engaged_once = true; // the pre-fix unconditional latch (no in_room_any gate)
                }
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
                if target.is_defend {
                    // ── BUG B2 (defender garrison). A Defend squad ARRIVES in its OWNED room — but the threat
                    // roams a NEIGHBOUR, so the owned room is CLEAR: `decide_squad` finds NO in-room focus
                    // (has_focus stays false). It garrisons the empty owned-room centre. Pre-fix the
                    // focus-less in-room defender past its lease GaveUp → Phase C re-fielded the SAME
                    // defender → Gen churn (the dominant live waste). Post-fix the `holding_station` signal
                    // (built below) refreshes its lease while the Defend objective persists. The kernel
                    // verdict (KeepRefreshLease vs Retire{GaveUp}) below decides which — no early return here.
                } else if tick >= dtos_clear_at {
                    // OFFENSE: once the room DTOs are readable (the focus-on-arrival fix forces this on
                    // arrival), a focus is computed and the squad ENGAGES — the deep bug is absent.
                    return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                }
                // Still no focus this tick — feed the kernel `has_focus = false` so the lease behaviour
                // reflects the in-room-no-focus state.
            }
        }
        // `has_focus` stays false in this model (an offense squad that finds a focus exits the driver; a
        // defender's owned room is clear) — assigned for clarity that the garrison branch is focus-less.
        let _ = &mut has_focus;

        // 3. The reconcile kernel decides keep / refresh / retire — the SAME kernel the bot's Phase A runs.
        //    `forming` mirrors the bot's `forming_state`: members, not engaged, below the requested roster.
        let forming = has_members && !engaged_once && phase == Phase::Forming && present < n_slots;
        let forming_progress = forming && present > prev_present;
        let forming_budget_remaining = tick.saturating_sub(gen_start) < MAX_FORMING_BUDGET;
        let travel_budget_remaining = tick.saturating_sub(travel_start) < MAX_TRAVEL_BUDGET;
        let deadline_lapsed = tick >= deadline;
        // FIX B2: a Defend squad GARRISONING its clear owned room (arrived, no in-room focus) holds its lease
        // while the Defend objective persists. (`is_defend && in_target_room && !has_focus` — the manager's
        // exact signal.) `garrison_holds` toggles whether the adapter SUPPLIES the signal (RED→GREEN); for an
        // offense squad this is always false. The shared kernel is unchanged either way.
        let holding_station = target.garrison_holds && target.is_defend && in_target_room && !has_focus;
        let snapshot = ReconcileSnapshot {
            objective_gone: false,
            duplicate: false,
            is_defend: target.is_defend,
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
            holding_station,
            reassign_available: false, // ADR 0027 v1 reassign is exercised by `run_v1_flow`, not here
        };
        // BUG B2 (fixed state): a defender that has GARRISONED its owned room (in-room, focus-less) and held
        // its lease until the budget elapsed without churning — a single stable generation. Detected when the
        // garrison reaches the final tick still in-room (no re-field happened). Checked before reconcile so
        // the stable-hold case reports `Garrisoned` rather than running the loop to the bottom-of-fn classify.
        if holding_station && tick + 1 >= s.budget_ticks {
            return ChurnOutcome::Garrisoned { generations: generation };
        }
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, .. } => {
                // RE-FIELD: drop the partial roster, orphan in-flight spawns, bump the generation, reopen
                // the lease — the live churn loop. The new generation re-forms from scratch.
                if phase != Phase::Forming {
                    // Released the gate but lapsed before engaging. BUG B1: if the squad latched
                    // `engaged_once` EN ROUTE, it froze mid-travel (the travel lease needs `!engaged_once`) —
                    // report `LatchedEnRoute`. A defender that GaveUp in-room is the B2 churn (re-field).
                    if engaged_once && !in_target_room {
                        return ChurnOutcome::LatchedEnRoute { generations: generation };
                    }
                    if target.is_defend && in_target_room {
                        // BUG B2 pre-fix: the garrison gave up in-room → it RE-FIELDS (Phase C immediately
                        // re-fields the same defender). Loop back to Forming, bumping the generation (churn).
                        generation += 1;
                        filled = vec![false; n_slots];
                        syncing.clear();
                        completing.clear();
                        busy_until = vec![0; s.homes.len()];
                        deadline = tick + COMMITMENT_BUDGET;
                        prev_present = 0;
                        engaged_once = false;
                        phase = Phase::Forming;
                        gen_start = tick;
                        continue;
                    }
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
            // This driver never feeds `reassign_available=true` (reassignment is exercised by `run_v1_flow`).
            ReconcileAction::Reassign { .. } => unreachable!("reassign_available is always false here"),
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

// ═══ ADR 0027 v1: the WHOLE-FLOW driver — multi-objective queue + a MOVING threat + the PURE defense ════
// ═══ kernel + whole-squad REASSIGN, all offline + deterministic (the operator's #1 requirement) ════════
//
// The existing drivers prove ONE objective's lifecycle. This driver models the END-TO-END v1 flow the live
// servers/Docker could not reliably validate (memory: war-lifecycle-debug):
//   • a multi-objective QUEUE (claim / withdraw / best_unclaimed-by-priority over a `Vec`, mirroring
//     `objective_queue::best_unclaimed_near`);
//   • a THREAT that MOVES one room per scan along a fixed room path;
//   • the PURE `war_decision::emit_defense` kernel called EACH SCAN — re-emit `Secure{threat_room}` at the
//     threat's current room (asset-priority boost when in/adjacent owned + the over-extension leash) and
//     TTL-LAPSE the stale objective the threat left;
//   • ONE squad driven through the shared `lifecycle::reconcile` kernel + the real gather/travel/engage,
//     and — the new behaviour — REBOUND IN PLACE when `reconcile` returns `Reassign` (bodies reused, NO
//     Generation churn), vs the pre-reassign retire→re-field that climbs the generation counter.
//
// Rooms are `(i32, i32)` grid coords (Chebyshev distance), one squad-step per tick within a room and one
// room-step per tick across rooms. Deterministic: same scenario → same outcome (no `HashMap` in any path).

/// A room in the toy world (grid coords). Chebyshev distance; one room = one queue/threat unit.
type V1Room = (i32, i32);

fn v1_dist(a: V1Room, b: V1Room) -> u32 {
    (a.0 - b.0).unsigned_abs().max((a.1 - b.1).unsigned_abs())
}

/// One entry in the toy objective queue (a faithful slice of `objective_queue::CombatObjective`):
/// a `Secure` objective at a room, with a priority, a TTL `expires_at`, and a `claimed` flag (the
/// ephemeral within-session claim). A monotonic `id` keys reassignment exclusion (`exclude=[current]`).
#[derive(Clone, Copy, Debug)]
struct V1Objective {
    id: u32,
    room: V1Room,
    priority: f32,
    expires_at: u32,
    claimed: bool,
}

/// The toy objective queue — the multi-objective claim/withdraw/best_unclaimed surface the manager pulls.
#[derive(Default)]
struct V1Queue {
    objectives: Vec<V1Objective>,
    next_id: u32,
}

impl V1Queue {
    /// Upsert a `Secure{room}` objective (dedup by room, like `objective_queue::request`'s kind-keyed
    /// upsert): refresh priority (max) + extend the TTL; mint a new id if absent. Returns the id.
    fn request(&mut self, room: V1Room, priority: f32, now: u32, ttl: u32) -> u32 {
        let expires_at = now + ttl;
        if let Some(o) = self.objectives.iter_mut().find(|o| o.room == room) {
            o.priority = o.priority.max(priority);
            o.expires_at = o.expires_at.max(expires_at);
            o.id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            self.objectives.push(V1Objective { id, room, priority, expires_at, claimed: false });
            id
        }
    }

    /// TTL-lapse stale objectives — but keep a CLAIMED one alive past its TTL (the commitment immunity,
    /// `objective_queue::expire`: a squad is on it right now). The stale Secure the threat LEFT is unclaimed
    /// (its squad reassigned to the new room), so it lapses and vanishes — the `ObjectiveGone` signal.
    fn expire(&mut self, now: u32) {
        self.objectives.retain(|o| o.expires_at > now || o.claimed);
    }

    fn get(&self, id: u32) -> Option<&V1Objective> {
        self.objectives.iter().find(|o| o.id == id)
    }

    fn claim(&mut self, id: u32) {
        if let Some(o) = self.objectives.iter_mut().find(|o| o.id == id) {
            o.claimed = true;
        }
    }

    fn release(&mut self, id: u32) {
        if let Some(o) = self.objectives.iter_mut().find(|o| o.id == id) {
            o.claimed = false;
        }
    }

    fn withdraw(&mut self, id: u32) {
        self.objectives.retain(|o| o.id != id);
    }

    /// Best unclaimed objective excluding `exclude` (the manager's `best_unclaimed_near_excluding`): highest
    /// priority, then the smallest id as a deterministic tie-break (no `HashMap`).
    fn best_unclaimed_excluding(&self, exclude: u32) -> Option<u32> {
        self.objectives
            .iter()
            .filter(|o| !o.claimed && o.id != exclude)
            .max_by(|a, b| {
                a.priority
                    .partial_cmp(&b.priority)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.id.cmp(&a.id)) // smaller id wins ties
            })
            .map(|o| o.id)
    }
}

/// The v1-flow scenario: where the colony's homes are, the owned room(s) it defends, the threat's per-scan
/// ROOM PATH, and whether whole-squad REASSIGN is enabled (the RED→GREEN toggle: `false` reproduces the
/// pre-reassign retire→re-field churn; `true` is the ADR 0027 v1 in-place rebind).
#[derive(Clone, Debug)]
pub struct V1FlowScenario {
    /// The owned rooms the defender protects (with strategic value for the asset-priority boost).
    pub owned: Vec<(V1Room, f32)>,
    /// The squad's home room (where it forms; one home = one member, kept simple — the deep forming/spawn
    /// contention is proven by the other drivers; this driver isolates the multi-objective + reassign flow).
    pub home: V1Room,
    /// The threat's room at each SCAN (one room-step per scan). The defense kernel emits `Secure` at the
    /// threat's CURRENT room each scan; as the threat advances the objective moves with it.
    pub threat_path: Vec<V1Room>,
    /// Ticks between defense scans (war.rs scans every ~2 ticks). The threat advances one path step per scan.
    pub scan_period: u32,
    /// The objective TTL (lapses a stale Secure a few scans after the threat leaves; mirrors DEFEND_TTL).
    pub objective_ttl: u32,
    /// Enable whole-squad reassignment (ADR 0027 v1). `false` = the pre-reassign retire→re-field control.
    pub reassign_enabled: bool,
    /// Ticks the squad needs to form its single member (a small fixed cost; not the contention plateau).
    pub form_ticks: u32,
    /// Tick budget.
    pub budget_ticks: u32,
}

/// One squad's live state in the v1 flow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum V1Phase {
    /// Forming its member at home.
    Forming,
    /// Roster ready; traveling toward the claimed objective's room.
    Traveling,
    /// In the claimed objective's room.
    InRoom,
}

/// Drive the WHOLE ADR 0027 v1 flow offline + deterministically: a multi-objective queue, a threat that
/// MOVES between rooms, the pure `war_decision::emit_defense` kernel re-emitting `Secure` at the threat's
/// room each scan, and the squad's reconcile + (new) in-place REASSIGN. Returns the [`ChurnOutcome`]:
/// `Reassigned` (the squad followed the threat via in-place rebind — `from_gen == 0`, no churn) is the GREEN
/// acceptance result; the pre-reassign control churns (`ChurnedNeverDeployed`/`LapsedInTravel` with climbing
/// generations) because the stale objective vanishes underneath the squad and it re-fields from scratch.
pub fn run_v1_flow(s: &V1FlowScenario) -> ChurnOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot, RetireReason};
    use screeps_combat_decision::war_decision::{
        emit_defense, neighbour_threats, observe_neighbours, DefensePolicy, OwnedRoom, RawObservation, Threat,
    };

    let owned: Vec<OwnedRoom<V1Room>> = s.owned.iter().map(|&(r, v)| OwnedRoom { room: r, value: v }).collect();
    let policy = DefensePolicy::default();

    let mut queue = V1Queue::default();
    let mut generation: u32 = 0;
    let mut reassignments: u32 = 0;

    // The squad's per-generation state.
    let mut claimed_id: Option<u32> = None;
    let mut phase = V1Phase::Forming;
    let mut pos: V1Room = s.home; // the squad's current room
    let mut form_done_at: u32 = s.form_ticks; // tick forming completes for this generation
    let mut engaged_once = false;
    let mut deadline: u32 = COMMITMENT_BUDGET;
    let mut gen_start: u32 = 0;
    let mut travel_start: u32 = 0;
    let mut prev_dist: Option<u32> = None;
    // The threat's index into its room path (advances one step per scan); None once the path is exhausted
    // (the threat left the map → its last objective TTL-lapses → the squad's objective_gone fires).
    let mut threat_step: usize = 0;

    for tick in 0..s.budget_ticks {
        // ── DEFENSE SCAN (every scan_period ticks): advance the threat one room + run the FULL PRODUCTION
        //    CHAIN (ADR 0027 P0): synthetic room-with-hostile → observe_neighbours → neighbour_threats →
        //    emit_defense → queue. The threat is an ARMED (Attack) hostile creep occupying one room; when it
        //    is in an OWNED room it goes through the owned-room threat path (emit_defense directly, as war.rs
        //    does), when it roams a NEIGHBOUR it goes through the PURE observe_neighbours → neighbour_threats
        //    builder — exactly the live war.rs seam, now end-to-end offline + deterministic. ──
        if tick % s.scan_period == 0 {
            // The threat occupies path[threat_step] this scan (None once exhausted). Model it as one armed
            // hostile body (a single Attack part ⇒ danger 30 via the lifted estimate).
            let threat_room = s.threat_path.get(threat_step).copied();
            let owned_set: Vec<V1Room> = s.owned.iter().map(|&(r, _)| r).collect();

            // OWNED-ROOM threats (emit_defense directly): the threat is in an owned room this scan.
            let owned_threats: Vec<Threat<V1Room>> = threat_room
                .filter(|r| owned_set.contains(r))
                .map(|r| vec![Threat { room: r, danger: 30.0 }])
                .into_iter()
                .flatten()
                .collect();

            // NEIGHBOUR observation (the lifted P0 chain): build a synthetic RawObservation for the threat's
            // room (visible, non-owned, an armed Attack body) and run observe_neighbours → neighbour_threats.
            let bodies: Vec<Vec<screeps::Part>> = vec![vec![screeps::Part::Attack, screeps::Part::Move]];
            let neighbour_obs: Vec<RawObservation<V1Room>> = threat_room
                .filter(|r| !owned_set.contains(r))
                .map(|r| {
                    let nearest = owned_set.iter().map(|&o| v1_dist(o, r)).min();
                    vec![RawObservation { room: r, hostile_bodies: &bodies, visible: true, is_owned: false, nearest_owned_dist: nearest }]
                })
                .into_iter()
                .flatten()
                .collect();
            let observed = observe_neighbours(&neighbour_obs, policy);
            let neighbour_threats = neighbour_threats(&owned, &observed, policy, v1_dist);

            // Feed the owned-room threats AND the neighbour threats to the one proven emission kernel.
            let threats: Vec<Threat<V1Room>> = owned_threats.into_iter().chain(neighbour_threats).collect();
            // PURE defense emission: Secure at the threat's current room (boost + leash applied in-kernel).
            for emission in emit_defense(&owned, &threats, policy, v1_dist) {
                queue.request(emission.room, emission.priority, tick, s.objective_ttl);
            }
            // Lapse the stale objective the threat LEFT (unclaimed → TTL drops it; claimed → immune).
            queue.expire(tick);
            threat_step += 1;
        } else {
            queue.expire(tick);
        }

        // ── Phase C (claim): an unclaimed squad claims the best objective. ──
        if claimed_id.is_none() {
            if let Some(id) = queue.best_unclaimed_excluding(u32::MAX) {
                queue.claim(id);
                claimed_id = Some(id);
                phase = V1Phase::Forming;
                form_done_at = tick + s.form_ticks;
                pos = s.home;
                engaged_once = false;
                deadline = tick + COMMITMENT_BUDGET;
                gen_start = tick;
                prev_dist = None;
            }
        }

        let Some(cur_id) = claimed_id else {
            continue; // nothing to do this tick (no objective in the queue yet)
        };

        // Snapshot the claimed objective (it may have TTL-lapsed if the squad never claimed in time, or the
        // threat moved + the stale one we are NOT on vanished — but a claimed objective is TTL-immune, so a
        // gone claimed objective means it was WITHDRAWN, not lapsed).
        let obj = queue.get(cur_id).copied();
        let objective_gone = obj.is_none();
        let target_room = obj.map(|o| o.room);

        // ── Phase progression: form → travel → in-room → engage. ──
        let mut in_target_room = false;
        let mut traveling = false;
        let mut travel_progress = false;
        if let Some(target) = target_room {
            match phase {
                V1Phase::Forming => {
                    if tick >= form_done_at {
                        phase = if pos == target { V1Phase::InRoom } else { V1Phase::Traveling };
                        travel_start = tick;
                    }
                }
                V1Phase::Traveling => {
                    traveling = true;
                    // Step one room toward the target.
                    let before = v1_dist(pos, target);
                    pos = (pos.0 + (target.0 - pos.0).signum(), pos.1 + (target.1 - pos.1).signum());
                    let after = v1_dist(pos, target);
                    travel_progress = after < before;
                    prev_dist = Some(after);
                    if pos == target {
                        phase = V1Phase::InRoom;
                    }
                }
                V1Phase::InRoom => {
                    in_target_room = true;
                    // Arrived: engage. If the threat is STILL here (the objective is fresh — the threat
                    // hasn't moved on), latch engaged_once + clear it (the squad clears the room / the threat
                    // steps out next scan). Either way the latch marks "fought here".
                    engaged_once = true;
                }
            }
        }
        let _ = prev_dist;

        // ── REASSIGN AVAILABILITY (the snapshot input, computed exactly like holding_station). A sibling
        //    objective exists for this squad to take over (best_unclaimed excluding the current id). The
        //    capability gate is trivially satisfied here (all v1 objectives are the same broad Secure/Defend
        //    class). Gated on the scenario toggle so the pre-reassign control reproduces the churn. ──
        let reassign_available = s.reassign_enabled && queue.best_unclaimed_excluding(cur_id).is_some();

        // ── RECONCILE (the shared kernel) — the SAME give-up/keep/reassign logic the bot Phase A runs. ──
        let has_members = true; // single fielded member, always present after forming in this driver
        let forming = phase == V1Phase::Forming && tick < form_done_at;
        let deadline_lapsed = tick >= deadline;
        // A clean clear (Resolved) fires when engaged_once + in-room + no-focus: model "the threat left this
        // room" as has_focus=false once the threat has advanced past target_room (its objective will lapse).
        let threat_here = s.threat_path.get(threat_step.saturating_sub(1)).copied() == target_room;
        let has_focus = in_target_room && threat_here; // a focus only while the threat is actually here
        let snapshot = ReconcileSnapshot {
            objective_gone,
            duplicate: false,
            is_defend: true, // a defender (the threat-centric Secure is the defense arm)
            deadline_lapsed,
            wiped: false,
            has_focus,
            engaged_once,
            in_target_room,
            has_members,
            forming,
            forming_progress: forming,
            forming_in_flight: forming,
            forming_budget_remaining: tick.saturating_sub(gen_start) < MAX_FORMING_BUDGET,
            traveling,
            travel_progress,
            travel_budget_remaining: tick.saturating_sub(travel_start) < MAX_TRAVEL_BUDGET,
            holding_station: is_defend_holding(in_target_room, has_focus),
            reassign_available,
        };
        match reconcile(snapshot) {
            ReconcileAction::Reassign { withdraw_old } => {
                // ── IN-PLACE REBIND (no Generation churn): release/withdraw the old claim → claim the new
                //    → reset engaged_once/state/travel clocks → reopen the lease. Bodies reused. ──
                let new_id = queue.best_unclaimed_excluding(cur_id).expect("reassign_available implies a sibling");
                queue.release(cur_id);
                if withdraw_old {
                    queue.withdraw(cur_id);
                }
                queue.claim(new_id);
                claimed_id = Some(new_id);
                phase = V1Phase::Forming; // re-gather at the new objective's rally; pos stays (bodies reused)
                form_done_at = tick; // already-formed roster — no re-spawn; re-rally is immediate
                engaged_once = false;
                travel_start = tick;
                prev_dist = None;
                deadline = tick + COMMITMENT_BUDGET; // reopen the commitment lease (set_deadline)
                reassignments += 1;
                // NB: `generation` is NOT bumped — that is the whole point (reuse, not re-field).
                continue;
            }
            ReconcileAction::Retire { reason, withdraw, .. } => {
                if withdraw {
                    queue.withdraw(cur_id);
                } else {
                    queue.release(cur_id);
                }
                // A clean Resolved with the budget remaining + no sibling: the squad is done — report it as
                // engaged (it reached + cleared at least one objective). Otherwise this is the pre-reassign
                // churn: re-field a fresh generation (bump the counter) and try to re-claim next tick.
                if reason == RetireReason::Resolved && reassignments > 0 {
                    return ChurnOutcome::Reassigned { from_gen: generation, reassignments, engage_tick: tick };
                }
                generation += 1;
                claimed_id = None;
                phase = V1Phase::Forming;
                engaged_once = false;
                continue;
            }
            ReconcileAction::KeepRefreshLease => {
                deadline = tick + COMMITMENT_BUDGET;
            }
            ReconcileAction::Keep => {}
        }

        // ── ACCEPTANCE: the squad has REASSIGNED at least once AND is now engaging the new objective in its
        //    room (it followed the threat to the neighbour, reusing its bodies — no Generation churn). ──
        if reassignments > 0 && in_target_room && engaged_once && has_focus {
            return ChurnOutcome::Reassigned { from_gen: generation, reassignments, engage_tick: tick };
        }
    }

    // Budget elapsed. If the squad reassigned but the run ended mid-flight, still report the reuse (no
    // churn) — `from_gen == 0` is the proof the bodies were reused. Otherwise classify the churn.
    if reassignments > 0 {
        ChurnOutcome::Reassigned { from_gen: generation, reassignments, engage_tick: s.budget_ticks }
    } else if phase == V1Phase::Forming {
        ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present: 0 }
    } else {
        ChurnOutcome::LapsedInTravel { generations: generation }
    }
}

/// The defender hold-station signal (mirrors the manager's `is_defend && in_target_room && !has_focus`).
fn is_defend_holding(in_target_room: bool, has_focus: bool) -> bool {
    in_target_room && !has_focus
}

// ═══ ADR 0027 P0: run_offense_flow — the OFFENSE production layer, sim-able end-to-end ════════════════════
//
// `run_v1_flow` proves the DEFENSE production chain (observe → threats → emit_defense → queue → reconcile).
// This driver brings the OFFENSE production layer in too (ADR 0027 P0 line 326-328): an offense CANDIDATE
// (room + source + defense) flows through the SAME two pure decisions the live `war.rs::run_offense_evaluation`
// makes — the source→(DoctrineObjective, ObjectiveKind class, priority) MAP and the WINNABILITY/ROI gate (the
// real `decision::doctrine::plan_engagement` honoring the oracle's unwinnable defer) — into the objective
// queue, where ONE squad claims it, fields, travels, and engages via the shared `reconcile` kernel. So
// offense production is offline-provable: a winnable candidate yields an engaged squad; an unwinnable one is
// gated out (no objective, no squad). Pure + deterministic.

/// An offense candidate the toy `run_offense_evaluation` produced — the bot-agnostic slice of `war.rs`'s
/// `AttackCandidate` the production decision needs.
#[derive(Clone, Debug)]
pub struct OffenseCandidate {
    /// The target room (toy grid coords; Chebyshev distance from home).
    pub room: V1Room,
    /// The bot-agnostic engagement objective (the source→doctrine map: a level-0 core is
    /// `KillImmuneStructure`; an attack flag is `ClearCreeps`; a player remote is `RaidCreeps`).
    pub objective: screeps_combat_decision::doctrine::DoctrineObjective,
    /// Whether the doctrine HONORS the oracle's unwinnable verdict (a gated core/raid defers a hopeless room;
    /// an operator flag always-fields). Mirrors `Doctrine::honor_verdict`.
    pub honor_verdict: bool,
    /// The scouted defense the winnability gate judges (towers / objective hits / enemy dps). An undefended
    /// level-0 core is `DefenseProfile::default()` with `objective_hits` set.
    pub defense: screeps_combat_decision::force_sizing::DefenseProfile,
    /// The EV upside (the candidate score scaled) the optimizer maximizes against.
    pub target_value: f32,
}

/// The offense-flow scenario: where home is, the candidate(s) the scan produced, the per-member spawn energy
/// + the on-site window the winnability gate sizes against, and the lifecycle timing.
#[derive(Clone, Debug)]
pub struct OffenseFlowScenario {
    pub home: V1Room,
    pub candidates: Vec<OffenseCandidate>,
    /// Per-member spawn energy (the optimizer sizes each member to this).
    pub member_energy: u32,
    /// On-site window (ticks) the candidate has to deliver its kill (`CREEP_LIFE_TIME − spawn − travel`).
    pub onsite_window: u32,
    pub scan_period: u32,
    pub objective_ttl: u32,
    pub form_ticks: u32,
    pub budget_ticks: u32,
}

/// The PURE offense production decision (ADR 0027 P0): a candidate → an objective to queue, OR `None`
/// (the winnability/ROI gate defers a hopeless room). Mirrors `war.rs::run_offense_evaluation`'s
/// candidate→objective map + the `plan_engagement` winnability gate, in one sim-able place: a gated
/// doctrine (`honor_verdict`) commits ONLY when `optimize_composition` returns a comp (EV-positive +
/// winnable); an always-field doctrine commits regardless. Returns `(priority, member_count)` for the
/// objective when fielded. Deterministic (the decision crate's optimizer is bit-deterministic).
fn offense_candidate_to_objective(c: &OffenseCandidate, member_energy: u32, onsite_window: u32) -> Option<f32> {
    use screeps_combat_decision::composition::{optimize_composition, CompositionParams};
    use screeps_combat_decision::doctrine::{DoctrineObjective, EnemyCoordination};

    // The source→priority map (a slice of war.rs's mapping): a core is MEDIUM, a flag HIGH, a raid LOW.
    let priority = match c.objective {
        DoctrineObjective::KillImmuneStructure | DoctrineObjective::DismantleStructure => 50.0,
        DoctrineObjective::ClearCreeps => 75.0,
        _ => 25.0,
    };

    // The WINNABILITY/ROI gate: run the SAME EV optimizer war.rs's `plan_engagement` runs. A gated doctrine
    // (`honor_verdict`) defers (`None`) a hopeless / negative-EV room; an always-field doctrine fields the
    // best regardless. The enemy creep force is folded into the defense profile's `enemy_dps`.
    let comp = optimize_composition(
        c.objective,
        &c.defense,
        None,
        c.target_value,
        onsite_window,
        EnemyCoordination::Coordinated,
        0.0,
        c.honor_verdict,
        &CompositionParams { member_energy, ..Default::default() },
    );
    comp.map(|_| priority)
}

/// Drive the OFFENSE production layer end-to-end + deterministically (ADR 0027 P0): candidate(s) →
/// `offense_candidate_to_objective` (source map + winnability gate) → queue → ONE squad claims, forms,
/// travels, engages via the shared `reconcile` kernel. Returns the [`ChurnOutcome`]: `DeployedAndEngaged`
/// when a winnable candidate yields an engaged squad; `ChurnedNeverDeployed` (generation 0) when EVERY
/// candidate is gated out (no objective ever queued → nothing to field).
pub fn run_offense_flow(s: &OffenseFlowScenario) -> ChurnOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot};

    let mut queue = V1Queue::default();
    let mut generation: u32 = 0;

    let mut claimed_id: Option<u32> = None;
    let mut phase = V1Phase::Forming;
    let mut pos: V1Room = s.home;
    let mut form_done_at: u32 = s.form_ticks;
    // Never set true (the InRoom branch returns DeployedAndEngaged immediately) — kept mutable only for the
    // re-field reset symmetry below; the snapshot reads it as the always-forming/traveling pre-engage state.
    #[allow(unused_assignments)]
    let mut engaged_once = false;
    let mut deadline: u32 = COMMITMENT_BUDGET;
    let mut gen_start: u32 = 0;
    let mut travel_start: u32 = 0;
    let mut emitted_any = false;

    for tick in 0..s.budget_ticks {
        // ── OFFENSE SCAN: map each candidate through the production decision (source map + winnability gate)
        //    and upsert the surviving objectives. A gated, hopeless candidate yields nothing (deferred). ──
        if tick % s.scan_period == 0 {
            for c in &s.candidates {
                if let Some(priority) = offense_candidate_to_objective(c, s.member_energy, s.onsite_window) {
                    queue.request(c.room, priority, tick, s.objective_ttl);
                    emitted_any = true;
                }
            }
        }
        queue.expire(tick);

        // ── Claim: an unclaimed squad claims the best objective. ──
        if claimed_id.is_none() {
            if let Some(id) = queue.best_unclaimed_excluding(u32::MAX) {
                queue.claim(id);
                claimed_id = Some(id);
                phase = V1Phase::Forming;
                form_done_at = tick + s.form_ticks;
                pos = s.home;
                engaged_once = false;
                deadline = tick + COMMITMENT_BUDGET;
                gen_start = tick;
            }
        }

        let Some(cur_id) = claimed_id else {
            continue;
        };
        let obj = queue.get(cur_id).copied();
        let objective_gone = obj.is_none();
        let target_room = obj.map(|o| o.room);

        // ── Phase progression: form → travel → in-room → engage. ──
        // The squad ENGAGES (exits the driver) on reaching the target room, so `in_target_room` stays false
        // at the snapshot below (a pre-engage forming/traveling state) — the offense flow isolates the
        // production→field→reach decision, not the in-room fight.
        let in_target_room = false;
        let mut traveling = false;
        let mut travel_progress = false;
        if let Some(target) = target_room {
            match phase {
                V1Phase::Forming => {
                    if tick >= form_done_at {
                        phase = if pos == target { V1Phase::InRoom } else { V1Phase::Traveling };
                        travel_start = tick;
                    }
                }
                V1Phase::Traveling => {
                    traveling = true;
                    let before = v1_dist(pos, target);
                    pos = (pos.0 + (target.0 - pos.0).signum(), pos.1 + (target.1 - pos.1).signum());
                    travel_progress = v1_dist(pos, target) < before;
                    if pos == target {
                        phase = V1Phase::InRoom;
                    }
                }
                V1Phase::InRoom => {
                    // Arrived + engaging the offense target — the production layer drove a squad to the kill.
                    return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                }
            }
        }

        let forming = phase == V1Phase::Forming && tick < form_done_at;
        let snapshot = ReconcileSnapshot {
            objective_gone,
            duplicate: false,
            is_defend: false,
            deadline_lapsed: tick >= deadline,
            wiped: false,
            has_focus: in_target_room,
            engaged_once,
            in_target_room,
            has_members: true,
            forming,
            forming_progress: forming,
            forming_in_flight: forming,
            forming_budget_remaining: tick.saturating_sub(gen_start) < MAX_FORMING_BUDGET,
            traveling,
            travel_progress,
            travel_budget_remaining: tick.saturating_sub(travel_start) < MAX_TRAVEL_BUDGET,
            holding_station: false,
            reassign_available: false, // offense reassign is v1.2+; this driver isolates production→engage
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason, withdraw, .. } => {
                if withdraw {
                    queue.withdraw(cur_id);
                } else {
                    queue.release(cur_id);
                }
                let _ = reason;
                generation += 1;
                claimed_id = None;
                phase = V1Phase::Forming;
                engaged_once = false;
                continue;
            }
            ReconcileAction::KeepRefreshLease => deadline = tick + COMMITMENT_BUDGET,
            ReconcileAction::Keep => {}
            ReconcileAction::Reassign { .. } => unreachable!("offense flow never feeds reassign_available=true"),
        }
    }

    // Budget elapsed. If nothing was ever emitted, EVERY candidate was gated out (the deferred-hopeless case).
    let _ = emitted_any;
    ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present: 0 }
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
        let target = ChurnTarget { travel_ticks: 30, uncontested: false, empty_dtos_on_arrival_ticks: 0, ..Default::default() };
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
        let target = ChurnTarget { travel_ticks: 500, uncontested: true, empty_dtos_on_arrival_ticks: 0, ..Default::default() };
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
        let target = ChurnTarget { travel_ticks: 20, uncontested: true, empty_dtos_on_arrival_ticks: 600, ..Default::default() };
        let out = run_lifecycle_churn(&scenario, &target);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "an arrived squad must force a DTO re-read + engage, not lapse on empty DTOs, got {out:?}"
        );
    }

    #[test]
    fn lifecycle_churn_is_deterministic() {
        let target = ChurnTarget { travel_ticks: 30, uncontested: false, empty_dtos_on_arrival_ticks: 0, ..Default::default() };
        let s = contended(oversized_defense_comp());
        assert_eq!(run_lifecycle_churn(&s, &target), run_lifecycle_churn(&s, &target));
    }

    /// An easily-fielded single-slot scenario for the B1/B2 lifecycle repros (no spawn contention — the bug
    /// under test is the LATCH / GARRISON wiring, not the forming plateau).
    fn easy_single_slot() -> ColonyFormingScenario {
        let comp = assemble_force(&RequiredForce { immune_struct_parts: 4, ..Default::default() }, 3000)
            .expect("a single-slot core-killer");
        ColonyFormingScenario {
            composition: comp,
            homes: vec![Home { energy_capacity: 5300, income: 300, start_energy: 3000 }],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 87.5,
            per_member_cap: 3000,
            budget_ticks: 2000,
            member_ttl: 1500,
            renew: false,
        }
    }

    /// BUG B1 (engaged-en-route latch): a squad whose VISIBLE target room has a hostile latches
    /// `engaged_once` while STILL TRAVELING (a proximity-free focus, no in-room gate). The PERMANENT latch
    /// kills the travel lease (`traveling` needs `!engaged_once`) → the squad FREEZES mid-hop on a long
    /// (> COMMITMENT_BUDGET) approach. The FIX gates the latch on in-room presence so it keeps its lease.
    #[test]
    fn engaged_en_route_latch_freezes_then_fixed_keeps_lease() {
        // PRE-FIX: latch from focus.is_some() regardless of distance (latch_engaged_in_room_only = false).
        let buggy = run_lifecycle_churn(
            &easy_single_slot(),
            &ChurnTarget {
                travel_ticks: 600, // > COMMITMENT_BUDGET (400) so the lapse is exercised mid-hop
                uncontested: true, // releases the gate quickly so we are squarely in TRAVEL
                target_visible_with_hostile_en_route: true,
                latch_engaged_in_room_only: false,
                ..Default::default()
            },
        );
        assert!(
            matches!(buggy, ChurnOutcome::LatchedEnRoute { .. }),
            "pre-fix: latching engaged_once en route kills the travel lease → freeze, got {buggy:?}"
        );
        // FIXED: latch only when in the target room (latch_engaged_in_room_only = true, the default).
        let fixed = run_lifecycle_churn(
            &easy_single_slot(),
            &ChurnTarget {
                travel_ticks: 600,
                uncontested: true,
                target_visible_with_hostile_en_route: true,
                latch_engaged_in_room_only: true,
                ..Default::default()
            },
        );
        assert!(
            matches!(fixed, ChurnOutcome::DeployedAndEngaged { .. }),
            "fixed: gating the latch on in-room presence keeps the travel lease → arrives + engages, got {fixed:?}"
        );
    }

    /// BUG B2 (defender garrison churn): a Defend squad ARRIVES in its clear OWNED room, finds no in-room
    /// focus (the threat roams a neighbour), and — pre-fix (the adapter never supplies `holding_station`) —
    /// GaveUp+RE-FIELDS the SAME defender every lease window → Generation churn. The FIX's `holding_station`
    /// signal garrisons it (one stable generation, no re-field). RED → GREEN.
    #[test]
    fn defender_garrison_churns_then_fixed_holds_station() {
        // PRE-FIX: the adapter does NOT supply holding_station → the focus-less in-room defender past its
        // lease GaveUp → re-field → churn (generations climb over the budget).
        let buggy = run_lifecycle_churn(
            &easy_single_slot(),
            &ChurnTarget {
                travel_ticks: 20,
                uncontested: true, // owned room is clear → quorum gate + no in-room focus on arrival
                is_defend: true,
                garrison_holds: false, // pre-fix: no holding_station signal
                ..Default::default()
            },
        );
        match buggy {
            ChurnOutcome::Garrisoned { generations } => panic!("pre-fix must CHURN, not garrison ({generations} gens)"),
            other => assert!(
                matches!(other, ChurnOutcome::LapsedOnArrival { generations } if generations >= 1),
                "pre-fix: the focus-less in-room defender GaveUp + re-fields → churn, got {other:?}"
            ),
        }
        // FIXED: the holding_station signal garrisons the SAME defender — one stable generation, no churn.
        let fixed = run_lifecycle_churn(
            &easy_single_slot(),
            &ChurnTarget {
                travel_ticks: 20,
                uncontested: true,
                is_defend: true,
                garrison_holds: true, // the fix supplies holding_station
                ..Default::default()
            },
        );
        assert!(
            matches!(fixed, ChurnOutcome::Garrisoned { generations: 0 }),
            "fixed: the Defend garrison HOLDS its clear owned room in a single stable generation, got {fixed:?}"
        );
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
            enemy_held_rooms: vec![], // no in-transit attrition in the baseline movement-stall repro
            latch_assault: true,      // the fixed assault-latch (Fix A) by default
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

    // ── BUG A: CONTESTED boundary oscillation (the W4N7 multi-home defender) ──

    /// Scattered homes whose members must CROSS an enemy-held neighbour to reach the rally, plus a CONTESTED
    /// target (so the gather quorum demands the near-full roster). A member stepping into the enemy-held room
    /// DIES + re-spawns → `present` oscillates → the non-latched per-tick gather re-eval never stabilises.
    fn contested_scatter(latch_assault: bool) -> SpatialTravel {
        SpatialTravel {
            // Two homes co-located near the rally room (rx=-4, ry=-5) → both members reach the rally without
            // attrition + form the contested quorum (so the assault DOES commit at least once).
            homes: vec![
                WPos { wx: -4 * 50 + 20, wy: -5 * 50 + 25 }, // W3N4 (near the rally)
                WPos { wx: -4 * 50 + 30, wy: -5 * 50 + 25 }, // W3N4 (near the rally)
            ],
            rally: WPos { wx: -4 * 50 + 25, wy: -5 * 50 + 25 }, // W3N4 staging
            target: WPos { wx: -6 * 50 + 25, wy: -5 * 50 + 25 }, // W5N4 target (two rooms away)
            uncontested: false,                                  // CONTESTED → near-full quorum required
            use_shared_rally: true,                              // the shared-rally travel is in place (ADR 0028)
            // The enemy HOLDS the intermediate room (rx=-5, ry=-5 = W4N4) the ASSAULT must cross from the
            // rally (rx=-4) to the target (rx=-6). A member stepping into it during the assault DIES → present
            // drops 2→1 → the NON-LATCHED per-tick gather re-eval loses quorum → reverts assault→travel → the
            // dead member re-spawns at home → re-gathers → re-assaults → dies again: the in_room<->travel
            // OSCILLATION that never commits. The latch keeps the assault committed through the same attrition.
            enemy_held_rooms: vec![(-5, -5)],
            latch_assault,
        }
    }

    /// RED on the pre-fix (non-latched re-eval), GREEN on Fix A (latch the assault on first quorum + count
    /// in-room members as gathered): a CONTESTED multi-home defender whose members die crossing enemy-held
    /// neighbours OSCILLATES (present thrashes → the per-tick gather never stabilises) and never commits the
    /// assault → `OscillatedNeverGathered`. The latch commits the assault on the first quorum and rides it to
    /// the target despite later attrition → `DeployedAndEngaged`.
    #[test]
    fn contested_scatter_oscillates_then_latch_commits_the_assault() {
        let buggy = run_lifecycle_churn_spatial(&two_home_offense(), &contested_scatter(false));
        assert!(
            matches!(buggy, ChurnOutcome::OscillatedNeverGathered { .. }),
            "pre-fix (non-latched gather re-eval): the contested defender oscillates + never commits, got {buggy:?}"
        );
        let fixed = run_lifecycle_churn_spatial(&two_home_offense(), &contested_scatter(true));
        assert!(
            matches!(fixed, ChurnOutcome::DeployedAndEngaged { .. }),
            "Fix A: latch the assault on the first quorum → commit + reach the target despite attrition, got {fixed:?}"
        );
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

    // ── ADR 0027 v1: the WHOLE-FLOW offline acceptance + reassign repros (the operator's #1 requirement) ──
    //
    // These drive `run_v1_flow` — the multi-objective queue + a MOVING threat + the pure
    // `war_decision::emit_defense` kernel + whole-squad REASSIGN — entirely offline + deterministically,
    // since this class of system has been unreliable to validate on live servers / Docker (memory:
    // war-lifecycle-debug). The acceptance test is a single deterministic run.

    /// A one-owned-room base whose threat walks from the owned room out into a neighbour. `reassign` toggles
    /// the RED (pre-reassign retire→re-field churn) vs GREEN (in-place rebind) arms.
    fn v1_threat_walks_to_neighbour(reassign: bool) -> V1FlowScenario {
        V1FlowScenario {
            owned: vec![((0, 0), 1.0)],                 // one owned room at the origin
            home: (0, 0),                               // the squad forms in the owned room
            // The threat sits in the owned room a couple of scans (the squad forms + engages there), then
            // steps to the neighbour (1,0) and stays — so the owned Secure resolves + a neighbour Secure
            // appears, and the squad must FOLLOW it (reassign).
            threat_path: vec![(0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)],
            scan_period: 2,
            objective_ttl: 6, // a few scans — the stale owned Secure lapses once the squad reassigns off it
            reassign_enabled: reassign,
            form_ticks: 2,
            budget_ticks: 400,
        }
    }

    /// THE END-TO-END ACCEPTANCE TEST (ADR 0027 v1): a defender forms + clears its owned room (the threat
    /// then steps out to a neighbour) → the PURE defense kernel emits `Secure{neighbour}` → the SAME squad
    /// (same generation — NO entity/Generation churn, `from_gen == 0`) REASSIGNS, reaches, and engages the
    /// threat at the neighbour. Single deterministic offline run. RED before the build (no `Reassign`
    /// action), GREEN after.
    #[test]
    fn defender_reassigns_to_follow_a_moving_threat_end_to_end() {
        let out = run_v1_flow(&v1_threat_walks_to_neighbour(true));
        match out {
            ChurnOutcome::Reassigned { from_gen, reassignments, .. } => {
                assert_eq!(from_gen, 0, "the SAME squad followed the threat — NO Generation churn (bodies reused)");
                assert!(reassignments >= 1, "the squad rebound in place to the neighbour Secure at least once");
            }
            other => panic!("the defender must reassign + follow the moving threat end-to-end, got {other:?}"),
        }
        // Deterministic: a single run reproduces.
        assert_eq!(run_v1_flow(&v1_threat_walks_to_neighbour(true)), out, "the v1 flow is deterministic");
    }

    /// REASSIGN-ON-RESOLVE = REUSE (same generation), vs the pre-reassign control that CHURNS (climbing
    /// generations): the threat-follow scenario with reassignment DISABLED retires the freed defender +
    /// re-fields a fresh generation each time the objective moves — `generations` climbs, no reuse.
    #[test]
    fn reassign_reuses_same_generation_vs_control_churns() {
        let reused = run_v1_flow(&v1_threat_walks_to_neighbour(true));
        assert!(
            matches!(reused, ChurnOutcome::Reassigned { from_gen: 0, .. }),
            "reassign reuses the same generation, got {reused:?}"
        );
        let churned = run_v1_flow(&v1_threat_walks_to_neighbour(false));
        match churned {
            ChurnOutcome::Reassigned { .. } => panic!("the control (reassign disabled) must NOT reassign — it churns"),
            ChurnOutcome::ChurnedNeverDeployed { generations, .. } | ChurnOutcome::LapsedInTravel { generations } => {
                assert!(generations >= 1, "the pre-reassign control re-fields a fresh generation (churn), got {generations}");
            }
            other => panic!("the control must churn (climbing generations), got {other:?}"),
        }
    }

    /// REASSIGN-ON-EXPIRE + the NO-SIBLING CONTROL: when the squad's objective vanishes (the threat left
    /// the map → no new Secure emitted → `ObjectiveGone`) AND there is NO sibling, reassignment cannot fire
    /// and the squad falls back to the existing retire (reassign is strictly ADDITIVE). Here the threat
    /// walks out of the owned room then off the map entirely, so the only objective vanishes with no sibling.
    #[test]
    fn no_sibling_falls_back_to_retire_not_reassign() {
        let scenario = V1FlowScenario {
            owned: vec![((0, 0), 1.0)],
            home: (0, 0),
            // Threat in the owned room briefly, then walks FAR off the map (beyond the leash → no Secure
            // emitted) → the claimed objective is the only one + it resolves/vanishes with NO sibling.
            threat_path: vec![(0, 0), (0, 0), (9, 9), (9, 9), (9, 9)],
            scan_period: 2,
            objective_ttl: 6,
            reassign_enabled: true, // enabled, but there is no sibling to reassign TO
            form_ticks: 2,
            budget_ticks: 200,
        };
        let out = run_v1_flow(&scenario);
        assert!(
            !matches!(out, ChurnOutcome::Reassigned { .. }),
            "with no sibling available, the squad must NOT reassign — it falls back to retire, got {out:?}"
        );
    }

    /// ADR 0027 P0: the FULL DEFENSE PRODUCTION CHAIN is sim-able — a MOVING ARMED NEIGHBOUR hostile flows
    /// through observe_neighbours → neighbour_threats → emit_defense → queue → reconcile and produces the
    /// Secure objective chain end-to-end (the squad reassigns to follow). This is the same `run_v1_flow`
    /// acceptance, but it now exercises the LIFTED `observe_neighbours` kernel on the neighbour leg (the
    /// threat walks from the owned room into a neighbour), proving the whole observation LAYER offline.
    #[test]
    fn full_defense_production_chain_drives_secure_via_observe_neighbours() {
        // The threat starts in the owned room, then walks to the neighbour (1,0) — the neighbour leg runs
        // through observe_neighbours (armed Attack body, visible, non-owned, within leash).
        let out = run_v1_flow(&v1_threat_walks_to_neighbour(true));
        match out {
            ChurnOutcome::Reassigned { from_gen, reassignments, .. } => {
                assert_eq!(from_gen, 0, "same squad followed the threat via the lifted observe chain (no churn)");
                assert!(reassignments >= 1, "the squad rebound to the neighbour Secure produced by observe_neighbours");
            }
            other => panic!("the full defense production chain must drive Secure end-to-end, got {other:?}"),
        }
    }

    // ── ADR 0027 P0: run_offense_flow — the offense production layer, sim-able ──

    use screeps_combat_decision::doctrine::DoctrineObjective;
    use screeps_combat_decision::force_sizing::DefenseProfile;

    /// An UNDEFENDED level-0 invader core (a 50k-hit dismantle-immune structure, no towers/defenders) is a
    /// WINNABLE candidate: the production layer maps it to a `KillImmuneStructure` objective, the winnability
    /// gate passes, a squad claims + forms + travels + ENGAGES it. The offense production chain drives a kill.
    #[test]
    fn offense_flow_winnable_core_fields_and_engages() {
        let s = OffenseFlowScenario {
            home: (0, 0),
            candidates: vec![OffenseCandidate {
                room: (2, 0),
                objective: DoctrineObjective::KillImmuneStructure,
                honor_verdict: true, // a gated doctrine — must pass the winnability gate to field
                defense: DefenseProfile { objective_hits: 50_000, ..Default::default() },
                target_value: 1_000_000.0,
            }],
            member_energy: 5600,
            onsite_window: 1000,
            scan_period: 2,
            objective_ttl: 100,
            form_ticks: 4,
            budget_ticks: 400,
        };
        let out = run_offense_flow(&s);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { generations: 0, .. }),
            "a winnable undefended core must field + engage end-to-end, got {out:?}"
        );
        assert_eq!(run_offense_flow(&s), out, "the offense flow is deterministic");
    }

    /// A SAFE-MODED room is UNWINNABLE (zero damage possible): the gated doctrine's winnability gate DEFERS
    /// it — no objective is ever queued, so no squad is fielded. The production layer never feeds a squad to
    /// a hopeless room (the ROI/winnability gate working in the sim).
    #[test]
    fn offense_flow_unwinnable_safe_mode_is_gated_out() {
        let s = OffenseFlowScenario {
            home: (0, 0),
            candidates: vec![OffenseCandidate {
                room: (2, 0),
                objective: DoctrineObjective::KillImmuneStructure,
                honor_verdict: true,
                // Safe mode → zero damage possible → unwinnable → the gate defers (no comp).
                defense: DefenseProfile { objective_hits: 50_000, safe_mode: true, ..Default::default() },
                target_value: 1_000_000.0,
            }],
            member_energy: 5600,
            onsite_window: 1000,
            scan_period: 2,
            objective_ttl: 100,
            form_ticks: 4,
            budget_ticks: 200,
        };
        let out = run_offense_flow(&s);
        assert!(
            matches!(out, ChurnOutcome::ChurnedNeverDeployed { generations: 0, .. }),
            "an unwinnable safe-moded room must be gated out (no squad fielded), got {out:?}"
        );
        assert_eq!(run_offense_flow(&s), out, "the offense flow is deterministic");
    }
}
