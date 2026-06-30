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
    /// ADR 0035 H1 (the VACUOUS-INTEL commit). At COMMIT time the room's combat DTOs are EMPTY (an
    /// empty-Cached snapshot — no towers were VISIBLE last scout), so during Forming/Travel the squad is
    /// classified UNCONTESTED off this empty view (`uncontested_intel = false` ⇒ but the boolean logic still
    /// yields `uncontested=true` when no hostiles/towers + reliable; D3's fix makes the manager pass
    /// `uncontested_intel=false` ⇒ `uncontested=false`). The harness threads `uncontested` from the EMPTY
    /// commit view while traveling and FLIPS to the REAL view (`!arrival_has_towers`) on arrival.
    pub commit_intel_empty: bool,
    /// ADR 0035 H1 (cannot-win-on-arrival). On arrival the room's LIVE DTOs reveal REAL towers, so the
    /// in-room P(win) = LOSE: `engaged_once` latches (a member is in-room + engages) but
    /// `present_force_wins_or_stalls = false` ⇒ `retreated_from_contact = true` ⇒ the squad retreats.
    /// Combined with `commit_intel_empty` this is the W4N5 vacuous-commit cascade.
    pub arrival_has_towers: bool,
    /// ADR 0035 D3/D4/D5 TOGGLE (RED→GREEN). `false` reproduces the PRE-FIX bot: the uncontested classifier
    /// trusts the empty commit view (`uncontested=true` ⇒ trickle into the towers), the reconcile kernel is
    /// fed `retreated_from_contact=false` (a retreat mis-resolves as a clean clear → withdraw → re-field),
    /// and the producer does NOT consult the backoff (instant re-upsert) ⇒ `LapsedOnVacuousCommit`,
    /// generations climbing. `true` is the FIXED bot: D3 classifies contested off the vacuous view (stage
    /// short), D4 feeds the real `retreated_from_contact` (abandon with backoff, NOT resolve), D5 suppresses
    /// the re-field via `is_unwinnable_now` ⇒ `AbandonedOnContact`, generations stable.
    pub abandon_fixes_enabled: bool,
    /// ADR 0035 D4 (E1 false-abandon fix) — the WINNABLE-RETREAT case. On arrival the squad is IN the
    /// target room and `decide_squad` would enter the Retreating STATE (a focus-fired member dipped to
    /// critical HP / the squad-average is low), BUT the real in-room `present_force_wins_or_stalls = TRUE`:
    /// the present force still WINS-OR-STALLS — it is NOT losing. The bot stamps the GENUINE lose verdict
    /// `lost_in_room = engaged_once && in_room_any && !present_force_wins_or_stalls`, so
    /// `retreated_from_contact = FALSE` even though the squad is Retreating. Pre-fix the bot derived
    /// `retreated_from_contact` from `ctx.state == Retreating` (a SUPERSET of the lose verdict) → it would
    /// have read TRUE → the kernel would ABANDON a winning room mid-fight (the false-abandon). With the fix
    /// the kernel sees `retreated_from_contact = false` → it does NOT abandon (the squad holds/wins →
    /// `DeployedAndEngaged`). Models the divergence the kernel unit test
    /// `winnable_fight_with_critical_member_retreats_but_does_not_lose_so_does_not_abandon` proves.
    pub winnable_retreat_in_room: bool,
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
            commit_intel_empty: false,
            arrival_has_towers: false,
            abandon_fixes_enabled: true, // default to the FIXED bot behaviour
            winnable_retreat_in_room: false,
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
            declaiming: false,      // ADR 0027 v1.1 P2 declaim is exercised by `run_declaim_flow`, not here
            reassign_available: false, // ADR 0027 v1 reassign is exercised by `run_v1_flow`, not here
            retreated_from_contact: false, // ADR 0035 D4 — not exercised by this driver
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
    /// ADR 0035 (RED — the vacuous-intel engage cascade). The squad COMMITTED to a towered room on an EMPTY
    /// commit view (classified uncontested → trickled in), REACHED it, found real towers (P(win)=LOSE), and
    /// RETREATED — but the pre-fix reconcile MIS-RESOLVED the retreat as a clean clear → withdraw (no
    /// backoff) → the producer instantly re-upserted (no `is_unwinnable_now` consult) → Phase C re-fielded
    /// the same squad on the same vacuous intel. `generations > 1` is the oscillation (false-resolve →
    /// re-upsert → re-field). Fixed by D3 (contested classification) + D4 (abandon-not-resolve) + D5
    /// (producer backoff consult).
    LapsedOnVacuousCommit { generations: u32 },
    /// ADR 0035 (GREEN — abandon-on-contact). With the fixes, the reached-and-losing squad ABANDONS the
    /// objective: the reconcile kernel returns `GaveUp + mark_unwinnable` (NOT a clean resolve), the producer
    /// suppresses re-upsert via `is_unwinnable_now`, and NO re-field happens within the backoff window —
    /// `generations` STABLE (the room sits in backoff, to be re-scouted when it expires). The de-commit is
    /// clean and bounded, ending the oscillation.
    AbandonedOnContact { generations: u32 },
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

    // ── ADR 0035 H1/D3 — the COMMIT-TIME uncontested view (vacuous intel). During Forming/Travel the squad
    // commits on the (possibly EMPTY) cached snapshot; the manager's D3 fix decides `uncontested` from the
    // REAL-intel notion (`uncontested_intel = !hostiles.is_empty() || !structures.is_empty() || LiveVisible`).
    // For a `commit_intel_empty` room that is empty-Cached (NOT live-visible), `uncontested_intel = false`,
    // so the FIXED bot classifies it CONTESTED (stages short + masses) even though no hostiles/towers are in
    // the view. The PRE-FIX bot (the bug) used `is_reliable()` → uncontested=true → trickled in. We model the
    // RED/GREEN split via `abandon_fixes_enabled`: fixed ⇒ contested on the vacuous view; pre-fix ⇒ honor the
    // declared `target.uncontested` (the empty view looked clear). On ARRIVAL the view flips to LIVE — see the
    // `Arrived` branch (`arrival_has_towers`).
    let commit_uncontested = if target.commit_intel_empty {
        if target.abandon_fixes_enabled {
            false // D3: empty-Cached towered room is NOT real intel → contested → stage short, mass
        } else {
            target.uncontested // pre-fix: trusted the empty view (is_reliable) → trickled in
        }
    } else {
        target.uncontested
    };
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
        // ADR 0035 D4 — the in-room LOST-FIGHT signal fed to the reconcile snapshot (the inverse of
        // `present_force_wins_or_stalls` over the REAL arrival DTOs). Set only in the `Arrived` branch on a
        // towered room; false otherwise (en route / clear room).
        let mut retreated_from_contact = false;
        match phase {
            Phase::Forming => {
                // The REAL rally gate over the present roster (full roster, or min-viable quorum if the
                // target is proven-uncontested). ADR 0035 D3: the COMMIT-time view (`commit_uncontested`)
                // drives the gate — for a `commit_intel_empty` room the FIXED bot classifies it CONTESTED
                // (holds for the full roster + stages short) rather than trickling a sub-roster into towers.
                let positions: Vec<Option<Position>> = vec![Some(dummy_home_pos()); present];
                if rally::ready_to_depart_gate(&positions, n_slots, commit_uncontested) {
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
                    // ADR 0035 H3 — ARRIVAL flips to the LIVE view. The room's real DTOs are now readable.
                    if target.winnable_retreat_in_room {
                        // ADR 0035 D4 (E1 false-abandon REGRESSION GUARD) — the WINNABLE-RETREAT case. A
                        // member is in-room and engages (`engaged_once` LATCHES), and `decide_squad` enters
                        // the Retreating STATE (a focus-fired member dipped to critical HP). The DANGER: the
                        // pre-fix bot derived `retreated_from_contact` from `ctx.state == Retreating` (a
                        // SUPERSET of the lose verdict), so it would feed the kernel TRUE here → the kernel
                        // would ABANDON a WINNABLE room mid-fight. The FIX carries the GENUINE lose verdict
                        // `lost_in_room = engaged_once && in_room_any && !present_force_wins_or_stalls`; here
                        // `present_force_wins_or_stalls = TRUE` (the present force still wins-or-stalls) ⇒
                        // `retreated_from_contact = FALSE`. So the kernel does NOT abandon — the squad holds
                        // + wins. Modeled as DeployedAndEngaged (it engaged and is winning, not abandoned).
                        // NOTE: `retreated_from_contact` stays FALSE (the lose-verdict signal, NOT the
                        // Retreating state) — that is the whole point; we short-circuit to the winning
                        // terminal rather than feeding the kernel an abandon signal.
                        return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                    } else if target.arrival_has_towers {
                        // CANNOT-WIN-ON-ARRIVAL (the W4N5 case). A member is in-room and engages, so
                        // `engaged_once` LATCHES — but the LIVE assessment over the REAL towers is
                        // `present_force_wins_or_stalls = false`, so the squad RETREATS. This is the ground
                        // truth the commit was missing. Set `retreated_from_contact` and FALL THROUGH to the
                        // reconcile kernel (NO short-circuit DeployedAndEngaged) — the kernel decides
                        // abandon-vs-mis-resolve. PRE-FIX (`abandon_fixes_enabled=false`) we feed the kernel
                        // `retreated_from_contact=false`, so it mis-resolves the retreat as a clean clear →
                        // withdraw → re-field (LapsedOnVacuousCommit). FIXED we feed the real signal → the
                        // kernel returns GaveUp+mark_unwinnable (AbandonedOnContact).
                        engaged_once = true;
                        retreated_from_contact = target.abandon_fixes_enabled;
                        // (when fixes disabled, the kernel sees a focus-less in-room engaged squad = a "clear")
                    } else {
                        // OFFENSE, clear room: once the room DTOs are readable a focus is computed and the
                        // squad ENGAGES — the deep bug is absent.
                        return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                    }
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
            declaiming: false, // ADR 0027 v1.1 P2 declaim is exercised by `run_declaim_flow`, not here
            reassign_available: false, // ADR 0027 v1 reassign is exercised by `run_v1_flow`, not here
            // ADR 0035 D4 — the in-room LOST-FIGHT signal. Set only on arrival at a towered room with the
            // fixes enabled (the kernel then ABANDONS-with-backoff instead of mis-resolving the retreat).
            retreated_from_contact,
        };
        // BUG B2 (fixed state): a defender that has GARRISONED its owned room (in-room, focus-less) and held
        // its lease until the budget elapsed without churning — a single stable generation. Detected when the
        // garrison reaches the final tick still in-room (no re-field happened). Checked before reconcile so
        // the stable-hold case reports `Garrisoned` rather than running the loop to the bottom-of-fn classify.
        if holding_station && tick + 1 >= s.budget_ticks {
            return ChurnOutcome::Garrisoned { generations: generation };
        }
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, mark_unwinnable, .. } => {
                // ADR 0035 D4/D5 (GREEN — ABANDON-ON-CONTACT). The reached squad engaged a towered room it
                // cannot win and is retreating; the kernel returned GaveUp + mark_unwinnable (NOT a clean
                // resolve). The PRODUCER then SUPPRESSES the re-upsert via `is_unwinnable_now` (D5) — so NO
                // re-field happens within the backoff. The de-commit is clean + bounded: report
                // `AbandonedOnContact` with the generation count STABLE (the oscillation is ended).
                if engaged_once && in_target_room && retreated_from_contact && mark_unwinnable {
                    // The producer (re-field path) is SUPPRESSED by `is_unwinnable_now` (D5) — modeled by
                    // returning here with the generation count STABLE (no re-field within the backoff window).
                    return ChurnOutcome::AbandonedOnContact { generations: generation };
                }
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
            ReconcileAction::Retire { reason: RetireReason::Resolved, .. } => {
                // ADR 0035 (RED — the VACUOUS-INTEL spiral). PRE-FIX (`abandon_fixes_enabled=false`), a squad
                // that REACHED a towered room, engaged, and is RETREATING presents to the kernel as
                // engaged_once + in_room + focus-less with `retreated_from_contact=false` (the manager never
                // computed it) → the kernel MIS-RESOLVES the retreat as a CLEAN CLEAR → withdraw. The producer
                // (no D5 backoff consult) RE-UPSERTS the objective → Phase C RE-FIELDS the same squad on the
                // same vacuous intel → it reaches, retreats, mis-resolves again. The generation count CLIMBS
                // — the reach↔retreat oscillation. Model the re-upsert + re-field: loop back to Forming,
                // bumping the generation. After several cycles (the oscillation is established) report
                // `LapsedOnVacuousCommit`. (A GENUINE clear — a clear room, `arrival_has_towers=false` —
                // returns `DeployedAndEngaged` from the Arrived branch BEFORE reaching here, so this arm only
                // fires on the false-resolve of a towered-room retreat.)
                if engaged_once && in_target_room && target.arrival_has_towers {
                    generation += 1;
                    if generation >= 3 {
                        return ChurnOutcome::LapsedOnVacuousCommit { generations: generation };
                    }
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
                // A genuine resolve with no commit-cascade context (shouldn't occur in this driver) ends as
                // never-deployed.
                return ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present };
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

// ═══ ADR 0034 Phase 1: run_lifecycle_churn_EXTENDED — the PRODUCTION-PATH far-home stall (RC-3/4/8/10) ═══
//
// `run_lifecycle_churn` models travel as a pure TICK COUNTER (`Phase::Traveling { arrives_at }`); the
// SPATIAL driver (`run_lifecycle_churn_spatial`) positions members but over a TOY `WPos` grid with a GIVEN
// rally + a hand-rolled stepper — neither calls the PRODUCTION geometry. So the far-home stall (the rally is
// mis-computed for a far/cross-quadrant scatter, then a wrong/blocked rally turns into a SILENT PERMANENT
// stall) is invisible offline (sim gap G3/G5/G6).
//
// This EXTENDED driver folds spatial positioning into the production path and exercises the REAL geometry:
//   (a) per-member REAL `screeps::Position`s seeded at distinct cross-quadrant homes (not a `WPos` grid);
//   (b) the PRODUCTION rally — `cohesion::centroid` → `rally::shared_rally_point_for_members` (the SAME calls
//       `squad_manager` makes), re-derived FRESH each tick over the present members (no stored field);
//   (c) a SOLO step: each member steps one world-tile toward the rally per tick;
//   (d) per-tick `rally::gather_quorum_met` + the FIX-A assault latch (committed once the quorum first fires);
//   (e) `Arrived` is GATED on `gathered` (NOT a bare tick counter) — the assault advances rally→target only
//       after the gather quorum, and arrival = a member actually IN the target room;
//   (f) BOTH the min-over-members distance AND per-member distances are tracked, so RC-4 (one stuck member
//       pins the min / one moving lead masks a stuck bulk) and RC-8 (the lease lapses on a pinned min) are
//       assertable.
//
// THE BLOCKED-PATH MODEL (RC-3, sim gap G5): a `blocked_rooms` set the solo stepper checks. A member whose
// next world-step would ENTER a blocked room CANNOT advance (the live NO_PATH / impassable-terrain / hostile
// room a `MoveToRoom::move_to(rally)` silently retries). Pre-fix the manager has NO member-side movement
// feedback (RC-3): a blocked member sits forever, the min-distance never closes (RC-4/RC-8), and the squad
// gives up only when the coarse 1000-tick travel budget lapses — a SILENT permanent stall masquerading as a
// clean give-up.
//
// THE FIXES (D4/D5/D8), modeled by the `escalate` + `majority_progress` + `stall_window` flags:
//   • D4 (member-side failure detection + escalation, RC-3): a blocked member surfaces a `Blocked` signal;
//     after a BOUNDED stall the driver ESCALATES by RE-ASSESSING that member OUT of the gather quorum (it is
//     dropped from the quorum denominator) so the REACHABLE subset masses and the contested quorum fires,
//     rather than waiting forever on a member that cannot path. The rally is NOT moved — the blocked member
//     is excluded. No silent retry loop.
//   • D5 (per-member + MAJORITY travel progress, RC-4/RC-8): the travel lease refreshes while a MAJORITY of
//     present members are closing (per-member distances), NOT while the single closest is (the min). One
//     straggler can't pin the squad "stalled"; one moving lead can't mask a stuck bulk.
//   • D8 (tighter per-member stall window, RC-8): a SOLO-travel stall of `STALL_WINDOW` ticks with ZERO
//     members EVER gathered ESCALATES (D4) BEFORE the coarse `MAX_TRAVEL_BUDGET` — a wrong/blocked rally is
//     caught fast, not 1000 ticks later. The absolute `MAX_TRAVEL_BUDGET` bound is preserved as the backstop.

/// ADR 0034 D8: the TIGHTER per-member solo-travel stall window — ticks of solo travel with ZERO members ever
/// gathered after which the driver ESCALATES (D4), well before the coarse `MAX_TRAVEL_BUDGET` (1000). In the
/// 50–150 band per the ADR (RC-8 fast-catch). Ephemeral runtime state (a per-objective tracker like
/// `assault_latched`) — NOT serialized, no `WORLD_FORMAT_VERSION` bump.
pub const SOLO_TRAVEL_STALL_WINDOW: u32 = 100;

/// The PRODUCTION-PATH far-home travel scenario (ADR 0034 Phase 1). Real cross-quadrant member homes, the
/// production rally geometry, an optional blocked-room set, and the D4/D5/D8 fix toggles (RED with them off,
/// GREEN with them on).
#[derive(Clone, Debug)]
pub struct ExtendedTravel {
    /// Each member's home `Position` (real room names, cross-quadrant). One per composition slot.
    pub homes: Vec<Position>,
    /// The assault TARGET position (a room beyond the rally).
    pub target: Position,
    /// Proven-uncontested target → the gather quorum may trickle; contested → the near-full roster.
    pub uncontested: bool,
    /// RC-3 BLOCKED-PATH MODEL (sim gap G5): rooms a member CANNOT enter (impassable terrain / hostile /
    /// NO_PATH). A member whose next world-step toward the rally would land in one of these rooms does NOT
    /// advance — it surfaces a `Blocked` signal. Empty ⇒ the clean path (S1 plain far-home stall).
    pub blocked_rooms: Vec<screeps::RoomName>,
    /// D4 FIX toggle (RC-3 — member-side movement-failure detection + escalation): `false` reproduces the
    /// pre-fix bot (a blocked member silently retries `MoveTo(rally)` forever — no feedback); `true` is the
    /// fixed bot (a blocked member surfaces `Blocked`; after a bounded stall the driver re-assesses it OUT of
    /// the gather quorum so the reachable subset proceeds — the rally is unchanged). Default `false` (RED).
    pub escalate_on_block: bool,
    /// D5 FIX toggle (RC-4/RC-8 — majority travel progress): `false` reproduces the pre-fix MIN-over-members
    /// progress signal (one stuck member pins it / one moving lead masks a stuck bulk); `true` refreshes the
    /// travel lease while a MAJORITY of present members are closing. Default `false` (RED).
    pub majority_progress: bool,
    /// D8 FIX toggle (RC-8 — tighter stall window): `false` keeps ONLY the coarse `MAX_TRAVEL_BUDGET` (a
    /// wrong/blocked rally lapses ~1000 ticks later, looking like a clean give-up); `true` adds the tighter
    /// `SOLO_TRAVEL_STALL_WINDOW` that escalates (D4) fast. Requires `escalate_on_block` to do anything (the
    /// stall window TRIGGERS the D4 escalation). Default `false` (RED).
    pub tight_stall_window: bool,
    // ── ADR 0034 Phase 2 (RC-5/RC-6/RC-7): the lifetime-attrition axis + renew-in-transit (sim gap G4) ──
    /// RC-6 — is the RALLY ROOM within range of a friendly spawn (so a held-at-rally member can RENEW there)?
    /// `false` (default) = a forward rally with NO spawn — a member that arrives + holds there just ages out.
    /// `true` models the D6c renewable-rally bias (a staging room near a friendly spawn). When set, the renew
    /// pass (D6b) tops up members that are GATHERED at the rally, not just those holding at home.
    pub rally_spawn: bool,
    /// D6a FIX toggle (RC-7 — pre-departure LIFETIME GATE): `false` reproduces the pre-fix bot (a member is
    /// committed to `MoveTo(rally)` with NO check that its TTL covers the journey + fight — a far/low-TTL
    /// member is sent anyway and arrives below `FIGHT_BUFFER` / ages out mid-travel). `true` is the fixed bot:
    /// before releasing a member it runs `rally::lifetime_sufficient_for_deployment`; an insufficient member
    /// HOLDS at home (D6b renew tops it up) instead of departing doomed. Default `false` (RED).
    pub lifetime_gate: bool,
    /// D6b FIX toggle (RC-5 — RENEW-to-sufficiency while holding): `false` reproduces the FORMING-ONLY renew
    /// (a held/rallying member with low TTL is NOT renewed once `filled >= requested` — RC-5). `true` extends
    /// the renew past the forming gate: a present member with low TTL that is HOLDING at home (or GATHERED at
    /// a `rally_spawn` rally) is topped up toward `rally::RENEW_TARGET_TTL`, energy/lane-bounded. Default
    /// `false` (RED). Requires `member_ttl`/decay to bite (see `ColonyFormingScenario::member_ttl`).
    pub renew_in_transit: bool,
}

impl ExtendedTravel {
    /// The world-coord room of a `Position` as `(rx, ry)` (room-grid coords; Chebyshev room-distance).
    fn room_xy(p: Position) -> (i32, i32) {
        let (wx, wy) = p.world_coords();
        (wx.div_euclid(50), wy.div_euclid(50))
    }
    /// Chebyshev ROOM-distance between two positions — the SAME signal production's `target_dist` /
    /// `travel_progress` use (`room_distance(member_room, target_room)`).
    fn room_dist(a: Position, b: Position) -> u32 {
        let (ax, ay) = Self::room_xy(a);
        let (bx, by) = Self::room_xy(b);
        (ax - bx).unsigned_abs().max((ay - by).unsigned_abs())
    }
    /// One Chebyshev world-tile step from `from` toward `to` (a member's per-tick solo move).
    fn step_toward(from: Position, to: Position) -> Position {
        let (fx, fy) = from.world_coords();
        let (tx, ty) = to.world_coords();
        Position::from_world_coords(fx + (tx - fx).signum(), fy + (ty - fy).signum())
    }
    fn pos_options(members: &[Position]) -> Vec<Option<Position>> {
        members.iter().map(|p| Some(*p)).collect()
    }
}

/// ADR 0034 Phase 1 — drive the FULL bot lifecycle (lease / reconcile / re-field churn + the real rally gate)
/// with PRODUCTION-PATH spatial travel: members spawn at DISTINCT cross-quadrant homes (real `Position`s),
/// the rally is derived FRESH each tick by the production `cohesion::centroid` → `shared_rally_point_for_members`
/// geometry, each member steps SOLO toward it, the gather quorum + FIX-A latch are re-evaluated each tick, and
/// `Arrived` is gated on `gathered`. The reconcile DECISION is the shared `lifecycle::reconcile` kernel — no
/// live/sim drift. Deterministic: same (scenario, travel) → same outcome (integer world-coord math, no float
/// branch, no `HashMap`).
pub fn run_lifecycle_churn_extended(s: &ColonyFormingScenario, travel: &ExtendedTravel) -> ChurnOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot, RetireReason};

    let n_slots = s.composition.slots.len();
    assert_eq!(travel.homes.len(), n_slots, "one home per member slot in the extended spatial model");
    let best_capacity = s.homes.iter().map(|h| h.energy_capacity).max().unwrap_or(0);
    let blocked: BTreeSet<screeps::RoomName> = travel.blocked_rooms.iter().copied().collect();

    let mut generation: u32 = 0;
    let mut max_present: usize = 0;
    let mut max_gathered: usize = 0;
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

    // Spatial member state: each member starts AT its home `Position`; `member_pos[i]` is its live tile.
    let mut member_pos: Vec<Position> = travel.homes.clone();
    let mut departed = false; // the rally gate released (solo travel begins)
    let mut gathered = false; // the gather quorum fired (assault begins)
    // RC-4/RC-8: track the MIN-over-members room-distance (the pre-fix production signal) — used when
    // `majority_progress` is OFF, so RED reproduces the min-pinning.
    let mut prev_min_dist: Option<u32> = None;
    // D5: per-member previous room-distance to the rally (for the majority-closing signal).
    let mut prev_member_dist: Vec<Option<u32>> = vec![None; n_slots];
    // RC-3 / D8: per-member consecutive BLOCKED-tick counter (how long the member has surfaced `Blocked` from
    // `check_movement_failure` with no advance — the tight D8 stall window keys on this). Pre-fix the manager
    // IGNORES it (the silent retry loop, RC-3); D4 reads it to escalate.
    let mut block_streak: Vec<u32> = vec![0; n_slots];
    // D4 (RC-3) escalation latch: a member whose block streak exceeds the tight window is RE-ASSESSED OUT of
    // the gather quorum — the manager surfaces `Blocked`, concludes the member cannot path to the rally, and
    // proceeds with the REACHABLE subset (the ADR's "re-assess" escalation) rather than waiting on it forever.
    // An ephemeral per-objective decision (no serialized state).
    let mut excluded: Vec<bool> = vec![false; n_slots];

    // ── ADR 0034 Phase 2 (RC-5/RC-6/RC-7): per-member TTL + renew model (sim gap G4). EPHEMERAL runtime
    // state — `ttl[i]` MIRRORS the live `creep.ticks_to_live()` read fresh from the world each tick (NOT a
    // serialized field), and the renew self-heals on reload. A present member's TTL starts at `member_ttl`
    // and decays -1/tick (the engine ages every creep); a renew tops it back up toward `RENEW_TARGET_TTL`.
    let mut ttl: Vec<u32> = vec![0; n_slots];
    // D6a: a member HELD at home by the pre-departure lifetime gate (its TTL can't yet cover journey+fight).
    // It does not advance toward the rally; the D6b renew tops it up; once sufficient it is released.
    let mut held_home: Vec<bool> = vec![false; n_slots];
    // The rally for the lifetime gate's `dist_to_rally`/`dist_to_target` — recomputed once departed; while
    // forming the members sit at home so the gate uses home→rally (re-derived when present changes).

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
                    ttl[id as usize] = s.member_ttl; // a fresh member starts with a full life (RC-5 decay below)
                }
                false
            } else {
                true
            }
        });

        // ── Phase 2 TTL DECAY + AGE-OUT (RC-5): every present member ages -1/tick (the engine). A member whose
        // TTL hits 0 while still rallying/traveling DIES of old age → drops back to unfilled (and re-spawns) —
        // the slow-far-home churn the renew (D6b) prevents. `ttl[i]` here mirrors the live `ticks_to_live()`.
        for i in 0..n_slots {
            if filled[i] {
                ttl[i] = ttl[i].saturating_sub(1);
                if ttl[i] == 0 {
                    filled[i] = false;
                    member_pos[i] = travel.homes[i];
                    held_home[i] = false;
                }
            }
        }

        let present = filled.iter().filter(|f| **f).count();
        max_present = max_present.max(present);
        let has_members = present > 0 || !completing.is_empty() || !syncing.is_empty();

        let any_queued =
            !fielding::slots_to_spawn(&s.composition, &filled, best_capacity, s.per_member_cap, s.combat_priority, MoveProfile::Plains).is_empty();
        let forming_in_flight = !completing.is_empty() || !syncing.is_empty() || any_queued;

        let present_positions: Vec<Position> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();

        let mut in_target_room = false;
        let mut traveling = false;
        let mut travel_progress = false;

        if !departed {
            // FORMING / rally gate over the present roster.
            let positions = ExtendedTravel::pos_options(&present_positions);
            if rally::ready_to_depart_gate(&positions, n_slots, travel.uncontested) {
                departed = true;
                travel_start = tick;
            }
        }

        if departed {
            traveling = true;
            // PRODUCTION RALLY GEOMETRY (item (b)): the manager derives the rally FRESH each tick from the
            // present members' positions via `cohesion::centroid` (inside) + the scatter-robust selector. D4
            // escalation does NOT move this rally — it drops a persistently-blocked member from the gather
            // quorum below, so the reachable subset masses at this same rally instead.
            let present_opts = ExtendedTravel::pos_options(&present_positions);
            let rally =
                screeps_combat_decision::rally::shared_rally_point_for_members(&present_opts, travel.target, travel.uncontested);

            // GATHER DECISION (item (d)): the unified kernel over the CURRENT positions; FIX-A also counts an
            // in-target-room member as gathered.
            let pre_step: Vec<Position> = (0..n_slots).filter(|&i| filled[i]).map(|i| member_pos[i]).collect();
            let has_fighter = !pre_step.is_empty();
            let target_room = travel.target.room_name();
            let in_room_count = (0..n_slots).filter(|&i| filled[i] && member_pos[i].room_name() == target_room).count();
            // D4 RE-ASSESS: a member RE-ASSESSED OUT (`excluded`, persistently blocked from the rally) is
            // dropped from the gather denominator — the manager proceeds with the REACHABLE subset rather
            // than waiting forever on the unreachable one. `effective_slots` = the roster minus the excluded
            // (and the gather positions exclude them too), so the contested quorum can fire over who can mass.
            let excluded_count = excluded.iter().filter(|e| **e).count();
            let effective_slots = n_slots.saturating_sub(excluded_count);
            let reachable: Vec<Position> = (0..n_slots).filter(|&i| filled[i] && !excluded[i]).map(|i| member_pos[i]).collect();
            let quorum_now = rally::gather_quorum_met(
                &ExtendedTravel::pos_options(&reachable),
                rally,
                effective_slots,
                travel.uncontested,
                has_fighter,
                rally::RALLY_GATHER_RADIUS,
            ) || (in_room_count > 0 && has_fighter);
            let gathered_now = rally::members_gathered_at(&ExtendedTravel::pos_options(&pre_step), rally, rally::RALLY_GATHER_RADIUS);
            max_gathered = max_gathered.max(gathered_now);
            // FIX-A LATCH: commit the assault on the FIRST quorum.
            gathered |= quorum_now;
            if prev_gathered && !gathered {
                oscillations += 1;
            }
            prev_gathered = gathered;

            if gathered {
                // ASSAULT: advance toward the target as a bloc; members follow (gated on `gathered`, item (e)).
                for i in 0..n_slots {
                    if filled[i] {
                        // The assault crosses toward the target room centre; a blocked room still blocks the
                        // step (so a blocked-corridor assault can't teleport through — but escalation has
                        // already re-routed the rally off it by here).
                        let next = ExtendedTravel::step_toward(member_pos[i], travel.target);
                        if !blocked.contains(&next.room_name()) {
                            member_pos[i] = next;
                        }
                    }
                }
                // ARRIVAL (RC-7): a member that reaches the target room with TTL still ABOVE `FIGHT_BUFFER` can
                // sustain the fight → engage. A member that arrives BELOW the buffer (the pre-fix far/low-TTL
                // crawl) is dead-on-arrival — it cannot meaningfully fight; it drops to unfilled (the roster
                // attrition that keeps the contested quorum from stabilising). The D6 renew keeps it above the
                // buffer; without it the far member arrives spent. `ttl[i]` mirrors the live `ticks_to_live()`.
                let mut arrived_fit = false;
                for i in 0..n_slots {
                    if filled[i] && member_pos[i].room_name() == target_room {
                        if ttl[i] > rally::FIGHT_BUFFER {
                            arrived_fit = true;
                        } else {
                            // Dead-on-arrival: too spent to fight → drops out (re-spawn / churn).
                            filled[i] = false;
                            member_pos[i] = travel.homes[i];
                            held_home[i] = false;
                        }
                    }
                }
                in_target_room = arrived_fit;
                if in_target_room {
                    return ChurnOutcome::DeployedAndEngaged { generations: generation, engage_tick: tick };
                }
                let cur = (0..n_slots).filter(|&i| filled[i]).map(|i| ExtendedTravel::room_dist(member_pos[i], travel.target)).min();
                travel_progress = match (cur, prev_min_dist) {
                    (Some(c), Some(p)) => c < p,
                    (Some(_), None) => true,
                    _ => false,
                };
                prev_min_dist = cur;
            } else {
                // SOLO TRAVEL (item (c)): each member steps toward the rally. RC-3 BLOCKED-PATH: a step that
                // would ENTER a blocked room does NOT advance — the member surfaces a `Blocked` signal
                // (tracked per-member in `block_streak`). An EXCLUDED member (re-assessed out, D4) holds.
                // A member within `RALLY_GATHER_RADIUS` of the rally has ARRIVED — it holds (range 1) and
                // counts as PROGRESS for D5 (it is done, not stalled).
                let mut at_rally = vec![false; n_slots];
                // D6a PRE-DEPARTURE LIFETIME GATE (RC-7): rally→target room-distance (the assault leg cost).
                let rally_to_target = ExtendedTravel::room_dist(rally, travel.target);
                for i in 0..n_slots {
                    if !filled[i] || excluded[i] {
                        continue; // unspawned, or re-assessed out (left behind / recalled)
                    }
                    if member_pos[i].get_range_to(rally) <= rally::RALLY_GATHER_RADIUS {
                        at_rally[i] = true;
                        held_home[i] = false;
                        block_streak[i] = 0;
                        continue; // already gathered at the rally — hold (range 1)
                    }
                    // D6a (RC-7): before stepping a member toward the rally, gate on whether its TTL can cover
                    // the journey (dist→rally + rally→target) + the fight buffer. PRE-FIX (`lifetime_gate`
                    // off) it is sent regardless → it arrives below `FIGHT_BUFFER` / ages out mid-travel. FIXED
                    // (on) an insufficient-but-renewable member HOLDS at home (D6b tops it up); once sufficient
                    // it is released to travel. A hopeless member is held too (the bot would recycle it; the
                    // sim just keeps it home so it can't feed the oscillation — recycle is bot-side).
                    if travel.lifetime_gate {
                        let dist_to_rally = ExtendedTravel::room_dist(member_pos[i], rally);
                        match rally::lifetime_sufficient_for_deployment(
                            ttl[i],
                            dist_to_rally,
                            rally_to_target,
                            rally::FIGHT_BUFFER,
                            rally::RENEW_TARGET_TTL,
                        ) {
                            rally::CommitDecision::Commit => held_home[i] = false,
                            // Insufficient (renewable or hopeless) → HOLD at home, do not advance this tick.
                            _ => {
                                held_home[i] = true;
                                block_streak[i] = 0;
                                continue;
                            }
                        }
                    }
                    let next = ExtendedTravel::step_toward(member_pos[i], rally);
                    if blocked.contains(&next.room_name()) {
                        // RC-3: the direct step is into a blocked room — surface `Blocked`. The member is
                        // wedged; pre-fix the manager IGNORES this (the silent retry loop) until D4 re-assesses
                        // it out. No advance.
                        block_streak[i] += 1;
                    } else {
                        block_streak[i] = 0;
                        member_pos[i] = next;
                    }
                }

                // RC-4/RC-8 — TRAVEL-PROGRESS signal. Per-member distances to the RALLY (excluded members
                // are out of the signal — the squad no longer waits on them).
                let dists: Vec<Option<u32>> = (0..n_slots)
                    .map(|i| if filled[i] && !excluded[i] { Some(ExtendedTravel::room_dist(member_pos[i], rally)) } else { None })
                    .collect();
                if travel.majority_progress {
                    // D5: refresh the lease while a MAJORITY of present members are MAKING PROGRESS — either
                    // CLOSING distance (per-member) OR already AT the rally (arrived = done). So one straggler
                    // doesn't pin "stalled", and (conversely) a lead that arrived + holds at the rally doesn't
                    // pin the MIN at 0 and mask a still-closing bulk (RC-4/RC-8). Per-member, not the min.
                    let mut progressing = 0usize;
                    let mut counted = 0usize;
                    for i in 0..n_slots {
                        if dists[i].is_none() {
                            continue;
                        }
                        counted += 1;
                        let closing = match (dists[i], prev_member_dist[i]) {
                            (Some(c), Some(p)) => c < p,
                            (Some(_), None) => true, // first reading — assume progress for one reconcile
                            _ => false,
                        };
                        if closing || at_rally[i] {
                            progressing += 1;
                        }
                    }
                    travel_progress = counted > 0 && progressing * 2 > counted;
                } else {
                    // PRE-FIX (RC-4): the MIN-over-members signal. A lead that arrived + holds pins the min at
                    // 0 (flat) while the bulk still closes → `travel_progress=false` → the lease lapses though
                    // members ARE advancing (the masking failure). Symmetrically a single stuck member pins it.
                    let cur = dists.iter().filter_map(|d| *d).min();
                    travel_progress = match (cur, prev_min_dist) {
                        (Some(c), Some(p)) => c < p,
                        (Some(_), None) => true,
                        _ => false,
                    };
                    prev_min_dist = cur;
                }
                prev_member_dist = dists;

                // D8 + D4 — the TIGHTER stall window → ESCALATE. A member whose `block_streak` exceeds the
                // tight `SOLO_TRAVEL_STALL_WINDOW` (D8) has been DETECTED (RC-3 — `check_movement_failure`
                // surfaced `Blocked`) as unable to path to the rally. D4 escalates: RE-ASSESS it OUT of the
                // gather quorum (`excluded`) so the squad proceeds with the REACHABLE subset instead of the
                // pre-fix silent forever-wait. Bounded; fires WELL BEFORE the coarse `MAX_TRAVEL_BUDGET` (the
                // backstop, preserved). Without `tight_stall_window` the manager never re-assesses → the
                // member sits, the contested quorum never fires, and the squad lapses at the budget (RED).
                if travel.escalate_on_block && travel.tight_stall_window {
                    for i in 0..n_slots {
                        if filled[i] && !excluded[i] && block_streak[i] >= SOLO_TRAVEL_STALL_WINDOW {
                            excluded[i] = true;
                        }
                    }
                }
            }
        }

        // ── D6b RENEW-TO-SUFFICIENCY while holding (RC-5/RC-6). The bot's Phase-B-renew tops up a present
        // member with low TTL that is HOLDING at a free spawn — extended PAST the forming-only `filled >=
        // requested` gate. A member is RENEWABLE here when it is either (a) HELD AT HOME by the D6a gate (a
        // home spawn is free to top it up), OR (b) GATHERED at the rally AND `rally_spawn` (the D6c renewable-
        // rally bias — a forward staging room near a friendly spawn). Each renew adds `RENEW_PER_TICK` toward
        // `RENEW_TARGET_TTL`, bounded by ONE renew per home spawn lane per tick (no spawn monopolization — the
        // economy lane is preserved exactly as the pre-fix model). PRE-FIX (`renew_in_transit` off) only the
        // forming-renew below fires, so a held/rallying member with low TTL just ages out (RC-5). The renew
        // self-heals on reload — `ttl[i]` mirrors the live creep, no serialized state.
        if travel.renew_in_transit {
            // Which present members are eligible for a top-up this tick (low TTL + renewable location)?
            let renewable = |i: usize| -> bool {
                if !filled[i] || ttl[i] >= rally::RENEW_TARGET_TTL {
                    return false;
                }
                let at_home = held_home[i] || (!departed && member_pos[i] == travel.homes[i]);
                let at_renewable_rally = departed && travel.rally_spawn && {
                    let r = screeps_combat_decision::rally::shared_rally_point_for_members(
                        &ExtendedTravel::pos_options(&present_positions),
                        travel.target,
                        travel.uncontested,
                    );
                    member_pos[i].get_range_to(r) <= rally::RALLY_GATHER_RADIUS
                };
                at_home || at_renewable_rally
            };
            // One renew per home spawn lane per tick (the bounded, no-monopolization model). Lowest-TTL first
            // (the most urgent member), deterministic (slot order tie-break, no `HashMap`).
            let free_lanes = (0..s.homes.len()).filter(|&h| tick >= busy_until[h]).count();
            let mut order: Vec<usize> = (0..n_slots).filter(|&i| renewable(i)).collect();
            order.sort_by_key(|&i| (ttl[i], i));
            for &i in order.iter().take(free_lanes) {
                ttl[i] = (ttl[i] + RENEW_PER_TICK).min(rally::RENEW_TARGET_TTL);
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
            holding_station: false,
            declaiming: false,
            reassign_available: false,
            retreated_from_contact: false, // ADR 0035 D4 — not exercised by this driver
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason: RetireReason::GaveUp, .. } => {
                if departed {
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
                departed = false;
                gathered = false;
                prev_min_dist = None;
                prev_member_dist = vec![None; n_slots];
                block_streak = vec![0; n_slots];
                excluded = vec![false; n_slots];
                ttl = vec![0; n_slots];
                held_home = vec![false; n_slots];
                continue;
            }
            ReconcileAction::Retire { .. } => {
                return ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present };
            }
            ReconcileAction::KeepRefreshLease => deadline = tick + COMMITMENT_BUDGET,
            ReconcileAction::Keep => {}
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
        if oscillations > 0 {
            ChurnOutcome::OscillatedNeverGathered { generations: generation, max_gathered }
        } else {
            ChurnOutcome::LapsedInTravel { generations: generation }
        }
    } else {
        ChurnOutcome::ChurnedNeverDeployed { generations: generation, max_present }
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
            declaiming: false, // ADR 0027 v1.1 P2 declaim is exercised by `run_declaim_flow`, not here
            reassign_available,
            retreated_from_contact: false, // ADR 0035 D4 — not exercised by the v1 reassign flow
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
    // best regardless. These offense-flow beds carry no defender creeps (`enemy` = `None`), so the candidate's
    // STRUCTURE defense alone drives the gate (ADR 0031 #41 — the enemy-creep dps is the `enemy` arg now).
    let comp = optimize_composition(
        c.objective,
        &c.defense,
        None,
        c.target_value,
        onsite_window,
        EnemyCoordination::Coordinated,
        0.0,
        c.honor_verdict,
        false, // not confirmed-undefended (these gate beds have no scouted defender creeps)
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
            declaiming: false, // ADR 0027 v1.1 P2 declaim is exercised by `run_declaim_flow`, not here
            reassign_available: false, // offense reassign is v1.2+; this driver isolates production→engage
            retreated_from_contact: false, // ADR 0035 D4 — not exercised by this driver
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

// ═══ ADR 0027 v1.1 P2 — the DECLAIM flow (sim-first) ═════════════════════════════════════════════════════
//
// The operator's sim-first requirement for the highest-risk salvage migration: prove the WHOLE declaim chain
// offline + deterministically — a `Declaim` objective is fielded as a CLAIM declaimer squad, it FORMS,
// TRAVELS, arrives, and `attackController`s the controller across the 1000-tick upgrade-block CADENCE,
// PERSISTING (the `declaiming` lease-hold refreshes the lease) through every cadence cycle until the
// controller is neutralized — NEVER giving up mid-neutralization. The terminal is the PRODUCER withdrawing
// the objective when the controller goes neutral (`objective_gone` retires the declaimer cleanly), exactly
// like the live `SalvageMission`. The reconcile DECISION is the SHARED `lifecycle::reconcile` kernel (no
// live/sim drift); the cadence + neutralization are modeled here (the engine `attackController` cadence is
// the soak's job, per the ADR — this proves the LIFECYCLE persistence the live bug would break).

/// The declaim-flow scenario: where home is, where the controller room is, the per-strike cadence, the number
/// of strikes to neutralize the controller (`−CONTROLLER_DOWNGRADE_PER_STRIKE` over `strikes_to_neutralize`),
/// and the lifecycle timing.
#[derive(Clone, Debug)]
pub struct DeclaimFlowScenario {
    pub home: V1Room,
    pub controller_room: V1Room,
    /// Ticks the upgrade-block lasts after a strike (the engine cadence, ~1000). Modeled as the gap between
    /// strikes a declaimer can land — DELIBERATELY longer than `COMMITMENT_BUDGET` so the base lease lapses
    /// BETWEEN strikes (the exact mid-cadence lapse the `declaiming` hold must bridge).
    pub cadence: u32,
    /// Strikes needed to drive the controller to neutral. The flow runs through this many cadence cycles,
    /// proving the squad persists across ALL of them.
    pub strikes_to_neutralize: u32,
    /// Ticks the declaimer needs to form (small — one CLAIM member).
    pub form_ticks: u32,
    pub objective_ttl: u32,
    pub budget_ticks: u32,
}

/// The declaim-flow outcome — did the declaimer reach the controller AND persist across the cadence to fully
/// neutralize it (the SUCCESS the P2 lease-hold must produce), or did it give up mid-neutralization (the
/// pre-fix failure: the base lease lapses between strikes → GaveUp → mark_unwinnable → re-field churn)?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaimOutcome {
    /// The declaimer formed, traveled, arrived, and STRUCK the controller across `cadence_cycles` cadence
    /// cycles WITHOUT ever giving up, until the controller went neutral and the producer withdrew the
    /// objective — the squad retired cleanly (`generations == 0`: no re-field churn). The P2 success path.
    Neutralized { cadence_cycles: u32, neutralized_tick: u32 },
    /// The declaimer gave up mid-neutralization (the lease lapsed between strikes and was NOT held) → it was
    /// re-fielded. `generations` counts the churn. The pre-`declaiming`-fix failure.
    GaveUpMidNeutralization { generations: u32, strikes_landed: u32 },
    /// The declaimer never reached the controller within the budget (a travel/form stall — not the P2 concern,
    /// but reported for completeness).
    NeverReached { generations: u32 },
}

/// Drive the DECLAIM lifecycle end-to-end + deterministically (ADR 0027 v1.1 P2): a `Declaim` objective → ONE
/// CLAIM declaimer squad claims, forms, travels, arrives, and strikes the controller on the 1000-tick cadence,
/// PERSISTING across every cadence cycle (the `declaiming` lease-hold) until the controller is neutralized and
/// the producer withdraws the objective. Returns [`DeclaimOutcome::Neutralized`] (with `generations == 0` — no
/// churn) on the success path. The reconcile is the SHARED `lifecycle::reconcile` kernel — the same Phase-A
/// logic the bot runs — so the persistence-across-cadence behavior cannot drift between sim and live.
pub fn run_declaim_flow(s: &DeclaimFlowScenario) -> DeclaimOutcome {
    use screeps_combat_decision::lifecycle::{reconcile, ReconcileAction, ReconcileSnapshot, RetireReason};

    let mut queue = V1Queue::default();
    let mut generation: u32 = 0;

    let mut claimed_id: Option<u32> = None;
    let mut phase = V1Phase::Forming;
    let mut pos: V1Room = s.home;
    let mut form_done_at: u32 = s.form_ticks;
    let mut deadline: u32 = COMMITMENT_BUDGET;
    let mut gen_start: u32 = 0;
    let mut travel_start: u32 = 0;

    // The controller's neutralization progress: it goes neutral after `strikes_to_neutralize` strikes, one
    // strike landing per cadence cycle (the engine upgrade-block). `next_strike_at` is when the in-room
    // declaimer can land its next strike (`None` until it arrives + the first strike fires).
    let mut strikes_landed: u32 = 0;
    let mut next_strike_at: Option<u32> = None;
    let mut cadence_cycles: u32 = 0;
    let mut controller_neutral = false;

    for tick in 0..s.budget_ticks {
        // ── PRODUCER (SalvageMission): emit the Declaim objective while the controller is still owned + the
        //    corridor is open (ReachableNow). Once the controller goes neutral, the producer STOPS emitting +
        //    WITHDRAWS — the `objective_gone` terminal that retires the declaimer cleanly. ──
        if !controller_neutral {
            queue.request(s.controller_room, 25.0 /* LOW */, tick, s.objective_ttl);
        } else if let Some(id) = queue.objectives.iter().find(|o| o.room == s.controller_room).map(|o| o.id) {
            queue.withdraw(id); // controller neutral → withdraw (the de-claim is done)
        }
        queue.expire(tick);

        // ── Claim: an unclaimed declaimer squad claims the Declaim objective. ──
        if claimed_id.is_none() {
            if let Some(id) = queue.best_unclaimed_excluding(u32::MAX) {
                queue.claim(id);
                claimed_id = Some(id);
                phase = V1Phase::Forming;
                form_done_at = tick + s.form_ticks;
                pos = s.home;
                deadline = tick + COMMITMENT_BUDGET;
                gen_start = tick;
            }
        }

        let Some(cur_id) = claimed_id else {
            if controller_neutral {
                // The controller is neutral AND the squad has retired (no claim) — the success terminal.
                return DeclaimOutcome::Neutralized { cadence_cycles, neutralized_tick: tick };
            }
            continue;
        };
        let obj = queue.get(cur_id).copied();
        let objective_gone = obj.is_none();
        let target_room = obj.map(|o| o.room);

        // ── Phase progression: form → travel → in-room (then strike on the cadence). ──
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
                    let before = v1_dist(pos, target);
                    pos = (pos.0 + (target.0 - pos.0).signum(), pos.1 + (target.1 - pos.1).signum());
                    travel_progress = v1_dist(pos, target) < before;
                    if pos == target {
                        phase = V1Phase::InRoom;
                    }
                }
                V1Phase::InRoom => {
                    in_target_room = true;
                    // ── DECLAIM DRIVE (the engine `attackController` cadence, modeled): land a strike when the
                    //    upgrade-block has cleared. The FIRST in-room tick arms the first strike. ──
                    let strike_due = next_strike_at.map(|t| tick >= t).unwrap_or(true);
                    if strike_due && !controller_neutral {
                        strikes_landed += 1;
                        cadence_cycles += 1;
                        next_strike_at = Some(tick + s.cadence); // upgrade-blocked for `cadence` ticks
                        if strikes_landed >= s.strikes_to_neutralize {
                            controller_neutral = true; // the controller goes neutral → producer withdraws next tick
                        }
                    }
                }
            }
        }

        // ── Phase A reconcile (the SHARED kernel). A declaimer has NO focus and never engages, so the
        //    `declaiming` hold (is_declaim && in_target_room && has_members) is what refreshes the lease across
        //    the cadence — without it the base lease lapses BETWEEN strikes (cadence > COMMITMENT_BUDGET) and
        //    the squad would GaveUp+mark_unwinnable mid-neutralization. ──
        let forming = phase == V1Phase::Forming && tick < form_done_at;
        let declaiming = in_target_room; // the manager's is_declaim && in_target_room && has_members
        let snapshot = ReconcileSnapshot {
            objective_gone,
            duplicate: false,
            is_defend: false,
            deadline_lapsed: tick >= deadline,
            wiped: false,
            has_focus: false,   // a quiet derelict room — a declaimer never has a combat focus
            engaged_once: false, // a declaimer never enters combat
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
            declaiming,
            reassign_available: false,
            retreated_from_contact: false, // ADR 0035 D4 — not exercised by the declaim flow
        };
        match reconcile(snapshot) {
            ReconcileAction::Retire { reason, withdraw, .. } => {
                // The ONLY clean terminal for a declaimer is ObjectiveGone (the producer withdrew on neutral).
                // A GaveUp here is the pre-fix failure: the lease lapsed mid-cadence and was NOT held.
                if reason == RetireReason::ObjectiveGone && controller_neutral {
                    return DeclaimOutcome::Neutralized { cadence_cycles, neutralized_tick: tick };
                }
                if reason == RetireReason::GaveUp {
                    return DeclaimOutcome::GaveUpMidNeutralization { generations: generation, strikes_landed };
                }
                let _ = withdraw;
                if withdraw {
                    queue.withdraw(cur_id);
                } else {
                    queue.release(cur_id);
                }
                generation += 1;
                claimed_id = None;
                phase = V1Phase::Forming;
            }
            ReconcileAction::KeepRefreshLease => deadline = tick + COMMITMENT_BUDGET,
            ReconcileAction::Keep => {}
            ReconcileAction::Reassign { .. } => unreachable!("declaim flow never feeds reassign_available=true"),
        }
    }

    // Budget elapsed without neutralizing → never reached / never finished.
    DeclaimOutcome::NeverReached { generations: generation }
}

// ═══ ADR 0032 v1.2 — run_auction_flow: the GLOBAL Hungarian matching flow (extends run_v1_flow to N
//     squads × M objectives, with a greedy-vs-global toggle) ════════════════════════════════════════════
//
// `run_v1_flow` proves the SINGLE-squad reassign LIFECYCLE (one defender following a moving threat).
// `run_auction_flow` proves the MULTI-squad ASSIGNMENT: given N already-fielded squads (heterogeneous
// caps) and M live objectives, the GLOBAL solve (`assignment::solve_assignment`) finds a strictly higher
// total-EV matching than the per-squad GREEDY baseline (a faithful model of the OLD v1.1 behaviour:
// iterate squads in order, each greedily claims its best still-unclaimed objective). This is the FLOW
// analog of the kernel headline test — it proves global-optimality through the harness's RED→GREEN toggle,
// not just in the kernel unit test.

use screeps_combat_decision::assignment::{
    build_ev_matrix, solve_assignment, CapClass, ColumnKind, MatrixParams, ObjectiveCell, SquadRow,
};
use screeps_combat_decision::composition::SquadCapabilities;
use screeps_combat_decision::objective_value::{ObjectiveIntel, ObjectiveValueKind};

/// One assignable squad in the auction flow: its surviving caps (structure DPS + heal) + its current
/// objective id (for the StayPut re-score; `None` = freshly freed/forming).
#[derive(Clone, Copy, Debug)]
pub struct AuctionSquad {
    pub structure_dps: u32,
    pub heal: u32,
    pub current_objective: Option<u32>,
}

/// One objective in the auction flow: a stable id + its energy-equivalent value + a per-row feasibility
/// override (so the flow can model "squad B cannot reach/take objective L" — the heterogeneity that makes
/// greedy wrong). All objectives are undefended (P(win)=1 for any dps>0) so VALUE drives the optimum, and
/// the greedy-vs-global gap is a pure assignment effect, not a winnability artifact.
#[derive(Clone, Debug)]
pub struct AuctionObjective {
    pub id: u32,
    pub value: f32,
    /// `feasible_per_row[r] == false` ⇒ squad `r` cannot take this objective (claimed/incompatible-tile).
    pub feasible_per_row: Vec<bool>,
}

/// The auction-flow scenario: the fielded squads, the live objectives, and the greedy-vs-global toggle.
#[derive(Clone, Debug)]
pub struct AuctionFlowScenario {
    pub squads: Vec<AuctionSquad>,
    pub objectives: Vec<AuctionObjective>,
    /// `true` ⇒ the GLOBAL Hungarian (`solve_assignment`); `false` ⇒ the per-squad GREEDY baseline (the
    /// RED arm modelling the OLD v1.1 behaviour).
    pub global_solve: bool,
}

/// The auction-flow outcome: the total EV the chosen selection achieved + the per-squad objective picks
/// (objective id, or `None` for StayPut/Recycle), in squad order — so a test can assert both the headline
/// total AND the assignment SHAPE.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuctionOutcome {
    pub total_ev: i64,
    pub picks: Vec<Option<u32>>,
}

/// Build the EV matrix for the scenario (shared by the greedy + global arms so both score the SAME cells —
/// a fair RED→GREEN comparison: only the SELECTION differs).
fn auction_matrix(s: &AuctionFlowScenario) -> screeps_combat_decision::assignment::EvMatrix {
    let rows: Vec<SquadRow> = s
        .squads
        .iter()
        .map(|sq| SquadRow {
            caps: SquadCapabilities { heal_per_tick: sq.heal, structure_dps: sq.structure_dps, tank_effective_hp: 2000 },
            class: CapClass::Offense,
            current_objective: sq.current_objective,
            recycle_ev: 0,
            ..Default::default()
        })
        .collect();
    let n_rows = rows.len();
    let cells: Vec<ObjectiveCell> = s
        .objectives
        .iter()
        .map(|o| ObjectiveCell {
            id: o.id,
            class: CapClass::Offense,
            value_kind: ObjectiveValueKind::Denial,
            // Denial value_e = denial_value × 0.5; pass 2× so value_e == o.value.
            intel: ObjectiveIntel { denial_value: o.value * 2.0, ..Default::default() },
            defense: Default::default(),
            enemy: None,
            travel_rooms_per_row: vec![0; n_rows],
            feasible_per_row: if o.feasible_per_row.is_empty() { vec![true; n_rows] } else { o.feasible_per_row.clone() },
        })
        .collect();
    build_ev_matrix(&rows, &cells, &MatrixParams::default())
}

/// The per-squad GREEDY baseline over the SAME matrix (a faithful model of the OLD v1.1 behaviour, ADR 0032
/// §Problem #2): iterate squads in row order; each claims its own best-EV still-unclaimed OBJECTIVE column
/// (StayPut/Recycle are per-row, never contended), marking the objective covered so a later squad cannot
/// take it. Returns the [`AuctionOutcome`].
fn auction_greedy(matrix: &screeps_combat_decision::assignment::EvMatrix) -> AuctionOutcome {
    use screeps_combat_decision::assignment::INFEASIBLE_EV;
    let mut covered = vec![false; matrix.cols()];
    let mut total = 0i64;
    let mut picks = vec![None; matrix.rows];
    for (r, pick) in picks.iter_mut().enumerate() {
        let mut best: Option<(usize, i64)> = None;
        for (c, col) in matrix.columns.iter().enumerate() {
            let is_obj = matches!(col, ColumnKind::Objective { .. });
            if is_obj && covered[c] {
                continue;
            }
            let ev = matrix.at(r, c);
            if ev == INFEASIBLE_EV {
                continue;
            }
            if best.map(|(_, b)| ev > b).unwrap_or(true) {
                best = Some((c, ev));
            }
        }
        if let Some((c, ev)) = best {
            if let ColumnKind::Objective { id } = matrix.columns[c] {
                covered[c] = true;
                *pick = Some(id);
            }
            total += ev;
        }
    }
    AuctionOutcome { total_ev: total, picks }
}

/// Drive the multi-squad ASSIGNMENT flow (ADR 0032 v1.2): build the shared EV matrix, then select via the
/// GLOBAL Hungarian (`global_solve == true`) or the per-squad GREEDY baseline (`false`). Returns the total
/// EV + the per-squad picks. The acceptance assertion (in the tests) is that GLOBAL strictly beats GREEDY
/// on total EV for a heterogeneous scenario — the same headline the kernel test proves, now in the flow.
pub fn run_auction_flow(s: &AuctionFlowScenario) -> AuctionOutcome {
    let matrix = auction_matrix(s);
    if !s.global_solve {
        return auction_greedy(&matrix);
    }
    let sol = solve_assignment(&matrix);
    let picks: Vec<Option<u32>> = sol
        .row_to_col
        .iter()
        .map(|c| {
            c.and_then(|c| match matrix.columns[c] {
                ColumnKind::Objective { id } => Some(id),
                _ => None,
            })
        })
        .collect();
    AuctionOutcome { total_ev: sol.total_ev, picks }
}

// ═══ ADR 0032 v2 / ADR 0027 — run_merge_flow: the MERGE/transfer pending-slot primitive (kernel + model) ══
//
// SCOPE — what this proves and what it does NOT. The kernel tests prove the Merge COLUMN (feasibility guard +
// EV) in isolation. `run_merge_flow` proves the KERNEL SELECTION (the solve picks a `Merge→Bk` when, and only
// when, it is EV-positive) plus an ABSTRACT transfer model mirroring the bot's `apply_merges` — (1) the
// donor's role-matched member is moved INTO the receiver's open pending slot, (2) that slot is marked filled,
// (3) the donor EMPTIES → clean retire (the creep was TRANSFERRED, never deleted; it ends owned by EXACTLY ONE
// squad). The "spawn-slot drop" in step (2) is asserted BY CONSTRUCTION (the model flips `filled = true`); it
// does NOT model the live SpawnQueue / Phase-B / deferred-`exec_mut` INTERLEAVING. In particular this harness
// does NOT exercise the same-tick DOUBLE-FILL race (a Phase-B spawn queued before the deferred transfer
// applied). That live no-double-fill is guarded in the BOT by the `create_spawn_callback` `is_slot_filled`
// recheck (`squad_manager::should_register_spawned_member` — Bug #1) and its bot `--lib` test; this harness
// validates only the kernel decision + the abstract transfer outcome, with NO ECS.

use screeps_combat_decision::assignment::{build_ev_matrix_with_merge, role_bit};
use screeps_combat_decision::composition::SquadRole;

/// One member-in-a-slot of an abstract squad in the merge flow: the role it fills + whether it is a real
/// (present, transferable) body or an unfilled PENDING spawn slot (queued, not yet spawned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeSlot {
    pub role: SquadRole,
    /// `true` ⇒ a present body (the donor sheds these); `false` ⇒ an OPEN pending spawn slot (the receiver's
    /// queue holds these; a transfer FILLS one, dropping it from the queue).
    pub filled: bool,
}

/// One abstract squad in the merge flow: its caps (for the receiver's marginal-lift P(win)), its current
/// objective id, and its slots (present bodies + open pending slots). The DONOR has present bodies it can
/// shed; the RECEIVER has open pending slots a transfer fills.
#[derive(Clone, Debug)]
pub struct MergeSquad {
    pub structure_dps: u32,
    pub heal: u32,
    /// The caps of this squad's SHEDDABLE (present) members — added to a receiver for the marginal lift.
    pub sheddable_dps: u32,
    pub sheddable_heal: u32,
    pub current_objective: u32,
    pub slots: Vec<MergeSlot>,
    /// Merge-eligible donor (terminal-with-survivors / over-rostered / forming-consolidate). A receiver does
    /// not need this set; an ineligible squad is never a donor.
    pub merge_eligible: bool,
}

impl MergeSquad {
    fn open_slot_roles(&self) -> u8 {
        self.slots.iter().filter(|s| !s.filled).fold(0u8, |acc, s| acc | role_bit(s.role))
    }
    fn sheddable_roles(&self) -> u8 {
        self.slots.iter().filter(|s| s.filled).fold(0u8, |acc, s| acc | role_bit(s.role))
    }
    fn present_count(&self) -> usize {
        self.slots.iter().filter(|s| s.filled).count()
    }
    fn open_count(&self) -> usize {
        self.slots.iter().filter(|s| !s.filled).count()
    }
}

/// The merge-flow scenario: the abstract squads, their objectives' values + defense (for the marginal lift),
/// and a toggle for the merge column (RED control = merge disabled ⇒ the donor can only solo-reassign/recycle).
#[derive(Clone, Debug)]
pub struct MergeFlowScenario {
    pub squads: Vec<MergeSquad>,
    /// Per-objective value (energy-equivalent) keyed by objective id == index.
    pub objective_values: Vec<f32>,
    /// Per-objective tower range (0 ⇒ undefended; >0 ⇒ a single energized tower at that range — so caps
    /// matter and there is real marginal lift). Index == objective id.
    pub objective_tower_range: Vec<u32>,
    /// Per-objective required kill hits (0 ⇒ undefended/trivial). Index == objective id.
    pub objective_required_hits: Vec<u32>,
    /// `true` ⇒ the Merge column is offered (GREEN); `false` ⇒ merge disabled (RED control — solo only).
    pub merge_enabled: bool,
}

/// The merge-flow outcome — the ABSTRACT-model state AFTER applying the chosen merge (or none): which squads
/// retired (emptied donors), the receiver's filled/open slot counts (the modeled slot-fill), and how many
/// members transferred. Lets a test assert the kernel SELECTION + the modeled transfer outcome (transfer +
/// slot-fill + clean retire). NOTE the slot-fill is asserted by construction — the live spawn-queue drop /
/// no-double-fill is guarded in the bot (see the section header / Bug #1), not here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MergeOutcome {
    /// `true` per squad index ⇒ that squad RETIRED (emptied donor). A receiver/partial donor stays `false`.
    pub retired: Vec<bool>,
    /// Per squad index: `(filled_slots, open_pending_slots)` AFTER the transfer. The receiver's open count
    /// DROPS by the number of transfers (the model marks the filled slots; the live queue-drop is the bot's).
    pub slots: Vec<(usize, usize)>,
    /// `(donor_idx, receiver_idx, transferred_count)` for the applied merge, or `None` if no merge fired.
    pub merge: Option<(usize, usize, usize)>,
}

fn merge_squad_row(sq: &MergeSquad) -> SquadRow {
    SquadRow {
        caps: SquadCapabilities { heal_per_tick: sq.heal, structure_dps: sq.structure_dps, tank_effective_hp: 2000 },
        class: CapClass::Offense,
        current_objective: Some(sq.current_objective),
        recycle_ev: 0,
        merge_eligible: sq.merge_eligible,
        sheddable: SquadCapabilities { heal_per_tick: sq.sheddable_heal, structure_dps: sq.sheddable_dps, tank_effective_hp: 2000 },
        sheddable_roles: sq.sheddable_roles(),
        // A receiver offers its open pending slots (a squad with present members AND open slots is forming).
        open_slot_roles: if sq.present_count() > 0 && sq.open_count() > 0 { sq.open_slot_roles() } else { 0 },
    }
}

fn merge_objective_cell(s: &MergeFlowScenario, id: u32, n_rows: usize) -> ObjectiveCell {
    use screeps_combat_decision::force_sizing::{DefenseProfile, TowerThreat};
    let i = id as usize;
    let value = s.objective_values[i];
    let tower_range = s.objective_tower_range.get(i).copied().unwrap_or(0);
    let required = s.objective_required_hits.get(i).copied().unwrap_or(0);
    let defense = if tower_range > 0 {
        DefenseProfile { objective_hits: required, towers: vec![TowerThreat { range_to_assault: tower_range, energy: 1000 }], ..Default::default() }
    } else {
        DefenseProfile { objective_hits: required, ..Default::default() }
    };
    ObjectiveCell {
        id,
        class: CapClass::Offense,
        value_kind: ObjectiveValueKind::Denial,
        intel: ObjectiveIntel { denial_value: value * 2.0, ..Default::default() },
        defense,
        enemy: None,
        travel_rooms_per_row: vec![0; n_rows],
        feasible_per_row: vec![true; n_rows], // current objectives are reachable via StayPut; merge is the move under test
    }
}

/// Drive the MERGE flow (ADR 0032 v2 / ADR 0027 — kernel selection + abstract transfer; see the section
/// header for the precise scope). Build the matrix WITH the merge column (or without, the RED control), solve,
/// then APPLY the chosen `Merge→Bk` to the ABSTRACT squad model the way the bot's `apply_merges` does: move
/// the donor's role-matched present member into the receiver's open pending slot, mark that slot filled (the
/// modeled spawn-slot drop — the live queue-drop / no-double-fill is the bot's, guarded by the callback
/// `is_slot_filled` recheck, NOT modeled here), and RETIRE the donor if it emptied. Returns the
/// [`MergeOutcome`].
pub fn run_merge_flow(s: &MergeFlowScenario) -> MergeOutcome {
    let rows: Vec<SquadRow> = s.squads.iter().map(merge_squad_row).collect();
    let n = rows.len();
    // Distinct objective ids per squad (each squad is on its own objective in these scenarios).
    let mut objective_ids: Vec<u32> = s.squads.iter().map(|sq| sq.current_objective).collect();
    objective_ids.sort_unstable();
    objective_ids.dedup();
    let cells: Vec<ObjectiveCell> = objective_ids.iter().map(|&id| merge_objective_cell(s, id, n)).collect();

    // RED control: zero out every receiver's open_slot_roles so NO merge column is built (merge disabled).
    let rows = if s.merge_enabled {
        rows
    } else {
        rows.into_iter().map(|mut r| { r.open_slot_roles = 0; r }).collect()
    };

    let matrix = build_ev_matrix_with_merge(&rows, &cells, &[], &MatrixParams::default());
    let sol = solve_assignment(&matrix);

    // Mutable copy of the slot state to apply the transfer to.
    let mut slots: Vec<Vec<MergeSlot>> = s.squads.iter().map(|sq| sq.slots.clone()).collect();
    let mut retired = vec![false; n];
    let mut merge: Option<(usize, usize, usize)> = None;

    // Find the chosen merge (a donor row matched to a Merge column). The commit gate mirrors the bot: the
    // merge must beat the donor's StayPut by >0 (a positive lift).
    let stay_base = cells.len();
    for (r, pick) in sol.row_to_col.iter().enumerate() {
        let Some(col) = pick else { continue };
        if let ColumnKind::Merge { receiver_row } = matrix.columns[*col] {
            let merge_ev = matrix.at(r, *col);
            let stay_ev = matrix.at(r, stay_base + r).max(0);
            if merge_ev <= stay_ev {
                continue;
            }
            // ── APPLY (ADR 0027) — transfer the donor's role-matched present member(s) into the receiver's
            //    OPEN pending slots, greedily in stable order; DROP the filled slot from the receiver's queue. ──
            let shed_roles = rows[r].sheddable_roles;
            let mut transferred = 0usize;
            // Open receiver slots whose role matches a shed role, stable order.
            let open_idxs: Vec<usize> = slots[receiver_row]
                .iter()
                .enumerate()
                .filter_map(|(i, sl)| (!sl.filled && (role_bit(sl.role) & shed_roles) != 0).then_some(i))
                .collect();
            for oi in open_idxs {
                let want = slots[receiver_row][oi].role;
                // First unused present donor body of the matching role.
                if let Some(di) = slots[r].iter().position(|sl| sl.filled && sl.role == want) {
                    // Move it: drop from donor, FILL the receiver's pending slot (the spawn queue drops it).
                    slots[r].remove(di);
                    slots[receiver_row][oi].filled = true; // the pending slot is now filled BY TRANSFER (queue drops it)
                    transferred += 1;
                }
            }
            if transferred > 0 {
                // Donor emptied (all present bodies shed) ⇒ clean retire (the creeps were TRANSFERRED).
                if slots[r].iter().all(|sl| !sl.filled) {
                    retired[r] = true;
                }
                merge = Some((r, receiver_row, transferred));
            }
            break; // these scenarios apply at most one merge
        }
    }

    let slot_counts: Vec<(usize, usize)> = slots
        .iter()
        .map(|sl| (sl.iter().filter(|s| s.filled).count(), sl.iter().filter(|s| !s.filled).count()))
        .collect();
    MergeOutcome { retired, slots: slot_counts, merge }
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
    // ADR 0031 #39 P2/P3 — the oracle picks the MODE. A winnable BREACH (or DRAIN) fields the oracle-sized
    // comp; otherwise the ceiling fallback keeps the chain running. The DRAIN comp carries the TOUGH+HEAL
    // soak buffer `from_assessment` sized (P2); the engage phase below runs it through the drain stance +
    // `breach_drain` tactics (P3 parity — the SAME `decide_squad` the live bot threads).
    let drain = assessment.winnable && assessment.mode == AssaultMode::Drain;
    let comp = match (assessment.winnable && assessment.mode == AssaultMode::Breach, drain, assemble_force(&required, sizing_energy)) {
        (true, _, Some(assembled)) => assembled,
        (_, true, Some(assembled)) => assembled,
        // The oracle deferred / the assembler couldn't field the required force at this energy — field the
        // ceiling so the chain still runs (the test then surfaces whether even the ceiling kills). The ceiling
        // fallback uses the HOME capacity (not the swept per-member cap) so the Default path is byte-identical
        // to the pre-sweep behaviour (which sized the fallback at `engage.member_energy`).
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

    // 4. Engage the FORMED + MOVING roster against the defended core. A BREACH dismantles through the gate
    //    while out-healing the tower + guards; a DRAIN (ADR 0031 #39 P3) holds the falloff standoff while the
    //    FINITE towers bleed dry, then advances + dismantles — fielded via the drain stance + `breach_drain`
    //    tactics (the SAME `decide_squad` the live bot threads through P3). The engaged comp == the formed comp.
    let engaged = if drain {
        crate::harness::validate::run_managed_assault_drain(&engage, obj, &comp, SquadTacticParams::breach_drain())
    } else {
        run_managed_assault_with(&engage, obj, &comp, SquadTacticParams::breach())
    };
    match engaged {
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
        let defense = DefenseProfile { towers: vec![], breach_hits: 0, objective_hits: 0, repair_per_tick: 0.0, safe_mode: false };
        // ADR 0031 #41: the threat creep dps the optimizer prices comes from this single `EnemyForce` (dps=30),
        // not a co-resident `defense.enemy_dps` (removed).
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
            false, // a REAL threat (enemy.dps=30) — NOT confirmed-undefended → floor retained
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

    /// ADR 0035 (E1, the VACUOUS-INTEL ENGAGE CASCADE — RED→GREEN). A squad COMMITS to a TOWERED room
    /// (W4N5) on an EMPTY commit snapshot (`commit_intel_empty`), REACHES it, finds REAL towers
    /// (`arrival_has_towers` ⇒ in-room P(win)=LOSE), and RETREATS.
    ///
    /// RED (`abandon_fixes_enabled=false`): the commit view looked clear so it trickled in; on arrival the
    /// retreat MIS-RESOLVES as a clean clear → withdraw → the producer re-upserts (no `is_unwinnable_now`
    /// consult) → Phase C re-fields → reach↔retreat. `LapsedOnVacuousCommit`, generations climbing.
    ///
    /// GREEN (`abandon_fixes_enabled=true`): D3 classifies the empty-Cached towered room CONTESTED (masses,
    /// stages short); on arrival D4 feeds the real `retreated_from_contact` → the kernel returns
    /// GaveUp+mark_unwinnable (ABANDON, not resolve); D5's producer backoff suppresses the re-field.
    /// `AbandonedOnContact`, generations STABLE.
    fn vacuous_commit_scenario() -> ColonyFormingScenario {
        // A 2-slot offense roster that forms readily (not the focus of this test — the commit/abandon
        // cascade is). Two healthy homes so the full roster banks well within the lease.
        let comp = assemble_force(&RequiredForce { immune_struct_parts: 4, ..Default::default() }, 3000)
            .expect("an offense force");
        ColonyFormingScenario {
            composition: comp,
            homes: vec![
                Home { energy_capacity: 5300, income: 300, start_energy: 5300 },
                Home { energy_capacity: 5300, income: 300, start_energy: 5300 },
            ],
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 85.0,
            per_member_cap: 3000,
            budget_ticks: 4000,
            member_ttl: 1500,
            renew: false,
        }
    }

    #[test]
    fn vacuous_commit_lapses_pre_fix() {
        let s = vacuous_commit_scenario();
        // RED: the commit view declared the room uncontested (empty-Cached looked clear); fixes OFF.
        let target = ChurnTarget {
            travel_ticks: 30,
            uncontested: true,
            commit_intel_empty: true,
            arrival_has_towers: true,
            abandon_fixes_enabled: false,
            ..Default::default()
        };
        let out = run_lifecycle_churn(&s, &target);
        match out {
            ChurnOutcome::LapsedOnVacuousCommit { generations } => {
                assert!(generations > 1, "the vacuous-commit oscillation must climb generations, got {generations}");
            }
            other => panic!("pre-fix must LapsedOnVacuousCommit (the reach<->retreat spiral), got {other:?}"),
        }
    }

    #[test]
    fn vacuous_commit_abandons_on_contact_post_fix() {
        let s = vacuous_commit_scenario();
        // GREEN: the SAME towered room + empty commit intel, but the D3/D4/D5 fixes are ON.
        let target = ChurnTarget {
            travel_ticks: 30,
            uncontested: true, // the declared (legacy) view; D3 overrides it to contested off the vacuous intel
            commit_intel_empty: true,
            arrival_has_towers: true,
            abandon_fixes_enabled: true,
            ..Default::default()
        };
        let out = run_lifecycle_churn(&s, &target);
        match out {
            ChurnOutcome::AbandonedOnContact { generations } => {
                // Stable: a single de-commit, no re-field within the backoff (the producer suppressed it).
                assert!(generations <= 1, "abandon-on-contact must NOT oscillate (stable generations), got {generations}");
            }
            other => panic!("post-fix must AbandonedOnContact (clean de-commit + backoff), got {other:?}"),
        }
    }

    /// ADR 0035 PRESERVE: a LEGITIMATE LiveVisible-empty room (genuinely clear, no towers on arrival) still
    /// deploys + engages even with the fixes ON — no false-abandon. `commit_intel_empty=false` (we have real
    /// intel it is clear) and `arrival_has_towers=false`.
    #[test]
    fn legitimate_clear_room_still_deploys_with_fixes_on() {
        let s = vacuous_commit_scenario();
        let target = ChurnTarget {
            travel_ticks: 30,
            uncontested: true,
            commit_intel_empty: false,
            arrival_has_towers: false,
            abandon_fixes_enabled: true,
            ..Default::default()
        };
        let out = run_lifecycle_churn(&s, &target);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "a genuinely-clear room must still deploy + engage (no false-abandon), got {out:?}"
        );
    }

    /// ADR 0035 D4 (E1 false-abandon REGRESSION) — the WINNABLE-RETREAT case. A squad reaches the target,
    /// engages, and `decide_squad` enters the Retreating STATE (a focus-fired member dipped to critical HP)
    /// — BUT the real in-room `present_force_wins_or_stalls = TRUE`: the present force is still WINNING.
    /// Because the bot now carries the GENUINE lose verdict (`!present_force_wins_or_stalls`) rather than the
    /// `ctx.state == Retreating` SUPERSET, `retreated_from_contact = FALSE` → the kernel does NOT abandon.
    /// The squad holds + wins (`DeployedAndEngaged`), NOT `AbandonedOnContact`. The pre-fix `ctx.state ==
    /// Retreating` signal would have read TRUE here and falsely abandoned a winnable room mid-fight.
    #[test]
    fn winnable_retreat_does_not_abandon_a_winning_room() {
        let s = vacuous_commit_scenario();
        let target = ChurnTarget {
            travel_ticks: 30,
            uncontested: true,
            // In-room + would-be Retreating (critical-HP member) but present_force_wins_or_stalls=TRUE — so
            // NOT a towered/unwinnable room (arrival_has_towers stays false) and the abandon fixes are ON.
            winnable_retreat_in_room: true,
            arrival_has_towers: false,
            abandon_fixes_enabled: true,
            ..Default::default()
        };
        let out = run_lifecycle_churn(&s, &target);
        assert!(
            matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "a WINNING-but-Retreating (critical-HP) squad must HOLD + win, NOT be abandoned mid-fight, got {out:?}"
        );
        assert!(
            !matches!(out, ChurnOutcome::AbandonedOnContact { .. }),
            "the winnable-retreat case must NOT trip AbandonedOnContact (the false-abandon regression), got {out:?}"
        );
    }

    #[test]
    fn vacuous_commit_is_deterministic() {
        let s = vacuous_commit_scenario();
        let target = ChurnTarget {
            travel_ticks: 30,
            uncontested: true,
            commit_intel_empty: true,
            arrival_has_towers: true,
            abandon_fixes_enabled: true,
            ..Default::default()
        };
        assert_eq!(run_lifecycle_churn(&s, &target), run_lifecycle_churn(&s, &target));
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

    // ── ADR 0034 Phase 1: the PRODUCTION-PATH far-home stall (RC-3/RC-4/RC-8/RC-10) ──────────────────────
    //
    // These drive `run_lifecycle_churn_extended` — REAL cross-quadrant `Position`s + the PRODUCTION
    // `cohesion::centroid` → `rally::shared_rally_point_for_members` geometry + a solo step + per-tick
    // gather + the FIX-A latch + `Arrived` gated on `gathered`. The RC-1/RC-2 headline (wrong-room rally)
    // is already fixed in Phase 0, so the remaining far-home failures are RC-4/RC-8 (the min-distance
    // progress signal mis-reads a converging squad as "stalled") and RC-3 (a blocked member silently
    // retries forever → a budget-lapse give-up). Each is RED with the D4/D5/D8 toggles OFF, GREEN with
    // them ON.

    /// A 3-slot offense roster (one anti-creep fighter + two healers) — three members for a clean MAJORITY
    /// progress test (D5): one lead + a two-member bulk.
    fn three_slot_offense(budget: u32) -> ColonyFormingScenario {
        // Three DISTINCT roles → three slots (a ranged fighter for the gather quorum's fighter requirement,
        // a dismantler, and a healer) so we have a clean lead + two-member bulk for the D5 majority test.
        let comp = assemble_force(
            &RequiredForce { anti_creep_parts: 4, dismantle_parts: 4, heal_parts: 4, ..Default::default() },
            3000,
        )
        .expect("a 3-slot ranged+dismantler+healer force");
        assert!(comp.slots.len() >= 3, "the far-home repro wants a >=3-slot roster, got {}", comp.slots.len());
        let n = comp.slots.len();
        ColonyFormingScenario {
            composition: comp,
            // One easily-fieldable home per slot — isolate the MOVEMENT stall from spawn contention.
            homes: (0..n).map(|_| Home { energy_capacity: 5300, income: 400, start_energy: 3000 }).collect(),
            economy: EconomyPressure { hauler: Some((75.0, 1000)), miner: None, miner_period: 0 },
            combat_priority: 87.5,
            per_member_cap: 3000,
            budget_ticks: budget,
            member_ttl: 1500,
            renew: false,
        }
    }

    /// A real `Position` at the centre (25,25) of room `(rx,ry)` in world-room coords, with an in-room
    /// offset. `rx,ry` map to the W/N quadrant the same way the spatial driver's `to_pos` does.
    fn wpos(rx: i32, ry: i32, x: i32, y: i32) -> Position {
        Position::from_world_coords(rx * 50 + x, ry * 50 + y)
    }

    /// S1 — FAR-HOME CONVERGENCE STALL (RC-4/RC-8, no blocking): a CONTESTED far target with a lead that
    /// reaches the one-room-short rally and HOLDS while a far two-member bulk is still closing. The pre-fix
    /// MIN-over-members progress signal reads the held lead's distance 0 (flat) as "stalled" — masking the
    /// bulk that IS advancing — so the travel lease lapses BEFORE the bulk gathers (the contested quorum
    /// needs all three). The fields are tuned so the bulk takes longer than `COMMITMENT_BUDGET` to arrive.
    fn far_home_s1(n_slots: usize) -> ExtendedTravel {
        // Axis-aligned corridor (same N row, ry=-6 = N5) so the bulk's solo step is a clean horizontal line
        // and a blocked room can be placed exactly ON it. Target at the far-left (rx=-6 = W5); the bulk far
        // to the right (rx=-18 = W17), the lead just off the rally.
        let mut homes = Vec::with_capacity(n_slots);
        // Slot 0 = the LEAD: one room past the rally toward the target side (arrives + holds quickly).
        homes.push(wpos(-7, -6, 25, 25)); // W6N5 — adjacent to the rally corridor
                                          // The BULK: far cross-quadrant homes, many rooms down the +x corridor (a long solo crawl).
        for _ in 1..n_slots {
            homes.push(wpos(-18, -6, 25, 25)); // W17N5 — far along the corridor
        }
        ExtendedTravel {
            homes,
            target: wpos(-6, -6, 25, 25), // W5N5 target (contested)
            uncontested: false,
            blocked_rooms: vec![],
            escalate_on_block: false,
            majority_progress: false,
            tight_stall_window: false,
            rally_spawn: false,
            lifetime_gate: false,
            renew_in_transit: false,
        }
    }

    /// S1-BLOCKED — RC-3 (silent NO_PATH stall): a far cross-quadrant scatter where ONE member's only path
    /// to the rally is through a BLOCKED room (impassable / hostile). Pre-fix the bot has no member-side
    /// movement feedback: the blocked member sits forever, the contested quorum never fires, and the squad
    /// gives up only when the coarse 1000-tick travel budget lapses — a SILENT permanent stall.
    fn far_home_s1_blocked(n_slots: usize) -> ExtendedTravel {
        let mut t = far_home_s1(n_slots);
        // Keep ONE member (slot 1) far down the corridor behind a wall, but put the OTHER bulk member
        // (slots 2..) on the rally's side so the squad can still reach a 2-member contested quorum once the
        // blocked one is re-assessed out (D4). Slot 1 stays at the far home W17N5; relocate the rest near
        // the rally corridor.
        for i in 2..n_slots {
            t.homes[i] = wpos(-8, -6, 25, 25); // W7N5 — on the rally's side of the wall (reachable)
        }
        // Block a room ON slot-1's horizontal corridor between its home (rx=-18) and the rally (~rx=-7):
        // rx=-12, ry=-6 = W11N5. Slot 1's clean horizontal solo step must cross it; pre-fix it sits forever
        // (RC-3 silent NO_PATH). Slot 0 (lead, rx=-7) + slots 2.. (rx=-8) never touch the blocked room.
        t.blocked_rooms = vec![wpos(-12, -6, 25, 25).room_name()]; // W11N5 — squarely on slot-1's corridor
        t
    }

    // ── ADR 0034 Phase 2: S3 RENEW-IN-TRANSIT (RC-5/RC-6/RC-7) ───────────────────────────────────────────
    //
    // A SLOW far-home form against a far CONTESTED target with a FINITE member life. The members spawn far
    // from the rally and must crawl ~12 rooms (~600 ticks at one world-tile/tick); the assault then adds the
    // rally→target leg. With a member life SHORT of (journey + fight), the pre-fix bot commits them anyway —
    // they arrive BELOW `FIGHT_BUFFER` (dead-on-arrival, drop out → re-spawn → churn) and never engage. D6
    // (the pre-departure lifetime gate + renew-to-sufficiency) holds them at home until a renew tops them up,
    // then they depart fit and arrive ABOVE the buffer → engage.

    /// A 3-slot offense roster with a SHORT member life (Phase-2 S3). Same fieldable homes as
    /// `three_slot_offense` (so forming is fast — the failure is LIFETIME, not spawn contention), but
    /// `member_ttl = 700` — short of the ~750-tick (12-room crawl + assault + fight) journey, so a renew is
    /// REQUIRED to deploy fit. Generous budget so the renew-hold + the long crawl both fit.
    fn s3_short_life_offense() -> ColonyFormingScenario {
        let mut s = three_slot_offense(3000);
        s.member_ttl = 700; // SHORT of journey(650) + FIGHT_BUFFER(100) = 750 → renew required (RC-5/RC-7)
        s
    }

    /// S3 geometry: all three members home ~12 rooms from the contested target's one-room-short rally, so the
    /// journey eats most of a member's life. All at the SAME far home (the lifetime axis is isolated from the
    /// lead-pins-the-min RC-4 axis S1 covers). `rally_spawn` is left default-`false` (the home-renew D6a/D6b
    /// fully carries S3; the renewable-rally D6c is exercised separately + is the documented follow-up).
    fn far_home_s3(n_slots: usize) -> ExtendedTravel {
        let homes = (0..n_slots).map(|_| wpos(-19, -6, 25, 25)).collect(); // W18N5 — ~12 rooms from the rally
        ExtendedTravel {
            homes,
            target: wpos(-6, -6, 25, 25), // W5N5 (contested) → rally one room short at W6N5
            uncontested: false,
            blocked_rooms: vec![],
            escalate_on_block: false,
            majority_progress: true, // isolate the LIFETIME axis (the long uniform crawl needs D5 to not min-pin)
            tight_stall_window: false,
            rally_spawn: false,
            lifetime_gate: false,
            renew_in_transit: false,
        }
    }

    /// RC-5/RC-6/RC-7 RED→GREEN: S3 renew-in-transit. PRE-FIX (no lifetime gate, no renew-in-transit) the
    /// short-lived members are committed to the long crawl anyway and arrive BELOW `FIGHT_BUFFER`
    /// (dead-on-arrival → drop out → churn) → never `DeployedAndEngaged`. FIXED (D6a lifetime gate + D6b
    /// renew-to-sufficiency) they HOLD at home, renew up to sufficiency, then depart fit and arrive ABOVE the
    /// buffer → `DeployedAndEngaged`.
    #[test]
    fn far_home_s3_short_life_churns_then_renew_deploys() {
        let s = s3_short_life_offense();
        let n = s.composition.slots.len();
        // RED: short-lived members sent on the long crawl arrive spent → no engage (churn / travel lapse).
        let red = run_lifecycle_churn_extended(&s, &far_home_s3(n));
        // The short-lived members crawl the long corridor but arrive BELOW FIGHT_BUFFER (dead-on-arrival →
        // drop out → the contested quorum never stabilises) → the travel lease lapses. Specifically NOT
        // DeployedAndEngaged — the lifetime attrition (RC-5/RC-7) blocks the engage.
        assert!(
            matches!(red, ChurnOutcome::LapsedInTravel { .. } | ChurnOutcome::OscillatedNeverGathered { .. } | ChurnOutcome::ChurnedNeverDeployed { .. }),
            "RC-5/RC-7: short-lived far members committed without a lifetime gate must NOT engage, got {red:?}"
        );
        // GREEN: D6a (pre-departure lifetime gate) + D6b (renew-to-sufficiency while holding at home) keep the
        // roster fit → it departs above sufficiency + arrives above FIGHT_BUFFER → engage.
        let green_travel = ExtendedTravel {
            lifetime_gate: true,
            renew_in_transit: true,
            ..far_home_s3(n)
        };
        let green = run_lifecycle_churn_extended(&s, &green_travel);
        assert!(
            matches!(green, ChurnOutcome::DeployedAndEngaged { .. }),
            "D6 lifetime gate + renew-to-sufficiency must deploy the roster fit → engage, got {green:?}"
        );
    }

    /// The lifetime GATE alone (no renew) cannot rescue a too-short member — proving D6b (renew) is
    /// load-bearing, not redundant with D6a. With the gate ON but renew OFF, the held members never get topped
    /// up → they hold at home forever (TTL keeps decaying) → never deploy.
    #[test]
    fn far_home_s3_gate_without_renew_still_fails() {
        let s = s3_short_life_offense();
        let n = s.composition.slots.len();
        let out = run_lifecycle_churn_extended(&s, &ExtendedTravel { lifetime_gate: true, ..far_home_s3(n) });
        assert!(
            !matches!(out, ChurnOutcome::DeployedAndEngaged { .. }),
            "D6a gate alone (no renew to top up the held members) can't deploy a too-short roster, got {out:?}"
        );
    }

    /// D6c (renewable-rally bias): with a `rally_spawn` rally, a member that REACHES the rally with low TTL is
    /// renewed THERE (not only at home) — the forward-staging top-up. Exercises the `rally_spawn` arm of the
    /// renew pass. (The home-renew path already carries S3; this pins the rally-spawn branch.)
    #[test]
    fn far_home_s3_renewable_rally_also_deploys() {
        let s = s3_short_life_offense();
        let n = s.composition.slots.len();
        let green = run_lifecycle_churn_extended(
            &s,
            &ExtendedTravel { lifetime_gate: true, renew_in_transit: true, rally_spawn: true, ..far_home_s3(n) },
        );
        assert!(
            matches!(green, ChurnOutcome::DeployedAndEngaged { .. }),
            "D6 with a renewable rally must deploy + engage, got {green:?}"
        );
    }

    /// RC-4/RC-8 RED→GREEN: S1 far-home stall. Pre-fix (min-distance progress, no D5) the held lead pins the
    /// min flat and the lease lapses mid-travel before the bulk gathers → `LapsedInTravel`. D5 (majority
    /// progress) keeps the lease alive while the bulk closes → the quorum fires → `DeployedAndEngaged`.
    #[test]
    fn far_home_s1_min_progress_lapses_then_majority_deploys() {
        let s = three_slot_offense(2000);
        let n = s.composition.slots.len();
        // RED: the pre-fix MIN signal mis-reads the converging squad as stalled.
        let red = run_lifecycle_churn_extended(&s, &far_home_s1(n));
        assert!(
            matches!(red, ChurnOutcome::LapsedInTravel { .. } | ChurnOutcome::OscillatedNeverGathered { .. }),
            "RC-4/RC-8: the pre-fix min-distance progress signal must lapse the far-home squad in travel, got {red:?}"
        );
        // GREEN: D5 majority progress carries the lease until the bulk gathers + the quorum fires.
        let green_travel = ExtendedTravel { majority_progress: true, ..far_home_s1(n) };
        let green = run_lifecycle_churn_extended(&s, &green_travel);
        assert!(
            matches!(green, ChurnOutcome::DeployedAndEngaged { .. }),
            "D5 majority-progress must keep the lease alive while the bulk closes → deploy + engage, got {green:?}"
        );
    }

    /// RC-3 RED→GREEN: S1-BLOCKED. Pre-fix (no member-side feedback, no D4/D8) a blocked member silently
    /// retries forever — the contested quorum never fires and the squad lapses at the coarse travel budget
    /// → `LapsedInTravel`. D4 (escalate on block) + D8 (tight stall window) detect the block fast and
    /// re-assess the blocked member OUT of the gather quorum → the reachable subset masses + `DeployedAndEngaged`.
    #[test]
    fn far_home_s1_blocked_silent_stall_then_escalation_deploys() {
        let s = three_slot_offense(2000);
        let n = s.composition.slots.len();
        // RED: a blocked member with no feedback → silent stall → budget-lapse give-up.
        let red = run_lifecycle_churn_extended(&s, &far_home_s1_blocked(n));
        assert!(
            matches!(red, ChurnOutcome::LapsedInTravel { .. } | ChurnOutcome::OscillatedNeverGathered { .. }),
            "RC-3: a blocked member with no member-side feedback must SILENTLY stall → budget lapse, got {red:?}"
        );
        // GREEN: D4 + D8 (+ D5) detect the block within the tight stall window + re-assess the blocked member
        // OUT of the gather quorum → the reachable subset masses → deploy + engage.
        let green_travel = ExtendedTravel {
            escalate_on_block: true,
            tight_stall_window: true,
            majority_progress: true,
            ..far_home_s1_blocked(n)
        };
        let green = run_lifecycle_churn_extended(&s, &green_travel);
        assert!(
            matches!(green, ChurnOutcome::DeployedAndEngaged { .. }),
            "D4+D8 must detect the block fast + escalate to a reachable rally → deploy + engage, got {green:?}"
        );
    }

    /// D5 ALONE cannot resolve the blocked stall (proving D4/D8 are load-bearing, not redundant with D5): a
    /// blocked member with majority-progress ON but escalation OFF still lapses — the contested quorum needs
    /// the member that can never reach the rally.
    #[test]
    fn far_home_s1_blocked_d5_alone_still_lapses() {
        let s = three_slot_offense(2000);
        let n = s.composition.slots.len();
        let out = run_lifecycle_churn_extended(&s, &ExtendedTravel { majority_progress: true, ..far_home_s1_blocked(n) });
        assert!(
            matches!(out, ChurnOutcome::LapsedInTravel { .. } | ChurnOutcome::OscillatedNeverGathered { .. }),
            "D5 alone can't unblock a NO_PATH member — D4/D8 escalation is required, got {out:?}"
        );
    }

    #[test]
    fn extended_lifecycle_is_deterministic() {
        let s = three_slot_offense(2000);
        let n = s.composition.slots.len();
        let t = far_home_s1_blocked(n);
        assert_eq!(run_lifecycle_churn_extended(&s, &t), run_lifecycle_churn_extended(&s, &t));
        // Phase 2: the TTL/renew/lifetime-gate path is equally deterministic (integer math, no float branch,
        // no HashMap) — same (scenario, travel) → same outcome.
        let s3 = s3_short_life_offense();
        let n3 = s3.composition.slots.len();
        let t3 = ExtendedTravel { lifetime_gate: true, renew_in_transit: true, rally_spawn: true, ..far_home_s3(n3) };
        assert_eq!(run_lifecycle_churn_extended(&s3, &t3), run_lifecycle_churn_extended(&s3, &t3));
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

    // ── ADR 0032 v1.2: the AUCTION FLOW (multi-squad GLOBAL assignment) acceptance + RED control ──

    /// A heterogeneous 2-squad × 2-objective scenario engineered so the per-squad GREEDY baseline is
    /// STRICTLY worse than the GLOBAL Hungarian (the flow analog of the kernel headline). Squad A (row 0) is
    /// a strong all-rounder; squad B (row 1) is weak. Objective H (id 0) and L (id 1) are both undefended +
    /// equally winnable. The trick: B can only take H (L is infeasible for B), and H edges L in value for A,
    /// so GREEDY (A iterated first) grabs H — stranding B with NO objective. The GLOBAL optimum routes A→L
    /// and B→H, claiming BOTH for a higher total. Toggled by `global_solve`.
    fn auction_greedy_suboptimal(global: bool) -> AuctionFlowScenario {
        AuctionFlowScenario {
            squads: vec![
                AuctionSquad { structure_dps: 1000, heal: 50, current_objective: None }, // A (row 0)
                AuctionSquad { structure_dps: 120, heal: 50, current_objective: None },  // B (row 1)
            ],
            objectives: vec![
                // H (id 0): high value, feasible for BOTH. Value edges L so greedy-A grabs it.
                AuctionObjective { id: 0, value: 50_001.0, feasible_per_row: vec![true, true] },
                // L (id 1): slightly lower value, feasible ONLY for A (B cannot reach/take it).
                AuctionObjective { id: 1, value: 50_000.0, feasible_per_row: vec![true, false] },
            ],
            global_solve: global,
        }
    }

    /// THE FLOW HEADLINE (ADR 0032 §Sim — "prove global-optimality in the FLOW, not just the kernel"): the
    /// GLOBAL solve STRICTLY beats the GREEDY baseline on total EV for the heterogeneous scenario. RED arm
    /// (greedy) leaves B unassigned; GREEN arm (global) claims both objectives.
    #[test]
    fn auction_global_strictly_beats_greedy_in_the_flow() {
        let greedy = run_auction_flow(&auction_greedy_suboptimal(false));
        let global = run_auction_flow(&auction_greedy_suboptimal(true));
        assert!(
            global.total_ev > greedy.total_ev,
            "the GLOBAL Hungarian must STRICTLY beat the per-squad GREEDY in the flow: global={} greedy={} (greedy picks={:?}, global picks={:?})",
            global.total_ev,
            greedy.total_ev,
            greedy.picks,
            global.picks
        );
        // Greedy: A→H, B→(stranded, L infeasible) → only H claimed. Global: A→L, B→H → BOTH claimed.
        assert_eq!(greedy.picks, vec![Some(0), None], "greedy strands B (A grabbed H; L infeasible for B)");
        assert_eq!(global.picks, vec![Some(1), Some(0)], "global routes A→L and B→H — the swap greedy misses");
    }

    /// The auction flow is DETERMINISTIC (ADR 0032 §Determinism): the same scenario yields a byte-identical
    /// outcome on repeat, for BOTH arms.
    #[test]
    fn auction_flow_is_deterministic() {
        for global in [false, true] {
            let a = run_auction_flow(&auction_greedy_suboptimal(global));
            let b = run_auction_flow(&auction_greedy_suboptimal(global));
            assert_eq!(a, b, "the auction flow is deterministic (global={global})");
        }
    }

    /// The EV-positive gate in the FLOW (ADR 0032 §EV-positive gate): a squad already on a high-value fight
    /// is offered only a tiny new objective; the global optimum keeps it on StayPut (no objective pick).
    #[test]
    fn auction_flow_respects_the_ev_positive_gate() {
        let s = AuctionFlowScenario {
            // One squad currently on the high-value objective (id 0); a tiny new objective (id 1) is the only
            // unclaimed one. The optimum is StayPut on 0, never the sub-threshold 1.
            squads: vec![AuctionSquad { structure_dps: 1000, heal: 50, current_objective: Some(0) }],
            objectives: vec![
                AuctionObjective { id: 0, value: 100_000.0, feasible_per_row: vec![true] }, // current (StayPut re-scores it)
                AuctionObjective { id: 1, value: 5.0, feasible_per_row: vec![true] },        // tiny new objective
            ],
            global_solve: true,
        };
        let out = run_auction_flow(&s);
        assert_ne!(out.picks[0], Some(1), "must NOT take the sub-threshold objective — StayPut/high-value wins");
    }

    // ── ADR 0032 v2 / ADR 0027: the MERGE FLOW (transfer pending-slot primitive END-TO-END) ──

    use crate::harness::lifecycle::{run_merge_flow, MergeFlowScenario, MergeSlot, MergeSquad};
    use screeps_combat_decision::composition::SquadRole;

    /// A receiver forming for a high-value DEFENDED objective (under-DPS alone, an OPEN Dismantler pending
    /// slot) + a terminal-with-survivors donor whose sheddable Dismantler fills that slot. Greedy on whether
    /// merge is enabled.
    fn merge_reinforce_scenario(merge_enabled: bool) -> MergeFlowScenario {
        MergeFlowScenario {
            squads: vec![
                // Receiver (idx 0): present RangedDPS + an OPEN Dismantler pending slot; under-DPS alone.
                MergeSquad {
                    structure_dps: 200,
                    heal: 50,
                    sheddable_dps: 0,
                    sheddable_heal: 0,
                    current_objective: 0,
                    slots: vec![
                        MergeSlot { role: SquadRole::RangedDPS, filled: true },
                        MergeSlot { role: SquadRole::Dismantler, filled: false }, // OPEN pending slot
                    ],
                    merge_eligible: false,
                },
                // Donor (idx 1): a present Dismantler to shed; merge-eligible; its own objective is low-value.
                MergeSquad {
                    structure_dps: 800,
                    heal: 0,
                    sheddable_dps: 800,
                    sheddable_heal: 0,
                    current_objective: 1,
                    slots: vec![MergeSlot { role: SquadRole::Dismantler, filled: true }],
                    merge_eligible: true,
                },
            ],
            objective_values: vec![200_000.0, 50.0],
            objective_tower_range: vec![20, 0],
            objective_required_hits: vec![400_000, 0],
            merge_enabled,
        }
    }

    /// THE MERGE FLOW HEADLINE (ADR 0027 lines 256-312 — kernel selection + abstract transfer; see the section
    /// header for scope): with the merge column enabled, the kernel picks the merge and the model TRANSFERS the
    /// donor's Dismantler into the receiver's open pending slot — the receiver's open slot is marked filled (by
    /// transfer, not spawn), the donor EMPTIES and cleanly RETIRES, and exactly one member transfers. The RED
    /// control (merge disabled) does NONE of this. (The live spawn-slot drop / no-double-fill is the bot's,
    /// guarded by the `create_spawn_callback` `is_slot_filled` recheck — not asserted here.)
    #[test]
    fn merge_flow_transfers_fills_the_pending_slot_and_retires_the_empty_donor() {
        let green = run_merge_flow(&merge_reinforce_scenario(true));
        // The merge fired: donor 1 → receiver 0, 1 member.
        assert_eq!(green.merge, Some((1, 0, 1)), "the donor merges into the receiver's open slot (1 transfer)");
        // The receiver's pending slot is now filled in the model (2 filled, 0 open — BY TRANSFER, not spawn).
        assert_eq!(green.slots[0], (2, 0), "the receiver's open pending slot is filled by transfer (model)");
        // The donor EMPTIED → clean retire (the creep was TRANSFERRED, not orphaned/deleted).
        assert!(green.retired[1], "the emptied donor retires cleanly");
        assert!(!green.retired[0], "the receiver is not retired");
        assert_eq!(green.slots[1], (0, 0), "the donor has no members left (all transferred)");

        // RED control: merge disabled ⇒ no transfer, no slot drop, no retire.
        let red = run_merge_flow(&merge_reinforce_scenario(false));
        assert_eq!(red.merge, None, "with merge disabled the donor cannot transfer (no merge column)");
        assert_eq!(red.slots[0], (1, 1), "the receiver's pending slot stays OPEN (must be spawned)");
        assert!(!red.retired[1], "the donor does not empty/retire without the transfer");
    }

    /// FORMING-CONSOLIDATION end-to-end (ADR 0027 lines 270-271 — "two squads stuck at 1/4 each → one at
    /// 2/4"): two forming squads each at partial strength consolidate into ONE via the transfer.
    #[test]
    fn merge_flow_consolidates_two_forming_squads() {
        let s = MergeFlowScenario {
            squads: vec![
                // Receiver: 1 present RangedDPS + 1 OPEN RangedDPS pending slot (1/2), under-DPS alone.
                MergeSquad {
                    structure_dps: 150,
                    heal: 50,
                    sheddable_dps: 0,
                    sheddable_heal: 0,
                    current_objective: 0,
                    slots: vec![
                        MergeSlot { role: SquadRole::RangedDPS, filled: true },
                        MergeSlot { role: SquadRole::RangedDPS, filled: false },
                    ],
                    merge_eligible: false,
                },
                // Donor: ALSO forming (1 present RangedDPS, an open slot of its own), merge-eligible, sheds RangedDPS.
                MergeSquad {
                    structure_dps: 150,
                    heal: 50,
                    sheddable_dps: 500,
                    sheddable_heal: 30,
                    current_objective: 1,
                    slots: vec![
                        MergeSlot { role: SquadRole::RangedDPS, filled: true },
                        MergeSlot { role: SquadRole::RangedDPS, filled: false },
                    ],
                    merge_eligible: true,
                },
            ],
            objective_values: vec![150_000.0, 40_000.0],
            objective_tower_range: vec![20, 20],
            objective_required_hits: vec![300_000, 300_000],
            merge_enabled: true,
        };
        let out = run_merge_flow(&s);
        assert_eq!(out.merge, Some((1, 0, 1)), "the donor consolidates its present member into the receiver");
        assert_eq!(out.slots[0], (2, 0), "the receiver is now at 2/2 (consolidated) — the pending slot dropped");
        assert!(out.retired[1], "the donor emptied and retired (two 1/2 squads became one 2/2)");
    }

    /// The merge flow is DETERMINISTIC (ADR 0032 §Determinism): the same scenario yields a byte-identical
    /// outcome on repeat (the donor→slot match is role-matched + stable order).
    #[test]
    fn merge_flow_is_deterministic() {
        for enabled in [false, true] {
            let a = run_merge_flow(&merge_reinforce_scenario(enabled));
            let b = run_merge_flow(&merge_reinforce_scenario(enabled));
            assert_eq!(a, b, "the merge flow is deterministic (enabled={enabled})");
        }
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

    /// ADR 0027 v1.1 P1 — the SALVAGE BREACH migration, proved offline end-to-end. A derelict room's
    /// controller/source is walled behind a dismantlable over-horizon seal: the salvage breach producer emits
    /// a `Dismantle{room, breach-blocker}` candidate (a `DismantleStructure` objective — the dormant
    /// `SiegeBreach` doctrine's FIRST live producer). The corridor's total hits (a feasible wall) are the
    /// `objective_hits` the doctrine sizes the WORK squad against. The production layer maps it through the
    /// SAME winnability gate, a squad claims + forms + travels + ENGAGES (razes) the blocker — the corridor
    /// opens. This is the migrated path: the mission no longer fields its own breach dismantler; the v1
    /// SquadManager sizes + fields it via `SiegeBreach`. Offline-provable + deterministic.
    #[test]
    fn offense_flow_salvage_breach_blocker_fields_sizes_and_dismantles() {
        let s = OffenseFlowScenario {
            home: (0, 0),
            candidates: vec![OffenseCandidate {
                room: (2, 0),
                // The breach corridor blocker is a dismantle-able structure ring → `SiegeBreach` (WORK).
                objective: DoctrineObjective::DismantleStructure,
                honor_verdict: true, // gated offense — must pass the winnability gate to field
                // A feasible dismantlable seal: the corridor's total hits (a single over-horizon wall here).
                defense: DefenseProfile { objective_hits: 30_000, ..Default::default() },
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
            "a feasible breach blocker must field a SiegeBreach squad + engage (dismantle) it end-to-end, got {out:?}"
        );
        assert_eq!(run_offense_flow(&s), out, "the salvage breach flow is deterministic");
    }

    /// The breach winnability gate: a breach corridor sealed past any feasible WORK budget (an enormous hit
    /// pool the `SiegeBreach` sizer can't crack within the on-site window) is DEFERRED — no objective, no
    /// squad. The producer never feeds a dismantler squad to an un-chewable seal (mirrors the mission only
    /// emitting on a feasible `breach_target`).
    #[test]
    fn offense_flow_salvage_breach_infeasible_seal_is_gated_out() {
        let s = OffenseFlowScenario {
            home: (0, 0),
            candidates: vec![OffenseCandidate {
                room: (2, 0),
                objective: DoctrineObjective::DismantleStructure,
                honor_verdict: true,
                // A wall hit-pool far beyond what a WORK squad can chew in the window → unwinnable → deferred.
                defense: DefenseProfile { objective_hits: u32::MAX, ..Default::default() },
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
            "an infeasible breach seal must be gated out (no squad fielded), got {out:?}"
        );
        assert_eq!(run_offense_flow(&s), out, "the salvage breach flow is deterministic");
    }

    // ── ADR 0027 v1.1 P2 — the DECLAIM flow ──────────────────────────────────────────────────────────────

    /// A declaim flow whose `cadence` (the 1000-tick upgrade-block between strikes) DELIBERATELY exceeds the
    /// `COMMITMENT_BUDGET` (400), so the base lease lapses BETWEEN strikes — the exact mid-cadence lapse the
    /// `declaiming` lease-hold must bridge. Needs several strikes to neutralize, proving persistence across
    /// MULTIPLE cadence cycles. The controller is one room from home.
    fn declaim_scenario() -> DeclaimFlowScenario {
        DeclaimFlowScenario {
            home: (0, 0),
            controller_room: (1, 0),
            cadence: 1000,             // the engine upgrade-block (>> COMMITMENT_BUDGET=400)
            strikes_to_neutralize: 3,  // multiple cadence cycles
            form_ticks: 4,
            objective_ttl: 100,        // << cadence — proves the objective survives via the lease, not the TTL
            budget_ticks: 4000,        // room for 3 cadence cycles + form + travel
        }
    }

    /// THE P2 success path: a `Declaim` objective fields a CLAIM declaimer that forms, travels, arrives, and
    /// strikes the controller across EVERY 1000-tick cadence cycle WITHOUT giving up, until the controller is
    /// neutralized + the producer withdraws — the squad retires cleanly (`generations`-free, no churn). This
    /// is precisely what the `declaiming` lease-hold buys: a declaimer has no focus and never engages, so the
    /// base lease lapses between strikes; without the hold this would `GaveUp` mid-neutralization.
    #[test]
    fn declaim_flow_persists_across_the_cadence_and_neutralizes() {
        let out = run_declaim_flow(&declaim_scenario());
        match out {
            DeclaimOutcome::Neutralized { cadence_cycles, .. } => {
                assert_eq!(cadence_cycles, 3, "all three cadence-cycle strikes landed (the squad persisted)");
            }
            other => panic!("declaim must persist across the cadence + neutralize, got {other:?}"),
        }
        assert_eq!(run_declaim_flow(&declaim_scenario()), out, "the declaim flow is deterministic");
    }

    /// The declaimer must persist EVEN THOUGH the lease lapses mid-cadence — i.e. the success above is NOT an
    /// artifact of the lease never lapsing. With `cadence` (1000) >> `COMMITMENT_BUDGET` (400), the lease
    /// DEMONSTRABLY lapses between strikes; the only thing keeping the squad alive is the `declaiming` hold. A
    /// control with the cadence SHORTENED below the budget would also pass — so this asserts the demanding case.
    #[test]
    fn declaim_flow_lease_actually_lapses_between_strikes() {
        let s = declaim_scenario();
        assert!(s.cadence > COMMITMENT_BUDGET, "the test must exercise the mid-cadence lapse the hold bridges");
        let out = run_declaim_flow(&s);
        assert!(
            matches!(out, DeclaimOutcome::Neutralized { .. }),
            "the declaimer holds across a lease that DOES lapse between strikes, got {out:?}"
        );
    }
}
