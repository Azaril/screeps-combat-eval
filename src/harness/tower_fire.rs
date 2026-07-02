//! U7 — the **combined squad+tower fire** scenario (combat-overhaul-plan.md §5, U-TOWER).
//!
//! A defending managed squad + our towers vs a HEALY attacker, so the combined-fire-beats-either-alone
//! win is **measured**, not asserted. The pure `decide_towers`
//! ([`screeps_combat_decision::tower_fire`]) sizes the tower commit to close the REMAINING heal gap
//! after the squad's own fire; this scenario proves the sizing actually kills a healer-backed target
//! FASTER than the squad alone or the towers alone (each of which barely out-heals it, so alone they
//! grind while combined they finish).
//!
//! Three variants over the SAME defender + target:
//! - **squad alone** — the managed squad only (no towers fire); its DPS barely beats the target's heal.
//! - **towers alone** — the towers only (scripted nearest-enemy fire, no squad); ditto.
//! - **combined** — the squad's shared focus drives `decide_towers` so squad+tower fire concentrates on
//!   ONE target and overwhelms the heal.
//!
//! The gate: combined kills the target in STRICTLY FEWER ticks than either alone (and both alone are
//! slower — the split-fire win the prior separate-system tower logic could not achieve, because the
//! squad and towers could focus different enemies / neither overcame the aggregate heal fast).

use screeps::{Direction, Part, Position, RoomCoordinate, RoomName};
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_agent::SimView;
use screeps_combat_decision::tower_fire::{creep_dps_on_focus, decide_towers, SquadFocus, TowerDto, TowerTarget};
use screeps_combat_decision::{decide_squad, SquadMemberView, SquadView, SquadOrderState};
use screeps_combat_engine::{
    resolve_tick, CombatAction, CombatWorld, Intents, MovementState, PlayerId, SimBody, SimCreep,
    SimTower, TowerAction,
};

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos(x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room())
}

/// The healy attacker: a stationary high-HP, self-healing brick at (25,25). No attack parts (so the
/// defending squad never kites away and never reads a mortal-incoming retreat) — it just soaks + heals.
/// `heal_parts` self-heals `heal_parts × HEAL_POWER` per tick (range 0). Enough HP + heal that a
/// half-strength fire source grinds while combined fire finishes fast.
const TARGET_ID: u32 = 100;
fn healy_target(heal_parts: usize, tough_parts: usize) -> SimCreep {
    let mut body = Vec::new();
    body.extend(std::iter::repeat_n(Part::Tough, tough_parts));
    body.extend(std::iter::repeat_n(Part::Heal, heal_parts));
    SimCreep {
        id: TARGET_ID,
        owner: 1,
        pos: pos(25, 25),
        body: SimBody::unboosted(&body),
        fatigue: 0,
        carry_used: 0,
    }
}

/// The defending squad: `count` ranged attackers of `ra_parts` RANGED + matching MOVE, in a file just
/// north of the target so they advance into shooting range and hold. Owner 0 (ours).
fn defender_creeps(count: u32, ra_parts: usize) -> Vec<SimCreep> {
    let body: Vec<Part> = std::iter::repeat_n(Part::RangedAttack, ra_parts)
        .chain(std::iter::repeat_n(Part::Move, ra_parts))
        .collect();
    (0..count)
        .map(|i| SimCreep {
            id: 1 + i,
            owner: 0,
            pos: pos(20 + i as u8, 20),
            body: SimBody::unboosted(&body),
            fatigue: 0,
            carry_used: 0,
        })
        .collect()
}

/// Our defending tower(s) (owner 0), positioned at FAR falloff from the target (range 20 → 150 dmg/tick,
/// the engine floor) so tower fire ALONE is out-healed by the target's 240/tick self-heal — the tower
/// only overcomes the heal when the squad's fire is ADDED (the whole point of U-TOWER: combined fire).
fn defender_towers() -> Vec<SimTower> {
    // (25,45) is Chebyshev range 20 from the target at (25,25) → tower_attack_damage_at_range(20) = 150.
    vec![SimTower { id: 200, owner: 0, pos: pos(25, 45), energy: 100_000, hits: 3000, hits_max: 3000 }]
}

/// The target's self-heal action for the tick (a `Heal` on itself, range 0), if it's still alive.
fn target_self_heal(world: &CombatWorld, intents: &mut Intents) {
    if world.movement.creeps.iter().any(|c| c.id == TARGET_ID && c.is_alive()) {
        intents.set(TARGET_ID, vec![CombatAction::Heal(TARGET_ID)]);
    }
}

/// Ticks for the target to die, or `None` if it survives `max_ticks`. `alive` reads the world.
fn ticks_to_kill(world: &CombatWorld, max_ticks: u32, mut step: impl FnMut(&CombatWorld) -> Intents) -> Option<u32> {
    let mut w = world.clone();
    for t in 1..=max_ticks {
        let mut intents = step(&w);
        target_self_heal(&w, &mut intents);
        resolve_tick(&mut w, &intents);
        if !w.movement.creeps.iter().any(|c| c.id == TARGET_ID && c.is_alive()) {
            return Some(t);
        }
    }
    None
}

/// Compute the defending squad's [`SquadFocus`] for the tower this tick, from the world — the SAME
/// pure `decide_squad` focus the squad itself uses, plus the squad's DPS landed on it
/// ([`creep_dps_on_focus`]). This mirrors the live `TowerMission`'s game→DTO seam (which reads the
/// squad's persisted `focus_target_id` + members) — here we derive it directly from the sim world.
fn squad_focus_for_towers(world: &CombatWorld, member_ids: &[u32], owner: PlayerId) -> Option<SquadFocus> {
    let sim = SimView::from_world(world, owner, pos(25, 25), room());
    // Living in-room members → `SquadMemberView`s (the same shape `ManagedSimSquad` builds).
    let members: Vec<SquadMemberView> = member_ids
        .iter()
        .filter_map(|&id| sim.friend_index(id))
        .map(|fi| {
            let f = &sim.friends()[fi];
            SquadMemberView {
                hits: f.hits,
                hits_max: f.hits_max,
                heal_power: f.working_parts(Part::Heal) as u32,
                pos: Some(f.pos),
                has_ranged: f.has_working(Part::RangedAttack),
                melee_power: f.working_parts(Part::Attack) as u32 * screeps_combat_engine::constants::ATTACK_POWER,
                ranged_power: f.working_parts(Part::RangedAttack) as u32 * screeps_combat_engine::constants::RANGED_ATTACK_POWER,
                damage_taken_last_tick: 0,
                id: f.id,
                dismantle_power: 0,
                claim_power: 0,
            }
        })
        .collect();
    if members.is_empty() {
        return None;
    }
    let view = SquadView {
        members: &members,
        hostiles: sim.hostiles(),
        structures: sim.structures(),
        retreat_threshold: 0.3,
        current_state: SquadOrderState::Engaged,
        enemy_safe_mode: false,
        engage_objective: screeps_combat_decision::EngageObjective::Destroy,
        enemy_stalled: false,
        structure_stalled: false,
        drain_stance: false,
    };
    let focus = decide_squad(&view).focus?;
    // The DPS the squad lands on the focus this tick (the `decide_towers` sizing input) — over the
    // living member DTOs (matched by engine id via `friend_index`), exactly the members `decide_squad`
    // scored above.
    let squad_dps: u32 = member_ids
        .iter()
        .filter_map(|&id| sim.friend_index(id))
        .map(|fi| creep_dps_on_focus(&sim.friends()[fi], focus.pos))
        .sum();
    Some(SquadFocus { id: focus.id, squad_dps })
}

/// Drive our towers with `decide_towers` fed the squad's focus (the combined-fire path), returning the
/// tower intents this tick. Mirrors the live `TowerMission` executor: it resolves each order's target id
/// and fires. `member_ids` are the defending squad's creeps; `owner` is ours.
fn combined_tower_intents(world: &CombatWorld, member_ids: &[u32], owner: PlayerId, intents: &mut Intents) {
    let towers: Vec<TowerDto> = world
        .towers
        .iter()
        .filter(|t| t.is_alive() && t.owner == owner)
        .map(|t| TowerDto { pos: t.pos, energy: t.energy })
        .collect();
    if towers.is_empty() {
        return;
    }
    let tower_ids: Vec<_> = world.towers.iter().filter(|t| t.is_alive() && t.owner == owner).map(|t| t.id).collect();
    let sim = SimView::from_world(world, owner, pos(25, 25), room());
    let squad_focus = squad_focus_for_towers(world, member_ids, owner);
    let decision = decide_towers(&towers, sim.hostiles(), sim.structures(), squad_focus, &std::collections::HashSet::new());
    for order in &decision.orders {
        let Some(target) = order.target else { continue };
        let Some(&tower_id) = tower_ids.get(order.tower_idx) else { continue };
        // Resolve the target creep id (synthetic RawObjectId → engine CreepId) via the view's map.
        if let Some(cid) = sim.creep_for(target.id) {
            intents.set_tower(tower_id, TowerAction::Attack(cid));
        }
    }
}

/// Scripted "towers alone" fire: every owner tower shoots the nearest enemy (the passive-base baseline,
/// no squad coordination) — the `screeps_combat_agent::opponents::tower_intents` behavior, restricted to
/// our owner so it's the fair towers-only comparison.
fn towers_alone_intents(world: &CombatWorld, owner: PlayerId, intents: &mut Intents) {
    for tower in world.towers.iter().filter(|t| t.is_alive() && t.owner == owner) {
        let target = world
            .movement
            .creeps
            .iter()
            .filter(|c| c.is_alive() && c.owner != tower.owner)
            .min_by_key(|c| tower.pos.get_range_to(c.pos));
        if let Some(t) = target {
            intents.set_tower(tower.id, TowerAction::Attack(t.id));
        }
    }
}

/// The three ticks-to-kill measurements for the U7 scenario (`None` ⇒ the target survived the cap).
#[derive(Clone, Copy, Debug)]
pub struct U7Result {
    pub squad_alone: Option<u32>,
    pub towers_alone: Option<u32>,
    pub combined: Option<u32>,
}

impl U7Result {
    /// The combined-fire win: combined KILLS the target and does so in strictly fewer ticks than EITHER
    /// alone (a slower/failing single source). This is the plan's "combined fire beats either alone".
    pub fn combined_wins(&self) -> bool {
        let Some(combined) = self.combined else { return false };
        let beats = |other: Option<u32>| other.is_none_or(|o| combined < o);
        beats(self.squad_alone) && beats(self.towers_alone)
    }
}

/// Run the U7 combined-fire scenario: measure ticks-to-kill for squad-alone, towers-alone, and combined
/// squad+tower fire against a shared healy target. Tuned so each single source barely out-heals the
/// target (a slow grind) while combined fire overwhelms it fast.
pub fn run_u7() -> U7Result {
    // Target: 10 TOUGH + 20 HEAL → 3000 HP, self-heals 240/tick. NEITHER the 3×(7-RANGED) squad (210
    // dps < 240 heal → out-healed) NOR the single far-falloff tower (150 dps < 240 → out-healed) can
    // kill it alone. Only COMBINED fire (210 + 150 = 360 > 240 → net 120/tick) overwhelms the heal — the
    // exact split-fire win `decide_towers` enables (the towers close the gap the squad can't).
    const MAX_TICKS: u32 = 400;
    let member_ids = [1u32, 2, 3];

    // ── squad alone ── the managed squad only (no towers in the world). 3×(7-RANGED) = 210 dps < 240
    // heal → the squad alone cannot kill it (out-healed; it disengages once stalled). Full cap → None.
    let squad_alone = {
        let world = CombatWorld {
            movement: MovementState {
                creeps: {
                    let mut c = defender_creeps(3, 7);
                    c.push(healy_target(20, 10));
                    c
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut squad = ManagedSimSquad::new(0, member_ids.to_vec(), pos(25, 25));
        ticks_to_kill(&world, MAX_TICKS, |_w| squad.step(_w))
    };

    // ── towers alone ── the tower only (scripted nearest-enemy fire), no squad creeps. One far-falloff
    // tower does 150 dps < 240 heal → out-healed → None (the tower alone can't overcome the heal).
    let towers_alone = {
        let world = CombatWorld {
            movement: MovementState {
                creeps: vec![healy_target(20, 10)],
                ..Default::default()
            },
            towers: defender_towers(),
            ..Default::default()
        };
        ticks_to_kill(&world, MAX_TICKS, |w| {
            let mut intents = Intents::new();
            towers_alone_intents(w, 0, &mut intents);
            intents
        })
    };

    // ── combined ── the squad + the towers, the towers driven by `decide_towers` fed the squad's focus.
    let combined = {
        let world = CombatWorld {
            movement: MovementState {
                creeps: {
                    let mut c = defender_creeps(3, 7);
                    c.push(healy_target(20, 10));
                    c
                },
                ..Default::default()
            },
            towers: defender_towers(),
            ..Default::default()
        };
        let mut squad = ManagedSimSquad::new(0, member_ids.to_vec(), pos(25, 25));
        ticks_to_kill(&world, MAX_TICKS, |w| {
            let mut intents = squad.step(w);
            combined_tower_intents(w, &member_ids, 0, &mut intents);
            intents
        })
    };

    U7Result { squad_alone, towers_alone, combined }
}

/// The result of the U7 **lag re-resolution** step: the position the tower PRICED its shot at (the
/// focus's tile when the decision was made), the focus's tile AFTER it moved one step, and whether the
/// `TowerTarget`'s stable id still re-resolves to the (now-moved) focus creep in the new world.
#[derive(Clone, Copy, Debug)]
pub struct U7LagResult {
    /// The focus tile the tower decision priced its shot from (world before the move).
    pub priced_pos: Position,
    /// The focus tile after it stepped one tile (world after the move).
    pub moved_pos: Position,
    /// Whether the captured [`TowerTarget`]'s stable id re-resolves to the focus creep in the moved world
    /// (the property the live executor's `target_id.resolve()` depends on — it fires by id, not the
    /// stale priced position).
    pub id_reresolves_to_focus: bool,
}

/// U7 lag re-resolution (the BEHAVIORAL half of the "targets carry the stable id for lag-safe fire"
/// pin): the live `TowerMission` reads LAST tick's squad focus, so the position it prices is one tick
/// stale by the time it fires — it re-resolves the STABLE creep id to hit the creep's CURRENT tile. The
/// pure `tower_fire` unit test proves every order carries an id; this proves the id actually re-resolves
/// to the moved creep engine-faithfully (which the pure test cannot — `.resolve()` is host-untestable,
/// but the sim's `SimView::creep_for` is the same id→creep map the live executor's resolve performs).
///
/// Capture the `TowerTarget` at world W0 (focus at tile P), step the focus ONE tile to P', then resolve
/// the captured id in a fresh view of the moved world W1 and confirm it still finds the focus creep now
/// at P' ≠ P. A position-only target would miss (fire at empty ground P); the id does not.
pub fn run_u7_lag_reresolution() -> U7LagResult {
    let owner: PlayerId = 0;
    let member_ids = [1u32, 2, 3];
    // A movable healy focus with a MOVE part (so it can step) and no attack (so the squad holds range
    // and does not kite). High HP so it survives the single priced tick — we need it alive at W1.
    const FOCUS_ID: u32 = 100;
    let focus_at = |p: Position| -> SimCreep {
        let mut body = vec![Part::Move];
        body.extend(std::iter::repeat_n(Part::Tough, 40));
        body.extend(std::iter::repeat_n(Part::Heal, 10));
        SimCreep { id: FOCUS_ID, owner: 1, pos: p, body: SimBody::unboosted(&body), fatigue: 0, carry_used: 0 }
    };

    let start = pos(25, 25);
    let world = CombatWorld {
        movement: MovementState {
            creeps: {
                let mut c = defender_creeps(3, 7);
                c.push(focus_at(start));
                c
            },
            ..Default::default()
        },
        towers: defender_towers(),
        ..Default::default()
    };

    // W0: capture the tower's chosen TowerTarget for the focus (id + the priced tile). This is exactly
    // what the live executor holds after `decide_towers` (before it resolves the id and fires).
    let towers: Vec<TowerDto> = world
        .towers
        .iter()
        .filter(|t| t.is_alive() && t.owner == owner)
        .map(|t| TowerDto { pos: t.pos, energy: t.energy })
        .collect();
    let sim0 = SimView::from_world(&world, owner, start, room());
    let squad_focus = squad_focus_for_towers(&world, &member_ids, owner);
    let decision = decide_towers(&towers, sim0.hostiles(), sim0.structures(), squad_focus, &std::collections::HashSet::new());
    let captured: TowerTarget = decision
        .orders
        .iter()
        .find_map(|o| o.target)
        .expect("the tower fires the focus this tick (a target is committed)");
    let priced_pos = captured.pos;

    // Step the focus ONE tile (it has a MOVE part → moves, no fatigue). Resolve the tick so the world
    // advances and the focus is now at P' ≠ P.
    let mut moved = world.clone();
    let mut intents = Intents::new();
    intents.set_move(FOCUS_ID, Direction::Right);
    resolve_tick(&mut moved, &intents);
    let moved_pos = moved
        .movement
        .creeps
        .iter()
        .find(|c| c.id == FOCUS_ID && c.is_alive())
        .expect("the focus is still alive after one tick")
        .pos;

    // W1: rebuild the view over the MOVED world and re-resolve the captured id (the same id→creep map
    // the live executor's `target_id.resolve()` walks). It must still find the focus creep — now at P'.
    let sim1 = SimView::from_world(&moved, owner, moved_pos, room());
    let id_reresolves_to_focus = sim1.creep_for(captured.id) == Some(FOCUS_ID);

    U7LagResult { priced_pos, moved_pos, id_reresolves_to_focus }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// U7 (the combined-fire-beats-either-alone win, MEASURED): combined squad+tower fire kills the
    /// healy target in strictly fewer ticks than either the squad or the towers alone. This is the
    /// split-fire win the prior separate-system tower logic couldn't achieve (the squad and towers
    /// could split fire onto different enemies / neither overcame the aggregate heal fast enough).
    #[test]
    fn u7_combined_fire_beats_either_alone() {
        let r = run_u7();
        // NEITHER single source kills the healy target alone (both out-healed): the strongest form of
        // the win — combined fire overcomes a heal that defeats each source individually.
        assert_eq!(r.squad_alone, None, "the squad alone is out-healed (210 dps < 240 heal)");
        assert_eq!(r.towers_alone, None, "the tower alone is out-healed (150 dps < 240 heal)");
        // Combined must KILL the target.
        assert!(r.combined.is_some(), "combined squad+tower fire kills the healy target (combined={:?})", r.combined);
        // And strictly faster than either single source (each of which grinds or fails).
        assert!(
            r.combined_wins(),
            "combined fire beats either alone — combined={:?}, squad_alone={:?}, towers_alone={:?}",
            r.combined,
            r.squad_alone,
            r.towers_alone
        );
    }

    /// U7 lag re-resolution (the BEHAVIORAL lag-safety proof the pure `tower_fire` pin defers to): the
    /// `TowerTarget` captured at world W0 (priced at the focus's tile P) still re-resolves by its stable
    /// id to the focus creep after it steps one tile to P' ≠ P. This exercises the same id→creep map the
    /// live executor's `target_id.resolve()` walks — a position-only target would fire at empty ground P.
    #[test]
    fn u7_tower_target_id_reresolves_after_the_focus_moves() {
        let r = run_u7_lag_reresolution();
        // The focus actually MOVED — the priced position is now stale (P' ≠ P), so this is a real lag test
        // and not a no-op where re-resolution would trivially hit the same tile.
        assert_ne!(r.moved_pos, r.priced_pos, "the focus stepped one tile → the priced position is now stale");
        // And the captured id still re-resolves to the (moved) focus creep — the load-bearing property.
        assert!(
            r.id_reresolves_to_focus,
            "the TowerTarget's stable id re-resolves to the moved focus (priced={:?}, moved={:?})",
            r.priced_pos,
            r.moved_pos
        );
    }
}
