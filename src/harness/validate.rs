//! Stage 3 — validation (ADR 0023a). A [`Validator`] judges a [`Scenario`], driving evaluation as it
//! sees fit, and is swappable independent of the generator. [`OracleCalibration`] is the P-FORCE WIN
//! gate (Move B) re-expressed on the seams: derive the oracle [`DefenseProfile`] FROM the scenario's
//! world, assess against the fieldable ceiling, size the REAL force, field it on the objective's
//! staging tiles, evaluate the siege, and classify false-positive / false-negative.

use crate::harness::evaluate::{evaluate, AnyOf, ObjectivesDestroyed, SideWiped, StopReason};
use crate::harness::scenario::{Objective, Scenario};
use screeps::Position;
use screeps_combat_agent::objective_bed::defense_intents;
use screeps_combat_decision::bodies::{build_combat_body, CombatBodySpec, MoveProfile};
use screeps_combat_decision::composition::{BodyType, SquadComposition, SquadRole, SquadSlot};
use screeps_combat_decision::damage::tower_repair_at_range;
use screeps_combat_decision::force_sizing::{assess, AssaultMode, DefenseProfile, ForceBudget, RequiredForce, TowerThreat};
use screeps_combat_engine::constants::TOWER_ENERGY_COST;
use screeps_combat_engine::{CombatAction, CombatWorld, Intents, PlayerId, SimBody, SimCreep, StructureId};

/// A validator's judgement of one scenario.
#[derive(Clone, Debug)]
pub struct Verdict {
    pub pass: bool,
    pub label: String,
    pub detail: String,
}

/// A scenario judge — swappable independent of the generator (ADR 0023a stage 3).
pub trait Validator {
    fn label(&self) -> &str;
    fn validate(&mut self, scenario: &Scenario) -> Verdict;
}

// ── OracleCalibration: the P-FORCE WIN gate on the seams ───────────────────────────────────────────

/// How the chosen attacking composition (the breach archetype) is sized + fielded for the FP rows. The
/// FN rows falsify a defer with the same ceiling the verdict referenced.
const CEILING_DISMANTLERS: usize = 3;
const CEILING_HEALERS: usize = 5;

/// The running FP/FN tally `OracleCalibration` accumulates across scenarios.
#[derive(Clone, Copy, Debug, Default)]
pub struct Calibration {
    pub scenarios: u32,
    pub fielded: u32,
    pub false_positives: u32,
    pub deferred: u32,
    pub false_negatives: u32,
    pub unfieldable: u32,
    pub drain_winnable: u32,
}

impl Calibration {
    pub fn fp_rate(&self) -> f64 {
        if self.fielded == 0 {
            0.0
        } else {
            self.false_positives as f64 / self.fielded as f64
        }
    }
    pub fn fn_rate(&self) -> f64 {
        if self.deferred == 0 {
            0.0
        } else {
            self.false_negatives as f64 / self.deferred as f64
        }
    }
    pub fn report(&self) -> String {
        format!(
            "oracle calibration — {} scenarios\n  fielded: {} | false positives: {} (fp_rate {:.3}, gate <= 0.010)\n  deferred: {} | false negatives: {} (fn_rate {:.3}, gate <= 0.200)\n  unfieldable: {} | drain-winnable (diag): {}",
            self.scenarios, self.fielded, self.false_positives, self.fp_rate(), self.deferred, self.false_negatives, self.fn_rate(), self.unfieldable, self.drain_winnable
        )
    }
}

/// The oracle-calibration validator. Holds the cross-scenario [`Calibration`] tally (read via
/// [`OracleCalibration::tally`] after a suite run); each [`Validator::validate`] updates it.
#[derive(Default)]
pub struct OracleCalibration {
    tally: Calibration,
}

impl OracleCalibration {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn tally(&self) -> &Calibration {
        &self.tally
    }
}

impl Validator for OracleCalibration {
    fn label(&self) -> &str {
        "oracle-calibration"
    }

    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        self.tally.scenarios += 1;
        let objective = &scenario.objectives[0]; // single-objective for now (multi-room: Phase C)
        let profile = derive_profile(&scenario.world, scenario.defender_owner, objective);

        // Assess what the sizing SYSTEM can field (the ceiling), not the bare template.
        let ceiling = siege_ceiling(scenario.member_energy);
        let caps = ceiling.capabilities(scenario.member_energy);
        let budget = ForceBudget {
            max_heal_per_tick: caps.heal_per_tick as f32,
            max_dismantle_dps: caps.structure_dps as f32,
            tank_effective_hp: caps.tank_effective_hp as f32,
            onsite_budget_ticks: scenario.onsite_budget,
        };
        let a = assess(&profile, &budget);

        if a.winnable {
            if a.mode == AssaultMode::Drain {
                self.tally.drain_winnable += 1;
                return Verdict { pass: true, label: self.label().into(), detail: "winnable (drain) — diagnostic, not breach-graded".into() };
            }
            match SquadComposition::siege_quad().sized_for(RequiredForce::from_assessment(&a), scenario.member_energy) {
                Some(sized) => match breaches(scenario, objective, &sized) {
                    Some(true) => {
                        self.tally.fielded += 1;
                        Verdict { pass: true, label: self.label().into(), detail: "winnable + fielded → breached".into() }
                    }
                    Some(false) => {
                        self.tally.fielded += 1;
                        self.tally.false_positives += 1;
                        Verdict { pass: false, label: self.label().into(), detail: "FALSE POSITIVE: winnable + fielded but did NOT breach".into() }
                    }
                    None => {
                        self.tally.unfieldable += 1;
                        Verdict { pass: true, label: self.label().into(), detail: "winnable but bed geometry can't place it (excluded)".into() }
                    }
                },
                None => {
                    self.tally.unfieldable += 1;
                    Verdict { pass: true, label: self.label().into(), detail: "winnable but required force exceeds one squad at this energy (defer-by-affordability)".into() }
                }
            }
        } else {
            self.tally.deferred += 1;
            if let Some(true) = breaches(scenario, objective, &ceiling) {
                self.tally.false_negatives += 1;
                Verdict { pass: false, label: self.label().into(), detail: "FALSE NEGATIVE: deferred but the ceiling squad breached".into() }
            } else {
                Verdict { pass: true, label: self.label().into(), detail: "deferred — the ceiling squad could not breach (correct)".into() }
            }
        }
    }
}

/// Derive the oracle [`DefenseProfile`] FROM the bed (consistency by construction): attackers = ALL
/// energized defender towers (the kill-phase worst case, when no tower repairs); `repair_per_tick` =
/// the maintainer's (last energized) `tower_repair_at_range` to the breach rampart; `enemy_dps` = Σ the
/// defender creeps' attack+ranged power; safe-mode from the world. Tower ranges measured to the
/// objective's assault tile.
fn derive_profile(world: &CombatWorld, defender: PlayerId, obj: &Objective) -> DefenseProfile {
    let energized: Vec<(Position, u32)> = world
        .towers
        .iter()
        .filter(|t| t.owner == defender && t.is_alive() && t.energy >= TOWER_ENERGY_COST)
        .map(|t| (t.pos, t.energy))
        .collect();
    let ramparts: Vec<(Position, u32)> = world
        .structures
        .iter()
        .filter(|s| s.kind == screeps_combat_engine::StructureKind::Rampart && s.owner == Some(defender) && s.is_alive())
        .map(|s| (s.pos, s.hits))
        .collect();
    let breach_hits: u32 = ramparts.iter().map(|(_, h)| *h).sum();
    let objective_hits = world.structures.iter().find(|s| s.id == obj.id).map(|s| s.hits).unwrap_or(0);
    let repair_per_tick = match (breach_hits > 0, ramparts.first(), energized.last()) {
        (true, Some((rampart_pos, _)), Some((maintainer_pos, _))) => tower_repair_at_range(maintainer_pos.get_range_to(*rampart_pos)) as f32,
        _ => 0.0,
    };
    let enemy_dps: u32 = world
        .creeps
        .iter()
        .filter(|c| c.owner == defender && c.is_alive())
        .map(|c| c.body.attack_power() + c.body.ranged_attack_power())
        .sum();
    DefenseProfile {
        towers: energized.iter().map(|(p, e)| TowerThreat { range_to_assault: p.get_range_to(obj.assault_pos), energy: *e }).collect(),
        breach_hits,
        objective_hits,
        enemy_dps: enemy_dps as f32,
        repair_per_tick,
        safe_mode: world.safe_mode_owner == Some(defender),
    }
}

/// Place `comp`'s members as attacker creeps on the objective's staging tiles (dismantlers → front,
/// healers → support). `false` ⇒ can't be placed faithfully (a body that won't build, or more members
/// than tiles) → the row is excluded from grading (a geometry limit, not an oracle error).
fn place_squad(world: &mut CombatWorld, obj: &Objective, comp: &SquadComposition, attacker: PlayerId, member_energy: u32) -> bool {
    let (mut d, mut h, mut next_id) = (0usize, 0usize, 1u32);
    for slot in &comp.slots {
        let Some(body) = slot.body_type.build_body(member_energy, MoveProfile::Plains) else {
            return false;
        };
        let tile = match slot.role {
            SquadRole::Dismantler => {
                if d >= obj.front_tiles.len() {
                    return false;
                }
                let t = obj.front_tiles[d];
                d += 1;
                t
            }
            SquadRole::Healer => {
                if h >= obj.support_tiles.len() {
                    return false;
                }
                let t = obj.support_tiles[h];
                h += 1;
                t
            }
            _ => return false,
        };
        world.creeps.push(SimCreep { id: next_id, owner: attacker, pos: tile, body: SimBody::unboosted(&body), fatigue: 0 });
        next_id += 1;
    }
    true
}

/// The scripted siege attacker (sizing-pure, movement-free): dismantlers break the nearest living
/// rampart then the core; healers heal the most-wounded ally (adjacent → Heal, ≤3 → RangedHeal).
fn siege_intents(world: &CombatWorld, attacker: PlayerId, core_id: StructureId, core_pos: Position) -> Intents {
    let mut intents = Intents::new();
    let wounded = world
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner == attacker)
        .min_by_key(|c| c.body.hits)
        .map(|c| (c.id, c.pos));
    for c in world.creeps.iter().filter(|c| c.is_alive() && c.owner == attacker) {
        if c.body.dismantle_power() > 0 {
            let rampart = world
                .structures
                .iter()
                .filter(|s| s.is_alive() && s.kind == screeps_combat_engine::StructureKind::Rampart)
                .min_by_key(|s| c.pos.get_range_to(s.pos));
            let (tpos, tid) = match rampart {
                Some(r) => (r.pos, r.id),
                None => (core_pos, core_id),
            };
            if c.pos.get_range_to(tpos) <= 1 {
                intents.set(c.id, vec![CombatAction::Dismantle(tid)]);
            }
        } else if c.body.heal_power() > 0 {
            if let Some((wid, wpos)) = wounded {
                let r = c.pos.get_range_to(wpos);
                if r <= 1 {
                    intents.set(c.id, vec![CombatAction::Heal(wid)]);
                } else if r <= 3 {
                    intents.set(c.id, vec![CombatAction::RangedHeal(wid)]);
                }
            }
        }
    }
    intents
}

/// Field `comp` against `scenario`'s bed (fresh clone) and run the scripted siege via the generic
/// evaluator; `Some(true)` ⇒ the objective fell within the on-site budget, `None` ⇒ couldn't place.
fn breaches(scenario: &Scenario, obj: &Objective, comp: &SquadComposition) -> Option<bool> {
    let mut world = scenario.world.clone();
    if !place_squad(&mut world, obj, comp, scenario.attacker_owner, scenario.member_energy) {
        return None;
    }
    let attacker = scenario.attacker_owner;
    let defender = scenario.defender_owner;
    let (core_id, core_pos) = (obj.id, obj.pos);
    let run_until = AnyOf(vec![Box::new(ObjectivesDestroyed(vec![core_id])), Box::new(SideWiped(attacker))]);
    let outcome = evaluate(
        world,
        &mut |w| siege_intents(w, attacker, core_id, core_pos),
        &mut |w, intents| defense_intents(w, defender, core_pos, intents),
        &run_until,
        scenario.onsite_budget,
    );
    Some(outcome.stop == StopReason::ObjectivesComplete)
}

/// Largest single-role part count one member can carry at `energy` (reuses the real builder).
fn max_role_parts(spec_of: impl Fn(u32) -> CombatBodySpec, energy: u32) -> u32 {
    (1..=25)
        .rev()
        .find(|&n| build_combat_body(&spec_of(n), MoveProfile::Plains, energy).is_some())
        .unwrap_or(0)
}

/// The strong-but-fieldable SIEGE CEILING — the oracle's BUDGET and the FN falsifier in one (so the
/// verdict and the falsifier reference the same force): siege_quad grown to its practical max within
/// the 8-member cap + the bed geometry (3 dismantlers + 5 healers), each at its per-member part cap.
fn siege_ceiling(energy: u32) -> SquadComposition {
    let work = max_role_parts(|n| CombatBodySpec { work: n, ..Default::default() }, energy);
    let heal = max_role_parts(|n| CombatBodySpec { heal: n, ..Default::default() }, energy);
    let mut slots = Vec::new();
    if work > 0 {
        for _ in 0..CEILING_DISMANTLERS {
            slots.push(SquadSlot { role: SquadRole::Dismantler, body_type: BodyType::Sized(CombatBodySpec { work, ..Default::default() }) });
        }
    }
    if heal > 0 {
        for _ in 0..CEILING_HEALERS {
            slots.push(SquadSlot { role: SquadRole::Healer, body_type: BodyType::Sized(CombatBodySpec { heal, ..Default::default() }) });
        }
    }
    SquadComposition { label: "Siege Ceiling".into(), slots, formation_shape: Default::default(), formation_mode: Default::default(), retreat_threshold: 0.3 }
}
