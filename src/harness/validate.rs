//! Stage 3 — validation (ADR 0023a). A [`Validator`] judges a [`Scenario`], driving evaluation as it
//! sees fit, and is swappable independent of the generator. [`OracleCalibration`] is the P-FORCE WIN
//! gate (Move B) re-expressed on the seams: derive the oracle [`DefenseProfile`] FROM the scenario's
//! world, assess against the fieldable ceiling, size the REAL force, field it on the objective's
//! staging tiles, evaluate the siege, and classify false-positive / false-negative.

use crate::harness::evaluate::{evaluate, evaluate_recorded, AnyOf, ObjectivesDestroyed, SideWiped, StopReason};
use crate::harness::scenario::{Objective, Scenario};
use crate::harness::visualize::{replay_to_html, ReplayMeta};
use screeps::Position;
use screeps_combat_agent::objective_bed::defense_intents;
use screeps_combat_agent::opponents::tower_intents;
use screeps_combat_agent::squad::ManagedSimSquad;
use screeps_combat_decision::bodies::{build_combat_body, CombatBodySpec, MoveProfile};
use screeps_combat_decision::composition::{assemble_force, formation_for, optimizer_ceiling_budget, BodyType, SquadComposition, SquadRole, SquadSlot};
use screeps_combat_decision::damage::tower_repair_at_range;
use screeps_combat_decision::doctrine::{decide_doctrine, default_doctrines, DoctrineObjective, EnemyCoordination, EnemyForce, EngagementContext, ForcePlan};
use screeps_combat_decision::force_sizing::{clear_force, AssaultMode, DefenseProfile, ForceBudget, TowerThreat, COORDINATED_DPS_MARGIN};
use screeps_combat_engine::constants::TOWER_ENERGY_COST;
use screeps_combat_engine::{CombatAction, CombatWorld, CreepId, Intents, PlayerId, SimBody, SimCreep, StructureId};

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
/// [`OracleCalibration::tally`] after a suite run); each [`Validator::validate`] updates it. The
/// `params` field is the TUNING SEAM (ADR 0031 D16/D17): a sweep injects a non-Default `CompositionParams`
/// to grade alternative knob sets; `Default`/`new()` reproduce the shipped fielding seeds.
#[derive(Default)]
pub struct OracleCalibration {
    tally: Calibration,
    params: screeps_combat_decision::composition::CompositionParams,
}

impl OracleCalibration {
    pub fn new() -> Self {
        Self::default()
    }
    /// Build a calibration validator that grades against the given knob set (the sweep injector).
    pub fn with_params(params: screeps_combat_decision::composition::CompositionParams) -> Self {
        Self { tally: Calibration::default(), params }
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

        // Select + size VIA THE DOCTRINE REGISTRY (parity with the bot). Assess against what the sizing
        // SYSTEM can field (the ceiling), not the bare template.
        let ceiling = siege_ceiling(scenario.member_energy);
        let budget = ceiling.force_budget(scenario.member_energy, scenario.onsite_budget);
        let plan = siege_doctrine_plan_with(profile, budget, scenario.member_energy, defender_force(scenario), &self.params);

        if plan.winnable() {
            if plan.assessment.mode == AssaultMode::Drain {
                self.tally.drain_winnable += 1;
                return Verdict { pass: true, label: self.label().into(), detail: "winnable (drain) — diagnostic, not breach-graded".into() };
            }
            match plan.composition {
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
/// the maintainer's (last energized) `tower_repair_at_range` to the breach rampart; safe-mode from the world.
/// Tower ranges measured to the objective's assault tile. ADR 0031 #41 — STRUCTURE-only: the defender CREEP
/// dps is carried separately on the single [`EnemyForce`] channel (`defender_force`), not on this profile.
pub(crate) fn derive_profile(world: &CombatWorld, defender: PlayerId, obj: &Objective) -> DefenseProfile {
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
    DefenseProfile {
        towers: energized.iter().map(|(p, e)| TowerThreat { range_to_assault: p.get_range_to(obj.assault_pos), energy: *e }).collect(),
        breach_hits,
        objective_hits,
        // ADR 0031 #41: the defender CREEP dps is no longer carried on the profile — it lives on the single
        // [`EnemyForce`] channel (`defender_force`), threaded into `assess` via `siege_doctrine_plan`'s
        // `enemy_force` arg. `derive_profile` is now STRUCTURE-only (towers / breach / objective / repair /
        // safe-mode); `defender_force` (and `enemy_force_of`) derive the SAME defender attack+ranged power,
        // so the value `assess` reads is read-equivalent — the calibration input is unchanged.
        repair_per_tick,
        safe_mode: world.safe_mode_owner == Some(defender),
        // ADR 0035 D1: the harness derives a profile FROM a known scenario world, so the towers are always
        // genuinely SEEN (empty ⇒ ScoutedEmpty, non-empty ⇒ Seen) — never the vacuous never-confirmed case.
        tower_intel: screeps_combat_decision::force_sizing::tower_intel_from(energized.is_empty(), true),
    }
}

/// Decide + size the siege force for a structure-breach scenario VIA THE DOCTRINE REGISTRY (ADR 0026 §9)
/// — the SAME selection + sizing path the bot's offense runs (parity; no divergent inline `assess` +
/// `siege_quad().sized_for` in the eval). A bed objective is a dismantle-able structure breach →
/// `DoctrineObjective::DismantleStructure` → the `SiegeBreach` doctrine sizes a `siege_quad` to the
/// oracle's required force against `budget` (the siege ceiling's, the calibration lens). The returned
/// `ForcePlan` carries the verdict (`assessment`) + the sized `composition` (`None` = defer / drain /
/// unfieldable). `importance: 0.0` matches the eval's base-force sizing (`importance_margin(0)` = 1×).
/// A deliberately HIGH target value so the EV optimizer (ADR 0031 D16) commits ANY winnable bed (EV > 0 ⇒
/// EV > commit threshold 0) — preserving the OracleCalibration FP/FN semantics (the calibration grades
/// winnable→fielded→breached; a low value must not turn winnable beds into defers).
const CALIBRATION_TARGET_VALUE: f32 = 1_000_000.0;

pub(crate) fn siege_doctrine_plan(profile: DefenseProfile, budget: ForceBudget, member_energy: u32, enemy_force: Option<EnemyForce>) -> ForcePlan {
    // The Default-knob plan (the seed): the calibration gates + every existing caller route through here, so
    // Default must reproduce the shipped fielding seeds (ADR 0031 D16 — Default is behavior-preserving).
    siege_doctrine_plan_with(profile, budget, member_energy, enemy_force, &screeps_combat_decision::composition::CompositionParams::default())
}

/// As [`siege_doctrine_plan`] but with chosen [`CompositionParams`] — the TUNING SEAM the param sweep
/// injects through (ADR 0031 D16/D17 / 0031a §4). `params.member_energy` is clamped to the bed's home
/// capacity (`member_energy`) — the swept per-member cap can never exceed what the home affords (matching
/// `optimize_composition`'s `min(member_energy, …)` probe). The rest of `params` (hold/over-power/dynamic/
/// commit-EV/cost weights) is threaded verbatim into the doctrine's `EngagementContext` → `plan_engagement`
/// → `optimize_composition`/`emit_requirement` (already param-driven since ADR 0031 P2/P3).
pub(crate) fn siege_doctrine_plan_with(
    profile: DefenseProfile,
    budget: ForceBudget,
    member_energy: u32,
    enemy_force: Option<EnemyForce>,
    params: &screeps_combat_decision::composition::CompositionParams,
) -> ForcePlan {
    // Coordination from the OBSERVED defenders: grouped (count > 1) or self-healing → Coordinated over-match;
    // none → Individual. (ADR 0031 P1b: feeding `enemy_force` is what triggers the SiegeBreach anti-creep
    // fusion on a defended bed; a creep-free bed passes `None` → the structure path is unperturbed.)
    let coordination = match enemy_force {
        Some(ef) if ef.count > 1 || ef.heal > 0.0 => EnemyCoordination::Coordinated,
        _ => EnemyCoordination::Individual,
    };
    // The swept per-member cap never exceeds the home's capacity (the bed's `member_energy`).
    let effective_member_energy = params.member_energy.min(member_energy);
    let ctx = EngagementContext {
        objective: DoctrineObjective::DismantleStructure,
        coordination,
        defense: profile,
        enemy_force,
        importance: 0.0,
        member_energy: effective_member_energy,
        // ADR 0031 D16: target_value high enough that "EV > commit" ⇔ "winnable", so the OracleCalibration
        // FP/FN semantics (winnable→fielded→breached) are PRESERVED — a low value must NOT turn winnable
        // beds into defers. window = the scenario's on-site budget (carried on `budget`).
        target_value: CALIBRATION_TARGET_VALUE,
        onsite_window: budget.onsite_budget_ticks,
        params: screeps_combat_decision::composition::CompositionParams { member_energy: effective_member_energy, ..*params },
        // The eval bed is a fully-specified, KNOWN defense (reliable intel by construction). Inert here
        // (DismantleStructure → the gated SiegeBreach never reaches the always-field floor), but set
        // truthfully so the field's meaning is unambiguous.
        defense_intel_reliable: true,
    };
    let doctrines = default_doctrines();
    let doctrine = decide_doctrine(&ctx, &doctrines).expect("DismantleStructure routes to the siege-breach doctrine");
    screeps_combat_decision::doctrine::plan_engagement(doctrine, &ctx, Some(budget))
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

// ── ManagedSquadIntegration: drive the REAL squad brain (movement/pathing) — the traversal lens ────

/// Decide what to FIELD for a scenario (the sized squad when winnable+breach, else the ceiling), with a
/// human label. Shared by the replay + managed-assault paths.
fn choose_fielded_comp(scenario: &Scenario, obj: &Objective) -> (SquadComposition, String) {
    let profile = derive_profile(&scenario.world, scenario.defender_owner, obj);
    let ceiling = siege_ceiling(scenario.member_energy);
    let budget = ceiling.force_budget(scenario.member_energy, scenario.onsite_budget);
    let plan = siege_doctrine_plan(profile, budget, scenario.member_energy, defender_force(scenario));
    if plan.winnable() && plan.assessment.mode == AssaultMode::Breach {
        match plan.composition {
            Some(sized) => (sized, "winnable → fielded the sized force".to_string()),
            None => (ceiling, "winnable but the comp can't field the required force; showing the ceiling".to_string()),
        }
    } else if plan.winnable() {
        (ceiling, "winnable (drain mode); showing the ceiling".to_string())
    } else {
        (ceiling, format!("deferred ({}); showing the ceiling", plan.assessment.reason))
    }
}

/// The composition the MANAGED lens fields: a ranged+heal combat quad the squad brain can actually
/// drive (advance, kite, focus-fire creeps, shoot structures), auto-sized to the home's energy. The
/// catalog `quad_ranged` it used to field is gone (ADR 0031 P4b), so it is reconstructed template-free
/// from `Sized` bodies — the same 2×RangedDPS + 2×Healer Box2x2 the positioning gates are calibrated
/// against. (The sizing-pure siege force is `choose_fielded_comp`/`OracleCalibration`.)
fn managed_assault_comp(scenario: &Scenario) -> SquadComposition {
    use screeps_combat_decision::composition::{FormationMode, FormationShape};
    let energy = scenario.member_energy;
    let ranged = max_role_parts(|n| CombatBodySpec { ranged_attack: n, ..Default::default() }, energy);
    let heal = max_role_parts(|n| CombatBodySpec { heal: n, ..Default::default() }, energy);
    let mut slots = Vec::new();
    for _ in 0..2 {
        slots.push(SquadSlot { role: SquadRole::RangedDPS, body_type: BodyType::Sized(CombatBodySpec { ranged_attack: ranged, ..Default::default() }) });
    }
    for _ in 0..2 {
        slots.push(SquadSlot { role: SquadRole::Healer, body_type: BodyType::Sized(CombatBodySpec { heal, ..Default::default() }) });
    }
    SquadComposition {
        label: "Quad Ranged".into(),
        slots,
        formation_shape: FormationShape::Box2x2,
        formation_mode: FormationMode::Strict,
        retreat_threshold: 0.3,
    }
}

/// Place `comp`'s members as attacker creeps clustered at the objective's ENTRY (a MOVING assault),
/// returning their ids in slot order (the `ManagedSimSquad` roster). `None` ⇒ a body wouldn't build.
fn place_at_entry(world: &mut CombatWorld, obj: &Objective, comp: &SquadComposition, attacker: PlayerId, energy: u32) -> Option<Vec<CreepId>> {
    let (ex, ey, rm) = (obj.entry.x().u8() as i32, obj.entry.y().u8() as i32, obj.entry.room_name());
    let need = comp.slots.len();
    // DISTINCT in-bounds, non-wall, unoccupied tiles — one creep per tile, ALWAYS. The previous version
    // spread the original 9 offsets then `.clamp(0,49)`'d them, which COLLAPSED out-of-bounds offsets onto
    // the edge whenever the entry sat on a room boundary (the multi-room base scenarios stage the squad on
    // the border tile) — spawning two creeps on one tile. That illegal start state (a creep stack the real
    // server can never produce) was then transported across the border by the occupancy-blind edge-exit
    // relocation; it is the true source of the sim's "two creeps on a tile". Fix: keep the legacy 9-offset
    // SHAPE first (so a normal in-bounds entry is placed byte-identically — no dynamics change for the
    // single-room beds), but SKIP (never clamp) any out-of-bounds / wall / already-taken tile and fall back
    // to expanding rings only when an edge entry leaves the 9 short. Collision-free either way.
    const OFF: [(i32, i32); 9] = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)];
    let tiles: Vec<(u8, u8)> = {
        let terrain = world.terrain_for(rm);
        let mut taken: std::collections::HashSet<(u8, u8)> =
            world.creeps.iter().filter(|c| c.pos.room_name() == rm).map(|c| (c.pos.x().u8(), c.pos.y().u8())).collect();
        let mut offsets: Vec<(i32, i32)> = OFF.to_vec();
        for r in 2..=10i32 {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs().max(dy.abs()) == r {
                        offsets.push((dx, dy));
                    }
                }
            }
        }
        let mut out: Vec<(u8, u8)> = Vec::with_capacity(need);
        for (dx, dy) in offsets {
            if out.len() == need {
                break;
            }
            let (x, y) = (ex + dx, ey + dy);
            if !(0..=49).contains(&x) || !(0..=49).contains(&y) {
                continue; // SKIP off-room (never clamp — clamping is what stacked creeps)
            }
            let (x, y) = (x as u8, y as u8);
            if terrain.is_wall(x, y) || !taken.insert((x, y)) {
                continue; // wall or already taken → skip (never reuse a tile)
            }
            out.push((x, y));
        }
        out
    };
    if tiles.len() < need {
        return None; // not enough free tiles near the entry to field the whole squad
    }
    let mut ids = Vec::new();
    for (i, slot) in comp.slots.iter().enumerate() {
        let body = slot.body_type.build_body(energy, MoveProfile::Plains)?;
        let (x, y) = tiles[i];
        let id = i as u32 + 1;
        let pos = Position::new(screeps::RoomCoordinate::new(x).unwrap(), screeps::RoomCoordinate::new(y).unwrap(), rm);
        world.creeps.push(SimCreep { id, owner: attacker, pos, body: SimBody::unboosted(&body), fatigue: 0 });
        ids.push(id);
    }
    Some(ids)
}

/// The run-until stop condition for an objective, BY KIND (ADR 0025 §12 Stage 2). Always wrapped with
/// the attacker-wiped guard (a destroyed attacker ends the run). `Raze`/`Breach` destroy the target
/// structure (`Breach`'s `id` is the rampart gate); `Secure` clears the defender side; `Declaim`
/// neutralizes the controller at the objective tile; `Farm` has no terminal (holds to the on-site budget).
fn run_until_for(scenario: &Scenario, obj: &Objective) -> AnyOf {
    use crate::harness::scenario::ObjectiveKind;
    let mut conds: Vec<Box<dyn crate::harness::evaluate::RunUntil>> = vec![Box::new(SideWiped(scenario.attacker_owner))];
    match obj.kind {
        ObjectiveKind::Raze | ObjectiveKind::Breach => conds.push(Box::new(ObjectivesDestroyed(vec![obj.id]))),
        ObjectiveKind::Secure => conds.push(Box::new(SideWiped(scenario.defender_owner))),
        ObjectiveKind::Declaim => conds.push(Box::new(crate::harness::evaluate::ControllerNeutralized(obj.pos))),
        ObjectiveKind::Farm => {}
    }
    AnyOf(conds)
}

/// Field `comp` as a MOVING managed squad at the entry and run the REAL `decide_squad_with_pathing`
/// brain to the objective (recording each tick). `None` ⇒ couldn't field.
fn run_managed_assault(scenario: &Scenario, obj: &Objective, comp: &SquadComposition) -> Option<(crate::harness::evaluate::EvalOutcome, screeps_combat_engine::CombatRecording)> {
    run_managed_assault_with(scenario, obj, comp, screeps_combat_decision::kite::SquadTacticParams::default())
}

/// As [`run_managed_assault`] but with chosen squad tactics — the seam the base-attack tuning pass uses to
/// field the assault under each `KernelParams` candidate (ADR 0025 basket enrichment: base attack/defend).
pub(crate) fn run_managed_assault_with(
    scenario: &Scenario,
    obj: &Objective,
    comp: &SquadComposition,
    tactics: screeps_combat_decision::kite::SquadTacticParams,
) -> Option<(crate::harness::evaluate::EvalOutcome, screeps_combat_engine::CombatRecording)> {
    let mut world = scenario.world.clone();
    let members = place_at_entry(&mut world, obj, comp, scenario.attacker_owner, scenario.member_energy)?;
    let mut squad = ManagedSimSquad::new(scenario.attacker_owner, members, obj.assault_pos).with_tactics(tactics);
    let defender = scenario.defender_owner;
    let core_pos = obj.pos;
    let run_until = run_until_for(scenario, obj);
    let out = evaluate_recorded(
        world,
        &mut |w| squad.step(w),
        &mut |w, intents| defense_intents(w, defender, core_pos, intents),
        &run_until,
        scenario.onsite_budget,
    );
    Some(out)
}

/// ADR 0031 #39 P2/P3 — as [`run_managed_assault_with`] but fields the squad in the tower-DRAIN STANCE
/// (`with_drain_stance(true)`): the TOUGH+HEAL tank holds the falloff standoff while the FINITE towers bleed
/// dry (the engine's `defense_intents` fires them, decrementing energy 10/shot/tick), then the squad advances
/// and dismantles the dead base. This is the END-TO-END oracle-driven drain (the oracle picked `Drain` + sized
/// the comp; here the SAME comp runs through the SAME `decide_squad` the live bot threads via P3) — the runtime
/// proof that complements the tactic-layer `a_drain_squad_bleeds_finite_towers_dry_then_breaches`.
pub(crate) fn run_managed_assault_drain(
    scenario: &Scenario,
    obj: &Objective,
    comp: &SquadComposition,
    tactics: screeps_combat_decision::kite::SquadTacticParams,
) -> Option<(crate::harness::evaluate::EvalOutcome, screeps_combat_engine::CombatRecording)> {
    let mut world = scenario.world.clone();
    let members = place_at_entry(&mut world, obj, comp, scenario.attacker_owner, scenario.member_energy)?;
    let mut squad = ManagedSimSquad::new(scenario.attacker_owner, members, obj.assault_pos)
        .with_tactics(tactics)
        .with_drain_stance(true);
    let defender = scenario.defender_owner;
    let core_pos = obj.pos;
    let run_until = run_until_for(scenario, obj);
    let out = evaluate_recorded(
        world,
        &mut |w| squad.step(w),
        &mut |w, intents| defense_intents(w, defender, core_pos, intents),
        &run_until,
        scenario.onsite_budget,
    );
    Some(out)
}

/// The managed ATTACKER's **period-2 oscillation rate** over a scenario's recorded assault — the
/// durable ADR 0024 regression gate (replaces the ad-hoc node A-B-A script). `None` when the squad
/// couldn't be fielded at the entry (excluded from the metric, like the integration lens).
pub fn managed_oscillation_rate(scenario: &Scenario) -> Option<f64> {
    let obj = &scenario.objectives[0];
    let comp = managed_assault_comp(scenario);
    let (_, rec) = run_managed_assault(scenario, obj, &comp)?;
    Some(crate::metrics::oscillation_rate(&rec, scenario.attacker_owner))
}

/// An attacker squad's assault outcome on a defended base (ADR 0025 base attack/defend lens).
#[derive(Clone, Copy, Debug)]
pub struct AssaultScore {
    pub objective_hp_removed: u32,
    pub objective_destroyed: bool,
    pub attacker_hp_retained: u32,
    /// Ticks the assault took to resolve (breach / wipe / budget) — fewer = a better-positioned breach.
    pub ticks: u32,
    /// The aggregate the base-attack tuner ranks on. With a WINNABLE-sized force every config breaches the
    /// crackable bases identically, so a binary "did it breach" doesn't discriminate positioning; the
    /// quality signal is EFFICIENCY — breach FAST with MANY survivors (good positioning takes less tower/
    /// defender damage en route). So: objective razed + destroyed bonus + survival×2 − a per-tick penalty.
    pub score: i64,
}

/// Score the managed attacker squad's assault on `scenario`'s defended base under `tactics` — the
/// objective-aware base-attack lens (cf. the symmetric open-combat tournament): how much of the objective
/// it razed, whether it cracked it, and how much of itself it kept. `None` ⇒ couldn't field at the entry.
pub fn assault_score(scenario: &Scenario, tactics: screeps_combat_decision::kite::SquadTacticParams) -> Option<AssaultScore> {
    let obj = &scenario.objectives[0];
    // Field a WINNABLE-SIZED siege force (the force-sizing solver's breach comp) — not the weak
    // `quad_ranged` — so the squad can actually crack the mid bases and POSITIONING discriminates the
    // KernelParams (the quad only chipped → all configs tied). The turtle's required force exceeds a
    // single placeable squad, so it stays hard (correctly). (place_at_entry fields up to 9 of the comp.)
    let (comp, _) = choose_fielded_comp(scenario, obj);
    let obj_hits_0 = scenario.world.structures.iter().find(|s| s.id == obj.id).map(|s| s.hits).unwrap_or(0);
    let (outcome, rec) = run_managed_assault_with(scenario, obj, &comp, tactics)?;
    let final_hits = outcome.world.structures.iter().find(|s| s.id == obj.id).map(|s| s.hits);
    let destroyed = !matches!(final_hits, Some(h) if h > 0); // gone from the world OR at 0 hits
    let removed = obj_hits_0.saturating_sub(final_hits.unwrap_or(0));
    let attacker_hp: u32 = outcome.world.creeps.iter().filter(|c| c.owner == scenario.attacker_owner && c.is_alive()).map(|c| c.body.hits).sum();
    let ticks = rec.frames.len() as u32;
    // Efficiency-weighted: razed HP + destroyed bonus + survival×2 (the position-sensitive term — a
    // better-placed assault eats less tower/defender fire) − a per-tick penalty (faster breach is better).
    // Seeds; the tournament tunes the kernel against this, not the score weights themselves.
    let score = removed as i64 + if destroyed { 50_000 } else { 0 } + (attacker_hp as i64) * 2 - (ticks as i64) * 10;
    Some(AssaultScore { objective_hp_removed: removed, objective_destroyed: destroyed, attacker_hp_retained: attacker_hp, ticks, score })
}

/// The **traversal lens** (ADR 0023a stage 3): field the real moving squad and grade whether it
/// NAVIGATES to + ENGAGES the objective (the movement/pathing the operator validates) — distinct from
/// `OracleCalibration` (which is sizing-pure, in-range). Pass = the assault breached, was wiped engaging,
/// or reached the objective vicinity; fail = it never approached (a pathing break).
#[derive(Default)]
pub struct ManagedSquadIntegration;

impl Validator for ManagedSquadIntegration {
    fn label(&self) -> &str {
        "managed-squad-integration"
    }
    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        let obj = &scenario.objectives[0];
        // Field a COMBAT composition (ranged + heal) — the managed brain (`decide_combat`) engages
        // creeps + structures with it, unlike a WORK-only siege squad it can't drive. The sizing-pure
        // siege lens is `OracleCalibration`; this is the movement/engagement lens.
        let comp = managed_assault_comp(scenario);
        let Some((outcome, _rec)) = run_managed_assault(scenario, obj, &comp) else {
            return Verdict { pass: true, label: self.label().into(), detail: "could not field at the entry (excluded)".into() };
        };
        let reached_vicinity = outcome
            .world
            .creeps
            .iter()
            .any(|c| c.is_alive() && c.owner == scenario.attacker_owner && c.pos.room_name() == obj.room && c.pos.get_range_to(obj.pos) <= 8);
        let pass = match outcome.stop {
            StopReason::ObjectivesComplete | StopReason::ControllerNeutralized | StopReason::SideWiped(_) => true,
            StopReason::Timeout => reached_vicinity,
        };
        Verdict {
            pass,
            label: self.label().into(),
            detail: format!("managed ranged assault → {:?} @ t{} (reached={reached_vicinity})", outcome.stop, outcome.ticks),
        }
    }
}

// ── SelfPlay: BOTH sides run the real squad brain (the realistic, moving-on-both-sides engagement) ──

/// Merge `src` intents into `dst` (the engine merges both squads' intents into one resolved tick).
fn merge_intents(dst: &mut Intents, src: Intents) {
    dst.creeps.extend(src.creeps);
    dst.moves.extend(src.moves);
    dst.pulls.extend(src.pulls);
    dst.reasons.extend(src.reasons);
}

/// Run a self-play engagement: the attacker (a fielded combat quad) AND the defender (the scenario's
/// force) BOTH driven by the real `ManagedSimSquad` brain (`decide_squad_with_pathing` — advance / kite
/// / focus-fire), with the defender's towers firing (`tower_intents`). Recording. `None` ⇒ couldn't
/// field the attacker.
fn run_self_play(scenario: &Scenario, obj: &Objective) -> Option<(crate::harness::evaluate::EvalOutcome, screeps_combat_engine::CombatRecording, Vec<CreepId>)> {
    let mut world = scenario.world.clone();
    let attacker_comp = managed_assault_comp(scenario);
    let att_ids = place_at_entry(&mut world, obj, &attacker_comp, scenario.attacker_owner, scenario.member_energy)?;
    // The defender squad = the scenario's pre-placed defender creeps (the ForceSpec opponent).
    let def_ids: Vec<CreepId> = world.creeps.iter().filter(|c| c.is_alive() && c.owner == scenario.defender_owner).map(|c| c.id).collect();
    let mut att = ManagedSimSquad::new(scenario.attacker_owner, att_ids, obj.assault_pos);
    let mut def = ManagedSimSquad::new(scenario.defender_owner, def_ids.clone(), obj.pos); // defender holds the core
    // Stop on objective destroyed or the ATTACKER wiped; only stop on the DEFENDER wiped when there ARE
    // defender creeps — else `SideWiped(defender)` is true at tick 0 (a tower-only / no-creep defense)
    // and the engagement records ZERO frames (the empty-recording bug).
    let mut conditions: Vec<Box<dyn crate::harness::evaluate::RunUntil>> =
        vec![Box::new(ObjectivesDestroyed(vec![obj.id])), Box::new(SideWiped(scenario.attacker_owner))];
    if !def_ids.is_empty() {
        conditions.push(Box::new(SideWiped(scenario.defender_owner)));
    }
    let run_until = AnyOf(conditions);
    let (outcome, rec) = evaluate_recorded(
        world,
        &mut |w| att.step(w),
        &mut |w, intents| {
            let d = def.step(w);
            merge_intents(intents, d);
            tower_intents(w, intents); // the defender's towers fire at the attacker
        },
        &run_until,
        scenario.onsite_budget,
    );
    Some((outcome, rec, def_ids))
}

/// The **self-play lens** (operator-requested realism): BOTH sides run the bot's squad logic and MOVE.
/// Pass = both sides actually engaged + the DEFENDER moved (the opposing side is not static) — the
/// realism check. The replay (`render_self_play_replay`) is the deliverable.
#[derive(Default)]
pub struct SelfPlay;

impl Validator for SelfPlay {
    fn label(&self) -> &str {
        "self-play"
    }
    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        let obj = &scenario.objectives[0];
        // Defender start positions (to detect movement).
        let starts: std::collections::HashMap<CreepId, Position> =
            scenario.world.creeps.iter().filter(|c| c.owner == scenario.defender_owner).map(|c| (c.id, c.pos)).collect();
        let Some((outcome, _rec, def_ids)) = run_self_play(scenario, obj) else {
            return Verdict { pass: true, label: self.label().into(), detail: "could not field the attacker (excluded)".into() };
        };
        let defender_moved = outcome
            .world
            .creeps
            .iter()
            .filter(|c| def_ids.contains(&c.id))
            .any(|c| starts.get(&c.id).map(|s| s.get_range_to(c.pos) > 0).unwrap_or(false));
        let had_defenders = !def_ids.is_empty();
        // The opposing side is genuinely brain-driven (not a static dummy) if it MOVED, or if the fight
        // RESOLVED decisively (a wipe/breach — real combat, not a frozen standoff). Fail only on the old
        // failure mode: a defended fight that times out with the defender never acting.
        let decisive = outcome.stop != StopReason::Timeout;
        let pass = !had_defenders || defender_moved || decisive;
        Verdict {
            pass,
            label: self.label().into(),
            detail: format!("self-play → {:?} @ t{} (defenders={}, moved={defender_moved}, decisive={decisive})", outcome.stop, outcome.ticks, def_ids.len()),
        }
    }
}

/// A simpler swappable lens (ADR 0023a stage 3 — demonstrates generation ⊥ validation): "did the force
/// the sizing system fields actually WIN?" Over a generator the pass-rate is the **win rate**. Pass = a
/// winnable+fielded scenario breaches (the load-bearing claim) OR the scenario was a correct defer /
/// unfieldable / drain (not a sizing failure). Distinct from `OracleCalibration` (which separates the
/// FP and FN rates with asymmetric gates); this is the at-a-glance win rate.
#[derive(Default)]
pub struct SizingWins {
    pub attempted: u32,
    pub won: u32,
    /// Total spawn cost of the WINNING fielded forces — the efficiency signal the sweep ranks on (cheapest
    /// force that still wins). Summed only over wins (a lost force's cost is not a "cost per win").
    pub winning_spawn_cost: u64,
    /// The TUNING SEAM (ADR 0031 D16/D17): the knob set the sized force is built with. `Default` = the seed.
    pub params: screeps_combat_decision::composition::CompositionParams,
}
impl SizingWins {
    pub fn win_rate(&self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.won as f64 / self.attempted as f64
        }
    }
    /// Build a sizing validator that fields against the given knob set (the sweep injector).
    pub fn with_params(params: screeps_combat_decision::composition::CompositionParams) -> Self {
        Self { params, ..Default::default() }
    }
    /// Mean spawn cost per WIN — the efficiency metric (cheapest winning force; lower is better). 0 when
    /// nothing won.
    pub fn mean_cost_per_win(&self) -> f64 {
        if self.won == 0 {
            0.0
        } else {
            self.winning_spawn_cost as f64 / self.won as f64
        }
    }
}
impl Validator for SizingWins {
    fn label(&self) -> &str {
        "sizing-wins"
    }
    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        let obj = &scenario.objectives[0];
        let profile = derive_profile(&scenario.world, scenario.defender_owner, obj);
        let budget = siege_ceiling(scenario.member_energy).force_budget(scenario.member_energy, scenario.onsite_budget);
        let plan = siege_doctrine_plan_with(profile, budget, scenario.member_energy, defender_force(scenario), &self.params);
        if plan.winnable() && plan.assessment.mode == AssaultMode::Breach {
            if let Some(sized) = plan.composition {
                if let Some(breached) = breaches(scenario, obj, &sized) {
                    self.attempted += 1;
                    if breached {
                        self.won += 1;
                        // Efficiency: the spawn cost of the winning force (per-member energy capped like the
                        // sizing probe) — the "cheapest force that still wins" signal the sweep ranks on.
                        let probe = self.params.member_energy.min(scenario.member_energy);
                        self.winning_spawn_cost += sized.estimated_cost(probe) as u64;
                    }
                    return Verdict { pass: breached, label: self.label().into(), detail: format!("fielded → {}", if breached { "WON" } else { "lost" }) };
                }
            }
        }
        Verdict { pass: true, label: self.label().into(), detail: "deferred / drain / unfieldable (not a sizing attempt)".into() }
    }
}

/// The enemy creep force on a creep-clear bed (the §9.3 principle: size from OBSERVED bodies): aggregate
/// dps / hits / heal + whether it's `Coordinated` — grouped (count > 1, they focus-fire / cover each
/// other) or self-sustaining (mutual heal). A lone enemy is `Individual`.
fn enemy_force_of(scenario: &Scenario) -> (f32, u32, f32, bool) {
    let d: Vec<&SimCreep> = scenario.world.creeps.iter().filter(|c| c.owner == scenario.defender_owner && c.is_alive()).collect();
    let dps: f32 = d.iter().map(|c| (c.body.attack_power() + c.body.ranged_attack_power()) as f32).sum();
    let hits: u32 = d.iter().map(|c| c.body.hits).sum();
    let heal: f32 = d.iter().map(|c| c.body.heal_power() as f32).sum();
    (dps, hits, heal, d.len() > 1 || heal > 0.0)
}

/// The observed defender force as a doctrine `EnemyForce` (the §9.3 "size from OBSERVED bodies") — the seam
/// that drives the SiegeBreach anti-creep fusion (ADR 0031 P1b). `None` when the bed has no ATTACKING
/// defender (`dps == 0`), so a creep-free structure bed leaves the structure sizing unperturbed (the
/// calibration invariant: `SizingWins`/`OracleCalibration` beds are creep-free → `None`).
pub(crate) fn defender_force(scenario: &Scenario) -> Option<EnemyForce> {
    let (dps, hits, heal, _) = enemy_force_of(scenario);
    let count = scenario.world.creeps.iter().filter(|c| c.owner == scenario.defender_owner && c.is_alive()).count() as u32;
    (dps > 0.0).then_some(EnemyForce { dps, heal, hits, count, boosted: false })
}

/// The outcome of fielding a `clear_force`-sized attacker against a creep-clear bed (ADR 0026 §9.10 L6).
pub struct ClearOutcome {
    pub cleared: bool,
    pub ticks: u32,
    pub spawn_cost: u32,
    pub ranged: u32,
    pub heal: u32,
}

/// Size an attacker via `clear_force` at `dps_margin`, field the REAL moving brain on the (Secure) bed, and
/// return the outcome metrics — the shared core of the L6 gate ([`CreepClearWins`]) and the L6b margin
/// sweep. `None` = the sizing deferred (unwinnable) or couldn't be fielded. (Open-field ranged kiters evade
/// a merely-matching force, so the `Coordinated` over-match is what lets the attacker close + clear them.)
pub fn clear_outcome_at(scenario: &Scenario, dps_margin: f32) -> Option<ClearOutcome> {
    let obj = &scenario.objectives[0];
    let (enemy_dps, enemy_hits, enemy_heal, _) = enemy_force_of(scenario);
    let budget = optimizer_ceiling_budget(DoctrineObjective::ClearCreeps, scenario.member_energy, scenario.onsite_budget);
    let (assessment, required) = clear_force(vec![], enemy_dps, enemy_hits, enemy_heal, &budget, dps_margin, scenario.world.safe_mode_owner.is_some());
    if !assessment.winnable {
        return None;
    }
    let comp = assemble_force(&required, scenario.member_energy)?;
    let spawn_cost = comp.estimated_cost(scenario.member_energy);
    let (outcome, _) = run_managed_assault_with(scenario, obj, &comp, screeps_combat_decision::kite::SquadTacticParams::open_combat())?;
    Some(ClearOutcome {
        cleared: outcome.stop == StopReason::SideWiped(scenario.defender_owner),
        ticks: outcome.ticks,
        spawn_cost,
        ranged: required.anti_creep_parts, // clear_outcome_at sizes via clear_force (creep-clear) -> anti_creep_parts
        heal: required.heal_parts,
    })
}

/// As [`clear_outcome_at`] but driven by a [`CompositionParams`] knob set (the sweep seam, ADR 0031 D16/D17):
/// the over-match `dps_margin` is `params.over_power_margin` (when the bed's force is coordinated; a lone
/// enemy still uses `1.0`), and the per-member cap is `min(params.member_energy, bed capacity)` — so the
/// member_energy + over_power knobs drive the creep-clear bed exactly as they drive the structure bed.
pub fn clear_outcome_with(scenario: &Scenario, params: &screeps_combat_decision::composition::CompositionParams) -> Option<ClearOutcome> {
    let obj = &scenario.objectives[0];
    let (enemy_dps, enemy_hits, enemy_heal, coordinated) = enemy_force_of(scenario);
    let dps_margin = if coordinated { params.over_power_margin } else { 1.0 };
    let probe = params.member_energy.min(scenario.member_energy);
    let budget = optimizer_ceiling_budget(DoctrineObjective::ClearCreeps, probe, scenario.onsite_budget);
    let (assessment, required) = clear_force(vec![], enemy_dps, enemy_hits, enemy_heal, &budget, dps_margin, scenario.world.safe_mode_owner.is_some());
    if !assessment.winnable {
        return None;
    }
    let comp = assemble_force(&required, probe)?;
    let spawn_cost = comp.estimated_cost(probe);
    let (outcome, _) = run_managed_assault_with(scenario, obj, &comp, screeps_combat_decision::kite::SquadTacticParams::open_combat())?;
    Some(ClearOutcome {
        cleared: outcome.stop == StopReason::SideWiped(scenario.defender_owner),
        ticks: outcome.ticks,
        spawn_cost,
        ranged: required.anti_creep_parts,
        heal: required.heal_parts,
    })
}

/// CREEP-CLEAR sizing GATE (ADR 0026 §9.8/§9.10 L6a): does `force_sizing::clear_force` size a squad that
/// actually CLEARS the defender creep force it was sized against? Picks the coordination margin from the
/// observed force ([`enemy_force_of`]) and fields it ([`clear_outcome_at`]); `SideWiped(defender)` = cleared.
/// The offline validation of the creep-clear keystone before it wires into `PlayerDefend`/`PlayerRaid`.
#[derive(Default)]
pub struct CreepClearWins {
    pub attempted: u32,
    pub won: u32,
    /// Total spawn cost of the WINNING fielded forces (the efficiency signal — cheapest force that clears).
    pub winning_spawn_cost: u64,
    /// The TUNING SEAM (ADR 0031 D16/D17): `Default` = the seed (`COORDINATED_DPS_MARGIN`), else the swept knobs.
    pub params: screeps_combat_decision::composition::CompositionParams,
}

impl CreepClearWins {
    pub fn win_rate(&self) -> f64 {
        if self.attempted == 0 {
            1.0
        } else {
            self.won as f64 / self.attempted as f64
        }
    }
    /// Build a creep-clear validator that sizes against the given knob set (the sweep injector).
    pub fn with_params(params: screeps_combat_decision::composition::CompositionParams) -> Self {
        Self { params, ..Default::default() }
    }
    /// Mean spawn cost per cleared bed (cheapest winning force; lower is better). 0 when nothing cleared.
    pub fn mean_cost_per_win(&self) -> f64 {
        if self.won == 0 {
            0.0
        } else {
            self.winning_spawn_cost as f64 / self.won as f64
        }
    }
}

impl Validator for CreepClearWins {
    fn label(&self) -> &str {
        "creep-clear-wins"
    }
    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        // Default params reproduce the seed (`COORDINATED_DPS_MARGIN`); a swept params injects the knobs.
        let outcome = if self.params == screeps_combat_decision::composition::CompositionParams::default() {
            let (_, _, _, coordinated) = enemy_force_of(scenario);
            let dps_margin = if coordinated { COORDINATED_DPS_MARGIN } else { 1.0 };
            clear_outcome_at(scenario, dps_margin)
        } else {
            clear_outcome_with(scenario, &self.params)
        };
        match outcome {
            None => Verdict { pass: true, label: self.label().into(), detail: "deferred / unfieldable (excluded)".into() },
            Some(o) => {
                self.attempted += 1;
                if o.cleared {
                    self.won += 1;
                    self.winning_spawn_cost += o.spawn_cost as u64;
                }
                Verdict {
                    pass: o.cleared,
                    label: self.label().into(),
                    detail: format!("sized→{} @ t{} ({} ranged, {} heal)", if o.cleared { "cleared" } else { "FAILED" }, o.ticks, o.ranged, o.heal),
                }
            }
        }
    }
}

/// The recording + metadata for a SELF-PLAY engagement on `scenario` (both sides managed). Used by the
/// single-file render + the split-file dashboard writer.
pub fn self_play_replay_data(scenario: &Scenario) -> (screeps_combat_engine::CombatRecording, ReplayMeta) {
    let obj = &scenario.objectives[0];
    match run_self_play(scenario, obj) {
        Some((outcome, rec, _)) => {
            let result = match outcome.stop {
                StopReason::ObjectivesComplete => "objective destroyed",
                StopReason::ControllerNeutralized => "controller neutralized",
                StopReason::SideWiped(o) if o == scenario.attacker_owner => "attacker wiped",
                StopReason::SideWiped(_) => "defender wiped",
                StopReason::Timeout => "timed out",
            };
            let meta = ReplayMeta::from_world_and_recording(&scenario.world, Some(&rec), &scenario.label, Some(format!("self-play (both sides managed) → {result} @ t{}", outcome.ticks)));
            (rec, meta)
        }
        None => (
            screeps_combat_engine::CombatRecording::new(),
            ReplayMeta::from_world(&scenario.world, &scenario.label, Some("self-play — could not field the attacker".into())),
        ),
    }
}

/// Render an interactive single-file HTML replay of a self-play engagement.
pub fn render_self_play_replay(scenario: &Scenario) -> String {
    let (rec, meta) = self_play_replay_data(scenario);
    replay_to_html(&rec, &meta)
}

/// Run a self-play engagement where the ATTACKER's composition is supplied explicitly (`att_bodies`,
/// a free-form roster — e.g. `roster::random_squad`) instead of the fixed ranged quad. BOTH sides are
/// still driven by the real `ManagedSimSquad` brain over the scenario's world (the defender is the
/// pre-placed defender creeps); the defender's towers fire. This is the seam the render corpus uses to
/// VARY both sides' compositions on real terrain (ADR 0038 P1) while keeping the fight fully real. `None`
/// ⇒ the attacker roster wouldn't field at the entry (too crowded / all walls).
pub fn run_self_play_bodies(
    scenario: &Scenario,
    att_bodies: &[Vec<screeps::Part>],
) -> Option<(crate::harness::evaluate::EvalOutcome, screeps_combat_engine::CombatRecording, Vec<CreepId>)> {
    let obj = &scenario.objectives[0];
    let mut world = scenario.world.clone();
    // Place the supplied attacker roster at the objective's entry (distinct, non-wall tiles), then run
    // the same both-sides-managed loop `run_self_play` uses.
    let att_ids = place_bodies_at_entry(&mut world, obj, att_bodies, scenario.attacker_owner)?;
    let def_ids: Vec<CreepId> =
        world.creeps.iter().filter(|c| c.is_alive() && c.owner == scenario.defender_owner).map(|c| c.id).collect();
    let mut att = ManagedSimSquad::new(scenario.attacker_owner, att_ids, obj.assault_pos);
    let mut def = ManagedSimSquad::new(scenario.defender_owner, def_ids.clone(), obj.pos);
    let mut conditions: Vec<Box<dyn crate::harness::evaluate::RunUntil>> =
        vec![Box::new(ObjectivesDestroyed(vec![obj.id])), Box::new(SideWiped(scenario.attacker_owner))];
    if !def_ids.is_empty() {
        conditions.push(Box::new(SideWiped(scenario.defender_owner)));
    }
    let run_until = AnyOf(conditions);
    let (outcome, rec) = evaluate_recorded(
        world,
        &mut |w| att.step(w),
        &mut |w, intents| {
            let d = def.step(w);
            merge_intents(intents, d);
            tower_intents(w, intents);
        },
        &run_until,
        scenario.onsite_budget,
    );
    Some((outcome, rec, def_ids))
}

/// Place already-built bodies (not a `SquadComposition`) as `owner` creeps at the objective's entry,
/// reusing the collision-free ring placement of [`place_at_entry`]. Ids from 1 (attacker) never collide
/// with the defender creeps (placed from 10_000). `None` ⇒ not enough free tiles near the entry.
fn place_bodies_at_entry(world: &mut CombatWorld, obj: &Objective, bodies: &[Vec<screeps::Part>], owner: PlayerId) -> Option<Vec<CreepId>> {
    let (ex, ey, rm) = (obj.entry.x().u8() as i32, obj.entry.y().u8() as i32, obj.entry.room_name());
    const OFF: [(i32, i32); 9] = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)];
    let need = bodies.len();
    let tiles: Vec<(u8, u8)> = {
        let terrain = world.terrain_for(rm);
        let mut taken: std::collections::HashSet<(u8, u8)> =
            world.creeps.iter().filter(|c| c.pos.room_name() == rm).map(|c| (c.pos.x().u8(), c.pos.y().u8())).collect();
        let mut offsets: Vec<(i32, i32)> = OFF.to_vec();
        for r in 2..=12i32 {
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx.abs().max(dy.abs()) == r {
                        offsets.push((dx, dy));
                    }
                }
            }
        }
        let mut out: Vec<(u8, u8)> = Vec::with_capacity(need);
        for (dx, dy) in offsets {
            if out.len() == need {
                break;
            }
            let (x, y) = (ex + dx, ey + dy);
            if !(0..=49).contains(&x) || !(0..=49).contains(&y) {
                continue;
            }
            let (x, y) = (x as u8, y as u8);
            if terrain.is_wall(x, y) || !taken.insert((x, y)) {
                continue;
            }
            out.push((x, y));
        }
        out
    };
    if tiles.len() < need {
        return None;
    }
    let mut ids = Vec::new();
    for (i, body) in bodies.iter().enumerate() {
        let (x, y) = tiles[i];
        let id = i as u32 + 1;
        let pos = Position::new(screeps::RoomCoordinate::new(x).unwrap(), screeps::RoomCoordinate::new(y).unwrap(), rm);
        world.creeps.push(SimCreep { id, owner, pos, body: SimBody::unboosted(body), fatigue: 0 });
        ids.push(id);
    }
    Some(ids)
}

/// The recording + a one-line outcome descriptor for a self-play match with an EXPLICIT attacker roster
/// (`att_bodies`). The corpus driver uses the descriptor (`objective destroyed` / `attacker wiped` /
/// `defender wiped` / `timed out` / `could not field`) as the per-render outcome label.
pub fn self_play_bodies_replay_data(scenario: &Scenario, att_bodies: &[Vec<screeps::Part>]) -> (screeps_combat_engine::CombatRecording, ReplayMeta, String) {
    match run_self_play_bodies(scenario, att_bodies) {
        Some((outcome, rec, _)) => {
            let result = match outcome.stop {
                StopReason::ObjectivesComplete => "objective destroyed",
                StopReason::ControllerNeutralized => "controller neutralized",
                StopReason::SideWiped(o) if o == scenario.attacker_owner => "attacker wiped",
                StopReason::SideWiped(_) => "defender wiped",
                StopReason::Timeout => "timed out",
            };
            let meta = ReplayMeta::from_world_and_recording(
                &scenario.world,
                Some(&rec),
                &scenario.label,
                Some(format!("self-play (both sides managed) → {result} @ t{}", outcome.ticks)),
            );
            (rec, meta, format!("{result} @ t{}", outcome.ticks))
        }
        None => (
            screeps_combat_engine::CombatRecording::new(),
            ReplayMeta::from_world(&scenario.world, &scenario.label, Some("self-play — could not field the attacker".into())),
            "could not field".into(),
        ),
    }
}

/// Render an interactive single-file HTML replay of a self-play match with an explicit attacker roster,
/// returning the HTML paired with the one-line outcome descriptor (for the corpus index).
pub fn render_self_play_replay_bodies(scenario: &Scenario, att_bodies: &[Vec<screeps::Part>]) -> (String, String) {
    let (rec, meta, outcome) = self_play_bodies_replay_data(scenario, att_bodies);
    (replay_to_html(&rec, &meta), outcome)
}

/// Render an interactive HTML replay of a MOVING managed assault on `scenario` — the real squad brain
/// pathing from the entry to the objective + engaging (the movement-rich replay).
pub fn render_managed_replay(scenario: &Scenario) -> String {
    let obj = &scenario.objectives[0];
    let comp = managed_assault_comp(scenario);
    match run_managed_assault(scenario, obj, &comp) {
        Some((outcome, rec)) => {
            let result = match outcome.stop {
                StopReason::ObjectivesComplete => "objective destroyed",
                StopReason::ControllerNeutralized => "controller neutralized",
                StopReason::SideWiped(_) => "attackers wiped",
                StopReason::Timeout => "held (timed out)",
            };
            let meta = ReplayMeta::from_world_and_recording(&scenario.world, Some(&rec), &scenario.label, Some(format!("managed ranged assault → {result} @ t{}", outcome.ticks)));
            replay_to_html(&rec, &meta)
        }
        None => {
            let meta = ReplayMeta::from_world(&scenario.world, &scenario.label, Some("managed assault — could not field at the entry".into()));
            replay_to_html(&screeps_combat_engine::CombatRecording::new(), &meta)
        }
    }
}

/// Render an interactive HTML replay of the oracle's decision on `scenario`: derive the profile,
/// assess the fieldable ceiling, FIELD the force the validator would (the sized squad when winnable,
/// else the ceiling falsifier), record the scripted siege, and render it with a verdict header. The
/// full Generation → Evaluation(record) → Visualization chain in one call — for the operator's visual
/// validation of outcomes + permutation variety.
pub fn calibration_replay_data(scenario: &Scenario) -> (screeps_combat_engine::CombatRecording, ReplayMeta) {
    let obj = &scenario.objectives[0];
    let (comp, decision) = choose_fielded_comp(scenario, obj);

    let mut world = scenario.world.clone();
    if !place_squad(&mut world, obj, &comp, scenario.attacker_owner, scenario.member_energy) {
        let meta = ReplayMeta::from_world(&scenario.world, &scenario.label, Some(format!("{decision} — could not place on the bed")));
        return (screeps_combat_engine::CombatRecording::new(), meta);
    }
    let attacker = scenario.attacker_owner;
    let defender = scenario.defender_owner;
    let (core_id, core_pos) = (obj.id, obj.pos);
    let run_until = AnyOf(vec![Box::new(ObjectivesDestroyed(vec![core_id])), Box::new(SideWiped(attacker))]);
    let (outcome, rec) = evaluate_recorded(
        world,
        &mut |w| siege_intents(w, attacker, core_id, core_pos),
        &mut |w, intents| defense_intents(w, defender, core_pos, intents),
        &run_until,
        scenario.onsite_budget,
    );
    let result = match outcome.stop {
        StopReason::ObjectivesComplete => "BREACHED",
        StopReason::ControllerNeutralized => "controller neutralized",
        StopReason::SideWiped(_) => "attackers wiped",
        StopReason::Timeout => "held (timed out)",
    };
    let meta = ReplayMeta::from_world_and_recording(&scenario.world, Some(&rec), &scenario.label, Some(format!("{decision} → {result} @ t{}", outcome.ticks)));
    (rec, meta)
}

/// Render an interactive single-file HTML replay of the oracle's sizing-pure siege decision.
pub fn render_calibration_replay(scenario: &Scenario) -> String {
    let (rec, meta) = calibration_replay_data(scenario);
    replay_to_html(&rec, &meta)
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
pub(crate) fn siege_ceiling(energy: u32) -> SquadComposition {
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
    // Derive the formation from the member count (ADR 0031 D14) — NOT Default(None), which on an 8-member
    // force would imply a single (0,0) layout that stacks every member on the anchor.
    let (formation_shape, formation_mode) = formation_for(slots.len());
    SquadComposition { label: "Siege Ceiling".into(), slots, formation_shape, formation_mode, retreat_threshold: 0.3 }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;

    /// INVARIANT (operator): no two creeps ever occupy the same tile. The cross-room relocation is
    /// occupancy-blind (faithful to the engine), but each exit tile maps to a UNIQUE mirror and every tile
    /// holds ≤1 creep, so the cross is a PERMUTATION — a stack can't form from movement. The only historical
    /// source was a harness placement bug (`place_at_entry`'s clamp-collapse onto an edge), now fixed. This
    /// scans every recorded tick of a representative set of base assaults (which all cross rooms to reach
    /// the objective) for a same-tile pair.
    /// ADR 0031 P2 golden-output / byte-stability fence: the unified `emit_requirement` (which
    /// `siege_doctrine_plan` now routes through) must produce IDENTICAL sizing — verdict + `RequiredForce`
    /// + the full serialized composition — run-twice over EVERY realistic base. This is the bed-level
    /// determinism the bot/eval parity depends on (the doctrine-crate unit fence covers the math in
    /// isolation; this covers it over the real beds, defenders fed in). (ADR 0031 §3 Phase 2, §5.)
    #[test]
    fn emit_requirement_golden_output_is_stable_over_realistic_bases() {
        for scenario in crate::harness::generate::realistic_bases() {
            let obj = &scenario.objectives[0];
            let profile = derive_profile(&scenario.world, scenario.defender_owner, obj);
            let budget = siege_ceiling(scenario.member_energy).force_budget(scenario.member_energy, scenario.onsite_budget);
            let run = || {
                let p = siege_doctrine_plan(profile.clone(), budget, scenario.member_energy, defender_force(&scenario));
                (p.assessment, p.required, p.composition.map(|c| format!("{c:?}")))
            };
            assert_eq!(run(), run(), "{}: emitter golden-output is stable", scenario.label);
        }
    }

    #[test]
    fn sim_maintains_one_creep_per_tile() {
        let scenarios = crate::tournament::realistic_base_scenarios();
        let tactics = screeps_combat_decision::kite::SquadTacticParams::breach();
        for scenario in scenarios.iter().take(12) {
            let obj = &scenario.objectives[0];
            let (comp, _) = choose_fielded_comp(scenario, obj);
            if let Some((_, rec)) = run_managed_assault_with(scenario, obj, &comp, tactics) {
                for (t, frame) in rec.frames.iter().enumerate() {
                    let mut seen = std::collections::HashSet::new();
                    for c in &frame.creeps {
                        assert!(
                            seen.insert((format!("{}", c.room), c.x, c.y)),
                            "stack in {} at tick {t}: two creeps on ({},{},{})",
                            scenario.label, c.x, c.y, c.room
                        );
                    }
                }
            }
        }
    }
}
