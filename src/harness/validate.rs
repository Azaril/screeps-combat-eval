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
use screeps_combat_decision::composition::{BodyType, SquadComposition, SquadRole, SquadSlot};
use screeps_combat_decision::damage::tower_repair_at_range;
use screeps_combat_decision::doctrine::{decide_doctrine, default_doctrines, DoctrineObjective, EnemyCoordination, EngagementContext, ForcePlan};
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

        // Select + size VIA THE DOCTRINE REGISTRY (parity with the bot). Assess against what the sizing
        // SYSTEM can field (the ceiling), not the bare template.
        let ceiling = siege_ceiling(scenario.member_energy);
        let budget = ceiling.force_budget(scenario.member_energy, scenario.onsite_budget);
        let plan = siege_doctrine_plan(profile, budget, scenario.member_energy);

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
/// the maintainer's (last energized) `tower_repair_at_range` to the breach rampart; `enemy_dps` = Σ the
/// defender creeps' attack+ranged power; safe-mode from the world. Tower ranges measured to the
/// objective's assault tile.
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

/// Decide + size the siege force for a structure-breach scenario VIA THE DOCTRINE REGISTRY (ADR 0026 §9)
/// — the SAME selection + sizing path the bot's offense runs (parity; no divergent inline `assess` +
/// `siege_quad().sized_for` in the eval). A bed objective is a dismantle-able structure breach →
/// `DoctrineObjective::DismantleStructure` → the `SiegeBreach` doctrine sizes a `siege_quad` to the
/// oracle's required force against `budget` (the siege ceiling's, the calibration lens). The returned
/// `ForcePlan` carries the verdict (`assessment`) + the sized `composition` (`None` = defer / drain /
/// unfieldable). `importance: 0.0` matches the eval's base-force sizing (`importance_margin(0)` = 1×).
pub(crate) fn siege_doctrine_plan(profile: DefenseProfile, budget: ForceBudget, member_energy: u32) -> ForcePlan {
    let ctx = EngagementContext {
        objective: DoctrineObjective::DismantleStructure,
        coordination: EnemyCoordination::Individual,
        defense: profile,
        enemy_force: None,
        importance: 0.0,
        member_energy,
    };
    let doctrines = default_doctrines();
    decide_doctrine(&ctx, &doctrines)
        .expect("DismantleStructure routes to the siege-breach doctrine")
        .plan(&ctx, Some(budget))
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
    let plan = siege_doctrine_plan(profile, budget, scenario.member_energy);
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
/// drive (advance, kite, focus-fire creeps, shoot structures), auto-sized to the home's energy. (The
/// sizing-pure siege force is `choose_fielded_comp`/`OracleCalibration`.)
fn managed_assault_comp(_scenario: &Scenario) -> SquadComposition {
    SquadComposition::quad_ranged()
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
}
impl SizingWins {
    pub fn win_rate(&self) -> f64 {
        if self.attempted == 0 {
            0.0
        } else {
            self.won as f64 / self.attempted as f64
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
        let plan = siege_doctrine_plan(profile, budget, scenario.member_energy);
        if plan.winnable() && plan.assessment.mode == AssaultMode::Breach {
            if let Some(sized) = plan.composition {
                if let Some(breached) = breaches(scenario, obj, &sized) {
                    self.attempted += 1;
                    if breached {
                        self.won += 1;
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
    let budget = SquadComposition::quad_ranged().force_budget(scenario.member_energy, scenario.onsite_budget);
    let (assessment, required) = clear_force(vec![], enemy_dps, enemy_hits, enemy_heal, &budget, dps_margin, scenario.world.safe_mode_owner.is_some());
    if !assessment.winnable {
        return None;
    }
    let comp = SquadComposition::quad_ranged().sized_for(required, scenario.member_energy)?;
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

/// CREEP-CLEAR sizing GATE (ADR 0026 §9.8/§9.10 L6a): does `force_sizing::clear_force` size a squad that
/// actually CLEARS the defender creep force it was sized against? Picks the coordination margin from the
/// observed force ([`enemy_force_of`]) and fields it ([`clear_outcome_at`]); `SideWiped(defender)` = cleared.
/// The offline validation of the creep-clear keystone before it wires into `PlayerDefend`/`PlayerRaid`.
#[derive(Default)]
pub struct CreepClearWins {
    pub attempted: u32,
    pub won: u32,
}

impl CreepClearWins {
    pub fn win_rate(&self) -> f64 {
        if self.attempted == 0 {
            1.0
        } else {
            self.won as f64 / self.attempted as f64
        }
    }
}

impl Validator for CreepClearWins {
    fn label(&self) -> &str {
        "creep-clear-wins"
    }
    fn validate(&mut self, scenario: &Scenario) -> Verdict {
        let (_, _, _, coordinated) = enemy_force_of(scenario);
        let dps_margin = if coordinated { COORDINATED_DPS_MARGIN } else { 1.0 };
        match clear_outcome_at(scenario, dps_margin) {
            None => Verdict { pass: true, label: self.label().into(), detail: "deferred / unfieldable (excluded)".into() },
            Some(o) => {
                self.attempted += 1;
                if o.cleared {
                    self.won += 1;
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
            (rec, ReplayMeta::from_world(&scenario.world, &scenario.label, Some(format!("self-play (both sides managed) → {result} @ t{}", outcome.ticks))))
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
            let meta = ReplayMeta::from_world(&scenario.world, &scenario.label, Some(format!("managed ranged assault → {result} @ t{}", outcome.ticks)));
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
    let meta = ReplayMeta::from_world(&scenario.world, &scenario.label, Some(format!("{decision} → {result} @ t{}", outcome.ticks)));
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
    SquadComposition { label: "Siege Ceiling".into(), slots, formation_shape: Default::default(), formation_mode: Default::default(), retreat_threshold: 0.3 }
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
