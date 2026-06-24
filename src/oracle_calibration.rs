//! Oracle-calibration tournament (ADR 0022 P-FORCE — the WIN): prove the force-sizing oracle is
//! CALIBRATED against the engine ground truth. Over seeded, randomized defended-base scenarios we
//! generate BOTH a `CombatWorld` siege bed AND the matching [`DefenseProfile`] from the same draws,
//! size OUR force with the REAL pipeline (`assess` → [`RequiredForce::from_assessment`] →
//! [`SquadComposition::sized_for`] → [`BodyType::build_body`]), field it as `SimCreep`s, and run the
//! authoritative engine siege. The claim the gate enforces:
//!
//! - **winnable ⇒ we win** — when the oracle says winnable AND the chosen composition can field the
//!   required force, the engine confirms the core is breached. A non-breach is a **false positive**
//!   (gate HARD: `fp_rate ≤ 1%`). This is the load-bearing direction — "size + tactics wins".
//! - **defer ⇒ we couldn't anyway** — when the oracle defers (unwinnable for one squad), even the
//!   strongest single squad we can field doesn't breach in-budget. A breach there is a **false
//!   negative** (gate SOFT: `fn_rate ≤ 20%`). Asymmetric thresholds match the oracle's contract: a
//!   "yes" is safe to commit, a "no" merely defers to the multi-squad path — never the reverse.
//!
//! **Consistency by construction.** The [`DefenseProfile`] is computed FROM the bed with the engine's
//! own tower curves + the bed's defense AI ([`defense_intents`]): the attacking towers are the
//! energized ones MINUS the one `defense_intents` reserves for maintenance, and `repair_per_tick` is
//! that maintainer's `tower_repair_at_range` to the rampart — so the oracle judges exactly what the
//! engine resolves. Healers are placed **adjacent** to the single focused dismantler (the bed's
//! `focusClosest` concentrates tower fire on the hostile nearest the core), so heal lands at the full
//! adjacent `HEAL_POWER` the oracle assumes (not the 1/3 `RANGED_HEAL_POWER`).
//!
//! **Scope (v1) / the CAVEAT (encoded).** The grading closure is a SCRIPTED SIEGE (dismantlers break
//! the nearest rampart then the core; healers heal the most-wounded ally) — sizing-pure, so a *squad
//! pathing* gap can never masquerade as a *sizing* false positive (the managed squad's kite/engage
//! brain doesn't drive a pure breach yet; that's the auction/movement workstream). The closure
//! executes the **Breach** assault mode, so FP is graded only on `Breach`-mode winnable rows;
//! `Drain`-mode wins (tank-soak-then-breach) are reported as a diagnostic, not graded (the drain
//! tactic isn't scripted here). FN is measured against a breach-only max squad (drain-only wins are
//! not detected — a conservative under-count for the soft gate). Enemy creep defenders are out of v1
//! (`enemy_dps = 0`): the calibration targets the tower / rampart / repair / energy / heal / dismantle
//! dimensions the force-sizing solver owns.

use screeps::{Position, RoomCoordinate, RoomName};
use screeps_combat_agent::objective_bed::{run_siege, SiegeResult};
use screeps_combat_agent::scenario::ScenarioBuilder;
use screeps_combat_decision::bodies::{CombatBodySpec, MoveProfile};
use screeps_combat_decision::composition::{BodyType, SquadComposition, SquadRole, SquadSlot};
use screeps_combat_decision::damage::tower_repair_at_range;
use screeps_combat_decision::force_sizing::{assess, AssaultMode, DefenseProfile, ForceBudget, RequiredForce, TowerThreat};
use screeps_combat_engine::constants::TOWER_ENERGY_COST;
use screeps_combat_engine::{CombatAction, CombatWorld, Intents, PlayerId, SimBody, SimCreep, StructureId, StructureKind};

const DEFENDER: PlayerId = 1;
const ATTACKER: PlayerId = 0;

// ── Bed geometry (fixed; the DRAWS vary the defense, not the layout — keeps the siege movement-free
//    and deterministic) ────────────────────────────────────────────────────────────────────────────
//
// Core (spawn) at (25,25), the breach rampart at (24,25). The focused dismantler sits at the CORNER
// (24,24): range 1 to BOTH the rampart (orthogonal) and the core (diagonal), so it breaches then kills
// without moving. All dismantler tiles are range 1 to both rampart and core (so the whole squad
// dismantles both phases at the oracle's full structure-DPS); the focus is the FIRST-inserted one
// (unique core-range-1 by insertion order — `defense_intents` `min_by_key` picks it). Healer tiles are
// the focused dismantler's free neighbours at core-range 2 (so they're never the focus) → full
// adjacent HEAL on the creep under fire.
const CORE: (u8, u8) = (25, 25);
const RAMPART: (u8, u8) = (24, 25);
/// The assault tile — the focused dismantler stands here; tower ranges are measured to it.
const ASSAULT: (u8, u8) = (24, 24);
/// Dismantler tiles: each range 1 to BOTH rampart (24,25) and core (25,25). `[0]` = (24,24) = the focus.
const DISMANTLER_TILES: [(u8, u8); 4] = [(24, 24), (25, 24), (24, 26), (25, 26)];
/// Healer tiles: free neighbours of the focus (range 1 → full 12/part HEAL), all core-range 2.
const HEALER_TILES: [(u8, u8); 5] = [(23, 23), (24, 23), (25, 23), (23, 24), (23, 25)];
/// Tower tiles (a pool of 6 at varied ranges to ASSAULT: 16/12/8/4/4/8 → a spread of falloff/DPS),
/// all outside the assault box so they never collide with attacker placement.
const TOWER_TILES: [(u8, u8); 6] = [(24, 8), (24, 12), (24, 16), (20, 24), (28, 24), (24, 32)];

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos(x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room())
}
fn range(a: (u8, u8), b: (u8, u8)) -> u32 {
    pos(a.0, a.1).get_range_to(pos(b.0, b.1))
}

// ── Seeded RNG (SplitMix64 — per-index reproducible, no `rand`/`Date`/`Math.random`) ───────────────
struct Rng(u64);
impl Rng {
    fn seeded(index: u32) -> Self {
        // Mix the index so adjacent seeds don't produce near-identical streams.
        Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive range `[lo, hi]`.
    fn range(&mut self, lo: u32, hi: u32) -> u32 {
        debug_assert!(hi >= lo);
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
    /// True with probability `pct`%.
    fn chance(&mut self, pct: u32) -> bool {
        self.range(0, 99) < pct
    }
    fn pick(&mut self, xs: &[u32]) -> u32 {
        xs[(self.next_u64() % xs.len() as u64) as usize]
    }
}

// ── Scenario (the draws + the derived oracle profile) ──────────────────────────────────────────────
struct Scenario {
    core_hits: u32,
    rampart_hits: u32, // 0 ⇒ no rampart (the core is reached directly)
    /// `(tile, energy)` per placed tower (insertion order = `defense_intents` maintainer = last energized).
    towers: Vec<((u8, u8), u32)>,
    safe_mode: bool,
    member_energy: u32,
    onsite_budget: u32,
    /// The oracle profile, computed FROM the bed (consistency by construction).
    profile: DefenseProfile,
}

fn energized(towers: &[((u8, u8), u32)]) -> Vec<((u8, u8), u32)> {
    towers.iter().copied().filter(|(_, e)| *e >= TOWER_ENERGY_COST).collect()
}

fn generate(index: u32) -> Scenario {
    let mut rng = Rng::seeded(index);
    // RCL4..8-ish spawn capacities.
    let member_energy = rng.pick(&[1300, 1800, 2300, 3300, 5600, 12_900]);
    let onsite_budget = rng.range(600, 1400);
    let core_hits = rng.range(20_000, 100_000);
    let rampart_hits = if rng.chance(70) { rng.range(1, 80_000) } else { 0 };
    let n_towers = rng.range(0, 6);
    let safe_mode = rng.chance(5);

    let mut towers = Vec::new();
    for &tile in TOWER_TILES.iter().take(n_towers as usize) {
        let energy = if rng.chance(25) { rng.range(0, 9) } else { rng.range(100, 100_000) };
        towers.push((tile, energy));
    }

    // Derive the profile FROM these draws as the CONSERVATIVE UNION of the bed's two siege phases:
    // - BREACH phase: `defense_intents` reserves one tower to repair the damaged rampart, so only
    //   `n-1` towers attack — but `repair_per_tick` drags the breach. The oracle models the maintainer
    //   as an attacker too (over-counts breach incoming by one tower → conservative).
    // - KILL phase: once the rampart FALLS there's nothing to repair, so ALL `n` towers attack the
    //   focus (the binding survival constraint). The oracle MUST size heal for this — modelling only
    //   `n-1` was the false-positive bug (the squad got wiped on the core by the ex-maintainer tower).
    // So: attackers = ALL energized towers; repair = the maintainer's (last energized) repair while a
    // rampart stands. The maintainer is double-counted (attack + repair) — conservative on both phases.
    let live = energized(&towers);
    let repair_per_tick = if rampart_hits > 0 && !live.is_empty() {
        tower_repair_at_range(range(live.last().unwrap().0, RAMPART)) as f32
    } else {
        0.0
    };
    let profile = DefenseProfile {
        towers: live.iter().map(|(tile, e)| TowerThreat { range_to_assault: range(*tile, ASSAULT), energy: *e }).collect(),
        breach_hits: rampart_hits,
        objective_hits: core_hits,
        enemy_dps: 0.0,
        repair_per_tick,
        safe_mode,
    };

    Scenario { core_hits, rampart_hits, towers, safe_mode, member_energy, onsite_budget, profile }
}

/// Build a FRESH bed for `scenario` (run_siege consumes the world, so each grading rebuilds). Returns
/// `(world, core_id)`.
fn build_world(scenario: &Scenario) -> (CombatWorld, StructureId) {
    let mut b = ScenarioBuilder::empty(room());
    let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), CORE.0, CORE.1, scenario.core_hits, scenario.core_hits);
    for ((tx, ty), energy) in &scenario.towers {
        b.tower(DEFENDER, *tx, *ty, *energy);
    }
    let mut world = if scenario.rampart_hits > 0 {
        b.rampart(DEFENDER, RAMPART.0, RAMPART.1, scenario.rampart_hits).build()
    } else {
        b.build()
    };
    if scenario.safe_mode {
        world.safe_mode_owner = Some(DEFENDER);
    }
    (world, core_id)
}

/// Place `comp`'s members as attacker `SimCreep`s on the fixed dismantler/healer tiles. Dismantlers go
/// first (so tile `[0]` = the unique focus), healers onto adjacent tiles. Returns `false` if the
/// composition can't be placed FAITHFULLY at this bed (a body that won't build, or more dismantlers /
/// healers than there are valid tiles) — such a row is excluded from grading (a bed-geometry limit, not
/// an oracle error).
fn place_squad(world: &mut CombatWorld, comp: &SquadComposition, member_energy: u32) -> bool {
    let (mut d, mut h, mut next_id) = (0usize, 0usize, 1u32);
    for slot in &comp.slots {
        let Some(body) = slot.body_type.build_body(member_energy, MoveProfile::Plains) else {
            return false;
        };
        let tile = match slot.role {
            SquadRole::Dismantler => {
                if d >= DISMANTLER_TILES.len() {
                    return false;
                }
                let t = DISMANTLER_TILES[d];
                d += 1;
                t
            }
            SquadRole::Healer => {
                if h >= HEALER_TILES.len() {
                    return false;
                }
                let t = HEALER_TILES[h];
                h += 1;
                t
            }
            _ => return false, // a siege composition is Dismantler + Healer only
        };
        world.creeps.push(SimCreep { id: next_id, owner: ATTACKER, pos: pos(tile.0, tile.1), body: SimBody::unboosted(&body), fatigue: 0 });
        next_id += 1;
    }
    true
}

/// The scripted siege attacker (sizing-pure, movement-free — all members start in range): every
/// dismantler breaks the nearest living rampart, then the core; every healer heals the most-wounded
/// ally (adjacent → `Heal`, ≤3 → `RangedHeal`). Mirrors `objective_bed`'s force-sized closure.
fn siege_intents(world: &CombatWorld, core_id: StructureId, core_pos: Position) -> Intents {
    let mut intents = Intents::new();
    let wounded = world
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner == ATTACKER)
        .min_by_key(|c| c.body.hits)
        .map(|c| (c.id, c.pos));
    for c in world.creeps.iter().filter(|c| c.is_alive() && c.owner == ATTACKER) {
        if c.body.dismantle_power() > 0 {
            let rampart = world
                .structures
                .iter()
                .filter(|s| s.is_alive() && s.kind == StructureKind::Rampart)
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

/// Run the scripted siege of `comp` against `scenario`'s bed; `true` ⇒ the core was breached within the
/// on-site budget.
fn breaches(scenario: &Scenario, comp: &SquadComposition) -> Option<bool> {
    let (mut world, core_id) = build_world(scenario);
    if !place_squad(&mut world, comp, scenario.member_energy) {
        return None; // can't field this composition faithfully on the bed → exclude
    }
    let core_pos = pos(CORE.0, CORE.1);
    let out = run_siege(world, DEFENDER, core_id, core_pos, &mut |w| siege_intents(w, core_id, core_pos), scenario.onsite_budget);
    Some(out.result == SiegeResult::CoreBreached)
}

/// Largest single-role part count one member can carry at `energy` (reuses the real body builder, incl.
/// the MOVE ratio + 50-part cap). The falsifier's per-member size.
fn max_role_parts(spec_of: impl Fn(u32) -> CombatBodySpec, energy: u32) -> u32 {
    (1..=25)
        .rev()
        .find(|&n| screeps_combat_decision::bodies::build_combat_body(&spec_of(n), MoveProfile::Plains, energy).is_some())
        .unwrap_or(0)
}

/// The strong-but-fieldable SIEGE CEILING — the oracle's BUDGET and the FN falsifier in one. The oracle
/// must assess what the sizing SYSTEM can field, not the bare `siege_quad` template: `sized_for` grows
/// member COUNT (D3) up to [`MAX_SIZED_MEMBERS`], so assessing the template alone would defer every
/// room a GROWN squad wins (a spurious conservatism). This is `siege_quad` grown to its practical max
/// that still fits the bed geometry + the 8-member cap: 3 dismantlers + 5 healers, each at its
/// per-member part cap for `energy`. The oracle assesses "can the ceiling win?"; `sized_for` then
/// MINIMIZES within it (the FP rows); a DEFERRED scenario is falsified by fielding the ceiling itself
/// (the FN rows) — so the verdict and the falsifier reference the same force, and a defer means "even
/// the ceiling can't win".
const CEILING_DISMANTLERS: usize = 3;
const CEILING_HEALERS: usize = 5;
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
    SquadComposition {
        label: "Siege Ceiling".into(),
        slots,
        formation_shape: Default::default(),
        formation_mode: Default::default(),
        retreat_threshold: 0.3,
    }
}

/// The calibration tally over N seeded scenarios.
#[derive(Clone, Copy, Debug, Default)]
pub struct Calibration {
    pub scenarios: u32,
    /// Oracle said winnable (Breach mode), composition fielded, scripted siege ran.
    pub fielded: u32,
    /// Fielded rows where the engine did NOT breach (the HARD-gated false positives).
    pub false_positives: u32,
    /// Oracle deferred (unwinnable for one squad).
    pub deferred: u32,
    /// Deferred rows where the MAX single squad DID breach in-budget (the SOFT-gated false negatives).
    pub false_negatives: u32,
    /// Winnable but the composition couldn't field the required force at this energy, OR the bed
    /// geometry couldn't place it (a defer-by-affordability / geometry limit — not an oracle error).
    pub unfieldable: u32,
    /// Winnable via the Drain assault mode — reported, not FP-graded (the scripted closure breaches; it
    /// doesn't run the tank-soak drain tactic).
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
}

/// Run the oracle-calibration over `n` seeded scenarios with the chosen attacking composition
/// (`siege_quad` — the breach archetype). Deterministic: seed = scenario index.
pub fn calibrate(n: u32) -> Calibration {
    let mut c = Calibration { scenarios: n, ..Default::default() };
    for index in 0..n {
        let scenario = generate(index);
        // Assess what the sizing SYSTEM can field (the ceiling), not the bare template.
        let ceiling = siege_ceiling(scenario.member_energy);
        let caps = ceiling.capabilities(scenario.member_energy);
        let budget = ForceBudget {
            max_heal_per_tick: caps.heal_per_tick as f32,
            max_dismantle_dps: caps.structure_dps as f32,
            tank_effective_hp: caps.tank_effective_hp as f32,
            onsite_budget_ticks: scenario.onsite_budget,
        };
        let assessment = assess(&scenario.profile, &budget);

        if assessment.winnable {
            // Drain-mode wins aren't executed by the scripted breach closure → report, don't FP-grade.
            if assessment.mode == AssaultMode::Drain {
                c.drain_winnable += 1;
                continue;
            }
            // Field the MINIMAL squad the sizing produces for this verdict.
            match SquadComposition::siege_quad().sized_for(RequiredForce::from_assessment(&assessment), scenario.member_energy) {
                Some(sized) => match breaches(&scenario, &sized) {
                    Some(true) => c.fielded += 1,
                    Some(false) => {
                        c.fielded += 1;
                        c.false_positives += 1;
                    }
                    None => c.unfieldable += 1, // bed geometry couldn't place it faithfully
                },
                None => c.unfieldable += 1, // required force exceeds one squad at this energy → defer
            }
        } else {
            c.deferred += 1;
            // Falsify the defer with the SAME ceiling the verdict referenced: if even it breaches, the
            // oracle was too pessimistic.
            if let Some(true) = breaches(&scenario, &ceiling) {
                c.false_negatives += 1;
            }
        }
    }
    c
}

/// A readable report of a calibration run (the dashboard).
pub fn report(c: &Calibration) -> String {
    format!(
        "oracle calibration — {} scenarios\n  fielded (winnable+breach): {} | false positives: {} (fp_rate {:.3}, gate <= 0.010)\n  deferred (unwinnable):     {} | false negatives: {} (fn_rate {:.3}, gate <= 0.200)\n  unfieldable (comp/geom):   {}\n  drain-winnable (diag):     {}",
        c.scenarios,
        c.fielded,
        c.false_positives,
        c.fp_rate(),
        c.deferred,
        c.false_negatives,
        c.fn_rate(),
        c.unfieldable,
        c.drain_winnable,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The WIN gate (ADR 0022 P-FORCE): over 200 seeded defended-base scenarios, the force-sizing oracle
    /// is calibrated against the engine — winnable verdicts breach (fp ≤ 1%) and defers are real (fn ≤
    /// 20%). Run with `-- --nocapture` to see the dashboard.
    #[test]
    fn oracle_is_calibrated_against_the_engine() {
        let c = calibrate(200);
        println!("{}", report(&c));
        // The scenario mix must actually exercise BOTH verdicts (else the gate is vacuous).
        assert!(c.fielded >= 20, "too few fielded scenarios to calibrate FP ({}); the draw mix is off", c.fielded);
        assert!(c.deferred >= 20, "too few deferred scenarios to calibrate FN ({}); the draw mix is off", c.deferred);
        // HARD: a winnable+fielded verdict that does not breach is the oracle promising a win it can't
        // deliver — the load-bearing "size + tactics wins" claim.
        assert!(
            c.fp_rate() <= 0.01,
            "FALSE POSITIVES {}/{} (fp_rate {:.3} > 0.010): the oracle said winnable + we fielded the sized force, but the engine did NOT breach\n{}",
            c.false_positives,
            c.fielded,
            c.fp_rate(),
            report(&c)
        );
        // SOFT: a deferred verdict the max single squad could have won — over-conservative, tolerated.
        assert!(
            c.fn_rate() <= 0.20,
            "FALSE NEGATIVES {}/{} (fn_rate {:.3} > 0.200): the oracle deferred but the max single squad breached\n{}",
            c.false_negatives,
            c.deferred,
            c.fn_rate(),
            report(&c)
        );
    }

    /// Determinism: the same seed yields the same tally (no `Date`/`Math.random`; SplitMix64 over the
    /// index) — so the gate is reproducible and a regression is attributable.
    #[test]
    fn calibration_is_deterministic() {
        assert_eq!(format!("{:?}", calibrate(64)), format!("{:?}", calibrate(64)));
    }

    /// A hand-built winnable bed (one drained tower, a thin rampart, RCL7 energy) IS fielded and breaches
    /// — a smoke test that the pipeline (assess → sized_for → place → siege) is wired end to end.
    #[test]
    fn a_clearly_winnable_bed_is_fielded_and_breaches() {
        // Find the first seed the oracle deems winnable+Breach+fieldable and assert it breaches (the gate
        // already enforces this in aggregate; this pins the wiring with a clear message).
        let mut found = false;
        for index in 0..200 {
            let s = generate(index);
            let caps = siege_ceiling(s.member_energy).capabilities(s.member_energy);
            let budget = ForceBudget {
                max_heal_per_tick: caps.heal_per_tick as f32,
                max_dismantle_dps: caps.structure_dps as f32,
                tank_effective_hp: caps.tank_effective_hp as f32,
                onsite_budget_ticks: s.onsite_budget,
            };
            let a = assess(&s.profile, &budget);
            if a.winnable && a.mode == AssaultMode::Breach {
                if let Some(sized) = SquadComposition::siege_quad().sized_for(RequiredForce::from_assessment(&a), s.member_energy) {
                    if let Some(breached) = breaches(&s, &sized) {
                        assert!(breached, "seed {index}: oracle said winnable+fieldable but the engine did not breach");
                        found = true;
                        break;
                    }
                }
            }
        }
        assert!(found, "no winnable+fieldable Breach scenario in the first 200 seeds — the draw mix is off");
    }
}
