//! Invader-stronghold + border-gauntlet corpus (operator directive 2026-08-23: "a test corpus in
//! simulation that matches real invader strongholds and boosted creeps … multi room and challenging
//! room layouts … increasingly challenging scenarios").
//!
//! GROUND TRUTH — everything here is transcribed from the local canonical clones, not invented:
//! - Templates bunker1–5: `C:\code\screeps-common\lib\strongholds.js:3-246` — exact per-structure
//!   offsets (invader core, towers, containers, the full rampart blanket). Roads and containers are
//!   omitted (no combat effect in the sim; roads only touch fatigue, and combat squads size MOVE for
//!   plains — noted, not modeled).
//! - Rampart hits by level 100K/200K/500K/1M/2M, core 100K hits (`common/constants.js:842,849`).
//! - Defender BODIES + BOOSTS: `engine/…/invader-core/stronghold/creeps.js:96-126` — exact part
//!   counts and compounds (UH2O/KHO2 = T2 ×3; XUH2O/XKHO2/XZHO2 = T3 ×4).
//! - Defender POPULATIONS per level: `stronghold.js:399-489` — bunker1 none; bunker2 1×weakDefender;
//!   bunker3 2×fullDefender; bunker4 4 drawn from {1 fortifier, 4 boostedDefender, 3 boostedRanger};
//!   bunker5 fortifier + 8 drawn from {7 fullBoostedMelee, 9 fullBoostedRanger}. Draws are seeded
//!   here (the engine shuffles; we take a deterministic seeded sample of the same deck).
//! - Tower AI: focusClosest (L1–3) / focusMax (L4–5) (`stronghold.js:397-489`); modeled in
//!   [`stronghold_tower_intents`]. Towers are core-refilled on live (`refillTowers`), so DRAINING a
//!   stronghold is not viable — modeled as a large finite pool (100K) per tower.
//!
//! Deliberate approximations (documented, revisit on evidence): the fortifier's rampart REPAIR is
//! not resolved (the sim has no creep-repair intent; the body still fields as the eHP + T3-WORK blob
//! it is); defender micro is [`ManagedSimSquad`] with `Hold` intent (defend near the core) rather
//! than the engine's spot-walk `coordinated` behavior; L5's anti-nuke fortify is out of scope.

use crate::harness::generate::{breach_geometry, ATTACKER, DEFENDER};
use crate::harness::scenario::{Objective, ObjectiveKind, Scenario};
use screeps::{Part, Position, RoomCoordinate, RoomName};
use screeps_combat_agent::scenario::ScenarioBuilder;
use screeps_combat_engine::{CombatWorld, SimCreep, StructureKind};
use screeps_sim_core::rng::Rng;
use screeps_sim_core::{BodyPartDef, BoostTier as SimBoost, SimBody};

/// Rampart hits by stronghold level (index 1–5; `constants.js:849` STRONGHOLD_RAMPART_HITS).
pub const RAMPART_HITS: [u32; 6] = [0, 100_000, 200_000, 500_000, 1_000_000, 2_000_000];
/// The invader core's hit pool (`constants.js:842`).
pub const CORE_HITS: u32 = 100_000;
/// Modeled tower energy: strongholds refill towers from the core (`refillTowers`), so the pool is
/// effectively unbounded — a large finite number keeps the sim honest that drain is NOT viable.
pub const TOWER_ENERGY: u32 = 100_000;

/// One template structure: (is_tower, is_core, dx, dy). Ramparts are carried separately as the full
/// blanket (every offset in the template is ramparted).
struct Template {
    core: (i8, i8),
    towers: &'static [(i8, i8)],
    /// EVERY structure/road offset in the template — the rampart blanket covers all of them
    /// (`strongholds.js`: each entry is paired with a rampart at the same offset).
    rampart_blanket: &'static [(i8, i8)],
}

/// bunker1–5 exactly per `strongholds.js` (offsets relative to the core).
fn template(level: u8) -> Template {
    match level {
        1 => Template {
            core: (0, 0),
            towers: &[(1, 1)],
            rampart_blanket: &[(0, 0), (1, 1), (0, 1), (1, 0)],
        },
        2 => Template {
            core: (0, 0),
            towers: &[(1, 1), (-1, -1)],
            rampart_blanket: &[
                (0, 0), (1, 1), (-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 0), (-1, 0),
            ],
        },
        3 => Template {
            core: (0, 0),
            towers: &[(1, 1), (-1, -1), (-1, 1)],
            rampart_blanket: &[
                (0, 0), (1, 1), (-1, -1), (-1, 1), (-2, -1), (0, -1), (-1, 0), (1, 0), (-2, 1),
                (0, 1), (-2, 2), (-1, 2), (1, 2), (1, -1), (-2, 0), (0, 2),
            ],
        },
        4 => Template {
            core: (0, 0),
            towers: &[(1, 1), (-1, -1), (-1, 1), (1, -1)],
            rampart_blanket: &[
                (0, 0), (1, 1), (-1, -1), (-1, 1), (1, -1), (-2, -2), (-1, -2), (1, -2), (2, -2),
                (-2, -1), (0, -1), (2, -1), (-1, 0), (1, 0), (-2, 1), (0, 1), (2, 1), (-2, 2),
                (-1, 2), (1, 2), (2, 2), (2, 0), (-2, 0), (0, 2), (0, -2),
            ],
        },
        5 => Template {
            core: (0, 0),
            towers: &[(1, 1), (-1, -1), (-1, 1), (1, -1), (0, -2), (0, 2)],
            rampart_blanket: &[
                (0, 0), (1, 1), (-1, -1), (-1, 1), (1, -1), (0, -2), (0, 2), (-2, -3), (-1, -3),
                (0, -3), (1, -3), (2, -3), (-3, -2), (-1, -2), (1, -2), (3, -2), (-3, -1), (-2, -1),
                (0, -1), (2, -1), (3, -1), (-3, 0), (-2, 0), (-1, 0), (1, 0), (2, 0), (3, 0),
                (-3, 1), (-2, 1), (0, 1), (2, 1), (3, 1), (-3, 2), (-1, 2), (1, 2), (3, 2),
                (-2, 3), (-1, 3), (0, 3), (1, 3), (2, 3), (2, 2), (-2, -2), (2, -2), (-2, 2),
            ],
        },
        _ => panic!("stronghold levels are 1..=5"),
    }
}

// ── Defender bodies (creeps.js:96-126, exact) ──────────────────────────────────────────────────────

fn parts(reps: &[(Part, u32, SimBoost)]) -> SimBody {
    let mut v = Vec::new();
    for &(p, n, b) in reps {
        for _ in 0..n {
            v.push(BodyPartDef::boosted(p, b));
        }
    }
    SimBody::new(v)
}

fn body_weak_defender() -> SimBody {
    parts(&[(Part::Attack, 15, SimBoost::None), (Part::Move, 15, SimBoost::None)])
}
fn body_full_defender() -> SimBody {
    parts(&[(Part::Attack, 25, SimBoost::None), (Part::Move, 25, SimBoost::None)])
}
fn body_boosted_defender() -> SimBody {
    // ATTACK ×25 UH2O (T2 ×3).
    parts(&[(Part::Attack, 25, SimBoost::T2), (Part::Move, 25, SimBoost::None)])
}
fn body_boosted_ranger() -> SimBody {
    // RANGED ×25 KHO2 (T2 ×3).
    parts(&[(Part::RangedAttack, 25, SimBoost::T2), (Part::Move, 25, SimBoost::None)])
}
fn body_full_boosted_melee() -> SimBody {
    // ATTACK ×44 XUH2O + MOVE ×6 XZHO2 (both T3).
    parts(&[(Part::Attack, 44, SimBoost::T3), (Part::Move, 6, SimBoost::T3)])
}
fn body_full_boosted_ranger() -> SimBody {
    parts(&[(Part::RangedAttack, 44, SimBoost::T3), (Part::Move, 6, SimBoost::T3)])
}
fn body_fortifier() -> SimBody {
    // WORK ×15 XLH2O (T3 repair boost) + CARRY ×15 + MOVE ×15. Repair itself is unmodeled (doc top).
    parts(&[
        (Part::Work, 15, SimBoost::T3),
        (Part::Carry, 15, SimBoost::None),
        (Part::Move, 15, SimBoost::None),
    ])
}

/// The per-level defender population (`stronghold.js:399-489`), seeded where the engine shuffles.
fn defender_bodies(level: u8, rng: &mut Rng) -> Vec<SimBody> {
    match level {
        1 => Vec::new(),
        2 => vec![body_weak_defender()],
        3 => vec![body_full_defender(), body_full_defender()],
        4 => {
            // 4 drawn from {1 fortifier, 4 boostedDefender, 3 boostedRanger}.
            let deck: Vec<fn() -> SimBody> = vec![
                body_fortifier,
                body_boosted_defender,
                body_boosted_defender,
                body_boosted_defender,
                body_boosted_defender,
                body_boosted_ranger,
                body_boosted_ranger,
                body_boosted_ranger,
            ];
            seeded_draw(deck, 4, rng)
        }
        5 => {
            // fortifier + 8 drawn from {7 fullBoostedMelee, 9 fullBoostedRanger}.
            let mut out = vec![body_fortifier()];
            let deck: Vec<fn() -> SimBody> = std::iter::repeat_n(body_full_boosted_melee as fn() -> SimBody, 7)
                .chain(std::iter::repeat_n(body_full_boosted_ranger as fn() -> SimBody, 9))
                .collect();
            out.extend(seeded_draw(deck, 8, rng));
            out
        }
        _ => panic!("stronghold levels are 1..=5"),
    }
}

fn seeded_draw(mut deck: Vec<fn() -> SimBody>, take: usize, rng: &mut Rng) -> Vec<SimBody> {
    let mut out = Vec::with_capacity(take);
    for _ in 0..take.min(deck.len()) {
        let i = rng.range(0, deck.len() as u32 - 1) as usize;
        out.push(deck.swap_remove(i)());
    }
    out
}

// ── Terrain ────────────────────────────────────────────────────────────────────────────────────────

/// The terrain regimes a stronghold scenario can sit in (operator: "fully open layouts are not
/// interesting" — Open exists only as the calibration baseline rung of the gauntlet).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrongholdTerrain {
    Open,
    /// Procedural cave terrain (`screeps_sim_core::terrain_gen`, the same generator as
    /// `Bed::Generated`) with the stronghold footprint + a spawnable entry pocket carved clear and
    /// entry→core CONNECTIVITY guaranteed (seeds that wall the core off are skipped
    /// deterministically).
    Chokepoint,
}

/// Apply procedural cave terrain. The sim's terrain is ONE GLOBAL (x, y) grid shared by every room
/// in the world (`SimTerrain` carries no room key) — so in a multi-room scenario both rooms get the
/// SAME caves, and exit gaps align at the border by construction. Carves the stronghold footprint,
/// the entry pocket, and (multi-room) the staging pocket; scans seeds deterministically until the
/// approach is CONNECTED — including, for multi-room, a border column `y` that is open on BOTH edges
/// (exit `(0, y)` in the staging room = arrival `(49, y)` in the target room) with a path from the
/// staging centre to the exit and from the arrival to the core. Returns the accepted seed.
fn apply_chokepoint_terrain(world: &mut CombatWorld, seed: u32, core: (u8, u8), entry: (u8, u8), multi_room: bool) -> u32 {
    use screeps_sim_core::terrain_gen::{generate_terrain, Exits, TerrainGenParams};
    for probe in seed..seed + 32 {
        let gen = generate_terrain(probe, &TerrainGenParams { exits: Exits::all(), ..Default::default() });
        let mut walls: std::collections::HashSet<(u8, u8)> = gen.walls.iter().copied().collect();
        // Carve: stronghold footprint (template radius ≤ 3 → clear 4), the entry pocket, and the
        // multi-room staging pocket (the same (25,25) on the shared grid).
        let mut carves = vec![(core.0, core.1, 4i32), (entry.0, entry.1, 2)];
        if multi_room {
            carves.push((25, 25, 2));
        }
        for (cx, cy, r) in carves {
            for dx in -r..=r {
                for dy in -r..=r {
                    let (x, y) = (cx as i32 + dx, cy as i32 + dy);
                    if (0..50).contains(&x) && (0..50).contains(&y) {
                        walls.remove(&(x as u8, y as u8));
                    }
                }
            }
        }
        let ok = if multi_room {
            // Some y open on BOTH edges, reachable from the staging centre (to (0,y) — the WEST
            // exit of the EAST staging room) and from the arrival ((49,y)) to the core.
            (1..49u8).any(|y| {
                !walls.contains(&(0, y))
                    && !walls.contains(&(49, y))
                    && connected(&walls, (25, 25), (0, y))
                    && connected(&walls, (49, y), core)
            })
        } else {
            connected(&walls, entry, core)
        };
        if ok {
            for &(x, y) in &walls {
                world.movement.terrain.walls.insert((x, y));
            }
            for &(x, y) in &gen.swamps {
                if !walls.contains(&(x, y)) {
                    world.movement.terrain.swamps.insert((x, y));
                }
            }
            return probe;
        }
    }
    // Every probe walled off — fall back to open (visible via the returned seed == input).
    seed
}

/// 4-neighbour flood connectivity over non-wall tiles.
fn connected(walls: &std::collections::HashSet<(u8, u8)>, from: (u8, u8), to: (u8, u8)) -> bool {
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![from];
    while let Some((x, y)) = stack.pop() {
        if (x, y) == to {
            return true;
        }
        if !seen.insert((x, y)) {
            continue;
        }
        for (dx, dy) in [(0i32, 1i32), (0, -1), (1, 0), (-1, 0)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if (0..50).contains(&nx) && (0..50).contains(&ny) {
                let n = (nx as u8, ny as u8);
                if !walls.contains(&n) && !seen.contains(&n) {
                    stack.push(n);
                }
            }
        }
    }
    false
}

// ── The generator ──────────────────────────────────────────────────────────────────────────────────

/// A REAL invader stronghold (bunker1–5) as a sim scenario: exact template structures + rampart
/// blanket at the level's hits, exact defender population with real boosts, stronghold tower AI,
/// optionally embedded in procedural chokepoint terrain, optionally approached from a NEIGHBOUR room
/// (the multi-room border crossing the operator is worried about).
pub struct StrongholdScenario;

impl StrongholdScenario {
    /// Build one stronghold scenario. `level` 1–5; `terrain` per [`StrongholdTerrain`];
    /// `multi_room` stages the attacker in the EAST neighbour (it must cross the border under the
    /// stronghold's guns to engage); `seed` drives the defender draw + terrain probe.
    pub fn build(level: u8, terrain: StrongholdTerrain, multi_room: bool, seed: u32) -> Scenario {
        let rm: RoomName = "W5N5".parse().unwrap();
        // The east neighbour of W5N5 is W4N5 (W-coordinates DECREASE eastward).
        let staging: RoomName = "W4N5".parse().unwrap();
        const CORE: (u8, u8) = (25, 25);
        // Entry: in-room east edge for single-room; the staging room's centre for multi-room (the
        // squad then crosses the border itself — the machinery under test).
        let entry_xy: (u8, u8) = (46, 25);

        let t = template(level);
        let mut rng = Rng::seeded(seed.wrapping_mul(1069).wrapping_add(level as u32));

        let mut b = ScenarioBuilder::empty(rm);
        let at = |d: (i8, i8)| -> (u8, u8) {
            (
                (CORE.0 as i16 + d.0 as i16) as u8,
                (CORE.1 as i16 + d.1 as i16) as u8,
            )
        };
        let core_xy = at(t.core);
        let core_id = b.structure(
            StructureKind::InvaderCore,
            Some(DEFENDER),
            core_xy.0,
            core_xy.1,
            CORE_HITS,
            CORE_HITS,
        );
        for &d in t.towers {
            let (x, y) = at(d);
            b.tower(DEFENDER, x, y, TOWER_ENERGY);
        }
        let rampart_hits = RAMPART_HITS[level as usize];
        let mut world = t
            .rampart_blanket
            .iter()
            .fold(b, |b, &d| {
                let (x, y) = at(d);
                b.rampart(DEFENDER, x, y, rampart_hits)
            })
            .build();

        // Terrain (before defender placement so spots stay clear — the footprint carve keeps the
        // template + adjacent ring open). One global grid serves both rooms (see the fn doc).
        let mut terrain_seed = seed;
        if terrain == StrongholdTerrain::Chokepoint {
            terrain_seed = apply_chokepoint_terrain(&mut world, seed, CORE, entry_xy, multi_room);
        }

        // Defenders: the exact population, holding the ring — placed on the template's outer
        // rampart tiles nearest the east approach (deterministic order), inside their own ramparts
        // like the live `assignDefenders` spots.
        let bodies = defender_bodies(level, &mut rng);
        let mut spots: Vec<(u8, u8)> = t.rampart_blanket.iter().map(|&d| at(d)).collect();
        spots.sort_by_key(|&(x, y)| (std::cmp::Reverse(x), y)); // east side first
        // Defender ids start at 10_000 (the harness convention — attacker ids are 1..N).
        let next_id: u32 = world.movement.creeps.iter().map(|c| c.id).max().unwrap_or(9_999) + 1;
        for (i, body) in bodies.into_iter().enumerate() {
            let (x, y) = spots[i % spots.len()];
            world.movement.creeps.push(SimCreep {
                id: next_id + i as u32,
                owner: DEFENDER,
                pos: Position::new(
                    RoomCoordinate::new(x).unwrap(),
                    RoomCoordinate::new(y).unwrap(),
                    rm,
                ),
                body,
                fatigue: 0,
                carry_used: 0,
            });
        }

        let (assault_pos, front_tiles, support_tiles, _) = breach_geometry(rm, core_xy);
        let entry_room = if multi_room { staging } else { rm };
        let entry = Position::new(
            RoomCoordinate::new(if multi_room { 25 } else { entry_xy.0 }).unwrap(),
            RoomCoordinate::new(if multi_room { 25 } else { entry_xy.1 }).unwrap(),
            entry_room,
        );

        Scenario {
            world,
            objectives: vec![Objective {
                id: core_id,
                room: rm,
                pos: Position::new(
                    RoomCoordinate::new(core_xy.0).unwrap(),
                    RoomCoordinate::new(core_xy.1).unwrap(),
                    rm,
                ),
                assault_pos,
                front_tiles,
                support_tiles,
                entry,
                kind: ObjectiveKind::Raze,
            }],
            attacker_owner: ATTACKER,
            defender_owner: DEFENDER,
            member_energy: 5600,
            onsite_budget: 1400,
            label: format!(
                "stronghold-L{level}-{}{}#s{terrain_seed}",
                match terrain {
                    StrongholdTerrain::Open => "open",
                    StrongholdTerrain::Chokepoint => "choke",
                },
                if multi_room { "-multi" } else { "" }
            ),
            seed: seed as u64,
        }
    }
}

/// Stronghold tower AI (`stronghold.js`): focusClosest for L1–3 (identical to the default
/// `opponents::tower_intents` — each tower fires its closest hostile) and focusMax for L4–5 (ALL
/// towers focus the hostile taking maximum combined tower damage — the coordinated volley that
/// makes high strongholds dangerous; deterministic tie-break by creep id).
pub fn stronghold_tower_intents(
    world: &CombatWorld,
    level: u8,
    defender: screeps_combat_engine::PlayerId,
    intents: &mut screeps_combat_engine::Intents,
) {
    use screeps_combat_engine::damage::tower_attack_damage_at_range;
    use screeps_combat_engine::TowerAction;
    if level <= 3 {
        screeps_combat_agent::opponents::tower_intents(world, intents);
        return;
    }
    let hostiles: Vec<&SimCreep> = world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner != defender)
        .collect();
    if hostiles.is_empty() {
        return;
    }
    let towers: Vec<_> = world
        .towers
        .iter()
        .filter(|t| t.is_alive() && t.owner == defender && t.energy >= 10)
        .collect();
    let Some(target) = hostiles.iter().max_by_key(|h| {
        let dmg: u32 = towers.iter().map(|t| tower_attack_damage_at_range(t.pos.get_range_to(h.pos))).sum();
        (dmg, std::cmp::Reverse(h.id))
    }) else {
        return;
    };
    for t in &towers {
        intents.set_tower(t.id, TowerAction::Attack(target.id));
    }
}

// ── The end-to-end stronghold assault driver ───────────────────────────────────────────────────────

/// One gauntlet rung's outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RungOutcome {
    /// The core died — the rung is beaten.
    Killed { ticks: u32 },
    /// The sizing oracle refused to field (EV/winnability defer) — HONEST for what one 8-member
    /// squad cannot take; the rung reports it distinctly from a combat loss.
    Deferred,
    /// The oracle fielded but could not BUILD/place the force at the entry.
    Unfieldable,
    /// The fielded squad was wiped.
    AttackerWiped { ticks: u32 },
    /// Ran out the on-site budget; `reached` = a living attacker ended within range 8 of the core.
    Timeout { reached: bool },
}

/// Size (the REAL EV oracle, boost-aware) + field + fight one stronghold scenario end-to-end:
/// attacker = the oracle's composition driven by the managed squad brain (`Destroy`); defenders =
/// the exact invader population holding the core; towers = the stronghold AI
/// ([`stronghold_tower_intents`]). `boost_max_tier` is the attacker's supply clamp (T0 = unboosted
/// pipeline; T3 = full boost supply — the ADR 0041 boosted-vs-boosted self-play).
pub fn run_stronghold_assault(
    scenario: &Scenario,
    level: u8,
    boost_max_tier: screeps_combat_decision::bodies::BoostTier,
) -> RungOutcome {
    use crate::harness::evaluate::{evaluate_recorded, AnyOf, ObjectivesDestroyed, SideWiped, StopReason};
    use crate::harness::validate::{merge_intents, place_at_entry};
    use screeps_combat_agent::squad::ManagedSimSquad;
    use screeps_combat_decision::composition::{optimize_composition, CompositionParams};
    use screeps_combat_decision::doctrine::{DoctrineObjective, EnemyCoordination, EnemyForce};
    use screeps_combat_decision::force_sizing::{DefenseProfile, TowerThreat};
    use screeps_combat_engine::body_combat::SimBodyCombat;
    use screeps_combat_engine::constants::HEAL_POWER;

    let obj = &scenario.objectives[0];

    // The oracle's inputs from the ACTUAL world: tower ranges to the assault tile, the breach cost
    // (~2 blanket tiles on the approach line), and the defender force with REAL boosted output
    // (the engine's per-part boost math — not an estimate).
    let towers: Vec<TowerThreat> = scenario
        .world
        .towers
        .iter()
        .filter(|t| t.is_alive())
        .map(|t| TowerThreat { range_to_assault: t.pos.get_range_to(obj.assault_pos), energy: t.energy })
        .collect();
    let defenders: Vec<&SimCreep> = scenario
        .world
        .movement
        .creeps
        .iter()
        .filter(|c| c.is_alive() && c.owner == scenario.defender_owner)
        .collect();
    let enemy = (!defenders.is_empty()).then(|| EnemyForce {
        dps: defenders.iter().map(|c| (c.body.attack_power() + c.body.ranged_attack_power()) as f32).sum(),
        heal: defenders.iter().map(|c| (c.body.heal_power() * HEAL_POWER) as f32).sum(),
        hits: defenders.iter().map(|c| c.body.hits_max()).sum(),
        count: defenders.len() as u32,
        boosted: true,
    });
    // Breach cost derived from the WORLD: the strongest rampart adjacent to the core × 2 (~two
    // blanket tiles on the approach line). 0 for a bare core (the border-gauntlet scenarios).
    let breach_hits = scenario
        .world
        .structures
        .iter()
        .filter(|s| s.is_alive() && s.kind == StructureKind::Rampart && s.pos.get_range_to(obj.pos) <= 1)
        .map(|s| s.hits)
        .max()
        .unwrap_or(0)
        .saturating_mul(2);
    let defense = DefenseProfile {
        towers,
        breach_hits,
        objective_hits: obj_hits(scenario),
        repair_per_tick: 0.0,
        safe_mode: false,
        ..Default::default()
    };
    let params = CompositionParams {
        member_energy: scenario.member_energy,
        boost_max_tier,
        ..Default::default()
    };
    let Some(comp) = optimize_composition(
        DoctrineObjective::KillImmuneStructure,
        &defense,
        enemy,
        None,
        10_000_000.0, // commit ⇔ winnable (the calibration convention)
        scenario.onsite_budget,
        EnemyCoordination::Coordinated,
        0.0,
        true,
        false,
        &params,
    ) else {
        return RungOutcome::Deferred;
    };

    let mut world = scenario.world.clone();
    let Some(att_ids) = place_at_entry(&mut world, obj, &comp, scenario.attacker_owner, scenario.member_energy) else {
        return RungOutcome::Unfieldable;
    };
    let def_ids: Vec<u32> = defenders.iter().map(|c| c.id).collect();

    let mut att = ManagedSimSquad::new(scenario.attacker_owner, att_ids, obj.assault_pos);
    let mut def = ManagedSimSquad::new(scenario.defender_owner, def_ids.clone(), obj.pos)
        .with_intent(screeps_combat_decision::EngageObjective::Hold);

    let mut conditions: Vec<Box<dyn crate::harness::evaluate::RunUntil>> = vec![
        Box::new(ObjectivesDestroyed(vec![obj.id])),
        Box::new(SideWiped(scenario.attacker_owner)),
    ];
    if !def_ids.is_empty() {
        conditions.push(Box::new(SideWiped(scenario.defender_owner)));
    }
    let run_until = AnyOf(conditions);
    let (outcome, _rec) = evaluate_recorded(
        world,
        &mut |w| att.step(w),
        &mut |w, intents| {
            let d = def.step(w);
            merge_intents(intents, d);
            stronghold_tower_intents(w, level, DEFENDER, intents);
        },
        &run_until,
        scenario.onsite_budget,
    );
    match outcome.stop {
        StopReason::ObjectivesComplete => RungOutcome::Killed { ticks: outcome.ticks },
        StopReason::SideWiped(side) if side == scenario.attacker_owner => RungOutcome::AttackerWiped { ticks: outcome.ticks },
        StopReason::SideWiped(_) | StopReason::ControllerNeutralized => RungOutcome::Killed { ticks: outcome.ticks },
        StopReason::Timeout => {
            let reached = outcome.world.movement.creeps.iter().any(|c| {
                c.is_alive() && c.owner == scenario.attacker_owner && c.pos.room_name() == obj.room && c.pos.get_range_to(obj.pos) <= 8
            });
            RungOutcome::Timeout { reached }
        }
    }
}

/// The objective structure's current hits from the world (falls back to the stronghold core pool).
fn obj_hits(scenario: &Scenario) -> u32 {
    let obj = &scenario.objectives[0];
    scenario
        .world
        .structures
        .iter()
        .find(|s| s.id == obj.id)
        .map(|s| s.hits)
        .unwrap_or(CORE_HITS)
}

// ── The border gauntlet (the "picked off crossing the room" stress) ────────────────────────────────

/// A multi-room crossing under fire: the squad stages in the EAST neighbour and must cross the
/// border into chokepoint terrain guarded by a camper pack parked on the arrival side, then raze a
/// bare core deep in the room. Isolates the operator's stated worry — moving in and out of rooms
/// without being picked off — from base-siege mechanics. Grades:
/// 1 = 2 unboosted rangers · 2 = 4 unboosted rangers · 3 = 4 T2 rangers + 2 T2 melee ·
/// 4 = 6 T3 rangers (the full stronghold-tier edge camp).
pub struct BorderGauntlet;

impl BorderGauntlet {
    pub fn build(grade: u8, seed: u32) -> Scenario {
        let rm: RoomName = "W5N5".parse().unwrap();
        let staging: RoomName = "W4N5".parse().unwrap();
        const CORE: (u8, u8) = (10, 25);

        let mut b = ScenarioBuilder::empty(rm);
        let core_id = b.structure(StructureKind::InvaderCore, Some(DEFENDER), CORE.0, CORE.1, CORE_HITS, CORE_HITS);
        let mut world = b.build();
        let terrain_seed = apply_chokepoint_terrain(&mut world, seed, CORE, (46, 25), true);

        // The camper pack: parked 2–4 tiles inside the arrival edge (x = 49 side), bracketing the
        // open border columns so an arriving squad lands inside their range-3 envelope.
        fn ranged_camper() -> SimBody {
            parts(&[(Part::RangedAttack, 20, SimBoost::None), (Part::Move, 20, SimBoost::None)])
        }
        let bodies: Vec<SimBody> = match grade {
            1 => (0..2).map(|_| ranged_camper()).collect(),
            2 => (0..4).map(|_| ranged_camper()).collect(),
            3 => (0..4)
                .map(|_| body_boosted_ranger())
                .chain((0..2).map(|_| body_boosted_defender()))
                .collect(),
            4 => (0..6).map(|_| body_full_boosted_ranger()).collect(),
            _ => panic!("border gauntlet grades are 1..=4"),
        };
        // Deterministic camp tiles: walkable tiles at x ∈ 45..=47 nearest the open border columns.
        let walls = world.movement.terrain.walls.clone();
        let open_border_ys: Vec<u8> = (1..49u8).filter(|&y| !walls.contains(&(0, y)) && !walls.contains(&(49, y))).collect();
        let anchor_y = open_border_ys.get(open_border_ys.len() / 2).copied().unwrap_or(25);
        let mut camp_tiles: Vec<(u8, u8)> = Vec::new();
        'outer: for r in 0..20u8 {
            for x in 45..=47u8 {
                for dy in [0i32, 1, -1] {
                    let y = (anchor_y as i32 + dy * r as i32).clamp(1, 48) as u8;
                    if !walls.contains(&(x, y)) && !camp_tiles.contains(&(x, y)) {
                        camp_tiles.push((x, y));
                        if camp_tiles.len() >= bodies.len() {
                            break 'outer;
                        }
                    }
                }
            }
        }
        for (i, body) in bodies.into_iter().enumerate() {
            let (x, y) = camp_tiles[i % camp_tiles.len().max(1)];
            world.movement.creeps.push(SimCreep {
                id: 10_000 + i as u32,
                owner: DEFENDER,
                pos: Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), rm),
                body,
                fatigue: 0,
                carry_used: 0,
            });
        }

        let (assault_pos, front_tiles, support_tiles, _) = breach_geometry(rm, CORE);
        Scenario {
            world,
            objectives: vec![Objective {
                id: core_id,
                room: rm,
                pos: Position::new(RoomCoordinate::new(CORE.0).unwrap(), RoomCoordinate::new(CORE.1).unwrap(), rm),
                assault_pos,
                front_tiles,
                support_tiles,
                entry: Position::new(RoomCoordinate::new(25).unwrap(), RoomCoordinate::new(25).unwrap(), staging),
                kind: ObjectiveKind::Raze,
            }],
            attacker_owner: ATTACKER,
            defender_owner: DEFENDER,
            member_energy: 5600,
            onsite_budget: 1400,
            label: format!("border-gauntlet-g{grade}#s{terrain_seed}"),
            seed: seed as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use screeps_combat_decision::bodies::BoostTier;

    /// The templates match `strongholds.js` structurally: per level — 1 core, the exact tower
    /// count (1/2/3/4/6), a rampart on EVERY template offset, and the exact defender populations
    /// with the real boost tiers.
    #[test]
    fn templates_and_populations_match_the_engine_ground_truth() {
        let tower_counts = [0usize, 1, 2, 3, 4, 6];
        for level in 1..=5u8 {
            let s = StrongholdScenario::build(level, StrongholdTerrain::Open, false, 1);
            let towers = s.world.towers.len();
            assert_eq!(towers, tower_counts[level as usize], "L{level} tower count");
            let ramparts = s
                .world
                .structures
                .iter()
                .filter(|st| st.kind == StructureKind::Rampart)
                .count();
            assert_eq!(ramparts, template(level).rampart_blanket.len(), "L{level} rampart blanket");
            let cores = s
                .world
                .structures
                .iter()
                .filter(|st| st.kind == StructureKind::InvaderCore)
                .count();
            assert_eq!(cores, 1, "L{level} exactly one core");
            let defenders: Vec<_> = s
                .world
                .movement
                .creeps
                .iter()
                .filter(|c| c.owner == DEFENDER)
                .collect();
            let expected_defenders = [0usize, 0, 1, 2, 4, 9];
            assert_eq!(defenders.len(), expected_defenders[level as usize], "L{level} defender count");
            if level == 5 {
                // Every L5 combat defender is T3-boosted (fortifier WORK is T3 too).
                assert!(
                    defenders.iter().all(|c| c.body.parts.iter().any(|p| p.boost == SimBoost::T3)),
                    "L5 defenders carry T3 boosts"
                );
            }
        }
    }

    /// Chokepoint terrain keeps the approach CONNECTED (the deterministic seed scan) — entry→core
    /// single-room, and an open both-sides border column for multi-room.
    #[test]
    fn chokepoint_terrain_stays_connected() {
        let s = StrongholdScenario::build(3, StrongholdTerrain::Chokepoint, false, 7);
        assert!(!s.world.movement.terrain.walls.is_empty(), "caves actually generated");
        let walls: std::collections::HashSet<(u8, u8)> = s.world.movement.terrain.walls.clone();
        assert!(connected(&walls, (46, 25), (25, 25)), "entry→core connected");
        let m = StrongholdScenario::build(2, StrongholdTerrain::Chokepoint, true, 11);
        let walls: std::collections::HashSet<(u8, u8)> = m.world.movement.terrain.walls.clone();
        assert!(
            (1..49u8).any(|y| !walls.contains(&(0, y)) && !walls.contains(&(49, y))),
            "multi-room: an open both-sides border column exists"
        );
    }

    /// The CHECKED-IN floor — the HONEST current baseline, pinned so any change is LOUD in either
    /// direction:
    /// - L1@T0 DEFERS: with the PREFERRED_MEMBER_ENERGY=3000 clamp, the T0 heal ceiling cannot
    ///   out-sustain even one point-blank stronghold tower — which is exactly why live has only
    ///   ever killed towerless (level-0) cores. The gauntlet quantified the gap.
    /// - L1-open@T3 is **KILLED** (Phase 4.5 item 1's acceptance bar, achieved 2026-08-24 by the
    ///   cohesion-under-fire kernel work: deliverable-heal advance gating + the siege risk-currency
    ///   floor + lockstep healer advertising + evidence-gated heal triage — current run: 151 ticks,
    ///   ZERO members lost). Chokepoint/multi-room rungs stay dashboard-graded (`stronghold_gauntlet`):
    ///   the corridor trickle-in commit window and the border crossing are the open follow-ups.
    #[test]
    fn stronghold_floor_t0_defers_t3_kills_open() {
        let s = StrongholdScenario::build(1, StrongholdTerrain::Open, false, 1);
        assert_eq!(run_stronghold_assault(&s, 1, BoostTier::T0), RungOutcome::Deferred, "T0: the heal ceiling defers a stronghold tower (the pre-boost capability truth)");
        let boosted = run_stronghold_assault(&s, 1, BoostTier::T3);
        assert!(matches!(boosted, RungOutcome::Killed { .. }), "T3 must TAKE the open L1 stronghold (the Phase 4.5 item-1 bar): {boosted:?}");
    }

    /// WS-VAL — the ESCALATION GAUNTLET (operator 2026-08-23: "increasingly challenging scenarios
    /// to stress test"): every stronghold level × terrain × room-count × attacker boost supply,
    /// plus the border-gauntlet grades. Prints the ladder; asserts nothing above the checked-in
    /// floor (the dashboard is for reading where the pipeline breaks — Deferred is an HONEST
    /// verdict for what one squad cannot take). Run:
    /// `cargo test --release -p screeps-combat-eval --lib stronghold_gauntlet -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn stronghold_gauntlet() {
        println!("\n=== STRONGHOLD GAUNTLET (oracle-sized, managed brain, real invader populations) ===");
        for &(terrain, multi) in &[
            (StrongholdTerrain::Open, false),
            (StrongholdTerrain::Chokepoint, false),
            (StrongholdTerrain::Chokepoint, true),
        ] {
            for level in 1..=5u8 {
                for &tier in &[BoostTier::T0, BoostTier::T3] {
                    let s = StrongholdScenario::build(level, terrain, multi, 1);
                    let out = run_stronghold_assault(&s, level, tier);
                    println!("  {:>28}  attacker@{:?}  → {:?}", s.label, tier, out);
                }
            }
        }
        println!("\n=== BORDER GAUNTLET (crossing under fire) ===");
        for grade in 1..=4u8 {
            for &tier in &[BoostTier::T0, BoostTier::T3] {
                let s = BorderGauntlet::build(grade, 3);
                let out = run_stronghold_assault(&s, 1, tier);
                println!("  {:>28}  attacker@{:?}  → {:?}", s.label, tier, out);
            }
        }
    }

    /// Rung TRACE instrument — tick-by-tick positions + hp + squad state for ONE gauntlet rung.
    /// Point the `build(...)` line at whatever the dashboard says is broken and read the arc
    /// (approach cohesion, wall camp, retreat) directly. This is how the cohesion-under-fire
    /// defect chain was root-caused; keep it aimed at the top open rung (currently: chokepoint
    /// trickle-in).
    #[test]
    #[ignore]
    fn probe_rung() {
        use crate::harness::evaluate::{evaluate_recorded, AnyOf, ObjectivesDestroyed, SideWiped};
        use crate::harness::validate::{merge_intents, place_at_entry};
        use screeps_combat_agent::squad::ManagedSimSquad;
        use screeps_combat_decision::composition::{optimize_composition, CompositionParams};
        use screeps_combat_decision::doctrine::{DoctrineObjective, EnemyCoordination};
        use screeps_combat_decision::force_sizing::{DefenseProfile, TowerThreat};
        let s = StrongholdScenario::build(1, StrongholdTerrain::Chokepoint, false, 1);
        let obj = &s.objectives[0];
        let towers: Vec<TowerThreat> = s
            .world
            .towers
            .iter()
            .map(|t| TowerThreat { range_to_assault: t.pos.get_range_to(obj.assault_pos), energy: t.energy })
            .collect();
        eprintln!("core {:?} towers {:?} entry {:?} assault {:?}", obj.pos, s.world.towers.iter().map(|t| (t.pos.x().u8(), t.pos.y().u8())).collect::<Vec<_>>(), obj.entry, obj.assault_pos);
        let breach_hits = s
            .world
            .structures
            .iter()
            .filter(|st| st.is_alive() && st.kind == StructureKind::Rampart && st.pos.get_range_to(obj.pos) <= 1)
            .map(|st| st.hits)
            .max()
            .unwrap_or(0)
            .saturating_mul(2);
        let defense = DefenseProfile {
            towers,
            breach_hits,
            objective_hits: obj_hits(&s),
            repair_per_tick: 0.0,
            safe_mode: false,
            ..Default::default()
        };
        let comp = optimize_composition(
            DoctrineObjective::KillImmuneStructure,
            &defense,
            None,
            None,
            10_000_000.0,
            s.onsite_budget,
            EnemyCoordination::Coordinated,
            0.0,
            true,
            false,
            &CompositionParams { member_energy: s.member_energy, boost_max_tier: BoostTier::T3, ..Default::default() },
        )
        .expect("fields");
        let mut world = s.world.clone();
        let ids = place_at_entry(&mut world, obj, &comp, s.attacker_owner, s.member_energy).expect("places");
        let mut att = ManagedSimSquad::new(s.attacker_owner, ids.clone(), obj.assault_pos);
        let run_until = AnyOf(vec![
            Box::new(ObjectivesDestroyed(vec![obj.id])),
            Box::new(SideWiped(s.attacker_owner)),
        ]);
        let mut t = 0u32;
        let (outcome, _rec) = evaluate_recorded(
            world,
            &mut |w| {
                t += 1;
                let out = att.step(w);
                if t <= 4 || t % 10 == 0 {
                    let st = format!("{:?}", att.state());
                    let ps: Vec<String> = w
                        .movement
                        .creeps
                        .iter()
                        .filter(|c| ids.contains(&c.id))
                        .map(|c| format!("#{}@({},{}){}", c.id, c.pos.x().u8(), c.pos.y().u8(), c.body.hits))
                        .collect();
                    eprintln!("t{} [{}]: moves={} {:?}", t, st, out.moves.len(), ps);
                }
                out
            },
            &mut |w, intents| {
                stronghold_tower_intents(w, 1, DEFENDER, intents);
                let _ = merge_intents;
            },
            &run_until,
            s.onsite_budget,
        );
        eprintln!("OUTCOME: {:?} @ t{}", outcome.stop, outcome.ticks);
    }
}
