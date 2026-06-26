//! Stage 1 — scenario generation (ADR 0023a). A [`Generator`] produces [`Scenario`]s; it owns the
//! layout + opponent placement and is oracle-agnostic (a validator derives any oracle profile from the
//! world). [`RandomDefendedBase`] is the seeded single-room defended-base generator the oracle
//! calibration runs on (the Move B draws, now behind the seam). Phases B/C add permutation, designed,
//! and multi-room generators + opponent force specs.

use crate::harness::scenario::{Objective, ObjectiveKind, Scenario};
use crate::harness::terrain_import::{decode_terrain, fixtures, TerrainFixture};
use screeps::{Part, Position, RoomCoordinate, RoomName};
use screeps_combat_engine::{CombatTerrain, CombatWorld, PlayerId, SimBody, SimController, SimCreep, StructureKind};
use screeps_combat_agent::scenario::ScenarioBuilder;

pub const ATTACKER: PlayerId = 0;
pub const DEFENDER: PlayerId = 1;

// ── Bed geometry ───────────────────────────────────────────────────────────────────────────────────
// A core (spawn) with the breach gate (rampart) to its WEST. The focus dismantler sits at the NW
// corner: range 1 to BOTH rampart (orthogonal) and core (diagonal). All front tiles are range 1 to
// both (whole squad dismantles both phases); the focus is front[0] (unique core-range-1 by order).
// Support (healer) tiles are the focus's neighbours at core-range 2 (full adjacent HEAL, never focus).
/// Tower positions tried by `RandomDefendedBase` (varied ranges to the (24,24) assault).
const TOWER_TILES: [(u8, u8); 6] = [(24, 8), (24, 12), (24, 16), (20, 24), (28, 24), (24, 32)];

fn room() -> RoomName {
    "W1N1".parse().unwrap()
}
fn pos_in(room: RoomName, x: u8, y: u8) -> Position {
    Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), room)
}

/// Breach staging for a `core` with the gate to its WEST. Returns `(assault, front_tiles, support_tiles,
/// rampart_xy)`: the assault corner (NW of the core), the front (range 1 to BOTH gate + core), the
/// support (adjacent to the assault, core-range ≥ 2 → full adjacent HEAL, never the focus), and the
/// rampart tile. (For `core == (25,25)` this is exactly the Move B layout.)
pub(crate) fn breach_geometry(rm: RoomName, core: (u8, u8)) -> (Position, Vec<Position>, Vec<Position>, (u8, u8)) {
    let (cx, cy) = core;
    let p = |x: u8, y: u8| pos_in(rm, x, y);
    let assault = p(cx - 1, cy - 1);
    let front = vec![p(cx - 1, cy - 1), p(cx, cy - 1), p(cx - 1, cy + 1), p(cx, cy + 1)];
    let support = vec![p(cx - 2, cy - 2), p(cx - 1, cy - 2), p(cx, cy - 2), p(cx - 2, cy - 1), p(cx - 2, cy)];
    (assault, front, support, (cx - 1, cy))
}

/// A scenario source. Seeded by index → fully reproducible.
pub trait Generator {
    fn label(&self) -> &str;
    /// How many distinct scenarios this generator offers.
    fn count(&self) -> u32;
    fn generate(&self, index: u32) -> Scenario;
}

// ── Seeded RNG (SplitMix64 — per-index reproducible; no `rand`/`Date`/`Math.random`) ───────────────
pub(crate) struct Rng(u64);
impl Rng {
    pub(crate) fn seeded(index: u32) -> Self {
        Rng(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index as u64 + 1))
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive range `[lo, hi]`.
    pub(crate) fn range(&mut self, lo: u32, hi: u32) -> u32 {
        debug_assert!(hi >= lo);
        lo + (self.next_u64() % (hi - lo + 1) as u64) as u32
    }
    pub(crate) fn chance(&mut self, pct: u32) -> bool {
        self.range(0, 99) < pct
    }
    pub(crate) fn pick(&mut self, xs: &[u32]) -> u32 {
        xs[(self.next_u64() % xs.len() as u64) as usize]
    }
}

/// The seeded single-room defended-base generator (the oracle-calibration substrate): a core behind a
/// (usually present) rampart, 0–6 towers of varied energy/range, a small safe-mode chance, and RCL-ish
/// member energy + on-site budget. No opponent CREEPS yet (`enemy_dps` is the validator's concern; the
/// force-spec hook lands in Phase C).
pub struct RandomDefendedBase {
    pub n: u32,
}

impl Generator for RandomDefendedBase {
    fn label(&self) -> &str {
        "random-defended-base"
    }
    fn count(&self) -> u32 {
        self.n
    }
    fn generate(&self, index: u32) -> Scenario {
        let mut rng = Rng::seeded(index);
        let member_energy = rng.pick(&[1300, 1800, 2300, 3300, 5600, 12_900]);
        let onsite_budget = rng.range(600, 1400);
        let core_hits = rng.range(20_000, 100_000);
        let rampart_hits = if rng.chance(70) { rng.range(1, 80_000) } else { 0 };
        let n_towers = rng.range(0, 6);
        let safe_mode = rng.chance(5);

        let rm = room();
        const CORE: (u8, u8) = (25, 25);
        let (assault_pos, front_tiles, support_tiles, rampart_xy) = breach_geometry(rm, CORE);
        let mut b = ScenarioBuilder::empty(rm);
        let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), CORE.0, CORE.1, core_hits, core_hits);
        for &(tx, ty) in TOWER_TILES.iter().take(n_towers as usize) {
            let energy = if rng.chance(25) { rng.range(0, 9) } else { rng.range(100, 100_000) };
            b.tower(DEFENDER, tx, ty, energy);
        }
        let mut world: CombatWorld = if rampart_hits > 0 {
            b.rampart(DEFENDER, rampart_xy.0, rampart_xy.1, rampart_hits).build()
        } else {
            b.build()
        };
        if safe_mode {
            world.safe_mode_owner = Some(DEFENDER);
        }

        let objective = Objective {
            id: core_id,
            room: rm,
            pos: pos_in(rm, CORE.0, CORE.1),
            assault_pos,
            front_tiles,
            support_tiles,
            entry: pos_in(rm, CORE.0 - 10, CORE.1), // a clear western approach for a moving assault
            kind: ObjectiveKind::Raze,
        };

        Scenario {
            world,
            objectives: vec![objective],
            attacker_owner: ATTACKER,
            defender_owner: DEFENDER,
            member_energy,
            onsite_budget,
            label: format!("random-defended-base#{index}"),
            seed: index as u64,
        }
    }
}

// ── Phase B: terrain-rich layouts (walls/corridors/swamps) + a permutation enumerator ──────────────

/// A single-room layout shape — the room/wall structure the operator wants to SEE, and that a moving
/// assault (the `ManagedSquadIntegration` validator) navigates.
#[derive(Clone, Copy, Debug)]
pub enum Layout {
    /// No terrain — an open field (the calibration baseline).
    Open,
    /// A full wall column west of the core with a 3-wide gap (a choke the assault funnels through).
    Corridor,
    /// A swamp band between the western approach and the core (slows the assault — visible terrain).
    SwampApproach,
    /// A natural-wall bunker ring around the core with a west gap at the breach.
    Bunker,
}

/// A list of in-room tiles.
type Tiles = Vec<(u8, u8)>;

/// The terrain (natural wall + swamp tiles) for a layout around `core`, leaving the breach staging (NW
/// of the core) and the western approach clear.
fn layout_terrain(layout: Layout, core: (u8, u8)) -> (Tiles, Tiles) {
    let (cx, cy) = core;
    let mut walls = Vec::new();
    let mut swamps = Vec::new();
    match layout {
        Layout::Open => {}
        Layout::Corridor => {
            let wx = cx - 6;
            for y in 1..=48u8 {
                if !(cy - 1..=cy + 1).contains(&y) {
                    walls.push((wx, y));
                }
            }
        }
        Layout::SwampApproach => {
            for y in cy - 4..=cy + 4 {
                for x in cx - 8..=cx - 4 {
                    swamps.push((x, y));
                }
            }
        }
        Layout::Bunker => {
            // A box ring at radius 3, gap on the west column (cx-3, cy-1..cy+1) for the breach approach.
            for d in 0..=6u8 {
                walls.push((cx - 3 + d, cy - 3)); // north
                walls.push((cx - 3 + d, cy + 3)); // south
                walls.push((cx + 3, cy - 3 + d)); // east
            }
            for d in 0..=6u8 {
                let y = cy - 3 + d;
                if !(cy - 1..=cy + 1).contains(&y) {
                    walls.push((cx - 3, y)); // west wall, minus the gap
                }
            }
        }
    }
    (walls, swamps)
}

/// The opponent (defender) creep force guarding the objective — random or designed. Realized into
/// defender `SimCreep`s near the core; their attack/ranged output flows into the oracle's `enemy_dps`
/// (the validator's `derive_profile` sums it) and they fight the managed assault (the combat the
/// operator sees). Stationary (`defense_intents` issues no moves), so they don't perturb the sizing
/// calibration's movement-free purity.
#[derive(Clone, Copy, Debug)]
pub enum ForceSpec {
    /// No defender creeps (towers/structures only).
    None,
    /// `n` ranged skirmishers (RANGED_ATTACK + MOVE).
    Skirmishers(u32),
    /// `n` melee defenders (TOUGH + ATTACK + MOVE) + 1 healer.
    Guard(u32),
}

/// Place a [`ForceSpec`]'s defender creeps in `world` around `core`, owned by `defender`. Ids start at
/// 10_000 so they never collide with attacker ids (1..N).
fn place_force(world: &mut CombatWorld, rm: RoomName, core: (u8, u8), spec: ForceSpec, defender: PlayerId) {
    let (cx, cy) = core;
    // A ring of guard tiles around the core (skip the core + the western breach approach).
    let ring: [(i32, i32); 6] = [(1, 0), (2, 0), (1, 2), (2, 1), (1, -2), (2, -1)];
    let mut push = |i: usize, parts: &[Part], id: u32| {
        let (dx, dy) = ring[i % ring.len()];
        let x = (cx as i32 + dx).clamp(0, 49) as u8;
        let y = (cy as i32 + dy).clamp(0, 49) as u8;
        let pos = Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), rm);
        world.creeps.push(SimCreep { id, owner: defender, pos, body: SimBody::unboosted(parts), fatigue: 0 });
    };
    match spec {
        ForceSpec::None => {}
        ForceSpec::Skirmishers(n) => {
            let body = [Part::RangedAttack, Part::RangedAttack, Part::RangedAttack, Part::Move, Part::Move, Part::Move];
            for i in 0..n as usize {
                push(i, &body, 10_000 + i as u32);
            }
        }
        ForceSpec::Guard(n) => {
            let melee = [Part::Tough, Part::Tough, Part::Attack, Part::Attack, Part::Move, Part::Move, Part::Move, Part::Move];
            for i in 0..n as usize {
                push(i, &melee, 10_000 + i as u32);
            }
            let healer = [Part::Heal, Part::Heal, Part::Heal, Part::Move, Part::Move, Part::Move];
            push(n as usize, &healer, 10_000 + n);
        }
    }
}

/// Assemble a single-room defended-objective scenario with terrain, a core (+ optional rampart),
/// towers, and an opponent force. The breach staging + a clear western `entry` come from
/// [`breach_geometry`].
#[allow(clippy::too_many_arguments)]
fn assemble_single_room(
    label: String,
    seed: u64,
    member_energy: u32,
    onsite_budget: u32,
    core: (u8, u8),
    rampart_hits: u32,
    towers: &[((u8, u8), u32)],
    layout: Layout,
    force: ForceSpec,
    safe_mode: bool,
) -> Scenario {
    let rm = room();
    let (assault_pos, front_tiles, support_tiles, rampart_xy) = breach_geometry(rm, core);
    let (walls, swamps) = layout_terrain(layout, core);
    let mut b = ScenarioBuilder::empty(rm);
    for (x, y) in walls {
        b.world_mut().terrain_mut(rm).walls.insert((x, y));
    }
    for (x, y) in swamps {
        b.world_mut().terrain_mut(rm).swamps.insert((x, y));
    }
    let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), core.0, core.1, 50_000, 50_000);
    if rampart_hits > 0 {
        b.structure(StructureKind::Rampart, Some(DEFENDER), rampart_xy.0, rampart_xy.1, rampart_hits, rampart_hits);
    }
    for &((tx, ty), e) in towers {
        b.tower(DEFENDER, tx, ty, e);
    }
    let mut world = b.build();
    if safe_mode {
        world.safe_mode_owner = Some(DEFENDER);
    }
    place_force(&mut world, rm, core, force, DEFENDER);
    let objective = Objective {
        id: core_id,
        room: rm,
        pos: pos_in(rm, core.0, core.1),
        assault_pos,
        front_tiles,
        support_tiles,
        entry: pos_in(rm, core.0 - 11, core.1),
        kind: ObjectiveKind::Raze,
    };
    Scenario {
        world,
        objectives: vec![objective],
        attacker_owner: ATTACKER,
        defender_owner: DEFENDER,
        member_energy,
        onsite_budget,
        label,
        seed,
    }
}

/// A systematic permutation enumerator over a feature grid (layout × rampart × towers × RCL): coverage
/// is enumerable, not sampled. `count()` is the grid size; `generate(i)` decodes `i` into a feature tuple.
pub struct Permutations;

const PERM_LAYOUTS: [Layout; 4] = [Layout::Open, Layout::Corridor, Layout::SwampApproach, Layout::Bunker];
const PERM_RAMPARTS: [u32; 3] = [0, 20_000, 60_000];
const PERM_TOWERS: [u32; 3] = [0, 2, 4];
const PERM_ENERGY: [u32; 3] = [2300, 5600, 12_900];

impl Generator for Permutations {
    fn label(&self) -> &str {
        "permutations"
    }
    fn count(&self) -> u32 {
        (PERM_LAYOUTS.len() * PERM_RAMPARTS.len() * PERM_TOWERS.len() * PERM_ENERGY.len()) as u32
    }
    fn generate(&self, index: u32) -> Scenario {
        let i = index as usize;
        let layout = PERM_LAYOUTS[i % PERM_LAYOUTS.len()];
        let rampart = PERM_RAMPARTS[(i / PERM_LAYOUTS.len()) % PERM_RAMPARTS.len()];
        let n_towers = PERM_TOWERS[(i / (PERM_LAYOUTS.len() * PERM_RAMPARTS.len())) % PERM_TOWERS.len()];
        let energy = PERM_ENERGY[(i / (PERM_LAYOUTS.len() * PERM_RAMPARTS.len() * PERM_TOWERS.len())) % PERM_ENERGY.len()];
        let towers: Vec<((u8, u8), u32)> = TOWER_TILES.iter().take(n_towers as usize).map(|&t| (t, 100_000)).collect();
        assemble_single_room(
            format!("perm#{index} {layout:?} r{rampart} t{n_towers} e{energy}"),
            index as u64,
            energy,
            1200,
            (25, 25),
            rampart,
            &towers,
            layout,
            ForceSpec::None,
            false,
        )
    }
}

/// A small set of hand-authored fixtures (named, terrain-rich) — regression anchors + the variety the
/// operator eyeballs. Includes a multi-room siege (the objective in the east neighbour, the assault
/// staging in the home room).
pub struct Designed;

impl Generator for Designed {
    fn label(&self) -> &str {
        "designed"
    }
    fn count(&self) -> u32 {
        6
    }
    fn generate(&self, index: u32) -> Scenario {
        match index {
            0 => assemble_single_room("designed#0 open + skirmishers".into(), 0, 5600, 1200, (25, 25), 30_000, &[((24, 16), 100_000)], Layout::Open, ForceSpec::Skirmishers(3), false),
            1 => assemble_single_room("designed#1 wall-corridor + guard".into(), 1, 5600, 1200, (32, 25), 20_000, &[((32, 12), 100_000), ((28, 12), 100_000)], Layout::Corridor, ForceSpec::Guard(2), false),
            2 => assemble_single_room("designed#2 swamp-approach".into(), 2, 12_900, 1300, (25, 25), 40_000, &[((24, 14), 100_000)], Layout::SwampApproach, ForceSpec::None, false),
            3 => assemble_single_room("designed#3 bunker + guard".into(), 3, 12_900, 1300, (25, 25), 60_000, &[((25, 22), 100_000), ((25, 28), 100_000)], Layout::Bunker, ForceSpec::Guard(3), false),
            4 => twin_room_siege(),
            // Open field, NO towers, ranged defenders — a clean two-managed-squad skirmish (both sides
            // move + trade fire), the self-play realism case.
            _ => assemble_single_room("designed#5 open skirmish (self-play)".into(), 5, 5600, 600, (25, 25), 0, &[], Layout::Open, ForceSpec::Skirmishers(3), false),
        }
    }
}

/// A varied set of **realistic defended bases** for the ADR 0025 base-attack/defend tuning lens —
/// terrain + structures + defenders across the regimes a real attacker meets: an open base ringed by a
/// tower nest, a corridor-choked guarded base, bunkers behind thick ramparts with tower crossfire and a
/// guard+healer force, and a swamp-slowed turtle. Reuses [`assemble_single_room`] (so it shares the
/// breach geometry + clear-entry + apply layer with the rest of the harness). Distinct from the symmetric
/// open-combat basket: here the "opponent" is the base, scored by objective progress (`assault_score`).
pub fn realistic_bases() -> Vec<Scenario> {
    let nest = |cx: u8, cy: u8| -> Vec<((u8, u8), u32)> {
        // a tight 3-tower nest around (cx,cy), all energized.
        [(0i32, 0i32), (1, 0), (0, 1)].iter().map(|&(dx, dy)| (((cx as i32 + dx) as u8, (cy as i32 + dy) as u8), 100_000u32)).collect()
    };
    vec![
        // Open base, full tower nest, no rampart — the attacker must close under heavy crossfire.
        assemble_single_room("base#0 open tower-nest".into(), 100, 12_900, 1400, (25, 25), 0, &nest(24, 16), Layout::Open, ForceSpec::Skirmishers(3), false),
        // Corridor-choked guarded base behind a moderate rampart + two towers.
        assemble_single_room("base#1 corridor choke + guard".into(), 101, 12_900, 1400, (32, 25), 30_000, &[((32, 12), 100_000), ((28, 14), 100_000)], Layout::Corridor, ForceSpec::Guard(3), false),
        // Bunker behind a THICK rampart, tower crossfire N+S, a strong guard+healer — the turtle.
        assemble_single_room("base#2 bunker turtle (thick rampart)".into(), 102, 12_900, 1500, (25, 25), 120_000, &[((25, 21), 100_000), ((25, 29), 100_000), ((21, 25), 100_000)], Layout::Bunker, ForceSpec::Guard(4), false),
        // Bunker, lighter rampart, 2 towers, melee guard — the common mid-RCL base.
        assemble_single_room("base#3 bunker + 2 towers".into(), 103, 12_900, 1400, (25, 25), 50_000, &[((23, 23), 100_000), ((27, 27), 100_000)], Layout::Bunker, ForceSpec::Guard(2), false),
        // Swamp-slowed approach to a ramparted core + a tower — terrain attrition before the wall.
        assemble_single_room("base#4 swamp turtle".into(), 104, 12_900, 1500, (25, 25), 40_000, &[((24, 14), 100_000)], Layout::SwampApproach, ForceSpec::Guard(1), false),
    ]
}

/// A multi-room fixture: the assault stages in `W1N1` and the objective core sits behind a corridor in
/// the east neighbour `W2N1` — the managed assault paths across the room border to engage (visible
/// cross-room movement).
fn twin_room_siege() -> Scenario {
    let home: RoomName = "W1N1".parse().unwrap();
    let target: RoomName = "W2N1".parse().unwrap();
    let core = (10u8, 25u8); // near the west edge of the target room (just across the border)
    let (assault_pos, front_tiles, support_tiles, rampart_xy) = breach_geometry(target, core);
    let mut b = ScenarioBuilder::empty(home);
    // A corridor wall in the target room, west of the core, with a gap.
    for y in 1..=48u8 {
        if !(24..=26).contains(&y) {
            b.world_mut().terrain_mut(target).walls.insert((5, y));
        }
    }
    let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), core.0, core.1, 50_000, 50_000);
    b.structure(StructureKind::Rampart, Some(DEFENDER), rampart_xy.0, rampart_xy.1, 30_000, 30_000);
    let mut world = b.build();
    // No tower here — this fixture demonstrates cross-room MOVEMENT + engagement (the assault crosses
    // W1N1→W2N1 and fights the defenders) rather than a tower-turtle the squad would retreat from.
    place_force(&mut world, target, core, ForceSpec::Skirmishers(2), DEFENDER);
    let objective = Objective {
        id: core_id,
        room: target,
        pos: Position::new(RoomCoordinate::new(core.0).unwrap(), RoomCoordinate::new(core.1).unwrap(), target),
        assault_pos,
        front_tiles,
        support_tiles,
        // Stage near the W1N1 west border so the cross into W2N1 is a short, reliably-pathable hop.
        entry: Position::new(RoomCoordinate::new(5).unwrap(), RoomCoordinate::new(25).unwrap(), home),
        kind: ObjectiveKind::Raze,
    };
    Scenario {
        world,
        objectives: vec![objective],
        attacker_owner: ATTACKER,
        defender_owner: DEFENDER,
        member_energy: 12_900,
        onsite_budget: 1400,
        label: "designed#4 twin-room-siege".into(),
        seed: 4,
    }
}

// ── ADR 0025 §12 Stage 2: scenarios over REAL imported terrain ──────────────────────────────────────

/// The objective kinds the imported-room generator enumerates — a VARIETY of attacker goals over real
/// terrain (ADR 0025 §12 Stage 2).
const OBJECTIVE_KINDS: [ObjectiveKind; 5] =
    [ObjectiveKind::Raze, ObjectiveKind::Breach, ObjectiveKind::Secure, ObjectiveKind::Farm, ObjectiveKind::Declaim];

/// Nearest non-wall interior tile (1..=48) to `(tx,ty)` by 8-connected BFS — anchors a base on real
/// terrain. (The fixture's own object coords are not reliably aligned to the terrain index — the Stage 1
/// caveat — so placement is derived from the decoded terrain itself.)
fn nearest_open(terrain: &CombatTerrain, tx: u8, ty: u8) -> (u8, u8) {
    use std::collections::{HashSet, VecDeque};
    let open = |x: i32, y: i32| (1..=48).contains(&x) && (1..=48).contains(&y) && !terrain.is_wall(x as u8, y as u8);
    let mut seen: HashSet<(i32, i32)> = HashSet::new();
    let mut q = VecDeque::from([(tx as i32, ty as i32)]);
    while let Some((x, y)) = q.pop_front() {
        if !seen.insert((x, y)) {
            continue;
        }
        if open(x, y) {
            return (x as u8, y as u8);
        }
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)] {
            let (nx, ny) = (x + dx, y + dy);
            if (0..50).contains(&nx) && (0..50).contains(&ny) {
                q.push_back((nx, ny));
            }
        }
    }
    (25, 25)
}

/// The 4-connected open component reachable from `start` — the navigable region the squad shares with the
/// objective (guarantees a chosen entry can path to the core).
fn open_component(terrain: &CombatTerrain, start: (u8, u8)) -> Vec<(u8, u8)> {
    use std::collections::HashSet;
    let open = |x: i32, y: i32| (0..50).contains(&x) && (0..50).contains(&y) && !terrain.is_wall(x as u8, y as u8);
    let mut seen: HashSet<(u8, u8)> = HashSet::new();
    let mut stack = vec![(start.0 as i32, start.1 as i32)];
    let mut out = Vec::new();
    while let Some((x, y)) = stack.pop() {
        if !open(x, y) {
            continue;
        }
        let t = (x as u8, y as u8);
        if !seen.insert(t) {
            continue;
        }
        out.push(t);
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]);
    }
    out
}

/// The room one step WEST of `room` (the multi-room staging room): `E{n}`->`E{n-1}` (`E0`->`W0`),
/// `W{n}`->`W{n+1}`. Falls back to the synthetic home if parsing fails.
fn west_neighbor(room: RoomName) -> RoomName {
    let s = room.to_string();
    let parse = || -> Option<RoomName> {
        let ew = s.chars().next()?;
        let n_split = s[1..].find(['N', 'S'])? + 1;
        let x: i32 = s[1..n_split].parse().ok()?;
        let ns_and_y = &s[n_split..];
        let (new_ew, new_x) = match ew {
            'E' if x > 0 => ('E', x - 1),
            'E' => ('W', 0),
            'W' => ('W', x + 1),
            _ => return None,
        };
        format!("{new_ew}{new_x}{ns_and_y}").parse().ok()
    };
    parse().unwrap_or_else(|| "W1N1".parse().unwrap())
}

/// Realize a fixture's REAL terrain + a kind-appropriate objective + a seeded random DEFENDER force into a
/// [`Scenario`] (ADR 0025 §12 Stage 2). Base placement is derived from the decoded terrain (clear,
/// navigable tiles), not the fixture's object coords. `multi_room` stages the moving assault in the
/// western-neighbour room so it crosses the border to engage.
fn assemble_imported(fixture: &TerrainFixture, kind: ObjectiveKind, comp_seed: u32, multi_room: bool) -> Scenario {
    let mut rng = Rng::seeded(comp_seed.wrapping_mul(31).wrapping_add(kind as u32 + 1));
    let terrain = decode_terrain(&fixture.terrain);
    let target: RoomName = fixture.room.parse().unwrap_or_else(|_| room());
    let p = |x: u8, y: u8| pos_in(target, x, y);

    // Anchor the base in the navigable interior near room centre; derive the breach staging from clear tiles.
    let core = nearest_open(&terrain, 25, 25);
    let component = open_component(&terrain, core);
    let comp_set: std::collections::HashSet<(u8, u8)> = component.iter().copied().collect();
    let neigh = |c: (u8, u8)| {
        [(-1i32, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)].into_iter().filter_map(move |(dx, dy)| {
            let (nx, ny) = (c.0 as i32 + dx, c.1 as i32 + dy);
            ((0..50).contains(&nx) && (0..50).contains(&ny)).then_some((nx as u8, ny as u8))
        })
    };
    let mut front: Vec<(u8, u8)> = neigh(core).filter(|t| comp_set.contains(t)).take(4).collect();
    if front.is_empty() {
        front.push(core);
    }
    let assault = front[0];
    let support: Vec<(u8, u8)> = neigh(assault).filter(|t| comp_set.contains(t) && *t != core && !front.contains(t)).take(3).collect();
    let entry_xy = component.iter().copied().min_by_key(|&(x, _)| x).unwrap_or(core);

    // ── world: real terrain + a realistic base in the target room ──
    let mut b = ScenarioBuilder::empty(target).in_room(target);
    *b.world_mut().terrain_mut(target) = terrain;
    let core_id = b.structure(StructureKind::Spawn, Some(DEFENDER), core.0, core.1, 50_000, 50_000);
    let rampart_hits = rng.range(15_000, 70_000); // the breach gate shielding the core
    let breach_id = b.structure(StructureKind::Rampart, Some(DEFENDER), core.0, core.1, rampart_hits, rampart_hits);
    // 1-3 energized towers spread over open tiles a few rings out from the core.
    let tower_tiles: Vec<(u8, u8)> = component
        .iter()
        .copied()
        .filter(|&t| {
            let d = (t.0 as i32 - core.0 as i32).abs().max((t.1 as i32 - core.1 as i32).abs());
            (3..=8).contains(&d)
        })
        .collect();
    let n_towers = rng.range(1, 3).min(tower_tiles.len() as u32);
    let stride = (tower_tiles.len() / n_towers.max(1) as usize).max(1);
    for i in 0..n_towers as usize {
        if let Some(&(tx, ty)) = tower_tiles.get(i * stride) {
            b.tower(DEFENDER, tx, ty, 100_000);
        }
    }
    let mut world = b.build();

    // ── defenders: a seeded random force on near-core open tiles ──
    let n_def = rng.range(1, 3);
    let bodies = crate::harness::roster::random_squad(&mut rng, 2300, n_def as u8);
    let mut def_tiles: Vec<(u8, u8)> = component.iter().copied().filter(|&t| t != core && !front.contains(&t)).collect();
    def_tiles.sort_by_key(|&(x, y)| (x as i32 - core.0 as i32).abs().max((y as i32 - core.1 as i32).abs()));
    for (i, body) in bodies.iter().enumerate() {
        if let Some(&(dx, dy)) = def_tiles.get(i) {
            world.creeps.push(SimCreep { id: 10_000 + i as u32, owner: DEFENDER, pos: pos_in(target, dx, dy), body: SimBody::unboosted(body), fatigue: 0 });
        }
    }

    // ── declaim: a controller at the core ──
    if kind == ObjectiveKind::Declaim {
        world.controllers.push(SimController { pos: p(core.0, core.1), owner: Some(DEFENDER), downgrade_ticks: 50_000 });
    }

    // Breach targets the rampart gate; every other kind targets the spawn core. Declaim's stop condition
    // reads the controller at `pos` (= the core tile).
    let obj_id = if kind == ObjectiveKind::Breach { breach_id } else { core_id };
    let entry = if multi_room {
        // Stage at the western-neighbour room's EAST border so the cross into the target is a short hop.
        pos_in(west_neighbor(target), 48, entry_xy.1.clamp(1, 48))
    } else {
        p(entry_xy.0, entry_xy.1)
    };

    let objective = Objective {
        id: obj_id,
        room: target,
        pos: p(core.0, core.1),
        assault_pos: p(assault.0, assault.1),
        front_tiles: front.iter().map(|&(x, y)| p(x, y)).collect(),
        support_tiles: support.iter().map(|&(x, y)| p(x, y)).collect(),
        entry,
        kind,
    };
    Scenario {
        world,
        objectives: vec![objective],
        attacker_owner: ATTACKER,
        defender_owner: DEFENDER,
        member_energy: 12_900,
        onsite_budget: 1400,
        label: format!("imported-{}{}-{kind:?}#{comp_seed}", fixture.room, if multi_room { "-multi" } else { "" }),
        seed: comp_seed as u64,
    }
}

/// Scenarios over committed REAL terrain (Stage 1 fixtures) × objective kind × defender-comp seed (ADR
/// 0025 §12 Stage 2). Single-room by default; `multi_room` stages the assault in the western neighbour and
/// crosses the border. `count = fixtures × OBJECTIVE_KINDS × n_comps`.
pub struct ImportedRoom {
    pub multi_room: bool,
    pub n_comps: u32,
}

impl Generator for ImportedRoom {
    fn label(&self) -> &str {
        if self.multi_room {
            "imported-room-multi"
        } else {
            "imported-room"
        }
    }
    fn count(&self) -> u32 {
        fixtures().len() as u32 * OBJECTIVE_KINDS.len() as u32 * self.n_comps.max(1)
    }
    fn generate(&self, index: u32) -> Scenario {
        let fx = fixtures();
        let n = fx.len() as u32;
        let k = OBJECTIVE_KINDS.len() as u32;
        let fixture = &fx[(index % n) as usize];
        let kind = OBJECTIVE_KINDS[((index / n) % k) as usize];
        let comp_seed = index / (n * k);
        assemble_imported(fixture, kind, comp_seed, self.multi_room)
    }
}
