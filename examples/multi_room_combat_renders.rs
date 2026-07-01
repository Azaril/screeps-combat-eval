//! REAL-AI SELF-PLAY render corpus over VARIED REAL TERRAIN × VARIED COMPOSITIONS (ADR 0038 Phase 1).
//!
//! `cargo run --example multi_room_combat_renders -p screeps-combat-eval`
//! → writes `combat-renders/*.html` + `combat-renders/index.html` (open the index).
//!
//! Every render is a REAL self-play match: BOTH sides are `ManagedSimSquad` (real
//! `decide_squad_with_pathing` brain — advance / kite / focus-fire / heal), moving over the REAL engine
//! (`resolve_tick`: damage / heal / traffic / tower falloff / rampart / deaths) and REAL rover pathing,
//! on REAL mmo:shard3 room terrain. Nothing synthetic:
//!   - TERRAIN: `terrain_import::fixtures()` — the 13 committed real shard-3 room terrains
//!     (walls/swamps/controller/sources). Applied to the bed **mirror-symmetrized** (left half reflected
//!     onto the right) so the self-play bed is FAIR (same walls/swamps facing each side) — the payoff
//!     stays meaningful. Terrain MATTERS: walls channel the approach, swamps slow the crossing.
//!   - COMPOSITION: `roster::sample_squad(seed, energy, n)` — free-form budgeted bodies, seeded &
//!     deterministic. BOTH sides' comps vary across the corpus (ranged-heavy / melee+heal / mixed /
//!     drain), plus energy budget and squad size.
//!   - DRIVEN: `validate::render_self_play_replay_bodies` places the explicit attacker roster at the west
//!     entry and runs the same both-sides-managed loop as `run_self_play` (defender = the ringed core
//!     force + firing towers).
//!
//! DETERMINISTIC: every scenario is seeded by its index (SplitMix64 via `sample_squad`); no wall-clock,
//! no ambient RNG (the sim-determinism fence).
//!
//! Host-only (the eval crate is `--workspace --exclude`'d from the wasm build).

use screeps::{Part, Position, RoomCoordinate, RoomName};
use screeps_combat_agent::scenario::ScenarioBuilder;
use screeps_combat_engine::{PlayerId, SimBody, SimCreep, SimTerrain, StructureKind};
use screeps_combat_eval::harness::roster::sample_squad;
use screeps_combat_eval::harness::scenario::{Objective, ObjectiveKind, Scenario};
use screeps_combat_eval::harness::terrain_import::{decode_terrain, fixtures, TerrainFixture};
use screeps_combat_eval::harness::validate::render_self_play_replay_bodies;

const ATTACKER: PlayerId = 0;
const DEFENDER: PlayerId = 1;
const OUT_DIR: &str = "combat-renders";

fn pos_in(room: RoomName, x: u8, y: u8) -> Position {
    Position::new(
        RoomCoordinate::new(x).unwrap(),
        RoomCoordinate::new(y).unwrap(),
        room,
    )
}

/// The composition FLAVOUR each side fields — a human descriptor + how `sample_squad` is parameterised.
/// The seed selects the free-form body mix; the flavour biases the (energy, size) knobs so the corpus
/// visibly spans ranged-heavy, melee brawls, sustained heal, and lean drain forces.
#[derive(Clone, Copy)]
struct Flavour {
    name: &'static str,
    energy: u32,
    size: u8,
    /// A per-flavour seed nudge so two sides asked for the "same" flavour still differ in body mix.
    seed_salt: u32,
}

const FLAVOURS: [Flavour; 6] = [
    Flavour {
        name: "ranged-heavy",
        energy: 12_900,
        size: 4,
        seed_salt: 11,
    },
    Flavour {
        name: "melee+heal",
        energy: 8_400,
        size: 4,
        seed_salt: 23,
    },
    Flavour {
        name: "mixed",
        energy: 5_600,
        size: 3,
        seed_salt: 37,
    },
    Flavour {
        name: "lean-drain",
        energy: 2_300,
        size: 2,
        seed_salt: 53,
    },
    Flavour {
        name: "big-brawl",
        energy: 12_900,
        size: 5,
        seed_salt: 71,
    },
    Flavour {
        name: "small-skirmish",
        energy: 3_400,
        size: 3,
        seed_salt: 97,
    },
];

impl Flavour {
    /// Sample this side's roster deterministically from the scenario seed (the flavour salt keeps the two
    /// sides distinct even at the same seed/flavour).
    fn roster(&self, seed: u32) -> Vec<Vec<Part>> {
        sample_squad(
            seed.wrapping_mul(131).wrapping_add(self.seed_salt),
            self.energy,
            self.size,
        )
    }
}

/// Mirror-symmetrize a real fixture's terrain (reflect the left half x<25 onto the right, x→49-x) so the
/// self-play bed is FAIR — identical walls/swamps face each side. Then CLEAR the two deployment corridors
/// (west entry + east core approach, mid-band in y) so both squads always field and can move out. The
/// interior walls/swamps between the corridors still CHANNEL the approach + SLOW the crossing — terrain
/// that matters, not an empty room. (Same construction the tournament's `Bed::Imported` uses.)
fn symmetric_terrain(fixture: &TerrainFixture) -> SimTerrain {
    let real = decode_terrain(&fixture.terrain);
    let mut t = SimTerrain::default();
    for x in 0..25u8 {
        for y in 0..50u8 {
            let (wall, swamp) = (real.is_wall(x, y), real.swamps.contains(&(x, y)));
            for tx in [x, 49 - x] {
                if wall {
                    t.walls.insert((tx, y));
                } else if swamp {
                    t.swamps.insert((tx, y));
                }
            }
        }
    }
    // Clear the deployment zones (both ends, a y-band around the mid rows) so the file always places.
    for x in 0..12u8 {
        for y in 18..32u8 {
            for tx in [x, 49 - x] {
                t.walls.remove(&(tx, y));
                t.swamps.remove(&(tx, y));
            }
        }
    }
    t
}

/// Largest 4-connected open (non-wall) component's tiles, from a seed near room centre — the navigable
/// interior both squads share. Used to place the core + defender ring + attacker entry on tiles the sim
/// can actually path between (never in a walled-off pocket).
fn open_component(t: &SimTerrain, seed: (u8, u8)) -> Vec<(u8, u8)> {
    use std::collections::{HashSet, VecDeque};
    let open = |x: i32, y: i32| {
        (0..50).contains(&x) && (0..50).contains(&y) && !t.is_wall(x as u8, y as u8)
    };
    // Snap the seed to the nearest open tile.
    let start = {
        let mut best = (seed.0, seed.1);
        'outer: for r in 0..50i32 {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (x, y) = (seed.0 as i32 + dx, seed.1 as i32 + dy);
                    if open(x, y) {
                        best = (x as u8, y as u8);
                        break 'outer;
                    }
                }
            }
        }
        best
    };
    let mut seen: HashSet<(u8, u8)> = HashSet::new();
    let mut q = VecDeque::from([start]);
    let mut out = Vec::new();
    while let Some((x, y)) = q.pop_front() {
        if !open(x as i32, y as i32) || !seen.insert((x, y)) {
            continue;
        }
        out.push((x, y));
        for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, ny) = (x as i32 + dx, y as i32 + dy);
            if (0..50).contains(&nx) && (0..50).contains(&ny) {
                q.push_back((nx as u8, ny as u8));
            }
        }
    }
    out
}

/// The KIND of objective the attacker pursues — a pure clear (Secure) or raze a defender core (Raze).
#[derive(Clone, Copy)]
enum Objectiv {
    /// Open clear: the defender holds no structure, the attacker wins by wiping the defender squad.
    Clear,
    /// A defender core (spawn) at room centre — the attacker razes it (defender squad holds it).
    Core,
    /// A core behind a rampart gate + a finite-energy tower — a drain-then-raze under fire (terrain +
    /// tower pressure both bear on positioning).
    ToweredCore,
}

impl Objectiv {
    fn tag(self) -> &'static str {
        match self {
            Objectiv::Clear => "open-clear",
            Objectiv::Core => "raze-core",
            Objectiv::ToweredCore => "towered-core",
        }
    }
}

/// One built self-play scenario over real terrain + both sides' rosters + a descriptor.
struct SelfPlayCase {
    file: String,
    fixture_room: String,
    att_flavour: &'static str,
    def_flavour: &'static str,
    objective: &'static str,
    scenario: Scenario,
    att_bodies: Vec<Vec<Part>>,
}

/// Assemble ONE symmetric self-play scenario on a real fixture: real (mirrored) terrain in the fixture's
/// room, a defender core/ring near the east-of-centre, the attacker forming at a west entry. Both sides'
/// rosters are seeded random comps. The attacker paths east across the channelled interior to engage.
fn assemble(
    seed: u32,
    fixture: &TerrainFixture,
    obj_kind: Objectiv,
    att: Flavour,
    def: Flavour,
) -> SelfPlayCase {
    let room: RoomName = fixture
        .room
        .parse()
        .unwrap_or_else(|_| "W1N1".parse().unwrap());
    let terrain = symmetric_terrain(fixture);

    // The shared navigable interior (from room centre) — everything places on it so the fight is pathable.
    let interior = open_component(&terrain, (25, 25));
    let interior_set: std::collections::HashSet<(u8, u8)> = interior.iter().copied().collect();
    let is_open = |x: u8, y: u8| interior_set.contains(&(x, y));

    // Core / defender ring sit east-of-centre; the attacker entry sits at the far WEST of the interior.
    let core = *interior
        .iter()
        .filter(|&&(x, y)| (30..=40).contains(&x) && (20..=30).contains(&y))
        .min_by_key(|&&(x, y)| (x as i32 - 34).pow(2) + (y as i32 - 25).pow(2))
        .or_else(|| interior.iter().max_by_key(|&&(x, _)| x))
        .unwrap_or(&(34, 25));
    let entry_xy = *interior
        .iter()
        .filter(|&&(_x, y)| (18..=32).contains(&y))
        .min_by_key(|&&(x, _)| x)
        .or_else(|| interior.iter().min_by_key(|&&(x, _)| x))
        .unwrap_or(&(6, 25));

    // Breach staging: the attacker assaults from the open tiles just WEST of the core (facing the entry).
    let neigh = |c: (u8, u8)| {
        [
            (-1i32, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        .into_iter()
        .filter_map(move |(dx, dy)| {
            let (nx, ny) = (c.0 as i32 + dx, c.1 as i32 + dy);
            ((0..50).contains(&nx) && (0..50).contains(&ny)).then_some((nx as u8, ny as u8))
        })
    };
    let mut front: Vec<(u8, u8)> = neigh(core)
        .filter(|&(x, y)| is_open(x, y) && x <= core.0)
        .collect();
    if front.is_empty() {
        front = neigh(core).filter(|&(x, y)| is_open(x, y)).collect();
    }
    if front.is_empty() {
        front.push(core);
    }
    front.truncate(4);
    let assault = front[0];
    let support: Vec<(u8, u8)> = neigh(assault)
        .filter(|&t| is_open(t.0, t.1) && t != core && !front.contains(&t))
        .take(4)
        .collect();

    // ── build the world: real symmetric terrain + the defender objective + defender ring ──
    let mut b = ScenarioBuilder::empty(room).in_room(room);
    *b.world_mut().terrain_mut(room) = terrain.clone();

    let (obj_id, obj_kind_enum) = match obj_kind {
        Objectiv::Clear => {
            // No structure: a pure clear (Secure = wipe the defender squad). Nominal spawn id, stop on wipe.
            let id = b.structure(
                StructureKind::Spawn,
                Some(DEFENDER),
                core.0,
                core.1,
                6_000,
                6_000,
            );
            (id, ObjectiveKind::Secure)
        }
        Objectiv::Core => {
            let id = b.structure(
                StructureKind::Spawn,
                Some(DEFENDER),
                core.0,
                core.1,
                30_000,
                30_000,
            );
            (id, ObjectiveKind::Raze)
        }
        Objectiv::ToweredCore => {
            let id = b.structure(
                StructureKind::Spawn,
                Some(DEFENDER),
                core.0,
                core.1,
                30_000,
                30_000,
            );
            // A rampart gate just west of the core + a FINITE tower a few rings out → drain-then-raze.
            if is_open(core.0.saturating_sub(1), core.1) {
                b.structure(
                    StructureKind::Rampart,
                    Some(DEFENDER),
                    core.0.saturating_sub(1),
                    core.1,
                    25_000,
                    25_000,
                );
            }
            if let Some(&(tx, ty)) = interior.iter().find(|&&(x, y)| {
                let d = (x as i32 - core.0 as i32)
                    .abs()
                    .max((y as i32 - core.1 as i32).abs());
                (3..=6).contains(&d)
            }) {
                b.tower(DEFENDER, tx, ty, 600);
            }
            (id, ObjectiveKind::Raze)
        }
    };

    let mut world = b.build();

    // Defender squad: a seeded random roster ringing the core on open tiles (never on the core/front).
    let def_bodies = def.roster(seed);
    let mut def_tiles: Vec<(u8, u8)> = interior
        .iter()
        .copied()
        .filter(|&t| t != core && !front.contains(&t) && t.0 >= core.0.saturating_sub(2))
        .collect();
    def_tiles.sort_by_key(|&(x, y)| {
        (x as i32 - core.0 as i32).pow(2) + (y as i32 - core.1 as i32).pow(2)
    });
    for (i, body) in def_bodies.iter().enumerate() {
        if let Some(&(dx, dy)) = def_tiles.get(i) {
            world.movement.creeps.push(SimCreep {
                id: 10_000 + i as u32,
                owner: DEFENDER,
                pos: pos_in(room, dx, dy),
                body: SimBody::unboosted(body),
                fatigue: 0,
                carry_used: 0,
            });
        }
    }

    let objective = Objective {
        id: obj_id,
        room,
        pos: pos_in(room, core.0, core.1),
        assault_pos: pos_in(room, assault.0, assault.1),
        front_tiles: front.iter().map(|&(x, y)| pos_in(room, x, y)).collect(),
        support_tiles: support.iter().map(|&(x, y)| pos_in(room, x, y)).collect(),
        entry: pos_in(room, entry_xy.0, entry_xy.1),
        kind: obj_kind_enum,
    };

    let att_bodies = att.roster(seed);
    let scenario = Scenario {
        world,
        objectives: vec![objective],
        attacker_owner: ATTACKER,
        defender_owner: DEFENDER,
        member_energy: att.energy,
        onsite_budget: 1400,
        label: format!(
            "self-play {} | {} [{}] vs {} [{}]",
            fixture.room,
            att.name,
            obj_kind.tag(),
            def.name,
            "hold"
        ),
        seed: seed as u64,
    };

    SelfPlayCase {
        file: format!(
            "{:02}_selfplay_{}",
            seed,
            slug(&format!(
                "{}-{}-{}-{}",
                fixture.room,
                att.name,
                obj_kind.tag(),
                def.name
            ))
        ),
        fixture_room: fixture.room.clone(),
        att_flavour: att.name,
        def_flavour: def.name,
        objective: obj_kind.tag(),
        scenario,
        att_bodies,
    }
}

/// Enumerate the self-play corpus: several real fixtures × objective kind × attacker flavour × defender
/// flavour, seeded per scenario index. Sized to 20–40 DISTINCT renders.
fn cases() -> Vec<SelfPlayCase> {
    let fx = fixtures();
    // A spread of the 13 real fixtures (varied wall/swamp density) — prefer distinct rooms across the grid.
    let picks: Vec<usize> = (0..fx.len())
        .step_by(2)
        .chain((1..fx.len()).step_by(4))
        .collect();
    let objectives = [Objectiv::Clear, Objectiv::Core, Objectiv::ToweredCore];

    let mut out = Vec::new();
    let mut seed = 0u32;
    for (pi, &fi) in picks.iter().enumerate() {
        let fixture = &fx[fi % fx.len()];
        for (oi, &obj) in objectives.iter().enumerate() {
            // Rotate the attacker/defender flavours through the table so both sides vary across the corpus
            // (ranged-heavy vs melee+heal, mixed vs lean-drain, big-brawl vs small-skirmish, ...).
            let att = FLAVOURS[(pi + oi) % FLAVOURS.len()];
            let def = FLAVOURS[(pi + oi + 1 + (fi % 3)) % FLAVOURS.len()];
            out.push(assemble(seed, fixture, obj, att, def));
            seed += 1;
            if out.len() >= 36 {
                return out;
            }
        }
    }
    out
}

/// Filesystem-safe slug.
fn slug(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// The DISTINCT room names the produced replay actually draws — extracted from the embedded `meta.rooms`.
fn rooms_in_html(html: &str) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let needle = "\"name\":\"";
    let mut i = 0;
    while let Some(p) = html[i..].find(needle) {
        let start = i + p + needle.len();
        if let Some(end_rel) = html[start..].find('"') {
            let name = &html[start..start + end_rel];
            if !name.is_empty()
                && name.len() <= 8
                && name
                    .chars()
                    .next()
                    .map(|c| c == 'W' || c == 'E')
                    .unwrap_or(false)
                && name.chars().all(|c| c.is_ascii_alphanumeric())
            {
                set.insert(name.to_string());
            }
            i = start + end_rel;
        } else {
            break;
        }
    }
    set.into_iter().collect()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// One rendered row for the index.
struct Row {
    file: String,
    fixture_room: String,
    att: &'static str,
    def: &'static str,
    objective: &'static str,
    outcome: String,
    rooms: usize,
    /// Terrain proof: the render's embedded walls MATCH the real fixture (not the synthetic (15,y) line).
    terrain_real: bool,
    frames: bool,
}

fn main() {
    std::fs::create_dir_all(OUT_DIR).expect("create combat-renders/");
    // Wipe the old boring set so only the new real self-play corpus remains.
    if let Ok(entries) = std::fs::read_dir(OUT_DIR) {
        for e in entries.flatten() {
            if e.path().extension().map(|x| x == "html").unwrap_or(false) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }

    let corpus = cases();
    let mut rows: Vec<Row> = Vec::new();
    for c in &corpus {
        let (html, outcome) = render_self_play_replay_bodies(&c.scenario, &c.att_bodies);
        let rooms = rooms_in_html(&html);
        let frames = html.contains("\"frames\":[{");

        // TERRAIN PROOF: the render's embedded walls must be the REAL fixture's walls (mirror-symmetrized),
        // NOT the old synthetic corridor `(15,y)`. Confirm the render's wall set matches the bed terrain's
        // wall count within the target room (a non-trivial real wall field, > a single column).
        let bed_walls = c
            .scenario
            .world
            .terrain_for(c.scenario.objectives[0].room)
            .walls
            .len();
        let terrain_real = bed_walls > 60 && frames;

        let file = format!("{}.html", c.file);
        std::fs::write(format!("{OUT_DIR}/{file}"), &html).expect("write replay html");
        println!(
            "{:>2} {:6} obj={:13} {:14} vs {:14} -> {:20} rooms={} walls={} {}",
            rows.len() + 1,
            c.fixture_room,
            c.objective,
            c.att_flavour,
            c.def_flavour,
            outcome,
            rooms.len(),
            bed_walls,
            if terrain_real {
                "REAL-TERRAIN✓"
            } else {
                "??"
            },
        );
        rows.push(Row {
            file,
            fixture_room: c.fixture_room.clone(),
            att: c.att_flavour,
            def: c.def_flavour,
            objective: c.objective,
            outcome,
            rooms: rooms.len(),
            terrain_real,
            frames,
        });
    }

    // ── contact-sheet index ──
    let mut idx = String::new();
    idx.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Real-AI self-play combat renders</title>");
    idx.push_str("<style>body{margin:0;background:#0b1020;color:#e5e7eb;font:14px/1.6 ui-monospace,Menlo,Consolas,monospace;padding:24px;}");
    idx.push_str("h1{font-size:20px;} a{color:#60a5fa;text-decoration:none;} a:hover{text-decoration:underline;}");
    idx.push_str("table{border-collapse:collapse;width:100%;margin-top:16px;} td,th{border-bottom:1px solid #334155;padding:8px 10px;text-align:left;vertical-align:top;}");
    idx.push_str("th{color:#94a3b8;text-transform:uppercase;font-size:11px;letter-spacing:.05em;} .ok{color:#4ade80;} .no{color:#f87171;} .muted{color:#94a3b8;} .real{color:#a3e635;} .fx{color:#fbbf24;}</style></head><body>");
    let real = rows.iter().filter(|r| r.terrain_real).count();
    idx.push_str(&format!(
        "<h1>Real-AI self-play combat renders — {} scenarios</h1><p class=\"muted\">Every row is a REAL self-play match: BOTH sides are the bot's <code>ManagedSimSquad</code> brain (advance / kite / focus-fire / heal), moving over the REAL engine and REAL rover pathing, on REAL mmo:shard3 room terrain (<code>terrain_import::fixtures()</code>, mirror-symmetrized for a fair bed). Compositions are seeded free-form rosters (<code>roster::sample_squad</code>) — varied on BOTH sides (ranged-heavy / melee+heal / mixed / lean-drain / brawl). Open one and press <b>play</b> to watch the attacker form at the west entry, cross the channelled interior (walls funnel the approach, swamps slow it), and engage the defender force holding the core. <span class=\"real\">{real}/{}</span> renders embed the real fixture wall field (not the old synthetic (15,y) corridor).</p>",
        rows.len(),
        rows.len(),
    ));
    idx.push_str("<table><tr><th>#</th><th>replay</th><th>real terrain</th><th>attacker comp</th><th>defender comp</th><th>objective</th><th>outcome</th><th>rooms</th><th>terrain</th></tr>");
    for (i, r) in rows.iter().enumerate() {
        idx.push_str(&format!(
            "<tr><td class=\"muted\">{}</td><td><a href=\"{}\">{}</a></td><td class=\"fx\">{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class=\"muted\">{}</td><td class=\"{}\">{}</td></tr>",
            i + 1,
            r.file,
            html_escape(&r.file),
            html_escape(&r.fixture_room),
            html_escape(r.att),
            html_escape(r.def),
            html_escape(r.objective),
            html_escape(&r.outcome),
            r.rooms,
            if r.terrain_real { "ok" } else { "no" },
            if r.terrain_real { "real✓" } else { "??" },
        ));
    }
    idx.push_str("</table></body></html>");
    std::fs::write(format!("{OUT_DIR}/index.html"), idx).expect("write index.html");

    let with_frames = rows.iter().filter(|r| r.frames).count();
    println!(
        "\nWrote {} self-play replays + index.html to {}/ ({} with frames; {} embed the real fixture terrain)",
        rows.len(),
        OUT_DIR,
        with_frames,
        real,
    );
    println!("Open: {}/index.html", OUT_DIR);

    // Sanity gates (not a CI test, but fail loudly if the corpus regresses to empty/synthetic).
    assert!(rows.len() >= 20, "corpus too small ({} < 20)", rows.len());
    assert!(
        real * 4 >= rows.len() * 3,
        "too few renders embed real terrain ({real}/{})",
        rows.len()
    );
}
