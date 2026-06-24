//! Stage 4 — the interactive HTML replay player (ADR 0023a). Turns a [`CombatRecording`] + scenario
//! metadata into a single self-contained HTML file (frames embedded as JSON, vanilla JS, no external
//! deps) the operator opens and scrubs to visually validate tournament outcomes + permutation variety.
//!
//! Multi-room (rooms tiled into a labeled grid; each entity drawn in its room's panel). Terrain
//! backdrop (plain/swamp/wall) + **typed buildings** reusing `screeps-visual`'s per-`StructureType`
//! primitive templates (so buildings match the bot's own renderer) — Spawn/Tower (with a draining
//! energy bar)/Rampart (shield, opacity ∝ hits)/Wall. Creeps = owner-coloured discs, radius ∝ HP. A
//! tick scrubber + play/pause/step + a per-frame data panel (intents + "why", deaths, destroyed).
//!
//! Host-only (`combat-eval` is `--workspace --exclude`'d from the wasm build) — never in live bot code.

use screeps_combat_engine::{CombatRecording, CombatWorld, StructureKind};
use screeps_visual::render::{render_structure, VisualBackend};
use screeps_visual::StructureType;
use std::collections::BTreeSet;
use std::fmt::Write;

/// One room's static backdrop (in-room wall/swamp tiles) for the replay panel.
#[derive(Clone, Debug)]
pub struct RoomLayout {
    pub name: String,
    pub walls: Vec<(u8, u8)>,
    pub swamps: Vec<(u8, u8)>,
}

/// Static metadata the player needs beyond the per-tick frames.
#[derive(Clone, Debug)]
pub struct ReplayMeta {
    pub title: String,
    /// The validator verdict (shown in the header), if any.
    pub verdict: Option<String>,
    /// The rooms (terrain backdrops); frame entities reference these by name.
    pub rooms: Vec<RoomLayout>,
}

impl ReplayMeta {
    /// Build the per-room terrain backdrops by scanning a world for the rooms its entities occupy and
    /// reading each room's terrain. (Covers every room a replay will draw into.)
    pub fn from_world(world: &CombatWorld, title: impl Into<String>, verdict: Option<String>) -> Self {
        let mut names: BTreeSet<String> = BTreeSet::new();
        for c in &world.creeps {
            names.insert(c.pos.room_name().to_string());
        }
        for s in &world.structures {
            names.insert(s.pos.room_name().to_string());
        }
        for t in &world.towers {
            names.insert(t.pos.room_name().to_string());
        }
        let rooms = names
            .into_iter()
            .map(|name| {
                let rn = name.parse().expect("a room name from a live Position parses");
                let terrain = world.terrain_for(rn);
                RoomLayout {
                    name,
                    walls: terrain.walls.iter().copied().collect(),
                    swamps: terrain.swamps.iter().copied().collect(),
                }
            })
            .collect();
        ReplayMeta { title: title.into(), verdict, rooms }
    }
}

/// Map the sim's `StructureKind` to the `screeps` `StructureType` the shape templates key off.
fn kind_to_type(kind: StructureKind) -> StructureType {
    match kind {
        StructureKind::Spawn => StructureType::Spawn,
        StructureKind::Tower => StructureType::Tower,
        StructureKind::Rampart => StructureType::Rampart,
        StructureKind::Wall => StructureType::Wall,
    }
}
/// Stable index per kind into the embedded `SHAPES` array (and the frame `kind` tag).
fn kind_index(kind: StructureKind) -> u8 {
    match kind {
        StructureKind::Spawn => 0,
        StructureKind::Tower => 1,
        StructureKind::Rampart => 2,
        StructureKind::Wall => 3,
    }
}

/// A `VisualBackend` that serializes the primitives into compact JSON objects (relative tile coords),
/// so the player's JS can instance a structure's shape at every building position.
#[derive(Default)]
struct CollectBackend {
    prims: Vec<String>,
}
fn col(c: Option<&str>) -> String {
    c.map(|s| format!("\"{s}\"")).unwrap_or_else(|| "null".into())
}
impl VisualBackend for CollectBackend {
    fn circle(&mut self, x: f32, y: f32, radius: f32, fill: Option<&str>, stroke: Option<&str>, sw: f32, op: f32) {
        self.prims
            .push(format!("{{\"t\":\"c\",\"x\":{x},\"y\":{y},\"r\":{radius},\"f\":{},\"s\":{},\"sw\":{sw},\"o\":{op}}}", col(fill), col(stroke)));
    }
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, fill: Option<&str>, stroke: Option<&str>, sw: f32, op: f32) {
        self.prims
            .push(format!("{{\"t\":\"r\",\"x\":{x},\"y\":{y},\"w\":{w},\"h\":{h},\"f\":{},\"s\":{},\"sw\":{sw},\"o\":{op}}}", col(fill), col(stroke)));
    }
    fn poly(&mut self, points: &[(f32, f32)], fill: Option<&str>, stroke: Option<&str>, sw: f32, op: f32) {
        let pts: Vec<String> = points.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
        self.prims
            .push(format!("{{\"t\":\"p\",\"pts\":[{}],\"f\":{},\"s\":{},\"sw\":{sw},\"o\":{op}}}", pts.join(","), col(fill), col(stroke)));
    }
    fn line(&mut self, from: (f32, f32), to: (f32, f32), color: Option<&str>, w: f32, op: f32) {
        self.prims.push(format!(
            "{{\"t\":\"l\",\"x1\":{},\"y1\":{},\"x2\":{},\"y2\":{},\"c\":{},\"w\":{w},\"o\":{op}}}",
            from.0,
            from.1,
            to.0,
            to.1,
            col(color)
        ));
    }
}

fn shape_json(kind: StructureKind) -> String {
    let mut b = CollectBackend::default();
    render_structure(&mut b, 0.0, 0.0, kind_to_type(kind), 1.0);
    format!("[{}]", b.prims.join(","))
}

/// Minimal JSON string escape (room names / verdicts / intent reasons).
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Render a recording + metadata to a self-contained interactive HTML replay player.
pub fn replay_to_html(rec: &CombatRecording, meta: &ReplayMeta) -> String {
    // Room name → panel index (frame entities reference rooms by this index).
    let room_idx = |name: &str| meta.rooms.iter().position(|r| r.name == name).map(|i| i as i64).unwrap_or(-1);

    // ── META JSON ──
    let mut rooms_json = String::from("[");
    for (i, r) in meta.rooms.iter().enumerate() {
        if i > 0 {
            rooms_json.push(',');
        }
        let walls: Vec<String> = r.walls.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
        let swamps: Vec<String> = r.swamps.iter().map(|(x, y)| format!("[{x},{y}]")).collect();
        let _ = write!(rooms_json, "{{\"name\":\"{}\",\"walls\":[{}],\"swamps\":[{}]}}", esc(&r.name), walls.join(","), swamps.join(","));
    }
    rooms_json.push(']');
    let verdict_json = meta.verdict.as_deref().map(|v| format!("\"{}\"", esc(v))).unwrap_or_else(|| "null".into());
    let meta_json = format!("{{\"title\":\"{}\",\"verdict\":{verdict_json},\"rooms\":{rooms_json}}}", esc(&meta.title));

    // ── FRAMES JSON (positional arrays; schema documented in the JS) ──
    let mut frames_json = String::from("[");
    for (fi, f) in rec.frames.iter().enumerate() {
        if fi > 0 {
            frames_json.push(',');
        }
        let creeps: Vec<String> = f
            .creeps
            .iter()
            .map(|c| {
                format!(
                    "[{},{},{},{},{},{},{},{}]",
                    room_idx(&c.room.to_string()),
                    c.x,
                    c.y,
                    c.owner,
                    c.hits,
                    c.hits_max,
                    c.attack_power,
                    c.ranged_power
                )
            })
            .collect();
        let structs: Vec<String> = f
            .structures
            .iter()
            .map(|s| {
                format!(
                    "[{},{},{},{},{},{},{}]",
                    room_idx(&s.room.to_string()),
                    s.x,
                    s.y,
                    kind_index(s.kind),
                    s.owner.map(|o| o as i64).unwrap_or(-1),
                    s.hits,
                    s.hits_max
                )
            })
            .collect();
        let towers: Vec<String> = f
            .towers
            .iter()
            .map(|t| format!("[{},{},{},{},{},{}]", room_idx(&t.room.to_string()), t.x, t.y, t.owner, t.energy, t.hits))
            .collect();
        // Per-frame notes for the data panel: intents (+ "why"), deaths, destroyed.
        let mut notes: Vec<String> = Vec::new();
        for ir in &f.intents {
            if ir.actions.is_empty() && ir.mv.is_none() {
                continue;
            }
            let acts = if ir.actions.is_empty() { String::new() } else { format!("{:?}", ir.actions) };
            let mv = ir.mv.map(|d| format!(" mv:{:?}", d)).unwrap_or_default();
            let why = ir.reason.as_deref().map(|r| format!("  [{r}]")).unwrap_or_default();
            notes.push(format!("\"#{} {}{}{}\"", ir.id, esc(&acts), esc(&mv), esc(&why)));
        }
        if !f.deaths.is_empty() {
            notes.push(format!("\"\\u2620 deaths: {:?}\"", f.deaths));
        }
        if !f.destroyed_structures.is_empty() {
            notes.push(format!("\"\\u2691 destroyed: {:?}\"", f.destroyed_structures));
        }
        let _ = write!(
            frames_json,
            "{{\"t\":{},\"creeps\":[{}],\"structs\":[{}],\"towers\":[{}],\"notes\":[{}]}}",
            f.tick,
            creeps.join(","),
            structs.join(","),
            towers.join(","),
            notes.join(",")
        );
    }
    frames_json.push(']');

    let shapes_json = format!(
        "[{},{},{},{}]",
        shape_json(StructureKind::Spawn),
        shape_json(StructureKind::Tower),
        shape_json(StructureKind::Rampart),
        shape_json(StructureKind::Wall)
    );

    let mut html = String::new();
    html.push_str(HTML_HEAD);
    let _ = write!(html, "<script>\nconst META={meta_json};\nconst FRAMES={frames_json};\nconst SHAPES={shapes_json};\n");
    html.push_str(PLAYER_JS);
    html.push_str("</script>\n</body>\n</html>\n");
    html
}

const HTML_HEAD: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Combat replay</title>
<style>
  :root{--bg:#0b1020;--panel:#111827;--ink:#e5e7eb;--muted:#94a3b8;--line:#334155;}
  body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;}
  header{padding:8px 12px;border-bottom:1px solid var(--line);}
  header b{font-size:15px;} header .v{color:var(--muted);}
  #wrap{display:flex;gap:12px;padding:12px;align-items:flex-start;}
  #board{flex:1;}
  #side{width:320px;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px;max-height:80vh;overflow:auto;}
  #side h3{margin:6px 0;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;}
  #notes div{white-space:pre-wrap;border-bottom:1px solid #1f2937;padding:2px 0;}
  #controls{display:flex;gap:8px;align-items:center;padding:8px 12px;border-top:1px solid var(--line);position:sticky;bottom:0;background:var(--bg);}
  #controls input[type=range]{flex:1;}
  button{background:#1f2937;color:var(--ink);border:1px solid var(--line);border-radius:4px;padding:4px 10px;cursor:pointer;}
  .legend span{margin-right:12px;} .dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px;}
  canvas{background:#070b16;border:1px solid var(--line);border-radius:6px;}
</style></head><body>
<header><b id="title"></b> <span class="v" id="verdict"></span>
  <div class="legend"><span><i class="dot" style="background:#3b82f6"></i>attacker</span><span><i class="dot" style="background:#ef4444"></i>defender</span><span id="ticklabel" class="v"></span></div>
</header>
<div id="wrap"><div id="board"><canvas id="cv"></canvas></div>
  <div id="side"><h3>tick events</h3><div id="notes"></div></div></div>
<div id="controls"><button id="play">▶ play</button><button id="step">step ▸</button>
  <input id="scrub" type="range" min="0" value="0"><span id="pos" class="v"></span></div>
"#;

const PLAYER_JS: &str = r#"
// Frame schema (positional): creeps[ri,x,y,owner,hits,hmax,atk,rng]; structs[ri,x,y,kind,owner,hits,hmax]; towers[ri,x,y,owner,energy,hits]
const CELL=11, ROOM=50, GAP=18, PAD=18, COLS=Math.min(2, Math.max(1, META.rooms.length));
const RW=ROOM*CELL, ROWS=Math.ceil(META.rooms.length/COLS);
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
cv.width=PAD+COLS*(RW+GAP); cv.height=PAD+ROWS*(RW+GAP+14);
document.getElementById('title').textContent=META.title;
if(META.verdict){document.getElementById('verdict').textContent='— '+META.verdict;}
const scrub=document.getElementById('scrub'); scrub.max=Math.max(0,FRAMES.length-1);
function roomOrigin(ri){const c=ri%COLS, r=(ri/COLS)|0; return [PAD+c*(RW+GAP), PAD+r*(RW+GAP+14)+14];}
function ownerColor(o){return o===0?'#3b82f6':'#ef4444';}
function drawShape(prims,ox,oy,sx,sy,extraOp){ // sx,sy = structure tile coords
  for(const p of prims){const cxp=ox+(sx+0.5+(p.x||0))*CELL, cyp=oy+(sy+0.5+(p.y||0))*CELL;
    ctx.globalAlpha=(p.o==null?1:p.o)*extraOp;
    if(p.t==='c'){ctx.beginPath();ctx.arc(cxp,cyp,p.r*CELL,0,7);if(p.f){ctx.fillStyle=p.f;ctx.fill();}if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.stroke();}}
    else if(p.t==='r'){const w=p.w*CELL,h=p.h*CELL;if(p.f){ctx.fillStyle=p.f;ctx.fillRect(cxp-w/2,cyp-h/2,w,h);}if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.strokeRect(cxp-w/2,cyp-h/2,w,h);}}
    else if(p.t==='p'){ctx.beginPath();p.pts.forEach((pt,i)=>{const X=ox+(sx+0.5+pt[0])*CELL,Y=oy+(sy+0.5+pt[1])*CELL;i?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});ctx.closePath();if(p.f){ctx.fillStyle=p.f;ctx.fill();}if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.stroke();}}
  }
  ctx.globalAlpha=1;
}
function draw(fi){
  const f=FRAMES[fi]; ctx.clearRect(0,0,cv.width,cv.height);
  META.rooms.forEach((rm,ri)=>{const [ox,oy]=roomOrigin(ri);
    ctx.fillStyle='#070b16'; ctx.fillRect(ox,oy,RW,RW);
    ctx.fillStyle='#0f2a1c'; rm.swamps.forEach(([x,y])=>ctx.fillRect(ox+x*CELL,oy+y*CELL,CELL,CELL));
    ctx.fillStyle='#475569'; rm.walls.forEach(([x,y])=>ctx.fillRect(ox+x*CELL,oy+y*CELL,CELL,CELL));
    ctx.strokeStyle='#334155'; ctx.strokeRect(ox,oy,RW,RW);
    ctx.fillStyle='#94a3b8'; ctx.font='11px monospace'; ctx.fillText(rm.name,ox,oy-3);
  });
  // structures (shape templates) then towers (shape + energy bar) then creeps (discs)
  for(const s of f.structs){if(s[0]<0)continue;const [ox,oy]=roomOrigin(s[0]);const op=s[3]===2?Math.max(0.2,s[5]/Math.max(1,s[6])):1;drawShape(SHAPES[s[3]],ox,oy,s[1],s[2],op);}
  for(const t of f.towers){if(t[0]<0)continue;const [ox,oy]=roomOrigin(t[0]);drawShape(SHAPES[1],ox,oy,t[1],t[2],1);
    const frac=Math.max(0,Math.min(1,t[4]/100000));ctx.fillStyle='#22d3ee';ctx.fillRect(ox+t[1]*CELL,oy+t[2]*CELL+CELL-2,CELL*frac,2);}
  for(const c of f.creeps){if(c[0]<0)continue;const [ox,oy]=roomOrigin(c[0]);const frac=c[5]>0?c[4]/c[5]:0;
    ctx.beginPath();ctx.arc(ox+(c[1]+0.5)*CELL,oy+(c[2]+0.5)*CELL,2+frac*(CELL*0.45),0,7);ctx.fillStyle=ownerColor(c[3]);ctx.fill();
    ctx.lineWidth=1;ctx.strokeStyle='#0b1020';ctx.stroke();}
  document.getElementById('ticklabel').textContent='tick '+f.t+' / '+FRAMES[FRAMES.length-1].t;
  document.getElementById('pos').textContent=(fi+1)+'/'+FRAMES.length;
  const nd=document.getElementById('notes'); nd.innerHTML=''; (f.notes||[]).forEach(n=>{const d=document.createElement('div');d.textContent=n;nd.appendChild(d);});
  scrub.value=fi;
}
let cur=0,timer=null;
scrub.oninput=()=>{cur=+scrub.value;draw(cur);stop();};
document.getElementById('step').onclick=()=>{cur=Math.min(FRAMES.length-1,cur+1);draw(cur);};
function stop(){if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ play';}}
document.getElementById('play').onclick=()=>{if(timer){stop();return;}document.getElementById('play').textContent='⏸ pause';
  timer=setInterval(()=>{if(cur>=FRAMES.length-1){stop();return;}cur++;draw(cur);},120);};
if(FRAMES.length){draw(0);}else{ctx.fillStyle='#94a3b8';ctx.fillText('(empty recording)',20,20);}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, Position, RoomCoordinate};
    use screeps_combat_engine::{record_tick, CombatAction, CombatWorld, Intents, SimBody, SimCreep, SimStructure, StructureKind};

    fn pos(x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), "W1N1".parse().unwrap())
    }

    #[test]
    fn renders_a_self_contained_html_player() {
        let mut world = CombatWorld {
            creeps: vec![
                SimCreep { id: 1, owner: 0, pos: pos(23, 25), body: SimBody::unboosted(&[Part::Work, Part::Move]), fatigue: 0 },
                SimCreep { id: 2, owner: 0, pos: pos(23, 24), body: SimBody::unboosted(&[Part::Heal, Part::Move]), fatigue: 0 },
            ],
            structures: vec![
                SimStructure { id: 1_000_000, kind: StructureKind::Spawn, owner: Some(1), pos: pos(25, 25), hits: 5000, hits_max: 5000 },
                SimStructure { id: 1_000_001, kind: StructureKind::Rampart, owner: Some(1), pos: pos(24, 25), hits: 3000, hits_max: 3000 },
            ],
            ..Default::default()
        };
        let mut rec = CombatRecording::new();
        for _ in 0..3 {
            let mut i = Intents::new();
            i.set(1, vec![CombatAction::Dismantle(1_000_001)]);
            record_tick(&mut rec, &mut world, &i);
        }
        let meta = ReplayMeta::from_world(&world, "smoke", Some("winnable + fielded → breached".into()));
        let html = replay_to_html(&rec, &meta);
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("const FRAMES=["), "frames embedded");
        assert!(html.contains("const SHAPES=[["), "structure shapes embedded");
        assert!(html.contains("\"title\":\"smoke\""));
        assert!(html.contains("W1N1"), "the room is named");
        assert!(html.trim_end().ends_with("</html>"));
        assert_eq!(html.matches("\"t\":0").count() + html.matches("\"t\":1").count() + html.matches("\"t\":2").count() >= 3, true);
    }

    #[test]
    fn empty_recording_is_still_valid_html() {
        let meta = ReplayMeta { title: "empty".into(), verdict: None, rooms: vec![] };
        let html = replay_to_html(&CombatRecording::new(), &meta);
        assert!(html.starts_with("<!doctype html>") && html.trim_end().ends_with("</html>"));
        assert!(html.contains("const FRAMES=[]"));
    }
}
