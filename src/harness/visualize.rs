//! Stage 4 — the interactive replay player (ADR 0023a), split into separate, hand-editable resources
//! (operator-requested): a UI **shell** HTML, a **renderer** (`renderer.js` — pure drawing primitives),
//! a **player** (`player.js` — playback / scrub / interaction / the unit inspector), and per-replay
//! **frame data** (`<name>.data.js` → `window.REPLAY`). `write_replay` emits the split (the shared JS
//! once + the shell + data per replay); `replay_to_html` inlines all four into one self-contained file
//! (tests / single-file sharing). Both share the SAME `RENDERER_JS` / `PLAYER_JS` (no drift).
//!
//! The player shows a per-creep **inspector** (id, side, HP bar, body composition T/A/RA/W/C/H/M) so a
//! tanky creep is distinguishable from a real exchange, plus a tick-event log (intents + "why"). The
//! scrubber is clamp-guarded and stops playback on input (no mid-play race). Multi-room: rooms tile a
//! labeled grid; terrain + `screeps-visual` typed buildings + owner-coloured HP discs.
//!
//! Host-only (`combat-eval` is `--workspace --exclude`'d from the wasm build).

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
    pub verdict: Option<String>,
    pub rooms: Vec<RoomLayout>,
}

impl ReplayMeta {
    /// Build the per-room terrain backdrops by scanning a world for the rooms its entities occupy.
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

fn kind_to_type(kind: StructureKind) -> StructureType {
    match kind {
        StructureKind::Spawn => StructureType::Spawn,
        StructureKind::Tower => StructureType::Tower,
        StructureKind::Rampart => StructureType::Rampart,
        StructureKind::Wall => StructureType::Wall,
    }
}
fn kind_index(kind: StructureKind) -> u8 {
    match kind {
        StructureKind::Spawn => 0,
        StructureKind::Tower => 1,
        StructureKind::Rampart => 2,
        StructureKind::Wall => 3,
    }
}

/// A `VisualBackend` that serializes structure primitives to compact JSON (relative tile coords).
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
            from.0, from.1, to.0, to.1, col(color)
        ));
    }
}

fn shape_json(kind: StructureKind) -> String {
    let mut b = CollectBackend::default();
    render_structure(&mut b, 0.0, 0.0, kind_to_type(kind), 1.0);
    format!("[{}]", b.prims.join(","))
}

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

/// Build the three JSON blobs the player needs. `stride` downsamples frames (keep every Nth + the
/// last) — 1 = all frames. Creep schema: `[id, roomIdx, x, y, owner, hits, hitsMax, [T,A,RA,W,C,H,M]]`
/// (composition = alive part counts). struct: `[roomIdx,x,y,kind,owner,hits,hitsMax]`. tower:
/// `[roomIdx,x,y,owner,energy,hits]`. notes: per-frame event strings.
fn build_replay_json(rec: &CombatRecording, meta: &ReplayMeta, stride: usize) -> (String, String, String) {
    let room_idx = |name: &str| meta.rooms.iter().position(|r| r.name == name).map(|i| i as i64).unwrap_or(-1);

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

    let stride = stride.max(1);
    let n = rec.frames.len();
    let mut frames_json = String::from("[");
    let mut first = true;
    for (fi, f) in rec.frames.iter().enumerate() {
        if fi % stride != 0 && fi != n - 1 {
            continue;
        }
        if !first {
            frames_json.push(',');
        }
        first = false;
        let creeps: Vec<String> = f
            .creeps
            .iter()
            .map(|c| {
                let m = c.composition;
                format!(
                    "[{},{},{},{},{},{},{},[{},{},{},{},{},{},{}]]",
                    c.id, room_idx(&c.room.to_string()), c.x, c.y, c.owner, c.hits, c.hits_max, m[0], m[1], m[2], m[3], m[4], m[5], m[6]
                )
            })
            .collect();
        let structs: Vec<String> = f
            .structures
            .iter()
            .map(|s| {
                format!(
                    "[{},{},{},{},{},{},{}]",
                    room_idx(&s.room.to_string()), s.x, s.y, kind_index(s.kind), s.owner.map(|o| o as i64).unwrap_or(-1), s.hits, s.hits_max
                )
            })
            .collect();
        let towers: Vec<String> = f
            .towers
            .iter()
            .map(|t| format!("[{},{},{},{},{},{}]", room_idx(&t.room.to_string()), t.x, t.y, t.owner, t.energy, t.hits))
            .collect();
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
            f.tick, creeps.join(","), structs.join(","), towers.join(","), notes.join(",")
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
    (meta_json, frames_json, shapes_json)
}

/// Render a recording + metadata to a single self-contained interactive HTML file (renderer + player +
/// data inlined). Use [`write_replay`] for the split (separate `renderer.js`/`player.js`/`*.data.js`).
pub fn replay_to_html(rec: &CombatRecording, meta: &ReplayMeta) -> String {
    let (m, f, s) = build_replay_json(rec, meta, 1);
    let mut html = String::new();
    html.push_str(SHELL_HEAD);
    html.push_str(SHELL_BODY);
    let _ = write!(html, "<script>\n{RENDERER_JS}\n{PLAYER_JS}\nwindow.REPLAY={{\"meta\":{m},\"frames\":{f},\"shapes\":{s}}};\nIbexReplay.start(window.REPLAY);\n</script>\n</body>\n</html>\n");
    html
}

/// Render to a self-contained `show_widget` FRAGMENT (no doctype/html/head/body) with the frames
/// downsampled to ~`max_frames` — for an inline, scripts-run viewer. Wraps in a dark `#replay-root`.
pub fn replay_to_widget(rec: &CombatRecording, meta: &ReplayMeta, max_frames: usize) -> String {
    let stride = (rec.frames.len() / max_frames.max(1)).max(1);
    let (m, f, s) = build_replay_json(rec, meta, stride);
    let mut html = String::new();
    html.push_str(SHELL_STYLE);
    html.push_str("<div id=\"replay-root\">");
    html.push_str(SHELL_BODY);
    html.push_str("</div>");
    let _ = write!(html, "<script>\n{RENDERER_JS}\n{PLAYER_JS}\nwindow.REPLAY={{\"meta\":{m},\"frames\":{f},\"shapes\":{s}}};\nIbexReplay.start(window.REPLAY);\n</script>");
    html
}

/// Write the SPLIT replay (operator-requested): the shared `renderer.js` + `player.js` (once) + a
/// per-replay shell `<name>.html` and frame-data `<name>.data.js`. The shell `<script src>`s the three
/// — so the renderer/player are hand-editable and the frame data is isolated for introspection.
pub fn write_replay(dir: &str, name: &str, rec: &CombatRecording, meta: &ReplayMeta) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(format!("{dir}/renderer.js"), RENDERER_JS)?;
    std::fs::write(format!("{dir}/player.js"), PLAYER_JS)?;
    let (m, f, s) = build_replay_json(rec, meta, 1);
    std::fs::write(format!("{dir}/{name}.data.js"), format!("window.REPLAY={{\"meta\":{m},\"frames\":{f},\"shapes\":{s}}};\n"))?;
    let mut shell = String::new();
    shell.push_str(SHELL_HEAD);
    shell.push_str(SHELL_BODY);
    let _ = write!(
        shell,
        "<script src=\"renderer.js\"></script>\n<script src=\"player.js\"></script>\n<script src=\"{name}.data.js\"></script>\n<script>IbexReplay.start(window.REPLAY);</script>\n</body>\n</html>\n"
    );
    std::fs::write(format!("{dir}/{name}.html"), shell)?;
    Ok(())
}

const SHELL_STYLE: &str = r#"<style>
  #replay-root{background:#0b1020;color:#e5e7eb;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;border-radius:10px;padding:6px;}
  .rp-head{padding:8px 10px;border-bottom:1px solid #334155;}
  .rp-head b{font-size:15px;} .rp-muted{color:#94a3b8;}
  .rp-legend span{margin-right:12px;} .rp-dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px;}
  .rp-wrap{display:flex;gap:10px;padding:10px;align-items:flex-start;flex-wrap:wrap;}
  .rp-board canvas{background:#070b16;border:1px solid #334155;border-radius:6px;max-width:100%;height:auto;}
  .rp-side{flex:1;min-width:260px;display:flex;flex-direction:column;gap:10px;}
  .rp-panel{background:#111827;border:1px solid #334155;border-radius:6px;padding:8px;max-height:300px;overflow:auto;}
  .rp-panel h3{margin:0 0 6px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;}
  .rp-unit{display:flex;align-items:center;gap:6px;padding:2px 0;border-bottom:1px solid #1f2937;white-space:nowrap;}
  .rp-bar{display:inline-block;width:46px;height:6px;background:#1f2937;border-radius:3px;overflow:hidden;vertical-align:middle;}
  .rp-bar i{display:block;height:6px;}
  .rp-comp{color:#cbd5e1;} .rp-events div{white-space:pre-wrap;border-bottom:1px solid #1f2937;padding:2px 0;}
  .rp-ctrl{display:flex;gap:8px;align-items:center;padding:8px 10px;border-top:1px solid #334155;}
  .rp-ctrl input[type=range]{flex:1;} .rp-ctrl button{background:#1f2937;color:#e5e7eb;border:1px solid #334155;border-radius:4px;padding:4px 10px;cursor:pointer;}
</style>"#;

const SHELL_HEAD: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Combat replay</title>
<style>
  body{margin:0;background:#0b1020;color:#e5e7eb;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;}
  #replay-root{padding:0;}
  .rp-head{padding:8px 12px;border-bottom:1px solid #334155;}
  .rp-head b{font-size:15px;} .rp-muted{color:#94a3b8;}
  .rp-legend span{margin-right:12px;} .rp-dot{display:inline-block;width:10px;height:10px;border-radius:50%;vertical-align:middle;margin-right:4px;}
  .rp-wrap{display:flex;gap:12px;padding:12px;align-items:flex-start;flex-wrap:wrap;}
  .rp-board canvas{background:#070b16;border:1px solid #334155;border-radius:6px;}
  .rp-side{width:340px;display:flex;flex-direction:column;gap:10px;}
  .rp-panel{background:#111827;border:1px solid #334155;border-radius:6px;padding:8px;max-height:360px;overflow:auto;}
  .rp-panel h3{margin:0 0 6px;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;}
  .rp-unit{display:flex;align-items:center;gap:6px;padding:2px 0;border-bottom:1px solid #1f2937;white-space:nowrap;}
  .rp-bar{display:inline-block;width:46px;height:6px;background:#1f2937;border-radius:3px;overflow:hidden;vertical-align:middle;}
  .rp-bar i{display:block;height:6px;}
  .rp-comp{color:#cbd5e1;} .rp-events div{white-space:pre-wrap;border-bottom:1px solid #1f2937;padding:2px 0;}
  .rp-ctrl{display:flex;gap:8px;align-items:center;padding:8px 12px;border-top:1px solid #334155;position:sticky;bottom:0;background:#0b1020;}
  .rp-ctrl input[type=range]{flex:1;} .rp-ctrl button{background:#1f2937;color:#e5e7eb;border:1px solid #334155;border-radius:4px;padding:4px 10px;cursor:pointer;}
</style></head><body><div id="replay-root">"#;

const SHELL_BODY: &str = r#"<div class="rp-head"><b id="rp-title"></b> <span class="rp-muted" id="rp-verdict"></span>
  <div class="rp-legend"><span><i class="rp-dot" style="background:#3b82f6"></i>attacker</span><span><i class="rp-dot" style="background:#ef4444"></i>defender</span><span id="rp-tick" class="rp-muted"></span></div></div>
<div class="rp-wrap"><div class="rp-board"><canvas id="rp-cv"></canvas></div>
  <div class="rp-side"><div class="rp-panel"><h3>units (id · hp · composition)</h3><div id="rp-units"></div></div>
    <div class="rp-panel"><h3>tick events</h3><div id="rp-events" class="rp-events"></div></div></div></div>
<div class="rp-ctrl"><button id="rp-play">▶ play</button><button id="rp-step">step ▸</button>
  <input id="rp-scrub" type="range" min="0" value="0" step="1"><span id="rp-pos" class="rp-muted"></span></div>
"#;

const RENDERER_JS: &str = r#"
// Pure drawing primitives. Frame schema: creeps[id,ri,x,y,owner,hits,hmax,comp[T,A,RA,W,C,H,M]];
// structs[ri,x,y,kind,owner,hits,hmax]; towers[ri,x,y,owner,energy,hits].
window.IbexRenderer = (function(){
  const CELL=11, ROOM=50, GAP=18, PAD=18;
  // Place rooms by their TRUE world geometry, not array index: Screeps Wx INCREASES westward and Ny
  // INCREASES northward, so a name-sorted index layout mirrors the map (W2N1 drawn right of W1N1). Parse
  // WxNy/ExSy to a signed world cell — east=+col (right), west=-col (left); south=+row (down), north=
  // -row (up) — so cross-border movement reads continuously.
  function parseRoom(name){ const m=/^([WE])(\d+)([NS])(\d+)$/.exec(name||''); if(!m) return [0,0];
    const c=(m[1]==='E')?(+m[2]):(-(+m[2])-1), r=(m[3]==='S')?(+m[4]):(-(+m[4])-1); return [c,r]; }
  function layout(meta){ const cells=(meta.rooms||[]).map(function(rm){ return parseRoom(rm.name); });
    let minc=0,minr=0,maxc=0,maxr=0;
    cells.forEach(function(p,i){ if(i===0){minc=maxc=p[0];minr=maxr=p[1];} else {
      if(p[0]<minc)minc=p[0]; if(p[0]>maxc)maxc=p[0]; if(p[1]<minr)minr=p[1]; if(p[1]>maxr)maxr=p[1]; } });
    return { cells:cells, minc:minc, minr:minr, cols:(maxc-minc+1), rows:(maxr-minr+1) }; }
  function dims(meta){ const L=layout(meta), rw=ROOM*CELL;
    return { w:PAD+L.cols*(rw+GAP), h:PAD+L.rows*(rw+GAP+14), rw }; }
  function origin(meta, ri){ const L=layout(meta), rw=ROOM*CELL, p=L.cells[ri]||[L.minc,L.minr];
    const col=p[0]-L.minc, row=p[1]-L.minr;
    return [PAD+col*(rw+GAP), PAD+row*(rw+GAP+14)+14]; }
  function ownerColor(o){ return o===0?'#3b82f6':'#ef4444'; }
  function drawShape(ctx, prims, ox, oy, sx, sy, extraOp){
    for(const p of prims){ const cx=ox+(sx+0.5+(p.x||0))*CELL, cy=oy+(sy+0.5+(p.y||0))*CELL;
      ctx.globalAlpha=(p.o==null?1:p.o)*extraOp;
      if(p.t==='c'){ ctx.beginPath(); ctx.arc(cx,cy,p.r*CELL,0,7); if(p.f){ctx.fillStyle=p.f;ctx.fill();} if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.stroke();} }
      else if(p.t==='r'){ const w=p.w*CELL,h=p.h*CELL; if(p.f){ctx.fillStyle=p.f;ctx.fillRect(cx-w/2,cy-h/2,w,h);} if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.strokeRect(cx-w/2,cy-h/2,w,h);} }
      else if(p.t==='p'){ ctx.beginPath(); p.pts.forEach((pt,i)=>{ const X=ox+(sx+0.5+pt[0])*CELL, Y=oy+(sy+0.5+pt[1])*CELL; i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); }); ctx.closePath(); if(p.f){ctx.fillStyle=p.f;ctx.fill();} if(p.s){ctx.lineWidth=Math.max(1,p.sw*CELL);ctx.strokeStyle=p.s;ctx.stroke();} }
    }
    ctx.globalAlpha=1;
  }
  function draw(ctx, frame, meta, shapes){
    const d=dims(meta); ctx.clearRect(0,0,d.w,d.h);
    meta.rooms.forEach((rm,ri)=>{ const [ox,oy]=origin(meta,ri);
      ctx.fillStyle='#070b16'; ctx.fillRect(ox,oy,d.rw,d.rw);
      ctx.fillStyle='#0f2a1c'; (rm.swamps||[]).forEach(([x,y])=>ctx.fillRect(ox+x*CELL,oy+y*CELL,CELL,CELL));
      ctx.fillStyle='#475569'; (rm.walls||[]).forEach(([x,y])=>ctx.fillRect(ox+x*CELL,oy+y*CELL,CELL,CELL));
      ctx.strokeStyle='#334155'; ctx.strokeRect(ox,oy,d.rw,d.rw);
      ctx.fillStyle='#94a3b8'; ctx.font='11px monospace'; ctx.fillText(rm.name,ox,oy-3);
    });
    for(const s of frame.structs){ if(s[0]<0)continue; const [ox,oy]=origin(meta,s[0]);
      const op=s[3]===2?Math.max(0.2, s[5]/Math.max(1,s[6])):1; drawShape(ctx, shapes[s[3]], ox,oy, s[1], s[2], op); }
    for(const t of frame.towers){ if(t[0]<0)continue; const [ox,oy]=origin(meta,t[0]); drawShape(ctx, shapes[1], ox,oy, t[1], t[2], 1);
      const frac=Math.max(0,Math.min(1, t[4]/100000)); ctx.fillStyle='#22d3ee'; ctx.fillRect(ox+t[1]*CELL, oy+t[2]*CELL+CELL-2, CELL*frac, 2); }
    for(const c of frame.creeps){ const ri=c[1]; if(ri<0)continue; const [ox,oy]=origin(meta,ri);
      const hits=c[5], hmax=c[6], frac=hmax>0?hits/hmax:0;
      const base=Math.min(CELL*0.5, 2 + hmax/1400);            // bigger creep = bigger dot (tankiness cue)
      const r=Math.max(1.6, base*Math.max(0.3,frac));
      ctx.beginPath(); ctx.arc(ox+(c[2]+0.5)*CELL, oy+(c[3]+0.5)*CELL, r, 0, 7); ctx.fillStyle=ownerColor(c[4]); ctx.fill();
      ctx.lineWidth=1; ctx.strokeStyle='#0b1020'; ctx.stroke(); }
  }
  return { draw, dims, ownerColor };
})();
"#;

const PLAYER_JS: &str = r#"
// Playback + scrub + the unit inspector. Robust scrubbing: clamp the index, stop playback on input.
window.IbexReplay = (function(){
  const PART=['T','A','RA','W','C','H','M'];
  function compStr(m){ return m.map((n,i)=>n>0?PART[i]+n:'').filter(Boolean).join(' '); }
  function start(replay){
    const meta=replay.meta, frames=replay.frames||[], shapes=replay.shapes||[];
    const $=function(id){ return document.getElementById(id); };
    const cv=$('rp-cv'); if(!cv) return; const ctx=cv.getContext('2d');
    const d=IbexRenderer.dims(meta); cv.width=d.w; cv.height=d.h;
    $('rp-title').textContent=meta.title||'';
    if(meta.verdict) $('rp-verdict').textContent='— '+meta.verdict;
    const scrub=$('rp-scrub'); scrub.min=0; scrub.max=Math.max(0,frames.length-1); scrub.step=1;
    const last=frames.length?frames[frames.length-1].t:0;
    let cur=0, timer=null;
    function clamp(i){ i=i|0; if(i<0)i=0; if(i>frames.length-1)i=frames.length-1; return i; }
    function units(f){
      const rows=(f.creeps||[]).slice().sort((a,b)=> a[4]-b[4] || a[0]-b[0]).map(function(c){
        const owner=c[4], hits=c[5], hmax=c[6], comp=c[7]||[], frac=hmax>0?Math.round(100*hits/hmax):0;
        const color=IbexRenderer.ownerColor(owner);
        return '<div class="rp-unit"><i class="rp-dot" style="background:'+color+'"></i>'+
          '<span>#'+c[0]+'</span>'+
          '<span class="rp-bar"><i style="width:'+frac+'%;background:'+color+'"></i></span>'+
          '<span class="rp-muted">'+hits+'/'+hmax+'</span>'+
          '<span class="rp-comp">'+compStr(comp)+'</span></div>';
      });
      $('rp-units').innerHTML = rows.join('') || '<div class="rp-muted">(none)</div>';
    }
    function events(f){
      const nd=$('rp-events'); nd.innerHTML='';
      (f.notes||[]).forEach(function(n){ const x=document.createElement('div'); x.textContent=n; nd.appendChild(x); });
    }
    function render(){
      if(!frames.length){ ctx.fillStyle='#94a3b8'; ctx.font='13px monospace'; ctx.fillText('(empty recording)',20,20); return; }
      cur=clamp(cur); const f=frames[cur];
      IbexRenderer.draw(ctx, f, meta, shapes);
      $('rp-tick').textContent='tick '+f.t+' / '+last;
      $('rp-pos').textContent=(cur+1)+'/'+frames.length;
      if(scrub.value!=cur) scrub.value=cur;
      units(f); events(f);
    }
    function stop(){ if(timer){ clearInterval(timer); timer=null; $('rp-play').textContent='▶ play'; } }
    scrub.addEventListener('input', function(){ stop(); cur=clamp(+scrub.value); render(); });
    $('rp-step').addEventListener('click', function(){ stop(); cur=clamp(cur+1); render(); });
    $('rp-play').addEventListener('click', function(){
      if(timer){ stop(); return; }
      if(cur>=frames.length-1) cur=0;
      $('rp-play').textContent='⏸ pause';
      timer=setInterval(function(){ if(cur>=frames.length-1){ stop(); return; } cur++; render(); }, 100);
    });
    render();
  }
  return { start: start };
})();
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use screeps::{Part, Position, RoomCoordinate};
    use screeps_combat_engine::{record_tick, CombatAction, CombatWorld, Intents, SimBody, SimCreep, SimStructure, StructureKind};

    fn pos(x: u8, y: u8) -> Position {
        Position::new(RoomCoordinate::new(x).unwrap(), RoomCoordinate::new(y).unwrap(), "W1N1".parse().unwrap())
    }

    fn smoke_recording() -> (CombatRecording, CombatWorld) {
        let mut world = CombatWorld {
            creeps: vec![
                SimCreep { id: 1, owner: 0, pos: pos(23, 25), body: SimBody::unboosted(&[Part::Tough, Part::Work, Part::Move]), fatigue: 0 },
                SimCreep { id: 2, owner: 0, pos: pos(23, 24), body: SimBody::unboosted(&[Part::Heal, Part::Move]), fatigue: 0 },
            ],
            structures: vec![
                SimStructure { id: 1_000_000, kind: StructureKind::Spawn, owner: Some(1), pos: pos(25, 25), hits: 5000, hits_max: 5000 },
                SimStructure { id: 1_000_001, kind: StructureKind::Rampart, owner: Some(1), pos: pos(24, 25), hits: 3000, hits_max: 3000 },
            ],
            ..Default::default()
        };
        let mut rec = CombatRecording::new();
        for _ in 0..4 {
            let mut i = Intents::new();
            i.set(1, vec![CombatAction::Dismantle(1_000_001)]);
            record_tick(&mut rec, &mut world, &i);
        }
        (rec, world)
    }

    #[test]
    fn single_file_replay_embeds_composition_and_player() {
        let (rec, world) = smoke_recording();
        let meta = ReplayMeta::from_world(&world, "smoke", Some("test".into()));
        let html = replay_to_html(&rec, &meta);
        assert!(html.starts_with("<!doctype html>") && html.trim_end().ends_with("</html>"));
        assert!(html.contains("IbexRenderer") && html.contains("IbexReplay.start"));
        assert!(html.contains("window.REPLAY="));
        // Creep array carries the 7-slot composition (a tanky TOUGH+WORK creep).
        assert!(html.contains("[1,0,23,25,0,300,300,[1,0,0,1,0,0,1]]"), "creep id+composition embedded");
    }

    #[test]
    fn write_replay_splits_into_resources() {
        let (rec, world) = smoke_recording();
        let meta = ReplayMeta::from_world(&world, "split", None);
        let dir = std::env::temp_dir().join("ibex-replay-split-test");
        let d = dir.to_str().unwrap();
        write_replay(d, "engagement", &rec, &meta).unwrap();
        for f in ["renderer.js", "player.js", "engagement.html", "engagement.data.js"] {
            assert!(dir.join(f).exists(), "{f} written");
        }
        let shell = std::fs::read_to_string(dir.join("engagement.html")).unwrap();
        assert!(shell.contains("<script src=\"renderer.js\">") && shell.contains("<script src=\"engagement.data.js\">"));
        let data = std::fs::read_to_string(dir.join("engagement.data.js")).unwrap();
        assert!(data.starts_with("window.REPLAY="));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn widget_downsamples_frames() {
        let (rec, world) = smoke_recording();
        let meta = ReplayMeta::from_world(&world, "w", None);
        let frag = replay_to_widget(&rec, &meta, 2);
        assert!(!frag.contains("<!doctype"), "fragment has no doctype");
        assert!(frag.contains("IbexReplay.start") && frag.contains("window.REPLAY="));
    }
}
