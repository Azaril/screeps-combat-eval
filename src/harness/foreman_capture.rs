//! ADR 0025 §12 Stage 3a — **offline foreman base capture** (the reusable lib, factored so the bin +
//! tests share it instead of depending on the `screeps-foreman-bench` binary). Given a real-terrain
//! [`TerrainFixture`], run the foreman room planner (a clean host LIBRARY) OFFLINE and extract the
//! combat-relevant structure placements (spawns / towers / ramparts / walls) into a serializable
//! [`CapturedBase`]. This is the SLOW part (the planner escalates anchor beams, ~3.6–55s/room), so it
//! runs once in the `capture_base` bin → committed JSON cache; the fast [`ForemanGenerator`] only loads
//! that cache and never plans.
//!
//! [`ForemanGenerator`]: crate::harness::generate::ForemanGenerator

use crate::harness::terrain_import::{decode_fast, TerrainFixture};
use screeps_common::plan_location::PlanLocation;
use screeps_foreman::room_data::PlannerRoomDataSource;
use screeps_foreman::terrain::FastRoomTerrain;

/// A captured realistic base: real terrain + the foreman planner's combat-relevant structure placements,
/// owner-agnostic and serializable. The committed cache (`resources/captured-bases.json`) is an array of
/// these; [`ForemanGenerator`](crate::harness::generate::ForemanGenerator) realizes each into a scenario.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CapturedBase {
    pub room: String,
    /// 2500-char encoded terrain (row-major `y*50+x`; the Stage 1 decoder reads it).
    pub terrain: String,
    pub controller: (u8, u8),
    pub structures: Vec<CapturedStructure>,
}

/// One placed structure the combat sim models (foreman roads/extensions/labs/etc. are dropped).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CapturedStructure {
    /// `"spawn"` / `"tower"` / `"rampart"` / `"wall"`.
    pub kind: String,
    pub x: u8,
    pub y: u8,
}

/// A `PlannerRoomDataSource` built from a fixture (owns the terrain + object `PlanLocation`s, hands out
/// refs). The fixture's object coords are already snapped to clear tiles (the Stage-1 coordinate fix), so
/// the planner gets valid, terrain-aligned inputs.
struct FixtureDataSource {
    terrain: FastRoomTerrain,
    controllers: Vec<PlanLocation>,
    sources: Vec<PlanLocation>,
    minerals: Vec<PlanLocation>,
}

impl PlannerRoomDataSource for FixtureDataSource {
    fn get_terrain(&self) -> &FastRoomTerrain {
        &self.terrain
    }
    fn get_controllers(&self) -> &[PlanLocation] {
        &self.controllers
    }
    fn get_sources(&self) -> &[PlanLocation] {
        &self.sources
    }
    fn get_minerals(&self) -> &[PlanLocation] {
        &self.minerals
    }
}

/// Map a foreman/screeps `StructureType` to the combat-sim kind, dropping types the sim doesn't model
/// (roads / extensions / labs / links / containers / storage / terminal / …). `None` ⇒ skip.
fn combat_kind(t: screeps::StructureType) -> Option<&'static str> {
    use screeps::StructureType::*;
    match t {
        Spawn => Some("spawn"),
        Tower => Some("tower"),
        Rampart => Some("rampart"),
        Wall => Some("wall"),
        _ => None,
    }
}

/// Run the foreman planner OFFLINE on `fixture` and extract a [`CapturedBase`]. SLOW (the planner
/// escalates anchor beams). `Err` if the planner can't plan the room (skip it in the caller).
pub fn capture(fixture: &TerrainFixture) -> Result<CapturedBase, String> {
    let data_source = FixtureDataSource {
        terrain: decode_fast(&fixture.terrain),
        controllers: vec![PlanLocation::new(fixture.controller.0 as i8, fixture.controller.1 as i8)],
        sources: fixture.sources.iter().map(|&(x, y)| PlanLocation::new(x as i8, y as i8)).collect(),
        minerals: fixture.mineral.into_iter().map(|(x, y)| PlanLocation::new(x as i8, y as i8)).collect(),
    };
    let plan = screeps_foreman::planner::plan_room(&data_source).map_err(|e| format!("planning {} failed: {e}", fixture.room))?;
    let mut structures = Vec::new();
    for (location, items) in &plan.structures {
        for item in items {
            if let Some(kind) = combat_kind(item.structure_type()) {
                structures.push(CapturedStructure { kind: kind.to_string(), x: location.x(), y: location.y() });
            }
        }
    }
    Ok(CapturedBase {
        room: fixture.room.clone(),
        terrain: fixture.terrain.clone(),
        controller: fixture.controller,
        structures,
    })
}
