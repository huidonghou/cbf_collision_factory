import argparse
# The ONLY Isaac Lab import allowed at the absolute top of the file
# Command to run the script: C:\IsaacLab\isaaclab.bat -p spawn_stage.py
from isaaclab.app import AppLauncher

# 1. Initialize the AppLauncher first! (Required before importing other modules)
parser = argparse.ArgumentParser(description="Phase 1: Setting up the Collision Stage.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. NOW you can safely import the rest of Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

@configclass
class CollisionArenaSceneCfg(InteractiveSceneCfg):
    """Configuration for our simple bounded collision environment."""
    
    # Add a default ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # --- DEFINE THE 4 BOUNDING WALLS ---

    # North Wall: Horizontal boundary at Y = +5m
    # North Wall
    wall_north = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallNorth",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 0.5, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 5.0, 1.0))
    )
    
    # South Wall
    wall_south = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallSouth",
        spawn=sim_utils.CuboidCfg(
            size=(10.0, 0.5, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -5.0, 1.0))
    )
    
    # East Wall
    wall_east = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallEast",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10.0, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(5.0, 0.0, 1.0))
    )
    
    # West Wall
    wall_west = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallWest",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 10.0, 2.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-5.0, 0.0, 1.0))
    )


def main():
    """Main simulation loop."""
    # Configure the global simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # Set up the scene configuration
    scene_cfg = CollisionArenaSceneCfg(num_envs=1, env_spacing=15.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    print("[INFO]: Bounded Arena Spawned Successfully. Press Play in the GUI if it is paused.")
    
    # Keep the simulation window open
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()