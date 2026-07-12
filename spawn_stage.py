import argparse
import torch

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
from isaaclab_assets import FRANKA_PANDA_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
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

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    """Main simulation loop."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = CollisionArenaSceneCfg(num_envs=1, env_spacing=15.0)
    
    # Disable internal PID for raw torque control
    for actuator_name in scene_cfg.robot.actuators.keys():
        scene_cfg.robot.actuators[actuator_name].stiffness = 0.0
        scene_cfg.robot.actuators[actuator_name].damping = 0.0

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Phase 2: Cyclic Baseline PD Controller Initialized.")
    
    robot: Articulation = scene["robot"]
    
    # --- CYCLIC TARGET POSES ---
    # Home Pose (Tucked in/neutral)
    home_q = torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.78, 0.0, 0.0]], device=sim.device)
    # Reach Pose (Bent forward, holding something out)
    reach_q = torch.tensor([[0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.78, 0.0, 0.0]], device=sim.device)
    
    # --- [FIX 1] SAFE CYCLIC TARGET POSES ---
    # Home Pose: Standing upright, but leaning slightly FORWARD (-0.4) so it never falls backward
    home_q = torch.tensor([[0.0, -0.4, 0.0, -2.0, 0.0, 1.5, 0.78, 0.0, 0.0]], device=sim.device)
    
    # Reach Pose: Extended aggressively forward
    reach_q = torch.tensor([[0.0, -1.2, 0.0, -1.5, 0.0, 1.5, 0.78, 0.0, 0.0]], device=sim.device)
    
    # --- [FIX 2] HEAVY DAMPING ---
    Kp = torch.tensor([[1000.0, 1500.0, 1000.0, 800.0, 300.0, 200.0, 50.0, 50.0, 50.0]], device=sim.device)
    # Doubled the Kd on the heavy joints to stop the violent swinging and overshooting
    Kd = torch.tensor([[ 200.0,  300.0,  200.0,  150.0,   30.0,   20.0,   5.0,   5.0,   5.0]], device=sim.device)

    # State Machine Tracker
    cycle_duration = 9.0
    
    # Initialize the smoother exactly at the starting position
    smoothed_target = home_q.clone()

    while simulation_app.is_running():
        cycle_time = sim.current_time % cycle_duration
        
        # --- STATE MACHINE LOGIC ---
        if cycle_time < 3.0:
            current_target = reach_q
            disturbance = torch.zeros_like(Kp)
            phase_name = "Phase 1: Nominal Reach       "
            
        elif cycle_time < 6.0:
            current_target = reach_q
            disturbance = torch.zeros_like(Kp)
            # Pull down violently on shoulder (joint 1) and elbow (joint 3)
            # Increased slightly so the sag is visually obvious!
            disturbance[0, 1] = 50.0  
            disturbance[0, 3] = 20.0  
            phase_name = "Phase 2: SABOTAGE (Drooping) "
            
        else:
            current_target = home_q
            disturbance = torch.zeros_like(Kp)
            phase_name = "Phase 3: Returning Home      "

        # --- [FIX 3] SOFTER TARGET SMOOTHING ---
        # 0.98 makes it glide much slower, preventing physics glitches
        smoothed_target = 0.98 * smoothed_target + 0.02 * current_target

        # --- 1. READ STATE ---
        q = robot.data.joint_pos
        dq = robot.data.joint_vel
        
        # --- 2. BASELINE PD CONTROLLER ---
        position_error = smoothed_target - q
        velocity_error = 0.0 - dq
        
        # Calculate nominal torque to hold the pose
        tau = Kp * position_error + Kd * velocity_error
        
        # --- 3. APPLY DISTURBANCE ---
        total_physics_torque = tau - disturbance 
            
        # --- 4. TORQUE CLIPPING ---
        # Crucial to prevent physics engine explosions
        total_physics_torque = torch.clamp(total_physics_torque, -87.0, 87.0)

        # --- 5. WRITE COMMAND ---
        robot.set_joint_effort_target(total_physics_torque)
        
        # Debug Output
        if sim.current_time % 0.5 < 0.01:
            # Removed the absolute value so we can see if it falls backward (positive error)
            shoulder_err = (current_target[0, 1] - q[0, 1]).item()
            print(f"Time {sim.current_time:04.1f}s | {phase_name} | Shoulder Err: {shoulder_err:+.3f} rad")

        # Step the scene
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()