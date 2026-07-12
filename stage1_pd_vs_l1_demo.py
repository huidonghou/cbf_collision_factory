import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 3: The L1-Adaptive Inner Loop.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG 

# --- [FIX 1] ELEVATE THE ROBOT ---
# Spawn the robot 1.0 meters in the air so it doesn't smash into the floor!
franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
franka_cfg.init_state.pos = (0.0, 0.0, 1.0) 

@configclass
class CollisionArenaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )
    robot: ArticulationCfg = franka_cfg

# --- [NEW] THE 9D L1 ADAPTIVE SPINAL CORD ---
class L1AdaptiveFilterPyTorch:
    def __init__(self, num_joints, dt, device):
        self.dt = dt
        self.device = device
        
        # Track all 9 joints simultaneously on the GPU
        self.v_hat = torch.zeros((1, num_joints), device=device)
        self.d_hat = torch.zeros((1, num_joints), device=device)
        self.u_l1  = torch.zeros((1, num_joints), device=device)
        
        self.As = -10.0      
        self.Gamma = 500.0   # Fast adaptation to catch the payload drop
        self.cutoff = 15.0   # Smooth low-pass filter
        
    def get_correction(self, current_velocity, applied_torque):
        v_tilde = self.v_hat - current_velocity
        self.d_hat += -self.Gamma * v_tilde * self.dt
        
        alpha = math.exp(-self.cutoff * self.dt)
        self.u_l1 = alpha * self.u_l1 - (1.0 - alpha) * self.d_hat
        
        # Clamp to prevent physics explosions
        self.u_l1 = torch.clamp(self.u_l1, -40.0, 40.0)
        
        # Predictor assumes nominal physics. Unmodeled mass is captured in d_hat!
        v_hat_dot = (applied_torque + self.d_hat) / 1.0 + self.As * v_tilde
        self.v_hat += v_hat_dot * self.dt
        
        return self.u_l1


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = CollisionArenaSceneCfg(num_envs=1, env_spacing=15.0)
    
    for actuator_name in scene_cfg.robot.actuators.keys():
        scene_cfg.robot.actuators[actuator_name].stiffness = 0.0
        scene_cfg.robot.actuators[actuator_name].damping = 0.0

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # ========================================================
    # THE MAGIC SWITCH: Change this to True to cure the droop!
    ENABLE_L1_FILTER = False 
    # ========================================================
    
    print(f"[INFO]: L1 Filter Enabled: {ENABLE_L1_FILTER}")
    
    robot: Articulation = scene["robot"]
    l1_filter = L1AdaptiveFilterPyTorch(num_joints=robot.num_joints, dt=sim_cfg.dt, device=sim.device)
    
    home_q = torch.tensor([[0.0, -1.0, 0.0, -2.5, 0.0, 1.57, 0.78, 0.0, 0.0]], device=sim.device)
    reach_q = torch.tensor([[0.0, 0.2, 0.0, -1.0, 0.0, 1.57, 0.78, 0.0, 0.0]], device=sim.device)
    robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q))
    
    Kp = torch.tensor([[400.0, 200.0, 400.0, 150.0, 100.0, 100.0, 50.0, 10.0, 10.0]], device=sim.device)
    Kd = torch.tensor([[ 40.0,  20.0,  40.0,  15.0,  10.0,  10.0,  5.0,   1.0,  1.0]], device=sim.device)

    cycle_duration = 6.0  
    smoothed_target = home_q.clone()
    commanded_torque = torch.zeros((1, robot.num_joints), device=sim.device)

    while simulation_app.is_running():
        cycle_time = sim.current_time % cycle_duration
        
        if cycle_time < 2.0:
            current_target = reach_q
            disturbance = torch.zeros_like(Kp)
            phase_name = "Phase 1: Nominal Reach "
            
        elif cycle_time < 4.0:
            current_target = reach_q
            disturbance = torch.zeros_like(Kp)
            # The Sabotage! Pulls down violently on shoulder and elbow
            disturbance[0, 1] = 40.0  
            disturbance[0, 3] = 15.0  
            phase_name = "Phase 2: SABOTAGE      "
            
        else:
            current_target = home_q
            disturbance = torch.zeros_like(Kp)
            phase_name = "Phase 3: Returning Home"

        smoothed_target = 0.90 * smoothed_target + 0.10 * current_target

        q = robot.data.joint_pos
        dq = robot.data.joint_vel
        
        # --- 1. BASELINE PD CONTROLLER ---
        position_error = smoothed_target - q
        velocity_error = 0.0 - dq
        tau_pd = Kp * position_error + Kd * velocity_error
        
        # --- 2. L1 ADAPTIVE INNER LOOP ---
        # This is for the ENABLE filter set to be false
        #if ENABLE_L1_FILTER and cycle_time >= 2.0 and cycle_time < 4.0:
        
        # This is for the ENABLE_L1_FILTER set to be True
        if ENABLE_L1_FILTER:
            # Filter looks at true velocity vs the torque we commanded LAST step
            u_l1 = l1_filter.get_correction(current_velocity=dq, applied_torque=commanded_torque)
        
        else:
            # During normal movement, reset filter memory
            l1_filter.v_hat.zero_()
            l1_filter.d_hat.zero_()
            l1_filter.u_l1.zero_()
            u_l1 = torch.zeros_like(tau_pd)
            
        # --- 3. SYNTHESIS ---
        # The controller INTENDS to send the nominal torque + L1 compensation
        commanded_torque = tau_pd + u_l1
        commanded_torque = torch.clamp(commanded_torque, -87.0, 87.0)

        # The physics engine receives the command MINUS the unmodeled payload disturbance
        total_physics_torque = commanded_torque - disturbance 
        robot.set_joint_effort_target(total_physics_torque)
        
        if sim.current_time % 0.5 < 0.01:
            el_err = (current_target[0, 3] - q[0, 3]).item()
            # Print the L1 compensation effort on the elbow
            l1_el_effort = u_l1[0, 3].item()
            print(f"Time {sim.current_time:04.1f}s | {phase_name} | Elbow Err: {el_err:+.3f} rad | L1 Torque: {l1_el_effort:+.1f} N-m")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

if __name__ == "__main__":
    main()
    simulation_app.close()