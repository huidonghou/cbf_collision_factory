"""
Phase 4 : NS-MPPI with batched PhysX rollouts + L1 adaptive inner loop.
 
Architecture: 
- Env 0 is the "real" robot: it receives the sabotage disturbance and the L1 correction.
- ALL env (including env 0) double as rollout workers during planning. PhysX steps
  every env together, so the sim time-shares between two modes:
 
    PLANNING MODE (every `replan_every` real steps):
      1. Snapshot env 0's joint state (the "real" state).
      2. Broadcast it to all K env  -> fork the world into K identical copies.
      3. Sample K velocity-command sequences, track them with the SAME PD law used
         at runtime, and step physics H*decimation times with render=False.
         NO disturbance, NO L1 in here: rollouts use the NOMINAL model. The
         payload mismatch is deliberately invisible to the planner -- absorbing
         it is the L1 filter's job. This preserves the 2-timescale story.
      4. MPPI exponential-weighted update of the nominal sequence (+ ESS diagnostic).
      5. Broadcast the real state again -> "restore" is just a second broadcast.
 
    EXECUTION MODE (every step):
      Integrate the current velocity knot into a setpoint, PD-track it, add L1 on
      env 0 only, subtract the phantom payload from env 0 only, step once (rendered).
 
Notes
-----
- Run headless for speed:  python robot_mppi_physx_rollouts.py --headless --num_samples 200
  Smoke-test first with:   --num_samples 32
- Written against Isaac Lab 2.x (`Isaac Lab.*` imports, matching your Phase 3 script).
  Untested in this container -- expect to touch up minor API details on your box.
- The viewport will appear to "freeze" between real frames when running with the GUI:
  that's the rollout steps rendering nothing. Expected.
"""
import argparse
import os
import time
import sys
from isaaclab.app import AppLauncher
from Write_log import SimLoggerTee
from datetime import datetime
from l1_filter import L1AdaptiveFilter
from payload_manager import PayloadManager
from batched_mppi import BatchedPhysicsMPPI, snapshot_ee_local
parser = argparse.ArgumentParser(description="Phase 4: NS-MPPI with batched PhysX rollouts + L1 adaptive inner loop.")
parser.add_argument("--num_samples", type=int, default=200, help="K = number of rollout env.")
parser.add_argument("--hold_test", action="store_true", help="hold at home, no reach, no sabotage")
parser.add_argument("--condition", choices = ["nominal","known", "unknown"], default = "unknown")
parser.add_argument("--payload", type = float, default = 1.5, help = "grasped mass added to the hand link, unit is kg")
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
from franka_constants import Franka_constants
franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
franka_cfg.init_state.pos = (0.0, 0.0, 1.0)

@configclass
class RolloutFarmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = franka_cfg

# For the constant definition, please refer to the franka_constants.py file, which documents choicses
# Note that TAU_MAX has shape (1,9) where the rest is (1,7), as the last two units have different unit
ARM_DOF = Franka_constants.ARM_DOF
FRANKA_Q_LOWER = Franka_constants.FRANKA_Q_LOWER
FRANKA_Q_UPPER = Franka_constants.FRANKA_Q_UPPER
FRANKA_DQ_LIM  = Franka_constants.FRANKA_DQ_LIM
TAU_MAX = Franka_constants.TAU_MAX

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = RolloutFarmSceneCfg(num_envs=args_cli.num_samples, env_spacing=3.0)

    for name in scene_cfg.robot.actuators.keys():
        scene_cfg.robot.actuators[name].stiffness = 0.0
        scene_cfg.robot.actuators[name].damping = 0.0
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot: Articulation = scene["robot"]
    device = sim.device
    dt = sim_cfg.dt
    ee_idx = robot.find_bodies("panda_hand")[0][0]
 
    home_q = torch.tensor([[0.0, -1.0, 0.0, -2.5, 0.0, 1.57, 0.78, 0.0, 0.0]], device=device)
    reach_q = torch.tensor([[0.0, 0.2, 0.0, -1.0, 0.0, 1.57, 0.78, 0.0, 0.0]], device=device)
    lift_q = torch.tensor([[0.6, -0.5, 0.0, -1.8, 0.0, 1.40, 0.78, 0.0, 0.0]], device=device)

    Kp = torch.tensor([[400.0, 200.0, 400.0, 150.0, 100.0, 100.0, 50.0, 10.0, 10.0]], device=device)
    Kd = torch.tensor([[40.0, 20.0, 40.0, 15.0, 10.0, 10.0, 5.0, 1.0, 1.0]], device=device)

    tau_max = torch.tensor(TAU_MAX, device=device)
    l1_budget = torch.minimum(torch.full_like(tau_max, 25.0), 0.35 * tau_max)

 
    # Task-space goals, obtained by teleport-and-read instead of analytic FK.
    p_reach = snapshot_ee_local(robot, scene, sim, reach_q, ee_idx, dt)
    p_home = snapshot_ee_local(robot, scene, sim, home_q, ee_idx, dt)
    p_lift = snapshot_ee_local(robot, scene, sim, lift_q, ee_idx, dt)
    print(f"[INFO] p_reach (env-local): {p_reach.cpu().numpy()}")
    print(f"[INFO] p_home  (env-local): {p_home.cpu().numpy()}")
    print(f"[INFO] p_lift  (env-local): {p_lift.cpu().numpy()}")

    # Start everyone at home.
    robot.write_joint_state_to_sim(
        home_q.expand(scene.num_envs, -1).contiguous(),
        torch.zeros((scene.num_envs, robot.num_joints), device=device),
    )
    scene.update(dt)
 
    mppi = BatchedPhysicsMPPI(
        robot, sim, scene, Kp, Kd, ee_idx, physics_dt=dt, device=device,
        num_samples=args_cli.num_samples, horizon=16, decimation=2, lambda_=0.9,
    )
    l1_filter = L1AdaptiveFilter(robot.num_joints, dt, device)

    # This block needs to be set here since the num_envs is determined by scene
    payload = PayloadManager(robot)
    env0_ids = torch.tensor([0],dtype = torch.long, device=device)
    all_ids = torch.arange(scene.num_envs, dtype = torch.long, device=device)
    payload_on = False
    T_GRASP, T_END = 6.0, 16.0

    ENABLE_L1_FILTER = True
    print(f"[INFO]: L1 Filter Enabled: {ENABLE_L1_FILTER}")
    print(f"Num Samples: {args_cli.num_samples}")
    # print(f"[INFO]: flag attached")
 
    # Cadence: replan every 4 real steps (40 ms of sim time) = 2 knots consumed.
    replan_every = 4
    knots_per_replan = max(1, replan_every // mppi.decimation)
 
    # IMPORTANT: sim.current_time is now polluted by rollout stepping.
    # All scheduling runs off our own clock.
    real_time = 0.0
    step_count = 0
    exec_step = 0
    # cycle_duration = 10.0
 
    q_ref_real = home_q.clone()
    commanded_torque_prev = torch.zeros((1, robot.num_joints), device=device)
    U_exec = None
    plan_ms, last_ess, last_cost = 0.0, float(args_cli.num_samples), 0.0

    while simulation_app.is_running() and real_time < T_END:
        # Phase schedule
        if args_cli.hold_test:
            p_goal, disturb_on, phase_name = p_home, False, "HOLD TEST"
        else:
        # --- Payload / Sabotage Trigger ---
        # 1. Are we in the sabotage phase? (disturb_on)
        # 2. Is the arm at the target? (ee_err < 0.15)
        # 3. Has the payload NOT been attached yet? (not payload_on)
        # Note: Condition 3 is reqired since once the first two is met, it will be the state for the next couple seconds, and 
        # this is running at 100 times per second, so we need to avoid unnecessary loading
            if (not payload_on) and real_time >= T_GRASP and args_cli.condition != "nominal":
                # Determine which environments receive the payload based on CLI args
                ids = all_ids if args_cli.condition == "known" else env0_ids
                payload.apply(ids, args_cli.payload)
                payload_on = True
                
                current_mass = payload.view.get_masses()[0, payload.hand_idx].item()
                print(f"[CHECK] env0 hand mass now: {current_mass:.3f} kg")

                # Calculate EE error directly here for the warning log
                ee_now = (robot.data.body_pos_w[0, ee_idx] - scene.env_origins[0] - p_reach).norm().item()
                grade = "" if ee_now < 0.15 else "  [WARN: grasped while %.2f m from target]" % ee_now
                print(f"[EVENT] t={real_time:.2f}s GRASP: +{args_cli.payload} kg on "
                      f"{'ALL envs' if args_cli.condition == 'known' else 'env 0'}{grade}")          # Env 0 only
                    
            if real_time < T_GRASP:
                p_goal, phase_name = p_reach, "Phase 1: Approach "
            else:
                p_goal, phase_name = p_lift, "Phase 2: Transport"

        #Now planning
        if step_count % replan_every == 0:
            q_real = robot.data.joint_pos[0:1].clone()
            dq_real = robot.data.joint_vel[0:1].clone()
            if U_exec is not None:
                mppi.shift(knots_per_replan)
            t0 = time.perf_counter()
            hidden_for_plan = False
            if payload_on and args_cli.condition == "unknown":
                payload.clear(env0_ids)  # env 0 sees the payload drop, but the planner does not
                hidden_for_plan = True
            U_exec, last_ess, last_cost = mppi.plan(
                q_real, dq_real, p_goal, finger_ref=home_q[:, ARM_DOF:]
            )
            # Put the payload back onto the real robot for execution
            if hidden_for_plan:
                payload.apply(env0_ids, args_cli.payload)
            plan_ms = (time.perf_counter() - t0) * 1e3
            exec_step = 0
            # Re-anchor the executed reference to the state the plan assumed,
            # so setpoint windup under disturbance can't accumulate across replans.
            q_ref_real = q_real.clone()
        # Now executing 
        knot = min(exec_step // mppi.decimation, mppi.H - 1)
        v_cmd = U_exec[knot].unsqueeze(0)  # (1, 7)
        q_ref_real[:, :ARM_DOF] = torch.clamp(
            q_ref_real[:, :ARM_DOF] + v_cmd * dt, mppi.q_lower, mppi.q_upper
        )
 
        q = robot.data.joint_pos
        dq = robot.data.joint_vel
        # PD for all envs toward the same reference (workers just shadow env 0;
        # they get hard-reset at the next replan anyway).
        # tau = Kp * (q_ref_real - q) - Kd * dq  
        tau_g = robot.root_physx_view.get_gravity_compensation_forces()    # (K, 9)
        tau_net = Kp * (q_ref_real - q) - Kd * dq   
        tau_c = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
        M = robot.root_physx_view.get_generalized_mass_matrices()        # (K, 9, 9)
        # --- THE ORACLE FIX ---
        if payload_on and args_cli.condition == "unknown":
            tau_g = tau_g.clone()
            tau_g[0] = tau_g[1]  # Overwrite real gravity with nominal gravity
            
            tau_c = tau_c.clone()
            tau_c[0] = tau_c[1]  # Overwrite real coriolis with nominal coriolis
            
            # Use Env 1's inertia for the L1 filter
            I_diag = torch.clamp(M[1:2].diagonal(dim1=-2, dim2=-1), min=0.05)
        else:
            # Nominal or Known conditions: it is safe to use Env 0's true inertia
            I_diag = torch.clamp(M[0:1].diagonal(dim1=-2, dim2=-1), min=0.05)

        if ENABLE_L1_FILTER:
            u_l1 = l1_filter.get_correction(current_velocity = dq[0:1], applied_torque=commanded_torque_prev, 
                                            inertia_diag=I_diag, budget=l1_budget)
        else:
            l1_filter.reset()
            u_l1 = torch.zeros((1, robot.num_joints), device=device)
 
        tau_net[0:1] += u_l1
        tau = torch.clamp(tau_net + tau_g + tau_c, -tau_max, tau_max)
        commanded_torque_prev = (tau[0:1] - tau_g[0:1] - tau_c[0:1]).clone()  # BEFORE disturbance: L1 sees intent

        robot.set_joint_effort_target(tau)
        scene.write_data_to_sim()
        sim.step()  # rendered step
        scene.update(dt)
 
        real_time += dt
        step_count += 1
        exec_step += 1
 
        if step_count % 50 == 0:
            ee_now = robot.data.body_pos_w[0, ee_idx] - scene.env_origins[0]
            ee_err = (ee_now - p_goal).norm().item()
            l1_sh = u_l1[0, 1].item()          # NEW: shoulder = joint index 1
            l1_el = u_l1[0, 3].item()
            print(
                f"t={real_time:05.1f}s | {phase_name} | EE err: {ee_err:.3f} m | "
                f"L1 sh: {l1_sh:+5.1f} | L1 elbow: {l1_el:+5.1f} N-m | "
                f"ESS: {last_ess:5.1f}/{args_cli.num_samples} | plan: {plan_ms:6.1f} ms"
            )
        
if __name__ == "__main__":
    # 1. Create a clean, timestamped filename for tracking runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    # Check the L1 flag so you can ablate it later

    l1_status = "L1_on" 
    # l1_status ="L1_off"
    filename = f"franka_v2_{args_cli.num_samples}_{args_cli.condition}_{l1_status}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, filename)

    # 2. Redirect standard outputs and errors
    logger_tee = SimLoggerTee(log_filepath)
    sys.stdout = logger_tee
    sys.stderr = logger_tee  # Also catches Python exceptions and stack traces

    print(f"[INFO] Self-contained logging activated.")
    print(f"[INFO] Writing all stdout/stderr channels to: {log_filepath}")
    print("-" * 60)
    main()
    simulation_app.close()