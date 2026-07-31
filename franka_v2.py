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
from batched_mppi import BatchedPhysicsMPPI, measure_ee_goal
parser = argparse.ArgumentParser(description="Phase 4: NS-MPPI with batched PhysX rollouts + L1 adaptive inner loop.")
parser.add_argument("--num_samples", type=int, default=200, help="K = number of rollout env.")
parser.add_argument("--hold_test", action="store_true", help="hold at home, no reach, no sabotage")
parser.add_argument("--condition", choices = ["nominal","known", "unknown"], default = "unknown")
parser.add_argument("--payload", type = float, default = 1.5, help = "grasped mass added to the hand link, unit is kg")
parser.add_argument("--show_package", action="store_true", help = "render a visual package in the gripper after grasp")
parser.add_argument("--solo_view", action="store_true", help="render only env 0; all K still simulate")
parser.add_argument("--friction", type = float, default = 0.0, help="viscous coefficient c on env 0 (N·m·s/rad); 0 = off")
parser.add_argument("--l1_off", action="store_true", help="disable the L1 correction")
parser.add_argument("--t_grasp", type = float, default = 6.0, help = "when the grasp event happens")
parser.add_argument("--t_switch", type = float, default = 6.0, help = "when we will switch the case")
parser.add_argument("--t_end", type=float, default=16.0, help="episode length (s)")
parser.add_argument("--seed", type=int, default=-1, help="RNG seed for MPPI sampling; -1 = random")
parser.add_argument("--w_posture", type=float, default=0.0, help="posture regularization weight; 0 = off")
parser.add_argument("--log_rollouts", action="store_true", help="record per-replan rollout EE paths + weights")
parser.add_argument("--pin_nominal", action="store_true", help="sample 0 carries the zero-noise nominal sequence")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, Articulation, RigidObject, RigidObjectCfg
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
    package = RigidObjectCfg(
     prim_path="{ENV_REGEX_NS}/Package",
        spawn=sim_utils.CuboidCfg(
        size=(0.06, 0.06, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15)),
        # deliberately NO collision_props -- see warning below
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),  # parked underground until grasp
)


# For the constant definition, please refer to the franka_constants.py file, which documents choicses
# Note that TAU_MAX has shape (1,9) where the rest is (1,7), as the last two units have different unit
ARM_DOF = Franka_constants.ARM_DOF
FRANKA_Q_LOWER = Franka_constants.FRANKA_Q_LOWER
FRANKA_Q_UPPER = Franka_constants.FRANKA_Q_UPPER
FRANKA_DQ_LIM  = Franka_constants.FRANKA_DQ_LIM
TAU_MAX = Franka_constants.TAU_MAX

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01,
                                      physx = sim_utils.PhysxCfg(
                                        enable_external_forces_every_iteration = True,
                                        min_velocity_iteration_count = 1

                                      ))
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = RolloutFarmSceneCfg(num_envs=args_cli.num_samples, env_spacing=3.0)
    
    # Franka in Isaac Lab came with its own attributes, so to be accurate and see only our approach, need to set the zero
    for name in scene_cfg.robot.actuators.keys():
        scene_cfg.robot.actuators[name].stiffness = 0.0
        scene_cfg.robot.actuators[name].damping = 0.0
    scene = InteractiveScene(scene_cfg)
    package = scene["package"]
    sim.reset()
    if args_cli.solo_view and not args_cli.headless:
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        for i in range(1, scene.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{i}")
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        origin0 = scene.env_origins[0].cpu().numpy()
        sim.set_camera_view(eye=(origin0 + [2.2, 2.2, 2.4]).tolist(),
                            target=(origin0 + [0.0, 0.0, 1.2]).tolist())
    robot: Articulation = scene["robot"]
    device = sim.device
    dt = sim_cfg.dt
    ee_idx = robot.find_bodies("panda_hand")[0][0]
 
    # This is roughly the "distance" relative to the fixed/default position
    # Angles in rad from the URDF zero posture (all-zeros = arm vertical); fingers in m.
    # idx: 0 base-yaw | 1 shoulder | 2 arm-roll | 3 elbow (limits exclude 0!) | 4 forearm-roll | 5 wrist-pitch | 6 wrist-roll
    home_q = torch.tensor([[0.0, -1.0, 0.0, -2.5, 0.0, 1.57, 0.78, 0.0, 0.0]], device=device) # folded rest posture; start state and return target
    reach_q = torch.tensor([[0.0, 0.2, 0.0, -1.0, 0.0, 1.57, 0.78, 0.0, 0.0]], device=device) # extended pre-grasp posture; defines the Phase-1 goal
    lift_q = torch.tensor([[0.6, -0.5, 0.0, -1.8, 0.0, 1.40, 0.78, 0.0, 0.0]], device=device) # raised transport posture; defines the Phase-2 goal

    # Per-joint gain vectors
    Kp = torch.tensor([[400.0, 200.0, 400.0, 150.0, 100.0, 100.0, 50.0, 10.0, 10.0]], device=device)
    Kd = torch.tensor([[40.0, 20.0, 40.0, 15.0, 10.0, 10.0, 5.0, 1.0, 1.0]], device=device)

    tau_max = torch.tensor(TAU_MAX, device=device)
    l1_budget = torch.minimum(torch.full_like(tau_max, 25.0), 0.35 * tau_max)

 
    # Task-space goals, obtained by teleport-and-read instead of analytic forward kinematics.
    # Engine simulator decides the forward kinematics, and error calculation is consistent, avoid additional introduced bias
    # For each hand-written joint configuration, teleport the robot there, ask PhysX where the hand landed, 
    # store that 3-vector as the task-space goal; repeat for home/reach/lift; then place everyone at home and start.
    p_reach = measure_ee_goal(robot, scene, sim, reach_q, ee_idx, dt)
    p_home = measure_ee_goal(robot, scene, sim, home_q, ee_idx, dt)
    p_lift = measure_ee_goal(robot, scene, sim, lift_q, ee_idx, dt)

    print(f"[INFO] Plant B (ext_forces=True, min_vel_iter=1) | seed={args_cli.seed} | w_posture={args_cli.w_posture}")
    print(f"[INFO] p_reach (env-local): {p_reach.cpu().numpy()}")
    print(f"[INFO] p_home  (env-local): {p_home.cpu().numpy()}")
    print(f"[INFO] p_lift  (env-local): {p_lift.cpu().numpy()}")
    print(f"[INFO] Viscous friction on env 0: c={args_cli.friction}")


    # Start everyone at home.
    robot.write_joint_state_to_sim(
        home_q.expand(scene.num_envs, -1).contiguous(),
        torch.zeros((scene.num_envs, robot.num_joints), device=device),
    )
    scene.update(dt)

    if args_cli.seed >= 0:
        torch.manual_seed(args_cli.seed) # This was to ensure that the noise sequence is the same
        # Seeding gives us paired comparisons — same noise across configurations, 
        # so measured differences are the configuration and not sampling luck — and sweeping three seeds per cell ensures no result rests on a single draw. 

    mppi = BatchedPhysicsMPPI(
        robot, sim, scene, Kp, Kd, ee_idx, physics_dt=dt, device=device,
        num_samples=args_cli.num_samples, horizon=16, decimation=2, lambda_=0.9,
        w_posture=args_cli.w_posture,  q_rest=reach_q[0, :ARM_DOF], pin_nominal= args_cli.pin_nominal
    )
    l1_filter = L1AdaptiveFilter(robot.num_joints, dt, device)

    # This block needs to be set here since the num_envs is determined by scene
    payload = PayloadManager(robot)
    env0_ids = torch.tensor([0],dtype = torch.long, device=device)
    all_ids = torch.arange(scene.num_envs, dtype = torch.long, device=device)
    payload_on = False
    t_grasp, t_switch, T_END = args_cli.t_grasp, args_cli.t_switch, args_cli.t_end # This is the new change per the conversation on making sure one change at a time
    print(f"[INFO]: Event times: t_grasp = {t_grasp}, t_switch: {t_switch}, T_END:{T_END}")
    ENABLE_L1_FILTER = not args_cli.l1_off
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
    dq_wmax = torch.zeros(2, device=device); dq_wsum = torch.zeros(2, device=device)
    wn = 0

    q_lower_t = torch.tensor(FRANKA_Q_LOWER, device=device)
    q_upper_t = torch.tensor(FRANKA_Q_UPPER, device=device)
    rollout_bufs, rollout_ws, exec_path, replan_steps = [], [], [], []

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

            if (not payload_on) and real_time >= t_grasp and args_cli.condition != "nominal":
                # Determine which environments receive the payload based on CLI args
                # Note on the condition, nominal serves as the base test, mimicing behavior of picking things up
                # Known means that all K robots know the weight it grasped, and will take account into consideration: imagination matches reality
                ids = all_ids if args_cli.condition == "known" else env0_ids
                payload.apply(ids, args_cli.payload)
                payload_on = True
                
                current_mass = payload.view.get_masses()[0, payload.hand_idx].item()
                print(f"[CHECK] env0 hand mass now: {current_mass:.3f} kg")

                # Calculate EE error directly here for the warning log
                ee_dist_at_grasp = (robot.data.body_pos_w[0, ee_idx] - scene.env_origins[0] - p_reach).norm().item()
                grade = "" if ee_dist_at_grasp < 0.15 else "  [WARN: grasped while %.2f m from target]" % ee_dist_at_grasp
                print(f"[EVENT] t={real_time:.2f}s GRASP: +{args_cli.payload} kg on "
                      f"{'ALL envs' if args_cli.condition == 'known' else 'env 0'}{grade}")          # Env 0 only
                    
            if real_time < t_switch:
                p_goal, phase_name = p_reach, "Phase 1: Approach "
            else:
                p_goal, phase_name = p_lift, "Phase 2: Transport"

        #Now planning
        if step_count % replan_every == 0:
            q_real = robot.data.joint_pos[0:1].clone()

            # The real robot data we care about,  env 0 
            dq_real = robot.data.joint_vel[0:1].clone()
            if U_exec is not None:
                mppi.shift(knots_per_replan)
            t0 = time.perf_counter()
            hidden_for_plan = False

            # The unknown one is really the test part: only the real env0 
            # The key question we are asking: if the reality(unknown) is different from the physics, can the L1 filter, observing only robot 0's velocity
            # adapts the difference
            if payload_on and args_cli.condition == "unknown":
                payload.clear(env0_ids)  # the planner's rollouts (including env 0's own worker-self) see the nominal mass
                hidden_for_plan = True
            U_exec, last_ess, ro_buf, ro_w = mppi.plan(
                q_real, dq_real, p_goal, finger_ref=home_q[:, ARM_DOF:]
            )
            if args_cli.log_rollouts:
                rollout_bufs.append(ro_buf.cpu())
                rollout_ws.append(ro_w.cpu())
                replan_steps.append(step_count)
           
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

        if args_cli.show_package and payload_on:
            from isaaclab.utils.math import quat_apply
            hand_pos  = robot.data.body_pos_w[:, ee_idx]          # (K, 3), world frame
            hand_quat = robot.data.body_quat_w[:, ee_idx]         # (K, 4)
            grip_offset = torch.tensor([0.0, 0.0, 0.09], device=device).expand(hand_pos.shape[0], -1)
            pkg_pos = hand_pos + quat_apply(hand_quat, grip_offset)   # 9 cm along hand z = between the fingertips
            package.write_root_pose_to_sim(torch.cat([pkg_pos, hand_quat], dim=-1))
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

        if args_cli.friction > 0.0:                          # add: --friction, type=float, default=0.0
            tau[0, :7] -= args_cli.friction * dq[0, :7]      # env 0 only, after save line
        robot.set_joint_effort_target(tau)
        scene.write_data_to_sim()
        sim.step()  # rendered step
        scene.update(dt)

        d_w = dq[0, [1, 3]].abs()
        dq_wmax = torch.maximum(dq_wmax, d_w); dq_wsum += dq[0, [1, 3]]; wn += 1
 
        real_time += dt
        step_count += 1
        exec_step += 1
        ee_now = robot.data.body_pos_w[0, ee_idx] - scene.env_origins[0]
        ee_err = (ee_now - p_goal).norm().item()
        if args_cli.log_rollouts:
            exec_path.append(ee_now.cpu())
        if step_count % 50 == 0:
            
            l1_sh = u_l1[0, 1].item()          # NEW: shoulder = joint index 1
            l1_el = u_l1[0, 3].item()
            dq_sh = dq[0, 1].item()
            dq_el = dq[0, 3].item()
            q_arm = robot.data.joint_pos[0,: 7]
            margin_lo = q_arm - q_lower_t
            margin_hi = q_upper_t - q_arm
            margins = torch.minimum(margin_lo, margin_hi)
            j_tight = margins.argmin().item()
            
            print(
                f"t={real_time:05.1f}s | {phase_name} | EE err: {ee_err:.3f} m | "
                f"L1 sh: {l1_sh:+5.1f} | L1 elbow: {l1_el:+5.1f} N-m | "
                f"dq sh mx/mn: {dq_wmax[0]:.2f}/{dq_wsum[0]/wn:+.2f} | dq el mx/mn: {dq_wmax[1]:.2f}/{dq_wsum[1]/wn:+.2f} | "
                f"ESS: {last_ess:5.1f}/{args_cli.num_samples} | plan: {plan_ms:6.1f} ms |"
                f"EE: ({ee_now[0]:+.2f},{ee_now[1]:+.2f},{ee_now[2]:+.2f}) | "
                f"tightest: j{j_tight} {margins[j_tight]:.2f} rad | "
            )
            dq_wmax.zero_()
            dq_wsum.zero_()
            wn = 0

    if args_cli.log_rollouts and rollout_bufs:
        npz_path = os.path.splitext(log_filepath)[0] + ".npz"
        np.savez(
            npz_path,
            rollouts=torch.stack(rollout_bufs).numpy(),   # (R, H, K, 3) EE, env-origin-corrected
            weights=torch.stack(rollout_ws).numpy(),      # (R, K) normalized; sample 0 = pinned nominal if --pin_nominal
            exec_path=torch.stack(exec_path).numpy(),     # (S, 3) same frame, every control step
            replan_steps=np.array(replan_steps),          # (R,) step_count at each plan() call
        )
        print(f"[INFO] Rollout recording saved: {npz_path} "
              f"({len(rollout_bufs)} replans, {len(exec_path)} exec samples)")
        
if __name__ == "__main__":
    # 1. Create a clean, timestamped filename for tracking runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "./logs/trajectory_rollout"
    os.makedirs(log_dir, exist_ok=True)
    # Check the L1 flag so you can ablate it later

    l1_status = "L1_off" if args_cli.l1_off else "L1_on"
    fric_tag = f"_fric{args_cli.friction:g}" if args_cli.friction > 0 else ""
    seed_tag = f"_s{args_cli.seed}" if args_cli.seed >= 0 else ""
    wp_tag = f"_wp{args_cli.w_posture:g}" if args_cli.w_posture > 0 else ""
    pin_tag = "_pin" if args_cli.pin_nominal else ""
    filename = f"franka_v2_{args_cli.num_samples}_{args_cli.condition}_{l1_status}{fric_tag}_pB{seed_tag}{wp_tag}_{timestamp}_tg{args_cli.t_grasp:g}_ts{args_cli.t_switch:g}_te{args_cli.t_end:g}_pin_t{pin_tag}.log"    
    log_filepath = os.path.join(log_dir, filename)

    # 2. Redirect standard outputs and errors
    logger_tee = SimLoggerTee(log_filepath)
    sys.stdout = logger_tee
    sys.stderr = logger_tee  # Also catches Python exceptions and stack traces

    print(f"[INFO] Self-contained logging activated.")
    print(f"[INFO] Writing all stdout/stderr channels to: {log_filepath}")
    print("-" * 60)
    main()
    # simulation_app.close()
     # --- 1. Secure the log FIRST: after this point the file is safe ---
    sys.stdout.flush(); sys.stderr.flush()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    for m in ("close", "flush"):
        if hasattr(logger_tee, m):
            getattr(logger_tee, m)()
            break
    # --- 2. Watchdog: guarantees process death even if close() blocks ---
    import threading
    threading.Timer(10.0, lambda: os._exit(0)).start()
    # --- 3. Attempt polite shutdown; hang or crash here no longer matters ---
    try:
        simulation_app.close()
    except Exception:
        pass
    os._exit(0)