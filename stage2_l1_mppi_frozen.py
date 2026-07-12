"""
Phase 4 (Option B): NS-MPPI with batched PhysX rollouts + L1 adaptive inner loop.
 
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
import time
import sys
from isaaclab.app import AppLauncher



parser = argparse.ArgumentParser(description="Phase 4: NS-MPPI with batched PhysX rollouts + L1 adaptive inner loop.")
parser.add_argument("--num_samples", type=int, default=200, help="K = number of rollout env.")
parser.add_argument("--hold_test", action="store_true", help="hold at home, no reach, no sabotage")
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

franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
franka_cfg.init_state.pos = (0.0, 0.0, 1.0)

@configclass
class RolloutFarmSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    robot: ArticulationCfg = franka_cfg

# ========
# The 7 DOF arm constants:
# ========

ARM_DOF = 7
FRANKA_Q_LOWER = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
FRANKA_Q_UPPER = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
FRANKA_DQ_LIM  = [ 2.1750,  2.1750,  2.1750,  2.1750,  2.6100,  2.6100,  2.6100]
 
class L1AdaptiveFilterPyTorch:
    def __init__(self, num_joints, dt, device):
        self.dt = dt
        self.device = device
        
        # Track all 7 joints simultaneously on the GPU
        self.v_hat = torch.zeros((1, num_joints), device=device)
        self.d_hat = torch.zeros((1, num_joints), device=device)
        self.u_l1  = torch.zeros((1, num_joints), device=device)
        
        self.As = -30.0      
        self.Gamma = 150.0   # Fast adaptation to catch the payload drop
        self.cutoff = 15.0   # Smooth low-pass filter
        
    def get_correction(self, current_velocity, applied_torque, inertia_diag):
        v_tilde = self.v_hat - current_velocity
        # adaptation gain scaled by inertia -> same stability margin on EVERY joint
        self.d_hat += -self.Gamma * inertia_diag * v_tilde * self.dt
        alpha = math.exp(-self.cutoff * self.dt)
        self.u_l1 = alpha * self.u_l1 - (1.0 - alpha) * self.d_hat
        self.u_l1 = torch.clamp(self.u_l1, -25.0, 25.0)          # was 40 (Edit 4)
        v_hat_dot = (applied_torque + self.d_hat) / inertia_diag + self.As * v_tilde   # was / 1.0
        self.v_hat += v_hat_dot * self.dt
        return self.u_l1
    
    def reset(self):
        self.v_hat.zero_()
        self.d_hat.zero_()
        self.u_l1.zero_()

# Now we actually have the actual planner

class BatchedPhysicsMPPI:
    """MPPI that uses the PhysX farm itself as the rollout model.
    Sampling space: joint velocity commands for the 7 arm joints (not raw torques).
    Each sampled sequence is integrated into a setpoint trajectory and tracked by
    the same PD law used at runtime. This keeps the sample space low-variance
    (gravity is handled by physics + PD, not by the sampler) and mirrors the
    hierarchical design: the planner outputs references, the inner loop executes.
    """

    def __init__(self, robot: Articulation, sim, scene, Kp, Kd, ee_idx,
                 physics_dt, device, num_samples=200, horizon=16, decimation=2,
                 lambda_=0.05):
        self.robot, self.sim, self.scene = robot, sim, scene
        self.Kp, self.Kd = Kp, Kd
        self.ee_idx = ee_idx
        self.dt = physics_dt
        self.device = device

        self.K = num_samples
        self.H = horizon              # knots; lookahead = H * decimation * dt seconds
        self.decimation = decimation  # physics steps per knot
        self.lambda_ = lambda_
 
        # Those are the limits of the robotic arm positions
        self.q_lower = torch.tensor(FRANKA_Q_LOWER, device=device)
        self.q_upper = torch.tensor(FRANKA_Q_UPPER, device=device)
        self.dq_lim = torch.tensor(FRANKA_DQ_LIM, device=device)

        # Exploration noise (rad/s) per arm joint -- wrists get a bit more.
        self.sigma = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8], device=device)
 
        # The cost space terms: task-space error, joint velocity, control effort.
        self.w_endeff = 400.0     # ||p_e - p_goal||^2   (stage)
        self.w_term = 2000.0  # terminal task-space error
        self.w_dq = 0.05      # ||q_arm||^2 -> discourages singularity whipping
        self.w_u = 0.02       # ||v_diff||^2  -> discourages aggressive references
 
        self.U_nom = torch.zeros((horizon, ARM_DOF), device=device)

    def shift(self, n_knots: int):
        """ 
        Receding horizon, and discard the executed knots, replace it with zeros
        """
        n = min(n_knots, self.H)
        self.U_nom = torch.roll(self.U_nom, -n, dims = 0)
        self.U_nom[-n:] = 0.0

    @torch.no_grad()
    def plan(self, q_real, dq_real, p_goal, finger_ref):
        """q_real, dq_real: (1, 9) env-0 state. p_goal: (3,) env-LOCAL EE goal."""
        K,H,D = self.K, self.H, ARM_DOF
        robot, sim, scene = self.robot, self.sim, self.scene

        # Broadcast the real state to every env
        qb = q_real.expand(K, -1).contiguous()
        dqb = dq_real.expand(K, -1).contiguous()
        robot.write_joint_state_to_sim(qb, dqb)
        scene.update(self.dt)

        # Sample K velocity command sequences
        eps = torch.randn((K,H,D), device = self.device) * self.sigma
        eps[0] = 0.0 # env 0 is the "real" robot, no noise
        V = torch.clamp(self.U_nom.unsqueeze(0) + eps, -self.dq_lim, self.dq_lim)

        #Physics rollout under nominal dynamics
        q_ref = qb.clone()
        q_ref[:, D:] = finger_ref  # fingers pinned
        cost = torch.zeros(K, device=self.device)
        ee_err2 = torch.zeros(K, device=self.device)

        # Now we will roll toward the future with real physics
        # Each imaginary robot integrates its velocity command into a moving position target q_ref, 
        # and a PD controller chases it: K_p * (target − actual) is a spring pulling the joint toward the target, 
        # − K_d * velocity is a damper stopping it from overshooting. 
        # Note the broadcasting: K_p is (1, 9) and q is (200, 9)
        for t in range(H):
            v_t = V[:, t]
            for _ in range(self.decimation):
                # Integrate the velocity command
                q_ref[:, :D] = torch.clamp(
                    q_ref[:, :D] + v_t * self.dt, self.q_lower, self.q_upper
                )
                # Measuring the robot
                q, dq = robot.data.joint_pos, robot.data.joint_vel
                tau_g = robot.root_physx_view.get_gravity_compensation_forces()
                tau_c = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
                tau = torch.clamp(self.Kp * (q_ref - q) - self.Kd * dq + tau_g + tau_c, -87.0, 87.0)
                # tau = torch.clamp(self.Kp * (q_ref - q) - self.Kd * dq, -87.0, 87.0)
                robot.set_joint_effort_target(tau)
                scene.write_data_to_sim()
                sim.step(render = False)
                scene.update(self.dt)
            # Stage cost at knot boundary. NOTE: body_pos_w is WORLD frame and each
            # env has its own origin -- subtract env_origins or the cost is garbage.
            ee = robot.data.body_pos_w[:, self.ee_idx] - scene.env_origins
            ee_err2 = (ee - p_goal).square().sum(dim=1)
            dq_arm = robot.data.joint_vel[:,:D]
            cost += self.w_endeff * ee_err2 + self.w_dq * (dq_arm ** 2).sum(dim=1) + self.w_u * (v_t ** 2).sum(dim=1)

            # >>> DPNCBF / RBR hook <<<
            # h = self.ncbf(robot.data.joint_pos, robot.data.joint_vel, ee)
            # cost += barrier_penalty(h)
            # RBR rewiring is a state PERMUTATION between knots:
            #   idx = resample_indices(running_weights)      # (K,) ancestor ids
            #   robot.write_joint_state_to_sim(
            #       robot.data.joint_pos[idx], robot.data.joint_vel[idx])
            #   V[:, t + 1:] = V[idx, t + 1:]; cost = cost[idx] (+ log-weight bookkeeping)
        cost += self.w_term * ee_err2  # terminal cost

        #Now updating the MPPI
        beta = cost.min()
        scaled = (cost - beta) / (cost.std() + 1e-6)
        w = torch.exp(-scaled / self.lambda_)      # set self.lambda_ = 0.5
        w = w / (w.sum() + 1e-10) # normalize to sum to 1 and prevent NaN
        ess = 1.0 / (w ** 2).sum()  # effective sample size
        self.U_nom = (w.view(K, -1, 1) * V).sum(dim=0)  # weighted average of the velocity sequences

        # Restore and broadcast the real state to env 0 (the "real" robot) for execution
        robot.write_joint_state_to_sim(qb,dqb)
        scene.update(self.dt)

        return self.U_nom.clone(), ess.item(), beta.item()
    
def snapshot_ee_local (robot, scene, sim, q_pose, ee_idx, dt):
    """Teleport all envs to q_pose, take one settle step, read env-local EE position.
    Avoids writing a hand-rolled FK just to define the task-space goal."""
    K = scene.num_envs
    qb = q_pose.expand(K, -1).contiguous()
    robot.write_joint_state_to_sim(qb, torch.zeros_like(qb))
    scene.write_data_to_sim()
    sim.step(render = False)
    scene.update(dt)
    return (robot.data.body_pos_w[0, ee_idx] - scene.env_origins[0]).clone()  # env-local EE position

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
 
    Kp = torch.tensor([[400.0, 200.0, 400.0, 150.0, 100.0, 100.0, 50.0, 10.0, 10.0]], device=device)
    Kd = torch.tensor([[40.0, 20.0, 40.0, 15.0, 10.0, 10.0, 5.0, 1.0, 1.0]], device=device)
 
    # Task-space goals, obtained by teleport-and-read instead of analytic FK.
    p_reach = snapshot_ee_local(robot, scene, sim, reach_q, ee_idx, dt)
    p_home = snapshot_ee_local(robot, scene, sim, home_q, ee_idx, dt)
    print(f"[INFO] p_reach (env-local): {p_reach.cpu().numpy()}")
    print(f"[INFO] p_home  (env-local): {p_home.cpu().numpy()}")
 
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
    l1_filter = L1AdaptiveFilterPyTorch(robot.num_joints, dt, device)
    ENABLE_L1_FILTER = True
    print(f"[INFO]: L1 Filter Enabled: {ENABLE_L1_FILTER}")
    print(f"[INFO]: flag attached")
 
    # Cadence: replan every 4 real steps (40 ms of sim time) = 2 knots consumed.
    replan_every = 4
    knots_per_replan = max(1, replan_every // mppi.decimation)
 
    # IMPORTANT: sim.current_time is now polluted by rollout stepping.
    # All scheduling runs off our own clock.
    real_time = 0.0
    step_count = 0
    exec_step = 0
    cycle_duration = 10.0
 
    q_ref_real = home_q.clone()
    commanded_torque_prev = torch.zeros((1, robot.num_joints), device=device)
    U_exec = None
    plan_ms, last_ess, last_cost = 0.0, float(args_cli.num_samples), 0.0

    while simulation_app.is_running():
        # Phase schedule
        if args_cli.hold_test:
            p_goal, disturb_on, phase_name = p_home, False, "HOLD TEST"
        else:
            cycle_time = real_time % cycle_duration
            if cycle_time < 4.0:
                p_goal, disturb_on, phase_name = p_reach, False, "Phase 1: Nominal Reach "
            elif cycle_time < 7.0:
                p_goal, disturb_on, phase_name = p_reach, True, "Phase 2: SABOTAGE      "
            else:
                p_goal, disturb_on, phase_name = p_home, False, "Phase 3: Returning Home"

        #Now planning
        if step_count % replan_every == 0:
            q_real = robot.data.joint_pos[0:1].clone()
            dq_real = robot.data.joint_vel[0:1].clone()
            if U_exec is not None:
                mppi.shift(knots_per_replan)
            t0 = time.perf_counter()
            U_exec, last_ess, last_cost = mppi.plan(
                q_real, dq_real, p_goal, finger_ref=home_q[:, ARM_DOF:]
            )
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
        I_diag = torch.clamp(M[0:1].diagonal(dim1=-2, dim2=-1), min=0.05)  # (1, 9) 

        if ENABLE_L1_FILTER:
            u_l1 = l1_filter.get_correction(current_velocity = dq[0:1], applied_torque=commanded_torque_prev, 
                                            inertia_diag=I_diag)
        else:
            l1_filter.reset()
            u_l1 = torch.zeros((1, robot.num_joints), device=device)
 
        tau_net[0:1] += u_l1
        tau = torch.clamp(tau_net + tau_g + tau_c, -87.0, 87.0)
        commanded_torque_prev = (tau[0:1] - tau_g[0:1] - tau_c[0:1]).clone()  # BEFORE disturbance: L1 sees intent

        # The phantom payload exists ONLY in env 0's physics.
        if disturb_on:
            tau[0, 1] -= 20.0
            tau[0, 3] -= 15.0
 
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
    main()
    simulation_app.close()