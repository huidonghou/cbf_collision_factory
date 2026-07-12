import torch 
# from isaaclab.assets import Articulation
# from isaaclab.sim import SimulationContext
# from isaaclab.scene import InteractiveScene
from franka_constants import Franka_constants
ARM_DOF = Franka_constants.ARM_DOF
FRANKA_Q_LOWER = Franka_constants.FRANKA_Q_LOWER
FRANKA_Q_UPPER = Franka_constants.FRANKA_Q_UPPER
FRANKA_DQ_LIM  = Franka_constants.FRANKA_DQ_LIM
TAU_MAX = Franka_constants.TAU_MAX
class BatchedPhysicsMPPI:    
    """MPPI that uses the PhysX farm itself as the rollout model.
    Sampling space: joint velocity commands for the 7 arm joints (not raw torques).
    Each sampled sequence is integrated into a setpoint trajectory and tracked by
    the same PD law used at runtime. This keeps the sample space low-variance
    (gravity is handled by physics + PD, not by the sampler) and mirrors the
    hierarchical design: the planner outputs references, the inner loop executes.
    """
    def __init__(self, robot, sim, scene, Kp, Kd, ee_idx,
                 physics_dt, device, num_samples=200, horizon=16, decimation=2,
                 lambda_=0.9):
        
        # Everything else is standard, ee_idx is the integer index of the panda_hand body
        # This is needed to know what piece of the arm to meaasure distance to origin
        self.robot, self.sim, self.scene = robot, sim, scene
        self.ee_idx = ee_idx
        self.dt = physics_dt
        self.device = device

        # PD controllers
        self.Kp, self.Kd = Kp, Kd

        # Batch size, how far do I look into future(# of decisions)
        # How long to hold each decision, and how picky when blending futures
        # TIME VOCABULARY (three clocks, do not conflate):
        #   physics step : dt = 0.01 s, one PhysX tick -- the world's clock
        #   knot         : one decision node of the plan, held for `decimation`
        #                  physics steps (0.02 s) -- the planner's clock.
        #                  U_nom is a staircase of H knots = 0.32 s of future.
        #   replan       : every `replan_every` physics steps (0.04 s = 2 knots
        #                  consumed), U_nom is shifted and rewritten -- the
        #                  thinking clock.
        self.K = num_samples
        self.H = horizon              # knots; lookahead = H * decimation * dt seconds
        self.decimation = decimation  # physics steps per knot
        

        # Those are the limits of the robotic arm positions
        self.q_lower = torch.tensor(FRANKA_Q_LOWER, device=device)
        self.q_upper = torch.tensor(FRANKA_Q_UPPER, device=device)
        self.dq_lim = torch.tensor(FRANKA_DQ_LIM, device=device)
        self.tau_max = torch.tensor(TAU_MAX, device=device)

        # MPPI hyperparameters, which is the search strategdy
        self.sigma = torch.tensor([0.6, 0.6, 0.6, 0.6, 0.8, 0.8, 0.8], device=device)
        # The lambda value(higher means less picky, seems to be working better when larger)
        self.lambda_ = lambda_

        #Now to determine the cost function
        self.w_endeff = 400.0     # Stage cost: distance to target
        self.w_term = 2000.0      # Terminal cost: final distance to target
        self.w_dq = 0.05          # Regularization: penalize high joint speeds
        self.w_u = 0.02           # Regularization: penalize aggressive command changes
        self.w_vterm = 10.0       # Terminal velocity penalty

        self.U_nom = torch.zeros((horizon, ARM_DOF), device=device) # Initialization with 0

    def shift(self, n_knots: int):
        """
        Receding Horizon, discard the "past" timed knots
        Note that currently replan happens every 4 physics step, since decimation is set to be 2
        This indicates that real robot consumed exactly 2 units of time
        """
        # while this n is not necessary, but it is just stick here, in case it gets overflown
        # Also need to set the shifted ones (after filling to be 0)
        n = min(n_knots, self.H)
        self.U_nom = torch.roll(self.U_nom, -n, dims = 0)
        self.U_nom[-n:] = 0.0

    # Now the planning step
    @torch.no_grad()
    def plan(self, q_real, dq_real, p_goal, finger_ref):
        K, H, D = self.K, self.H, ARM_DOF
        robot, sim, scene = self.robot, self.sim, self.scene

        # Broadcast the real robot's state onto all K (command input determined) environments 
        qb = q_real.expand(K, -1).contiguous()
        dqb = dq_real.expand(K, -1).contiguous()
        robot.write_joint_state_to_sim(qb, dqb)

        # Need to add the additional some noise into the robots before planning
        eps = torch.randn((K, H, D), device = self.device) * self.sigma
        eps[0] = 0.0 # No disturbance on the initial robot
        V = torch.clamp(self.U_nom.unsqueeze(0) + eps, -self.dq_lim, self.dq_lim)

        q_ref = qb.clone() # This essentially acts like the target
        q_ref[:,D:] = finger_ref # This is the essentially for finger
        
        # Initialize the score calculation
        # ee_err2 holds the calculated distance (squared) between the end-effector and the goal for all robots
        cost = torch.zeros(K, device = self.device)
        ee_err2 = torch.zeros(K, device = self.device)

        # Now we will roll toward the future with real physics
        for t in range(H):
            v_t = V[:, t] # Select the specific time for the decision
            for _ in range(self.decimation):
                q_ref[:,:D] = torch.clamp(
                    q_ref[:,:D] + v_t * self.dt, self.q_lower, self.q_upper
                )

                #Now need to measure the robot
                q, dq = robot.data.joint_pos, robot.data.joint_vel
                tau_g = robot.root_physx_view.get_gravity_compensation_forces()
                tau_c = robot.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()

                # PD Controller
                tau = torch.clamp(self.Kp * (q_ref - q) - self.Kd * dq + tau_g + tau_c, -self.tau_max,self.tau_max)
                
                # Stepping in the engine
                robot.set_joint_effort_target(tau)
                scene.write_data_to_sim()
                sim.step(render = False)
                scene.update(self.dt)
            # Stage cost at knot boundary. NOTE: body_pos_w is WORLD frame and each
            # env has its own origin -- subtract env_origins (for consistency)
            ee = robot.data.body_pos_w[:, self.ee_idx] - scene.env_origins
            ee_err2 = (ee - p_goal).square().sum(dim=1)
            dq_arm = robot.data.joint_vel[:,:D]
            cost += self.w_endeff * ee_err2 + self.w_dq * (dq_arm ** 2).sum(dim=1) + self.w_u * (v_t ** 2).sum(dim=1)
        cost += self.w_term * ee_err2 + self.w_vterm * (dq_arm ** 2).sum(dim=1)  # terminal cost, which needs to be updated as well

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
