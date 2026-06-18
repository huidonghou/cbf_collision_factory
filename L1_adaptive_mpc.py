import torch
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sandbox_2d import DoubleIntegrator2D
from train_ncbf import NeuralCBF

class AdaptiveMPC:
    def __init__(self, env, ncbf_model, horizon=15, dt = 0.05):
        self.env = env
        self.horizon = horizon
        self.dt = dt
        self.ncbf = ncbf_model

    def simulate_trajectory_cost(self, u_sequence, current_state, target_state):
        u_sequence = u_sequence.reshape((self.horizon, 2))
        cost = 0
        state = current_state.copy()

        for i in range(self.horizon):
            force = u_sequence[i]
            # The MPC STILL ASSUMES the mass is 1.0!
            acceleration = force / 1.0
            state[0] += state[2] * self.dt 
            state[1] += state[3] * self.dt
            state[2] += acceleration[0] * self.dt
            state[3] += acceleration[1] * self.dt

            # 1. Linear Distance Penalty
            distance_penalty = np.linalg.norm([state[0] - target_state[0], state[1] - target_state[1]])
            effort_penalty = (force[0]**2 + force[1]**2) * 0.05
            
            h1 = np.linalg.norm([state[0] - 4.0, state[1] - 4.0]) - 1.5
            h2 = np.linalg.norm([state[0] - 4.0, state[1] - 9.0]) - 1.5 # New obstacle directly above
                
            cbf_penalty = 0
            if h1 < 0.8: 
                cbf_penalty += 50000 * (0.8 - h1)**3
            if h2 < 0.8:
                cbf_penalty += 50000 * (0.8 - h2)**3
            cost += distance_penalty + effort_penalty + cbf_penalty
        return cost
    
    def get_action(self, current_state, target_state):
        initial_guess = np.zeros(self.horizon * 2)
        bounds = [(-3.0, 3.0)] * (self.horizon * 2)
        
        result = minimize(
            self.simulate_trajectory_cost, 
            initial_guess, 
            args=(current_state, target_state), 
            bounds=bounds,
            method='SLSQP'
        )
        return result.x.reshape((self.horizon, 2))[0]
    
class L1AdaptiveFilter:
    """
    The L1 Adaptive Controller: Estimates unmodeled dynamics and cancels them out.
    """
    def __init__(self, dt=0.05):
        self.dt = dt
        self.v_hat = np.zeros(2) # The predictor's internal velocity
        self.d_hat = np.zeros(2) # The estimated disturbance (mass change)
        self.u_l1 = np.zeros(2)  # The corrective force
        
        # Tuning Parameters
        self.As = -10.0      # Predictor error dynamics (Hurwitz)
        self.Gamma = 200.0   # Adaptation gain (How fast it learns)
        self.cutoff = 5.0    # Low-pass filter bandwidth (Hz)
        
    def get_correction(self, current_velocity, applied_force):
        # 1. The Error: Difference between nominal physics and actual physics
        v_tilde = self.v_hat - current_velocity
        
        # 2. Adaptation Law: Rapidly estimate the disturbance
        self.d_hat += -self.Gamma * v_tilde * self.dt
        
        # 3. State Predictor: Simulating nominal 1.0kg physics
        v_hat_dot = (applied_force + self.d_hat) / 1.0 + self.As * v_tilde
        self.v_hat += v_hat_dot * self.dt
        
        # 4. Low-Pass Filter: The core of L1. Smooths the raw estimate to prevent chattering
        alpha = np.exp(-self.cutoff * self.dt)
        self.u_l1 = alpha * self.u_l1 - (1 - alpha) * self.d_hat
        
        return self.u_l1

if __name__ == "__main__":
    print("[INFO] Loading Neural CBF...")
    brain = NeuralCBF()
    brain.load_state_dict(torch.load("ncbf_weights.pth", weights_only=True))
    brain.eval()
    
    env = DoubleIntegrator2D(mass=1.0, dt=0.05)
    mpc = AdaptiveMPC(env, brain, horizon=30, dt=0.05)
    l1_filter = L1AdaptiveFilter(dt=0.05)
    
    state = env.reset(initial_state=[0.0, 0.5, 0.0, 0.0])
    target = np.array([8.0, 8.0])
    
    total_force = np.zeros(2) # Track the actual force hitting the motors
    
    print("[INFO] Starting L1 Adaptive Simulation...")
    for step in range(900):
        
        # *** THE SABOTAGE ***
        if step == 20:
            print(f"[WARNING] Step 20: System fault! Mass doubled from 1.0kg to 2.0kg!")
            env.mass = 2.0
            
        # 1. MPC computes the nominal command (Blind to the mass change)
        u_mpc = mpc.get_action(state, target)
        
        # 2. L1 computes the correction (Based on the previous step's total force)
        u_l1 = l1_filter.get_correction(current_velocity=state[2:4], applied_force=total_force)
        
        # 3. The actual force sent to the motors is the sum of both
        total_force = u_mpc + u_l1
        
        # 4. Step the physical environment
        state = env.step(total_force)
        
        dist = np.linalg.norm(state[:2] - target)
        if dist < 0.1:
            print(f"[SUCCESS] Target reached at step {step}!")
            break

    # --- Plotting ---
    history = np.array(env.history)
    plt.figure(figsize=(8, 6))
    
    circle = plt.Circle((4.0, 4.0), 1.0, color='red', alpha=0.3, label="Obstacle")
    circle2 = plt.Circle((4.0, 8.0), 1.0, color='red', alpha=0.3, label="Obstacle 2")
    plt.gca().add_patch(circle)
    plt.gca().add_patch(circle2)

    # Plot normal trajectory (0 to 20) and ADAPTED trajectory (20 onward)
    # Set this to exactly where your sabotage 'if' statement triggers
    fault_step = 20 
    
    # 1. Plot the nominal trajectory (before the mass changes)
    plt.plot(history[:fault_step, 0], history[:fault_step, 1], 
             label="Normal Trajectory (1.0kg)", color='blue', marker='.')
             
    # 2. Plot the adapted trajectory seamlessly from the fault point onward
    if len(history) > fault_step:
        plt.plot(history[fault_step-1:, 0], history[fault_step-1:, 1], 
                 label="L1 Adapted Trajectory (Sloshing Payload)", color='green', marker='x')
    plt.title("Milestone 5: NS-MPC with L1 Adaptation")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()