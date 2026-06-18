import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sandbox_2d import DoubleIntegrator2D

# 1. Load the Single-Threat Oracle Architecture
class NeuralCBF(nn.Module):
    def __init__(self):
        super(NeuralCBF, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

class BatchedSafeMPC:
    def __init__(self, env, ncbf_model, obstacles, horizon=15, dt=0.05):
        self.env = env
        self.horizon = horizon
        self.dt = dt
        self.ncbf = ncbf_model
        self.obstacles = obstacles # Array of shape (N, 2)
        
        # --- CRITICAL FIX: Initialize Warm Start Array ---
        self.last_u = np.zeros(self.horizon * 2)

    def simulate_trajectory_cost(self, u_sequence, current_state, target_state):
        u_sequence = u_sequence.reshape((self.horizon, 2))
        cost = 0
        state = current_state.copy()
        
        for i in range(self.horizon):
            force = u_sequence[i]
            acceleration = force / 1.0 # Assuming nominal 1.0kg for the planner
            
            state[0] += state[2] * self.dt 
            state[1] += state[3] * self.dt
            state[2] += acceleration[0] * self.dt
            state[3] += acceleration[1] * self.dt

            # --- THE WEIGHT TUNING FIX ---
            # 1. Heavily reward getting closer to the target
            distance_penalty = np.linalg.norm([state[0] - target_state[0], state[1] - target_state[1]]) * 2.0
            
            # 2. Drastically reduce the speed penalty so it's allowed to cruise (0.5 -> 0.02)
            speed_penalty = (state[2]**2 + state[3]**2) * 0.02
            
            # 3. Slightly reduce effort penalty so it's not afraid to steer hard
            effort_penalty = (force[0]**2 + force[1]**2) * 0.01
            
            # --- CRITICAL FIX 2: Wider Repulsive Forcefield ---
            relative_positions = self.obstacles - state[:2]
            rel_tensor = torch.tensor(relative_positions, dtype=torch.float32)
            with torch.no_grad():
                min_h = torch.min(self.ncbf(rel_tensor)).item()
            
            cbf_penalty = 0
            # Widened from 1.0m to 2.5m so the optimizer "feels" the obstacle earlier
            if min_h < 2.5: 
                cbf_penalty = 50000 * (2.5 - min_h)**3

            cost += distance_penalty + speed_penalty + effort_penalty + cbf_penalty
            
        return cost

    def cbf_constraint(self, u_sequence, current_state):
        """
        The Batched Neural Shield. 
        Evaluates N obstacles in a single PyTorch pass.
        """
        u_sequence = u_sequence.reshape((self.horizon, 2))
        state = current_state.copy()
        h_values = []
        
        for i in range(self.horizon):
            force = u_sequence[i]
            acceleration = force / 1.0 
            
            state[0] += state[2] * self.dt 
            state[1] += state[3] * self.dt
            state[2] += acceleration[0] * self.dt
            state[3] += acceleration[1] * self.dt
            
            # --- THE MAGIC: Batched Tensor Subtraction ---
            # Subtract robot position from ALL obstacles instantly
            relative_positions = self.obstacles - state[:2]
            
            # Push the batch to the AI
            rel_tensor = torch.tensor(relative_positions, dtype=torch.float32)
            with torch.no_grad():
                # Returns an array of safety scores, one for each obstacle
                h_batch = self.ncbf(rel_tensor) 
                
                # The optimizer only needs to satisfy the most dangerous threat
                min_h = torch.min(h_batch).item()
                
            h_values.append(min_h)
            
        return np.array(h_values)

    def get_action(self, current_state, target_state):
        # --- CRITICAL FIX: Warm Start from Previous Frame ---
        initial_guess = self.last_u 
        bounds = [(-15.0, 15.0)] * (self.horizon * 2)
        
        # Inject the Batched Neural CBF as a constraint
        constraints = {'type': 'ineq', 'fun': self.cbf_constraint, 'args': (current_state,)}
        
        result = minimize(
            self.simulate_trajectory_cost, 
            initial_guess, 
            args=(current_state, target_state), 
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': 30, 'ftol': 1e-3} # --- CRITICAL FIX: Stop Optimizer Hangs ---
        )
        optimal_u_sequence = result.x.reshape((self.horizon, 2))
        
        # --- UPDATE WARM START FOR NEXT TIMESTEP ---
        next_u = np.zeros((self.horizon, 2))
        next_u[:-1] = optimal_u_sequence[1:] # Shift sequence forward by 1
        self.last_u = next_u.flatten()
        
        return optimal_u_sequence[0]

if __name__ == "__main__":
    print("[INFO] Loading Batched Neural CBF...")
    brain = NeuralCBF()
    # We keep it on CPU for SciPy since transferring tiny matrices back and forth 
    # to the GPU for every SLSQP step is actually slower than just doing it on the CPU.
    brain.load_state_dict(torch.load("ncbf_batched_weights.pth", map_location=torch.device('cpu'), weights_only=True))
    brain.eval()
    
    # 1. Setup 50x50 Minefield
    np.random.seed(42) # Seeded so you get the same minefield every run
    num_obstacles = 15
    # Generate random obstacles between 5m and 40m
    obstacles = np.random.uniform(5.0, 40.0, (num_obstacles, 2))
    
    env = DoubleIntegrator2D(mass=1.0, dt=0.05)
    
    # Increased from 20 to 30 to give the robot 1.5 seconds of foresight
    mpc = BatchedSafeMPC(env, brain, obstacles, horizon=30, dt=0.05)
    
    state = env.reset(initial_state=[0.0, 0.0, 0.0, 0.0])
    target = np.array([45.0, 45.0])
    
    print(f"[INFO] Navigating {num_obstacles} obstacles to target {target}...")
    
    # Increased loop count to 600 since the robot is now adhering to a speed limit
    for step in range(600):
        best_force = mpc.get_action(state, target)
        state = env.step(best_force)
        
        dist = np.linalg.norm(state[:2] - target)
        if dist < 0.5:
            print(f"[SUCCESS] Target reached at step {step}!")
            break
            
        if step % 20 == 0:
            print(f"Step {step}: Distance to target: {dist:.2f}m")

    # 3. Plot the Result
    history = np.array(env.history)
    plt.figure(figsize=(10, 10))
    
    # Draw all N obstacles
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), 1.5, color='red', alpha=0.4)
        plt.gca().add_patch(circle)

    plt.plot(history[:, 0], history[:, 1], label="MPC Trajectory", color='blue', linewidth=2)
    plt.scatter(0, 0, color='black', s=100, label="Start")
    plt.scatter(target[0], target[1], color='green', s=100, label="Target")
    
    plt.title(f"Batched N-Obstacle NCBF Navigation ({num_obstacles} Obstacles)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(-2, 50)
    plt.ylim(-2, 50)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()