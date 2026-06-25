import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sandbox_2d import DoubleIntegrator2D

# 1. The Neural Shield (from Yin's DPNCBF)
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

# 2. Yin's Baseline: Sampling-based NS-MPPI 
class NS_MPPI:
    def __init__(self, env, ncbf_model, obstacles, horizon=30, dt=0.05, num_samples=200):
        self.env = env
        self.horizon = horizon
        self.dt = dt
        self.ncbf = ncbf_model
        self.obstacles = obstacles
        self.K = num_samples
        
        self.sigma = 2.5      
        self.lambda_ = 5.0    
        self.U_nominal = np.zeros((self.horizon, 2))

    def get_action(self, current_state, target_state):
        noise = np.random.normal(0, self.sigma, (self.K, self.horizon, 2))
        
        # --- ACTUATOR ALLOCATION ---
        # The originalMPPI is capped at 7.0N. 
        U_samples = np.clip(self.U_nominal + noise, -7.0, 7.0)
        
        states = np.tile(current_state, (self.K, 1)) 
        costs = np.zeros(self.K)
        
        for t in range(self.horizon):
            forces = U_samples[:, t, :]
            
            # The original MPPI assumes mass is ALWAYS 1.0kg
            accel = forces / 1.0 
            
            states[:, 0] += states[:, 2] * self.dt
            states[:, 1] += states[:, 3] * self.dt
            states[:, 2] += accel[:, 0] * self.dt
            states[:, 3] += accel[:, 1] * self.dt
            
            dist_to_target = np.linalg.norm(states[:, :2] - target_state, axis=1)
            costs += dist_to_target * 2.0
            
            # Smooth steering penalty to dampen high-frequency jitters
            costs += (forces[:, 0]**2 + forces[:, 1]**2) * 0.05
            
            # --- Geometric Penetration Penalty ---
            # Penalizes the depth of the crash so the MPPI fights for the shallowest impact
            rel_pos = self.obstacles[None, :, :] - states[:, None, :2] 
            distances = np.linalg.norm(rel_pos, axis=2)
            penetrations = np.clip(1.5 - distances, 0.0, None)
            costs += np.sum(penetrations, axis=1) * 5000.0
            
            # --- Neural Shield Soft Penalty ---
            rel_pos_flat = rel_pos.reshape(-1, 2)
            with torch.no_grad():
                h_flat = self.ncbf(torch.tensor(rel_pos_flat, dtype=torch.float32)).numpy()
            
            h_vals = h_flat.reshape(self.K, len(self.obstacles))
            
            # Sensor Range Masking (Ignore distant hallucinations > 4.0m away)
            h_vals[distances > 4.0] = 10.0
            
            min_h = np.min(h_vals, axis=1)
            violations = min_h < 1.2
            costs[violations] += 2000 * (1.2 - min_h[violations])**2
            
        # --- Terminal Cost ---
        # Distance + Speed penalty to prevent swirling at the target
        final_dist = np.linalg.norm(states[:, :2] - target_state, axis=1)
        costs += final_dist * 50.0
        final_speed_sq = states[:, 2]**2 + states[:, 3]**2
        costs += final_speed_sq * 20.0
            
        beta = np.min(costs)
        weights = np.exp(-1.0 / self.lambda_ * (costs - beta))
        weights /= np.sum(weights) 
        
        self.U_nominal = np.sum(weights[:, None, None] * U_samples, axis=0)
        
        action = self.U_nominal[0].copy()
        self.U_nominal[:-1] = self.U_nominal[1:]
        self.U_nominal[-1] = self.U_nominal[-2] 
        
        return action

if __name__ == "__main__":
    print("[INFO] Loading Yin's Neural Shield...")
    brain = NeuralCBF()
    brain.load_state_dict(torch.load("ncbf_batched_weights.pth", map_location=torch.device('cpu'), weights_only=True))
    brain.eval()
    
    # --- THE OVER-UNDER MOMENTUM TRAP ---
    obstacles = np.array([
        # Gate 1: Forces robot UP 
        [3.0, -1.5],  
        [3.0, -4.5],  
        [3.0, 4.5],   
        
        # Gate 2: Forces robot DOWN (with plugged loophole)
        [8.0, 1.5],   
        [8.0, 4.5],   
        [8.0, -3.0],  
        [8.0, -6.0]   
    ])
    
    # Sandbox strict physics enforcement
    env = DoubleIntegrator2D(mass=1.0, dt=0.05, obstacles=obstacles)
    
    # The naked Yin controller
    mppi = NS_MPPI(env, brain, obstacles, horizon=30, dt=0.05, num_samples=200)
    
    # Start and Target alignment
    state = env.reset(initial_state=[0.0, 1.5, 0.0, 0.0])
    target = np.array([14.0, -0.75])
    
    fault_triggered = False
    fault_step = 0
    
    print("[INFO] Starting Baseline NS-MPPI Simulation...")
    for step in range(250): 
        
        # *** THE SABOTAGE: Trigger right as the robot clears Gate 1 and tries to dive ***
        if state[0] > 4.5 and not fault_triggered: 
            print(f"[WARNING] Step {step}: System fault! Mass doubled from 1.0kg to 2.0kg!")
            env.mass = 2.0
            fault_triggered = True
            fault_step = step
            
        # 1. Yin's NS-MPPI computes trajectory (Assuming mass is still 1.0kg)
        u_mppi = mppi.get_action(state, target)
        
        # 2. Naked architecture: The motor command is ONLY what the MPPI asks for.
        # Clipped strictly to its 7.0N assumed limit.
        total_force = np.clip(u_mppi, -7.0, 7.0)
        
        state = env.step(total_force)
        
        if env.crashed:
            break
        
        dist = np.linalg.norm(state[:2] - target)
        if dist < 0.3:
            print(f"[SUCCESS] Target reached at step {step}!")
            break

    # --- Plotting ---
    history = np.array(env.history)
    plt.figure(figsize=(12, 6))
    
    for obs in obstacles:
        circle = plt.Circle((obs[0], obs[1]), 1.5, color='red', alpha=0.3)
        plt.gca().add_patch(circle)

    if fault_step > 0:
        plt.plot(history[:fault_step, 0], history[:fault_step, 1], 
                 label="Nominal NS-MPPI (1.0kg)", color='blue', marker='.')
                 
        if len(history) > fault_step:
            plt.plot(history[fault_step-1:, 0], history[fault_step-1:, 1], 
                     label="Sabotaged NS-MPPI (Crash!)", color='orange', marker='x')
    else:
        plt.plot(history[:, 0], history[:, 1], label="Normal Trajectory", color='blue', marker='.')
                 
    plt.scatter(target[0], target[1], color='green', s=100, label=f"Target ({target[0]}, {target[1]})", zorder=5)
    plt.title("Act 1: The Baseline Crash (Naked NS-MPPI)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.xlim(-1, 15)
    plt.ylim(-8, 6)
    plt.grid(True)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.axis('equal')
    plt.show()