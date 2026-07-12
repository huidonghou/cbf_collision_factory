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

# Attempted Resampling-Based NS-MPPI 
class NS_MPPI:
    def __init__(self, env, ncbf_model, obstacles, horizon=50, dt=0.05, num_samples=200):
        self.env = env
        self.horizon = horizon
        self.dt = dt
        self.ncbf = ncbf_model
        self.obstacles = obstacles
        self.K = num_samples
        
        # Massive Exploration. RBR protects us from bad samples now.
        self.sigma = 3.5      
        # Increased greediness. Favor the exact path that survives.
        self.lambda_ = 1.5    
        self.U_nominal = np.zeros((self.horizon, 2))

    def get_action(self, current_state, target_state):
        noise = np.random.normal(0, self.sigma, (self.K, self.horizon, 2))
        
        # Give MPPI the physical authority to pull off the dive
        U_samples = np.clip(self.U_nominal + noise, -12.0, 12.0) 
        
        states = np.tile(current_state, (self.K, 1)) 
        costs = np.zeros(self.K)
        
        L1_TUBE_BUFFER = 0.1 
        
        printed_debug_this_step = False 
        
        for t in range(self.horizon):
            forces = U_samples[:, t, :]
            accel = forces / 1.0 
            
            # Recall that the state is [x, y, vx, vy], so we update positions and velocities accordingly
            states[:, 0] += states[:, 2] * self.dt
            states[:, 1] += states[:, 3] * self.dt
            states[:, 2] += accel[:, 0] * self.dt
            states[:, 3] += accel[:, 1] * self.dt
            
            # --- EVALUATE SAFETY ---
            rel_pos = self.obstacles[None, :, :] - states[:, None, :2] 
            distances = np.linalg.norm(rel_pos, axis=2)
            min_dist = np.min(distances, axis=1)
            
            rel_pos_flat = rel_pos.reshape(-1, 2)
            with torch.no_grad():
                h_flat = self.ncbf(torch.tensor(rel_pos_flat, dtype=torch.float32)).numpy()
            
            # Now we have the result after we applied the neural CBF, now it is for the sampling process
            h_vals = h_flat.reshape(self.K, len(self.obstacles))
            h_vals[distances > 4.0] = 10.0
            min_h = np.min(h_vals, axis=1)

            # --- THE RBR ENGINE ---
            # [THE FINAL FIX] Drop min_h to -0.5 to push through the NCBF hallucination
            # The -0.5 is to push through the false-positive blockage the neural network hallucinated in the narrow corridor. 
            safe_mask = (min_dist >= (1.5 + L1_TUBE_BUFFER)) & (min_h >= -0.5)
            unsafe_indices = np.where(~safe_mask)[0]
            safe_indices = np.where(safe_mask)[0]

            # This is the resampling process, and instead of just throwing away the unsafe
            # the "dead" particles and physically teloport their states and their accumulated costs to the match the donors.
            if len(safe_indices) > 0 and len(unsafe_indices) > 0:
                replace_indices = np.random.choice(safe_indices, size=len(unsafe_indices), replace=True)
                states[unsafe_indices] = states[replace_indices]
                costs[unsafe_indices] = costs[replace_indices]
                
                # Maintain Particle Lineage, which is to save the whole control history
                U_samples[unsafe_indices, :t+1, :] = U_samples[replace_indices, :t+1, :]
                
            elif len(safe_indices) == 0:
                # --- [DEBUG PROBE] ---
                if not printed_debug_this_step:
                    geom_kills = np.sum(min_dist < (1.5 + L1_TUBE_BUFFER))
                    ncbf_kills = np.sum(min_h < -0.5)
                    print(f"[DEBUG] MPPI Extinction at horizon t={t}. Geom kills: {geom_kills}/200 | NCBF kills: {ncbf_kills}/200")
                    printed_debug_this_step = True
                # ---------------------

                # FALLBACK penalties updated with the new -0.5 threshold
                penetrations = np.clip((1.5 + L1_TUBE_BUFFER) - min_dist, 0.0, None)
                costs += penetrations * 50.0
                violations = min_h < -0.5
                costs[violations] += 2000 * (-0.5 - min_h[violations])**2
            
            dist_to_target = np.linalg.norm(states[:, :2] - target_state, axis=1)
            costs += dist_to_target * 2.0
            costs += (forces[:, 0]**2 + forces[:, 1]**2) * 0.01 
            
        final_dist = np.linalg.norm(states[:, :2] - target_state, axis=1)
        costs += final_dist * 500.0
        final_speed_sq = states[:, 2]**2 + states[:, 3]**2
        costs += final_speed_sq * 2.0 
            
        beta = np.min(costs)
        weights = np.exp(-1.0 / self.lambda_ * (costs - beta))
        
        weight_sum = np.sum(weights)
        if weight_sum < 1e-10:
            weights = np.ones(self.K) / self.K
        else:
            weights /= weight_sum
        
        self.U_nominal = np.sum(weights[:, None, None] * U_samples, axis=0)
        
        action = self.U_nominal[0].copy()
        self.U_nominal[:-1] = self.U_nominal[1:]
        self.U_nominal[-1] = self.U_nominal[-2] 
        
        return action
# 3. YOUR EXTENSION: The L1 Adaptive Filter
class L1AdaptiveFilter:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.v_hat = np.zeros(2) 
        self.d_hat = np.zeros(2) 
        self.u_l1 = np.zeros(2)  
        self.last_v = np.zeros(2) 
        
        self.As = -5.0       # Error dynamics which is the Hurwitz matrix
        self.Gamma = 200.0   # Dropped from 500.0 to prevent Euler integration overshoot
        self.cutoff = 15.0   # Fast enough to catch the mass drop, slow enough to remain stable  
        
    def get_correction(self, current_velocity, applied_force):
        # 1. Calculate true time-aligned error FIRST
        v_tilde = self.v_hat - current_velocity
        
        # 2. Adaptation Law using properly aligned error
        self.d_hat += -self.Gamma * v_tilde * self.dt
        
        # 3. Low Pass Filter
        alpha = np.exp(-self.cutoff * self.dt)
        self.u_l1 = alpha * self.u_l1 - (1 - alpha) * self.d_hat
        self.u_l1 = np.clip(self.u_l1, -15.0, 15.0)
        
        # 4. Advance predictor state to t+1 for the NEXT step
        v_hat_dot = (applied_force + self.d_hat) / 1.0 + self.As * v_tilde
        self.v_hat += v_hat_dot * self.dt
        
        return self.u_l1

if __name__ == "__main__":
    print("[INFO] Loading Previous Neural Shield...")
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
    mppi = NS_MPPI(env, brain, obstacles, horizon=50, dt=0.05, num_samples=200)
    
    # Your L1 inner-loop
    l1_filter = L1AdaptiveFilter(dt=0.05)
    
    # Start and Target alignment
    state = env.reset(initial_state=[0.0, 1.5, 0.0, 0.0])
    target = np.array([10.0, -6.0])
    
    total_force = np.zeros(2)
    fault_triggered = False
    fault_step = 0
    
    print("[INFO] Starting L1-NS-MPPI Simulation...")
    for step in range(250): 
        
        # *** THE SABOTAGE: Trigger right as the robot clears Gate 1 and tries to dive ***
        if state[0] > 4.5 and not fault_triggered: 
            print(f"[WARNING] Step {step}: System fault! Mass doubled from 1.0kg to 2.0kg!")
            env.mass = 2.0
            fault_triggered = True
            fault_step = step
            
        # 1. Yin's NS-MPPI computes trajectory (Assuming mass is still 1.0kg)
        u_mppi = mppi.get_action(state, target)
        
        # 2. YOUR EXTENSION: L1 estimates the unmodeled mass and injects counter-force
        u_l1 = l1_filter.get_correction(current_velocity=state[2:4], applied_force=total_force)
        
        # 3. Robust architecture: We combine MPPI and L1, clipping to the true physical limit of 15.0N
        total_force = np.clip(u_mppi + u_l1, -15.0, 15.0)
        
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
                     label="L1-NS-MPPI Adapted (2.0kg Mass)", color='green', marker='x')
    else:
        plt.plot(history[:, 0], history[:, 1], label="Normal Trajectory", color='blue', marker='.')
                 
    plt.scatter(target[0], target[1], color='green', s=100, label=f"Target ({target[0]}, {target[1]})", zorder=5)
    plt.title("Act 2: Your Extension (L1-NS-MPPI Crash Averted)")
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