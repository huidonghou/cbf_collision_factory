import torch
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sandbox_2d import DoubleIntegrator2D
from train_ncbf import NeuralCBF

class SafeMPC:
    def __init__(self, env, ncbf_model, horizon=15, dt=0.05):
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
            acceleration = force / 1.0 
            
            state[0] += state[2] * self.dt 
            state[1] += state[3] * self.dt
            state[2] += acceleration[0] * self.dt
            state[3] += acceleration[1] * self.dt

            # 1. Linear Distance Penalty
            distance_penalty = np.linalg.norm([state[0] - target_state[0], state[1] - target_state[1]])
            effort_penalty = (force[0]**2 + force[1]**2) * 0.05
            
            # 2. The 64-Bit Float Fix: Use pure NumPy to calculate h(x)
            # h(x) = distance from center (4,4) minus radius (1.5)
            h = np.linalg.norm([state[0] - 4.0, state[1] - 4.0]) - 1.5
                
            cbf_penalty = 0
            # Start reacting aggressively when the robot gets within 0.8 meters of the boundary
            if h < 0.8: 
                cbf_penalty = 50000 * (0.8 - h)**3

            cost += distance_penalty + effort_penalty + cbf_penalty
            
        return cost

    def cbf_constraint(self, u_sequence, current_state):
        """
        The Safety Filter: SciPy MUST keep all returned values >= 0.
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
            
            # Ask the PyTorch brain: "Is this future state safe?"
            state_tensor = torch.tensor([state[0], state[1]], dtype=torch.float32)
            with torch.no_grad():
                h = self.ncbf(state_tensor).item()
            h_values.append(h)       
        return np.array(h_values)

    def get_action(self, current_state, target_state):
        initial_guess = np.zeros(self.horizon * 2)
        bounds = [(-10.0, 10.0)] * (self.horizon * 2)
        
        # Inject the Neural CBF as a strict inequality constraint
        constraints = {'type': 'ineq', 'fun': self.cbf_constraint, 'args': (current_state,)}
        
        result = minimize(
            self.simulate_trajectory_cost, 
            initial_guess, 
            args=(current_state, target_state), 
            bounds=bounds,
            method='SLSQP'
        )
        optimal_u_sequence = result.x.reshape((self.horizon, 2))
        return optimal_u_sequence[0]

if __name__ == "__main__":
    # 1. Load the PyTorch Brain
    print("[INFO] Loading Neural CBF...")
    brain = NeuralCBF()
    brain.load_state_dict(torch.load("ncbf_weights.pth", weights_only=True))
    brain.eval()
    
    # 2. Setup Environment
    env = DoubleIntegrator2D(mass=1.0, dt=0.05)
    mpc = SafeMPC(env, brain, horizon=30, dt=0.05)
    
    state = env.reset(initial_state=[0.0, 0.5, 0.0, 0.0])
    target = np.array([8.0, 8.0])
    
    print("[INFO] Starting Sabotage Simulation...")
    for step in range(100):
        
        # *** THE SABOTAGE ***
        if step == 40:
            print(f"[WARNING] Step 40: System fault! Mass doubled from 1.0kg to 2.0kg!")
            env.mass = 2.0
            
        best_force = mpc.get_action(state, target)
        state = env.step(best_force)
        
        dist = np.linalg.norm(state[:2] - target)
        if dist < 0.1:
            print("[SUCCESS] Target reached!")
            break

    # 3. Plot the Failure
    history = np.array(env.history)
    plt.figure(figsize=(8, 6))
    
    # Draw the mathematical obstacle
    circle = plt.Circle((4.0, 4.0), 1.0, color='red', alpha=0.3, label="Obstacle")
    circle2 = plt.Circle((4.0, 8.0), 1.0, color='red', alpha=0.3, label="Obstacle 2")

    plt.gca().add_patch(circle)
    plt.gca().add_patch(circle2)

    plt.plot(history[:40, 0], history[:40, 1], label="Normal Trajectory", color='blue', marker='.')
    plt.plot(history[40:, 0], history[40:, 1], label="Sabotaged Trajectory", color='orange', marker='x')
    
    plt.scatter(target[0], target[1], color='green', s=100, label="Target (8,8)", zorder=5)
    plt.title("Milestone 4: The Baseline Crash (Without L1 Adaptation)")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()