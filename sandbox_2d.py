import numpy as np
import matplotlib.pyplot as plt

# This is the basic Euleer integration for the stepped simulating physics

class DoubleIntegrator2D:
    def __init__(self, mass = 1.0, dt = 0.05):
        """
        This initialize the original 2D environment with a double integrator
        """
        self.mass = mass
        self.dt = dt
        self.state = np.zeros(4)  # [x, y, vx, vy]
        self.history = []

    def reset(self, initial_state):
        """
        This reset the environment to certain particular state
        """
        self.state = np.array(initial_state)
        self.history = [self.state.copy()]
        return self.state
    
    def step(self, force):
        """
        Applied the force [Fx, Fy] for one time step and return the new state
        """
        acceleration = np.array(force) / self.mass

        # Update position
        self.state[0] += self.state[2] * self.dt 
        self.state[1] += self.state[3] * self.dt

        # Update velocity
        self.state[2] += acceleration[0] * self.dt
        self.state[3] += acceleration[1] * self.dt

        self.history.append(self.state.copy())
        return self.state
    
# Baseline Test
if __name__ == "__main__":
    env = DoubleIntegrator2D(mass = 1.0, dt = 0.05)
    env.reset([0, 0, 1.0,3.0])  # Start at the origin with zero velocity

    # Apply a constant force for 10 seconds
    for _ in range(40):  # 200 steps at dt=0.05s is 10 seconds
        env.step([0.0,-9.81])  # Apply a force of [0.0, -9.81]

    # Plot the trajectory
    history = np.array(env.history)
    plt.plot(history[:, 0], history[:, 1])
    plt.title("Trajectory of Double Integrator in 2D")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid()
    plt.axis('equal')
    plt.show()