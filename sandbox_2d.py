import numpy as np

class DoubleIntegrator2D:
    def __init__(self, mass=1.0, dt=0.05, obstacles=None, obstacle_radius=1.5):
        """
        A 2D physical simulator for a drone/robot.
        Now includes native collision detection for strict physical realism.
        """
        self.mass = mass
        self.dt = dt
        self.state = np.zeros(4) # [x, y, vx, vy]
        self.history = []
        
        # Environmental Hazards
        self.obstacles = obstacles if obstacles is not None else []
        self.obstacle_radius = obstacle_radius
        self.crashed = False

    def reset(self, initial_state=[0.0, 0.0, 0.0, 0.0]):
        self.state = np.array(initial_state, dtype=float)
        self.history = [self.state.copy()]
        self.crashed = False
        return self.state

    def step(self, force):
        """
        Applies force to the robot. If crashed, the motors are dead and physics stop.
        """
        # 1. If the robot is destroyed, ignore all MPC commands and do not move.
        if self.crashed:
            self.history.append(self.state.copy())
            return self.state
            
        # 2. Standard Kinematics
        acceleration = force / self.mass
        
        self.state[0] += self.state[2] * self.dt
        self.state[1] += self.state[3] * self.dt
        self.state[2] += acceleration[0] * self.dt
        self.state[3] += acceleration[1] * self.dt
        
        # 3. Native Physical Collision Check
        for obs in self.obstacles:
            # If the robot's physical center enters the obstacle's radius
            if np.linalg.norm(self.state[:2] - obs) <= self.obstacle_radius:
                self.crashed = True
                print(f"[ENVIRONMENT] FATAL PHYSICS EVENT: Collision at obstacle {obs}. Motors destroyed.")
                break # Stop checking, robot is already dead
                
        self.history.append(self.state.copy())
        
        return self.state