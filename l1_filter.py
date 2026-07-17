import torch, math
class L1AdaptiveFilter:
    def __init__(self, num_joints: int, dt: float, device):
        # First initialize all the devices and the DOF parameters
        # Also the device as we need to make sure the GPU stored ones are on GPU and CPU ones are on CPU
        self.num_joints = num_joints
        self.dt = dt
        self.device = device

        # Instantiate the 3 state variables
        # The state, the disturbance estimate, and the control signal
        self.v_hat = torch.zeros((1, num_joints), device = device)
        self.d_hat = torch.zeros((1, num_joints), device = device)
        self.u_l1 = torch.zeros((1, num_joints), device = device)

        # Defining the constants
        # 1: Hurwitz matrix which is a damping factor, trying to drag the robot back into the correct trajectory
        # Note that this was originally set to -10, but kinda broke the experiment setting, so we need to increase the actual matrix
        self.As = -30.0

        # 2: Adaptation Gain: control how agressively filter translate velocity error into disturbance estimate
        self.Gamma = 150.0 

        # 3: Low passfilter bandwidth: Gamma is too big, so it is going to take a lot of noise, so need to low pass filter it
        # Unit is rad/s, so this means that if it is fatser than 2.4Hz, it will be blocked as noise
        self.cutoff = 15.0

    def get_correction(self, current_velocity, applied_torque, inertia_diag, budget):
        # first calculate_error
        v_tilde = self.v_hat - current_velocity

        # adaption law
        self.d_hat += -self.Gamma * inertia_diag * v_tilde * self.dt

        # Now the low pass filter, then need to clamp it for safety issues
        alpha = math.exp(-self.cutoff * self.dt)
        self.u_l1 = alpha * self.u_l1 - (1.0 - alpha) * self.d_hat
        self.u_l1 = torch.clamp(self.u_l1, -25.0, 25.0)

        # Now to update the predictor 
        v_hat_dot = (applied_torque + self.d_hat) / inertia_diag + self.As * v_tilde
        self.v_hat += v_hat_dot * self.dt

        self.u_l1 = torch.clamp(self.u_l1, -budget, budget)
        return self.u_l1
    
    def reset(self):
        self.v_hat.zero_()
        self.d_hat.zero_()
        self.u_l1.zero_()
