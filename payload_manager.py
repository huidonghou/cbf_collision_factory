import torch
class PayloadManager: 
    """
    Manages structural mass changes (payloads) to the robot's end-effector.
    Interfaces directly with the PhysX backend to alter the physical mass 
    of the `panda_hand` mid-simulation. Maintains a pristine CPU-side snapshot 
    to ensure all payload applications are idempotent and safely reversible.
    """
    def __init__(self, robot):
        """
        Locates the hand link in the USD hierarchy and caches a permanent, 
        CPU-bound snapshot of the factory-default mass matrix to prevent drift.
        """
        self.view = robot.root_physx_view
        self.hand_idx = robot.find_bodies("panda_hand")[0][0]
        self.device = robot.device

        self.default = self.view.get_masses().clone().cpu()

    def _set(self, env_ids_cpu, new_hand_mass):
        """
        Internal helper: copy active mass matrix to CPU and apply the edit
        then push back to PhysX using strictly CPU tensor to prevent mismatch device
        """
        m = self.view.get_masses().clone().cpu()
        m[env_ids_cpu, self.hand_idx] = new_hand_mass

        all_end_indices = torch.arange(m.shape[0], dtype = torch.int32, device = 'cpu')
        self.view.set_masses(m, all_end_indices)

    def apply(self, env_ids, extra):
        """
        Attach the payload, and add the extra kg to the factory-default mass of targeted environments
        """
        env_ids_cpu = env_ids.cpu()
        self._set(env_ids_cpu, self.default[env_ids_cpu, self.hand_idx] + extra)

    def clear(self, env_ids):
        """
        Removes the payload. Restores the hand mass of the targeted 
        environments strictly back to the cached factory defaults.
        """
        env_ids_cpu = env_ids.cpu()
        self._set(env_ids_cpu, self.default[env_ids_cpu, self.hand_idx])