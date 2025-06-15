#%% landing_env.py

import numpy as np
import gym
from gym import spaces

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + np.pi) % (2*np.pi) - np.pi

def aircraft_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Continuous‐time aircraft dynamics:
      x = [x, y, h, V, psi, gamma]
      u = [n_x, n_z, mu]
    Returns ẋ = f(x,u).
    """
    g = 9.81
    X, Y, H, V, psi, gamma = x
    n_x, n_z, mu = u
    dx = np.zeros(6)
    dx[0] = V * np.cos(psi) * np.cos(gamma)             # x_dot
    dx[1] = V * np.sin(psi) * np.cos(gamma)             # y_dot
    dx[2] = V * np.sin(gamma)                           # h_dot
    dx[3] = g * (n_x - np.sin(gamma))                   # V_dot
    dx[4] = g * n_z / V * np.sin(mu) / np.cos(gamma)    # psi_dot
    dx[5] = g / V * (n_z * np.cos(mu) - np.cos(gamma))  # gamma_dot
    return dx

class LandingEnv(gym.Env):
    """
    Gym environment for fixed-wing landing with PD-based load-factor controllers.
    Action: desired 3D position [x_ref, y_ref, h_ref].
    Internally computes u = [n_x, n_z, mu] via PD loops, then integrates dynamics.
    """
    metadata = {}

    def __init__(self):
        super().__init__()
        # State: [x, y, h, V, psi, gamma]
        # Observation space
        obs_low  = np.array([-2000, -2000,    0,   0, -np.pi, -np.deg2rad(20)])
        obs_high = np.array([ 2000,  2000, 2000, 100,  np.pi,  np.deg2rad(20)])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        # Action: target [x_ref, y_ref, h_ref]
        act_low  = np.array([-2000, -2000,    0], dtype=np.float32)
        act_high = np.array([ 2000,  2000, 2000], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # integration timestep
        self.dt = 0.02  # 50 Hz

        # nominal cruise speed
        self.V_ref = 70.0

        # PD gains (tune as needed)
        self.Kp_psi,   self.Kd_psi   = 0.1, 0.5    # heading → bank angle mu
        self.Kp_h,     self.Kd_h     = 0.1, 0.2    # altitude → normal load factor n_z
        self.Kp_v                = 1.0            # speed → longitudinal load factor n_x

        # actuator limits
        self.nx_min, self.nx_max = -3.0,  5.0
        self.nz_min, self.nz_max = -3.0,  5.0
        self.mu_min, self.mu_max = -np.pi/4, np.pi/4

        # state placeholders
        self.x = np.zeros(6)      # current state
        self.x_prev = np.zeros(6) # previous state for derivative estimation

    def reset(self, x0: np.ndarray = None) -> np.ndarray:
        """Reset to default or provided initial state."""
        if x0 is None:
            # default starting conditions
            self.x = np.array([-500., 100., 300., 70., 0., -0.05])
        else:
            self.x = x0.copy()
        self.x_prev = self.x.copy()
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        action: [x_ref, y_ref, h_ref]
        Returns: obs, reward, done, info
        """
        x_ref, y_ref, h_ref = action
        x, y, h, V, psi, gamma = self.x

        # estimate derivatives
        psi_dot   = (psi   - self.x_prev[4]) / self.dt
        h_dot     = (h     - self.x_prev[2]) / self.dt

        # 1) Heading control → bank angle μ
        psi_ref = np.arctan2(y_ref - y, x_ref - x)
        e_psi   = wrap_to_pi(psi_ref - psi)
        mu = self.Kp_psi * e_psi - self.Kd_psi * psi_dot
        mu = np.clip(mu, self.mu_min, self.mu_max)

        # 2) Altitude control → normal load factor n_z
        e_h   = h_ref - h
        e_h_dot = 0.0 - h_dot
        # vertical accel a_z = Kp_h*e_h + Kd_h*e_h_dot
        a_z = self.Kp_h * e_h + self.Kd_h * e_h_dot
        # map to load factor: a_z ≈ g*(n_z - sinγ)
        n_z = np.sin(gamma) + a_z / 9.81 + 1/np.cos(mu)
        n_z = np.clip(n_z, self.nz_min, self.nz_max)

        # 3) Speed control → longitudinal load factor n_x
        e_V = self.V_ref - V
        n_x = np.sin(gamma) + (self.Kp_v * e_V) / 9.81
        n_x = np.clip(n_x, self.nx_min, self.nx_max)

        # assemble control and integrate
        u = np.array([n_x, n_z, mu])
        dx = aircraft_dynamics(self.x, u)
        self.x_prev = self.x.copy()
        self.x = self.x + dx * self.dt

        # observation
        obs = self._get_obs()
        # simple reward: negative 3D position error
        pos_err = np.linalg.norm(self.x[:3] - action)
        reward = -pos_err
        done = pos_err < 1.0  # within 1 m of target
        info = {'u': u}
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Return current state as observation."""
        return self.x.astype(np.float32)

    def close(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # simulation parameters
    sim_time = 10.0            # total time [s]
    env = LandingEnv()
    obs = env.reset()
    dt = env.dt
    n_steps = int(sim_time / dt)

    # fixed reference target
    action = np.array([500.0, 0.0, 50.0], dtype=np.float32)

    # storage
    traj = np.zeros((n_steps+1, 6))
    traj[0] = env.x.copy()

    # run loop
    for i in range(1, n_steps+1):
        obs, reward, done, info = env.step(action)
        traj[i] = env.x.copy()
        if done:
            print(f"Target reached in {i*dt:.2f} s (step {i})")
            break

    env.close()

    # plot 3D trajectory
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj[:i+1,0], traj[:i+1,1], traj[:i+1,2], marker='.')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('h [m]')
    ax.set_title('3D Landing Trajectory')
    plt.tight_layout()
    plt.show()