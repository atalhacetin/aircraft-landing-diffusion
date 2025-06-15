#!/usr/bin/env python3
# landing_env_v2.py

import numpy as np
import gym
from gym import spaces

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def aircraft_dynamics(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Continuous-time aircraft dynamics (unchanged):
      x = [x, y, h, V, psi, gamma]
      u = [n_x, n_z, mu]
    Returns ẋ = f(x,u).
    """
    g = 9.81
    X, Y, H, V, psi, gamma = x
    n_x, n_z, mu = u
    dx = np.zeros(6)
    dx[0] = V * np.cos(psi) * np.cos(gamma)
    dx[1] = V * np.sin(psi) * np.cos(gamma)
    dx[2] = V * np.sin(gamma)
    dx[3] = g * (n_x - np.sin(gamma))
    dx[4] = g * n_z / V * np.sin(mu) / np.cos(gamma)
    dx[5] = g / V * (n_z * np.cos(mu) - np.cos(gamma))
    return dx

class LandingEnv(gym.Env):
    """
    Gym environment for fixed-wing landing with moving obstacles.

    - Observation: Augmented state including aircraft state, obstacle positions, and car velocity.
    - Action: Desired 3D position [x_ref, y_ref, h_ref].
    - Internally computes u = [n_x, n_z, mu] via PD loops and integrates dynamics.
    - Obstacles move along the x-axis based on a constant car velocity.
    """
    metadata = {}

    def __init__(self, num_obstacles=8):
        super().__init__()
        self.num_obstacles = num_obstacles
        
        # --- NEW: Augmented Observation Space ---
        # 6 (aircraft) + num_obstacles * 3 (pos) + 1 (car_vel)
        obs_dim = 6 + self.num_obstacles * 3 + 1
        
        # Define bounds for each part of the observation
        obs_low_ac = np.array([-2000, -2000, 0, 0, -np.pi, -np.deg2rad(20)])
        obs_high_ac = np.array([2000, 2000, 2000, 100, np.pi, np.deg2rad(20)])
        
        obs_low_obstacles = np.full((self.num_obstacles * 3,), -2000)
        obs_high_obstacles = np.full((self.num_obstacles * 3,), 2000)
        
        obs_low_vel = np.array([-50])
        obs_high_vel = np.array([50])

        obs_low = np.concatenate([obs_low_ac, obs_low_obstacles, obs_low_vel])
        obs_high = np.concatenate([obs_high_ac, obs_high_obstacles, obs_high_vel])
        
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # --- UNCHANGED: Action Space ---
        act_low = np.array([-2000, -2000, 0], dtype=np.float32)
        act_high = np.array([2000, 2000, 2000], dtype=np.float32)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # --- UNCHANGED: Dynamics and Controller Parameters ---
        self.dt = 0.02  # 50 Hz
        self.V_ref = 70.0
        self.Kp_psi, self.Kd_psi = 0.1, 0.5
        self.Kp_h, self.Kd_h = 0.1, 0.2
        self.Kp_v = 1.0
        self.nx_min, self.nx_max = -3.0, 5.0
        self.nz_min, self.nz_max = -3.0, 5.0
        self.mu_min, self.mu_max = -np.pi / 4, np.pi / 4

        # --- State Placeholders ---
        self.x = np.zeros(6)
        self.x_prev = np.zeros(6)
        self.obstacle_centers = np.zeros((self.num_obstacles, 3))
        self.car_velocity = 0.0

    def reset(self, x0: np.ndarray = None, obstacle_centers: np.ndarray = None, car_velocity: float = 0.0) -> np.ndarray:
        """
        Reset the environment to a new initial state.
        
        Args:
            x0: Initial aircraft state [x, y, h, V, psi, gamma].
            obstacle_centers: Initial positions of obstacles (num_obstacles, 3).
            car_velocity: The constant velocity of the ground vehicle/obstacles.
        
        Returns:
            The initial augmented observation.
        """
        # Reset aircraft state
        if x0 is None:
            self.x = np.array([-500., 100., 300., 70., 0., -0.05])
        else:
            self.x = x0.copy()
        self.x_prev = self.x.copy()

        # Reset obstacle state
        if obstacle_centers is None:
            # Default obstacles if none provided
            self.obstacle_centers = np.array([[200 + i*100, np.random.uniform(-15, 15), 0] for i in range(self.num_obstacles)])
        else:
            assert obstacle_centers.shape == (self.num_obstacles, 3), "Obstacle centers must have shape (num_obstacles, 3)"
            self.obstacle_centers = obstacle_centers.copy()
            
        self.car_velocity = car_velocity
        
        return self._get_obs()

    def step(self, action: np.ndarray):
        """
        Take a step in the environment.
        
        Args:
            action: [x_ref, y_ref, h_ref] target position.
        
        Returns:
            obs, reward, done, info tuple.
        """
        x_ref, y_ref, h_ref = action
        x, y, h, V, psi, gamma = self.x

        # --- UNCHANGED: PD Controller Logic ---
        psi_dot = (psi - self.x_prev[4]) / self.dt
        h_dot = (h - self.x_prev[2]) / self.dt

        psi_ref = np.arctan2(y_ref - y, x_ref - x)
        e_psi = wrap_to_pi(psi_ref - psi)
        mu = self.Kp_psi * e_psi - self.Kd_psi * psi_dot
        mu = np.clip(mu, self.mu_min, self.mu_max)

        e_h = h_ref - h
        e_h_dot = 0.0 - h_dot
        a_z = self.Kp_h * e_h + self.Kd_h * e_h_dot
        n_z = np.sin(gamma) + a_z / 9.81 + 1 / np.cos(mu)
        n_z = np.clip(n_z, self.nz_min, self.nz_max)

        e_V = self.V_ref - V
        n_x = np.sin(gamma) + (self.Kp_v * e_V) / 9.81
        n_x = np.clip(n_x, self.nx_min, self.nx_max)

        # --- Integrate Aircraft Dynamics ---
        u = np.array([n_x, n_z, mu])
        dx = aircraft_dynamics(self.x, u)
        self.x_prev = self.x.copy()
        self.x = self.x + dx * self.dt

        # --- NEW: Update Obstacle Positions ---
        self.obstacle_centers[:, 0] += self.car_velocity * self.dt

        # --- Construct Observation and Reward ---
        obs = self._get_obs()
        pos_err = np.linalg.norm(self.x[:3] - action)
        reward = -pos_err
        done = pos_err < 1.0
        info = {'u': u, 'obstacle_centers': self.obstacle_centers.copy()}
        
        return obs, reward, done, info

    def _get_obs(self) -> np.ndarray:
        """
        Assemble the augmented observation vector.
        
        Returns:
            A 31-dimensional observation vector.
        """
        flat_obstacles = self.obstacle_centers.ravel()
        car_vel_arr = np.array([self.car_velocity])
        
        # Concatenate all parts into a single vector
        obs = np.concatenate([self.x, flat_obstacles, car_vel_arr])
        return obs.astype(np.float32)

    def close(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # --- Simulation Parameters ---
    sim_time = 15.0
    num_obstacles = 8
    env = LandingEnv(num_obstacles=num_obstacles)

    # --- Initial Conditions for this run ---
    initial_x0 = np.array([-500., 200., 400., 70., 0.1, -0.05])
    initial_obstacles = np.array([[100 + i*150, np.random.uniform(-20, 20), 0] for i in range(num_obstacles)])
    car_vel = 25.0 # m/s

    obs = env.reset(x0=initial_x0, obstacle_centers=initial_obstacles, car_velocity=car_vel)
    
    dt = env.dt
    n_steps = int(sim_time / dt)
    
    print(f"Initial Observation Shape: {obs.shape}")
    assert obs.shape == (6 + num_obstacles * 3 + 1,), "Observation dimension is incorrect!"

    # --- Fixed Reference Target ---
    action = np.array([1500.0, 0.0, 50.0], dtype=np.float32)

    # --- Storage for plotting ---
    traj = np.zeros((n_steps + 1, 6))
    traj[0] = env.x.copy()
    
    obstacle_hist = np.zeros((n_steps + 1, num_obstacles, 3))
    obstacle_hist[0] = env.obstacle_centers.copy()

    # --- Run Simulation Loop ---
    for i in range(1, n_steps + 1):
        obs, reward, done, info = env.step(action)
        traj[i] = env.x.copy()
        obstacle_hist[i] = info['obstacle_centers']
        if done:
            print(f"Target reached in {i*dt:.2f} s (step {i})")
            break
    
    last_step = i # Store the last step for plotting
    env.close()

    # --- Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot aircraft trajectory
    ax.plot(traj[:last_step, 0], traj[:last_step, 1], traj[:last_step, 2], 'b-', label='Aircraft Path')
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='g', marker='o', s=100, label='Start')
    ax.scatter(action[0], action[1], action[2], c='r', marker='x', s=100, label='Target')

    # Plot initial and final obstacle positions
    ax.scatter(obstacle_hist[0, :, 0], obstacle_hist[0, :, 1], obstacle_hist[0, :, 2], c='orange', marker='s', label='Obstacles (Initial)')
    ax.scatter(obstacle_hist[last_step-1, :, 0], obstacle_hist[last_step-1, :, 1], obstacle_hist[last_step-1, :, 2], c='purple', marker='s', label='Obstacles (Final)')
    
    ax.set_xlabel('X [m] (Runway Axis)')
    ax.set_ylabel('Y [m] (Lateral)')
    ax.set_zlabel('H [m] (Altitude)')
    ax.set_title('3D Landing Trajectory with Moving Obstacles')
    ax.legend()
    plt.tight_layout()
    plt.show()
