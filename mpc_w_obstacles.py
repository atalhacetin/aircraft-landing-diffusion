#%% Define the NMPC for aircraft landing with ellipsoidal obstacles
import numpy as np
import casadi as cs

# Aircraft landing dynamics
def aircraft_dynamics(x, u):
    g = 9.81
    V = x[3]
    psi = x[4]
    gamma = x[5]
    nx, nz, mu = u[0], u[1], u[2]

    dx = V * cs.cos(psi) * cs.cos(gamma)
    dy = V * cs.sin(psi) * cs.cos(gamma)
    dh = V * cs.sin(gamma)
    dV = g * (nx - cs.sin(gamma))
    dpsi = g * nz / V * cs.sin(mu) / cs.cos(gamma)
    dgamma = g / V * (nz * cs.cos(mu) - cs.cos(gamma))

    return cs.vertcat(dx, dy, dh, dV, dpsi, dgamma)

# Discretize with RK4
def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

# NMPC Class for Landing
class LandingNMPC:
    def __init__(self, T, N, m_steps_per_point, q_diag, r_diag, u_bounds):
        self.T = T
        self.N = N
        self.m_steps_per_point = m_steps_per_point
        self.dt = T / N / m_steps_per_point

        self.Q = cs.diag(q_diag)
        self.R = cs.diag(r_diag)
        self.min_u = (-u_bounds).tolist()
        self.max_u = u_bounds.tolist()

        self.x = cs.MX.sym('x', 6)
        self.u = cs.MX.sym('u', 3)
        self.x_dot = aircraft_dynamics(self.x, self.u)
        self.dynamics_f = cs.Function('x_dot', [self.x, self.u], [self.x_dot], ['x', 'u'], ['x_dot'])

        L = self.x.T @ self.Q @ self.x + self.u.T @ self.R @ self.u
        self.cost_f = cs.Function('q', [self.x, self.u], [L], ['x', 'u'], ['q'])

        self.integrator = self.build_integrator()

        # Define ellipsoid obstacles (cars)
        car_size =  np.array([10.0, 10.0, 10.0])
        self.car_velocity = np.array([80.0/3.6, 0.0, 0.0])  # m/s
        self.ellipsoids = [
            {'center': np.array([400, 2.5, 0]), 'axes':car_size},
            {'center': np.array([600, -3.0, 0]), 'axes': car_size},
            {'center': np.array([500, 12.0, 0]), 'axes': car_size},
            {'center': np.array([200, 6.0, 0]), 'axes': car_size},
            {'center': np.array([100, -3.0, 0]), 'axes': car_size},
            {'center': np.array([500, -3.0, 0]), 'axes': car_size}
        ]

    def build_integrator(self):
        x = self.x
        u = self.u
        dt = self.dt
        f = self.dynamics_f
        qf = self.cost_f

        for _ in range(self.m_steps_per_point):
            k1 = f(x=x, u=u)['x_dot']
            k2 = f(x=x + dt / 2 * k1, u=u)['x_dot']
            k3 = f(x=x + dt / 2 * k2, u=u)['x_dot']
            k4 = f(x=x + dt * k3, u=u)['x_dot']
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            q = qf(x=x, u=u)['q']

        return cs.Function('F', [self.x, self.u], [x, q], ['x0', 'p'], ['xf', 'qf'])

    def solve(self, x0, x_target):
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []
        J = 0

        Xk = cs.MX.sym('X_0', 6)
        w += [Xk]
        lbw += x0.tolist()
        ubw += x0.tolist()
        w0 += x0.tolist()

        for k in range(self.N):
            Uk = cs.MX.sym(f'U_{k}', 3)
            w += [Uk]
            lbw += self.min_u
            ubw += self.max_u
            w0 += [0.0, 0.0, 0.0]

            Fk = self.integrator(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            J += Fk['qf'] 

            Xk = cs.MX.sym(f'X_{k+1}', 6)
            w += [Xk]
            lbw += [-2000, -2000, 0, 50, -3.14, -np.deg2rad(20)]
            ubw += [2000, 2000, 2000, 100, 3.14, np.deg2rad(20)]
            w0 += [0, 0, 0, 70, 0, 0.0]

            g += [Xk_end - Xk]
            lbg += [0.0]*6
            ubg += [0.0]*6

            # Obstacle avoidance: enforce ellipsoid constraint at each step
            for obs in self.ellipsoids:
                xc, yc, zc = obs['center'] + self.car_velocity * k * self.dt
                a, b, c = obs['axes']
                ellipsoid_expr = ((Xk[0] - xc)/a)**2 + ((Xk[1] - yc)/b)**2 + ((Xk[2] - zc)/c)**2
                g.append(ellipsoid_expr)
                lbg.append(1.0)
                ubg.append(cs.inf)

        prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
        solver = cs.nlpsol('solver', 'ipopt', prob, {'ipopt.print_level': 1})

        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        return w_opt

#%% Main function to run the NMPC
if __name__ == "__main__":
    T = 20.0
    N = 20
    m_steps = 5
    q_diag = [0.0, 0.005, 0.01, 0, 20, 10]
    r_diag = [0.1, 0.1, 0.1]
    acc_limits = np.array([0.1, 3.0, np.pi/4])

    nmpc = LandingNMPC(T, N, m_steps, q_diag, r_diag, acc_limits)

    x0 = np.array([-200, 500, 100, 70, 0, -0.05])
    x_target = np.array([1000, 0, 0, 0, 0, 0])

    w_opt = nmpc.solve(x0, x_target)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    n_state = 6
    n_control = 3
    n_var_per_step = n_state + n_control
    n_steps = N

    x_opt = w_opt[0::n_var_per_step]
    y_opt = w_opt[1::n_var_per_step]
    h_opt = w_opt[2::n_var_per_step]
    V_opt = w_opt[3::n_var_per_step]
    psi_opt = w_opt[4::n_var_per_step]
    gamma_opt = w_opt[5::n_var_per_step]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_opt, y_opt, h_opt, marker='o')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('h (m)')
    ax.set_title('Optimal Landing Trajectory')

    plt.figure()
    plt.plot(x_opt, h_opt, marker='o')
    plt.xlabel('x (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude Profile During Landing')
    plt.grid(True)

    plt.figure()
    plt.plot(psi_opt, marker='o')
    plt.plot(gamma_opt, marker='o')
    plt.xlabel('N')
    plt.ylabel('Psi, Gamma (deg)')
    plt.title('Psi and Gamma Profiles During Landing')
    plt.legend(['Psi (rad)', 'Gamma (rad)'])
    plt.grid(True)

    plt.figure()
    plt.plot(V_opt, marker='o')
    plt.xlabel('N')
    plt.ylabel('Velocity (m/s)')
    plt.title('Velocity Profile During Landing')
    plt.grid(True)

    a1_opt = w_opt[n_state::n_var_per_step]
    a2_opt = w_opt[n_state+1::n_var_per_step]
    a3_opt = w_opt[n_state+2::n_var_per_step]

    tgrid = np.linspace(0, T, N)

    plt.figure()
    plt.plot(tgrid, a1_opt, label='n_x')
    plt.plot(tgrid, a2_opt, label='n_z')
    plt.plot(tgrid, a3_opt, label='mu (rad)')
    plt.xlabel('Time (s)')
    plt.ylabel('Controls')
    plt.legend()
    plt.title('Optimal Control Inputs')
    plt.grid(True)


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_opt, y_opt, h_opt, label='Trajectory')

    # === Visualize ellipsoidal obstacles ===
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    for obs in nmpc.ellipsoids:
        xc, yc, zc = obs['center']
        a, b, c = obs['axes']
        x = a * np.outer(np.cos(u), np.sin(v)) + xc
        y = b * np.outer(np.sin(u), np.sin(v)) + yc
        z = c * np.outer(np.ones_like(u), np.cos(v)) + zc
        ax.plot_surface(x, y, z, color='r', alpha=0.3, linewidth=0)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('h (m)')
    ax.set_title('Optimal Landing Trajectory with Obstacles')
    ax.legend()
    # Equal aspect ratio
    max_range = np.array([x_opt, y_opt, h_opt]).ptp(axis=1).max() / 2.0
    mid_x = (np.max(x_opt) + np.min(x_opt)) * 0.5
    mid_y = (np.max(y_opt) + np.min(y_opt)) * 0.5
    mid_h = (np.max(h_opt) + np.min(h_opt)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_h - max_range, mid_h + max_range)


    plt.show()