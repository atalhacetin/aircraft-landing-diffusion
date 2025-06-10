#!/usr/bin/env python3
# ------------------------------------------------------------------
# Fixed‑wing landing NMPC with ellipsoidal obstacle avoidance
# Compatible with acados template versions ≤2023‑10 and ≥2024‑xx
# ------------------------------------------------------------------
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ------------------------------------------------------------------
# 1)  Aircraft model
# ------------------------------------------------------------------
def build_model() -> AcadosModel:
    m = AcadosModel();  m.name = "landing"
    nx, nu = 6, 3
    x    = ca.MX.sym('x', nx)
    u    = ca.MX.sym('u', nu)
    xdot = ca.MX.sym('xdot', nx)

    g = 9.81
    V, psi, gamma = x[3], x[4], x[5]
    n_x, n_z, mu  = u[0], u[1], u[2]

    f = ca.vertcat(
        V*ca.cos(psi)*ca.cos(gamma),
        V*ca.sin(psi)*ca.cos(gamma),
        V*ca.sin(gamma),
        g*(n_x - ca.sin(gamma)),
        g*n_z/V * ca.sin(mu)/ca.cos(gamma),
        g/V * (n_z*ca.cos(mu) - ca.cos(gamma))
    )

    m.sym_x, m.sym_u, m.sym_xdot = x, u, xdot
    m.f_expl_expr, m.f_impl_expr = f, xdot - f
    m.x, m.u, m.xdot = x, u, xdot      # legacy aliases
    return m

# ------------------------------------------------------------------
# 2)  OCP description
# ------------------------------------------------------------------
def build_ocp(model: AcadosModel) -> AcadosOcp:
    ocp = AcadosOcp();       ocp.model = model
    nx, nu = 6, 3

    # ---------- horizon --------------------------------------------
    Tf, N = 2.0, 10
    ocp.solver_options.tf = Tf
    if hasattr(ocp.dims, "N_horizon"):
        ocp.dims.N_horizon = N
    else:
        ocp.dims.N = N                          # pre‑v0.3.3 template

    # integrator & QP solver
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.num_stages, ocp.solver_options.num_steps = 4, 5
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # supports one‑side

    # ---------- cost (linear LS) -----------------------------------
    Q = np.diag([0., 0.1, 0.01, 0.01, 20., 10.])
    R = np.diag([0.1, 0.1, 1])
    ny, ny_e = nx + nu, nx

    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W   = np.block([[Q,               np.zeros((nx, nu))],
                             [np.zeros((nu, nx)),         R     ]])
    ocp.cost.W_e = Q
    ocp.cost.Vx  = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu  = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.ny, ocp.cost.ny_e = ny, ny_e
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # ---------- box constraints ------------------------------------
    u_min = np.array([-0.5, -3., -np.pi/4]);  u_max = -u_min
    ocp.constraints.lbu, ocp.constraints.ubu = u_min, u_max
    ocp.constraints.idxbu = np.arange(nu)

    x_min = np.array([-2000, -2000, 0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000,  2000, 100,  np.pi,  np.deg2rad(20)])
    ocp.constraints.lbx, ocp.constraints.ubx = x_min, x_max
    ocp.constraints.idxbx = np.arange(nx)

    # --- initial‑state hard bounds (all states) --------------------
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = np.zeros(nx)   # dummy, overwritten at run‑time
    ocp.constraints.ubx_0   = np.zeros(nx)

    # ---------- ellipsoidal obstacles ------------------------------
    radii   = np.array([10., 10., 10.])
    centers = np.array([[400, 2.5, 0], [600,-3,0], [500,12,0],
                        [200, 6 , 0], [100,-3,0], [500,-3,0]])
    phi = [((ocp.model.sym_x[0]-cx)/radii[0])**2 +
           ((ocp.model.sym_x[1]-cy)/radii[1])**2 +
           ((ocp.model.sym_x[2]-cz)/radii[2])**2 for cx,cy,cz in centers]

    expr_h = ca.vertcat(*phi)
    ocp.model.expr_h     = expr_h   # new name
    ocp.model.con_h_expr = expr_h   # legacy name
    n_h = expr_h.shape[0]
    ocp.dims.nh          = int(n_h)
    ocp.constraints.lh   = np.ones(n_h)
    ocp.constraints.uh   = 2e3*np.ones(n_h)

    ocp.constraints.idxsh = np.arange(n_h, dtype=int)

    # ---------- slack-penalty weights ---------------------------------
    # Linear term  z . s   (acts like an L1 penalty, fast to tune)
    # Quadratic    Z . s^2  (acts like an L2 penalty)
    #
    # Here we use a *pure quadratic* penalty:
    slack_weight = 1e3                          # <-- tune! (larger => harder)
    ocp.cost.zl = np.zeros(n_h)                 # no L1 part on lower side
    ocp.cost.zu = np.zeros(n_h)                 # (upper side is unused)
    ocp.cost.Zl = slack_weight * np.ones(n_h)
    ocp.cost.Zu = np.zeros(n_h)                 # keep upper inactive


    ocp.code_export_directory = "c_generated_code"
    return ocp

# ------------------------------------------------------------------
# 3)  Build solver
# ------------------------------------------------------------------
def build_solver(ocp: AcadosOcp) -> AcadosOcpSolver:
    json = "landing_ocp.json"
    AcadosOcpSolver.generate(ocp, json_file=json, verbose=False)
    return AcadosOcpSolver(ocp, json_file=json)

# ------------------------------------------------------------------
# 4)  Main
# ------------------------------------------------------------------

def main_mpc():
    import time
    import matplotlib.pyplot as plt

    # Simulate one step forward using RK4
    def aircraft_dynamics_np(x, u):
        g = 9.81
        V, psi, gamma = x[3], x[4], x[5]
        n_x, n_z, mu = u
        dx = np.array([
            V * np.cos(psi) * np.cos(gamma),
            V * np.sin(psi) * np.cos(gamma),
            V * np.sin(gamma),
            g * (n_x - np.sin(gamma)),
            g * n_z / V * np.sin(mu) / np.cos(gamma),
            g / V * (n_z * np.cos(mu) - np.cos(gamma))
        ])
        return dx

    def rk4(x, u, dt):
        k1 = aircraft_dynamics_np(x, u)
        k2 = aircraft_dynamics_np(x + dt/2 * k1, u)
        k3 = aircraft_dynamics_np(x + dt/2 * k2, u)
        k4 = aircraft_dynamics_np(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

    # === Build model, ocp, and solver ===
    model  = build_model()
    ocp    = build_ocp(model)
    solver = build_solver(ocp)

    # === MPC Setup ===
    Tf, N = ocp.solver_options.tf, ocp.dims.N
    dt = Tf / N
    sim_time = 20.0                       # [s] total MPC run time
    sim_steps = int(sim_time / dt)

    # Initial and target states
    x0 = np.array([-200., 100., 100., 70., 0., -0.05])
    xdes = np.array([1000., 0., 0., 50., 0., 0.])
    nx, nu = 6, 3

    # Allocate logs
    X_log = np.zeros((sim_steps+1, nx))
    U_log = np.zeros((sim_steps, nu))
    T_log = np.zeros(sim_steps)
    X_log[0] = x0.copy()

    # === MPC Loop ===
    for sim_k in range(sim_steps):
        print(f"--- MPC Step {sim_k} ---")

        # Set current state as initial constraint
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        # Update references
        for k in range(N):
            solver.set(k, "y_ref", np.hstack((xdes, np.zeros(nu))))
        solver.set(N, "y_ref", xdes)

        # Warm start
        for k in range(N):
            solver.set(k, "x", x0)
            solver.set(k, "u", np.zeros(nu))
        solver.set(N, "x", x0)

        # Solve
        t0 = time.time()
        solver.solve()
        t1 = time.time()
        print(f"Solver time: {(t1 - t0)*1000:.2f} ms")
        if solver.get_status() != 0:
            print("Solver failed with status:", solver.get_status())

            

        # Extract optimal control
        u0 = solver.get(0, "u")
        U_log[sim_k] = u0
        T_log[sim_k] = sim_k * dt

        # Advance the state
        x0 = rk4(x0, u0, dt)
        X_log[sim_k+1] = x0

    # === Save and Plot ===
    np.savez("mpc_landing_sim.npz", X=X_log, U=U_log, T=T_log)

    # Plot 3D trajectory
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_log[:,0], X_log[:,1], X_log[:,2], marker='o')
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("h")
    ax.set_title("MPC aircraft landing trajectory")


    t_state   = np.arange(sim_steps + 1) * dt     # time grid for states
    t_control = np.arange(sim_steps) * dt         # time grid for inputs

    # ---- 1) all states -------------------------------------------------
    state_labels = [r"$x$ [m]",
                    r"$y$ [m]",
                    r"$h$ [m]",
                    r"$V$ [m/s]",
                    r"$\psi$ [rad]",
                    r"$\gamma$ [rad]"]

    fig_s, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    axs = axs.flatten()

    for i in range(nx):
        axs[i].plot(t_state, X_log[:, i], lw=1.5)
        axs[i].set_ylabel(state_labels[i])
        axs[i].grid(True)

    axs[-1].set_xlabel("time [s]")
    fig_s.suptitle("States along closed-loop trajectory", fontsize=14)
    fig_s.tight_layout(rect=[0, 0.03, 1, 0.97])

    # ---- 2) all control inputs ----------------------------------------
    input_labels = [r"$n_x$ [–]",
                    r"$n_z$ [–]",
                    r"$\mu$ [rad]"]

    fig_u, axu = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    for i in range(nu):
        axu[i].step(t_control, U_log[:, i], where="post", lw=1.5)
        axu[i].set_ylabel(input_labels[i])
        axu[i].grid(True)

    axu[-1].set_xlabel("time [s]")
    fig_u.suptitle("Control inputs applied by MPC", fontsize=14)
    fig_u.tight_layout(rect=[0, 0.03, 1, 0.97])

    plt.show()

def main_ocp():
    import time
    model  = build_model()
    ocp    = build_ocp(model)
    solver = build_solver(ocp)

    x0   = np.array([-1000., 100., 500., 70., 0., -0.05])
    xdes = np.array([1000., 0., 0., 0., 0., 0.])

    # horizon length (old vs new name)
    N = ocp.dims.N if hasattr(ocp.dims, "N") else ocp.dims.N_horizon
    nx, nu = 6, 3

    # hard‑fix initial state
    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    # running & terminal references
    for k in range(N):
        solver.set(k, "y_ref", np.hstack((xdes, np.zeros(nu))))  
        solver.set(k, "x", x0)          # warm start
        solver.set(k, "u", np.zeros(nu))
    solver.set(N, "y_ref", xdes)        # <‑‑ terminal stage uses same key
    solver.set(N, "x", x0)

    t_start = time.time()
    solver.solve()
    t_end = time.time()
    print("Solver time = %.3f ms" % ((t_end - t_start) * 1000))
    print("Optimal cost =", solver.get_cost())

    X = np.vstack([solver.get(k, "x") for k in range(N+1)])
    print(X)
    U = np.vstack([solver.get(k, "u") for k in range(N)])
    np.savez("landing_solution.npz", X=X, U=U)
    print("Saved landing_solution.npz")

    import matplotlib.pyplot as plt
    N = U.shape[0]
    Tf = 20.0
    dt = Tf / N
    t = np.arange(N) * dt

    # State components
    x, y, h, V, psi, gamma = X.T
    n_x, n_z, mu = U.T

    # 1) 3‑D trajectory --------------------------------------------------
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot(x, y, h, marker="o")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.set_zlabel("h [m]")
    ax1.set_title("Optimal landing trajectory (acados)")

    # 2) altitude vs ground‑track ---------------------------------------
    fig2 = plt.figure()
    plt.plot(x, h, marker="o")
    plt.xlabel("x [m]")
    plt.ylabel("Altitude h [m]")
    plt.title("Altitude profile along x")
    plt.grid(True)

    # 3) control inputs over time ---------------------------------------
    fig3 = plt.figure()
    plt.step(t, n_x, label="n_x")
    plt.step(t, n_z, label="n_z")
    plt.step(t, mu, label="mu [rad]")
    plt.xlabel("time [s]")
    plt.ylabel("Control inputs")
    plt.title("Applied controls vs time")
    plt.legend()
    plt.grid(True)

    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.plot(x, y, h, marker="o", label='trajectory')

    # -------- ellipsoids ------------ (new)
    radii   = np.array([10., 10., 10.])
    centers = np.array([[400, 2.5, 0], [600,-3,0], [500,12,0],
                        [200, 6 , 0], [100,-3,0], [500,-3,0]])
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0,     np.pi, 12)
    uu, vv = np.meshgrid(u, v)
    for cx, cy, cz in centers:
        xs = radii[0]*np.cos(uu)*np.sin(vv) + cx
        ys = radii[1]*np.sin(uu)*np.sin(vv) + cy
        zs = radii[2]*np.cos(vv)            + cz
        ax1.plot_surface(xs, ys, zs, color='crimson',
                        alpha=0.3, linewidth=0)
    # --------------------------------

    ax1.set_xlabel("x [m]"); ax1.set_ylabel("y [m]"); ax1.set_zlabel("h [m]")
    ax1.set_title("Optimal landing trajectory with obstacles")
    ax1.legend()

    plt.show()


if __name__ == "__main__":
    main_mpc()
    #main_ocp()  # Uncomment to run the OCP directly
