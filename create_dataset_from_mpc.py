#!/usr/bin/env python3
import numpy as np
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ------------------------------------------------------------------
# 1)  Aircraft model (from MPC code)
# ------------------------------------------------------------------
def build_model() -> AcadosModel:
    m = AcadosModel();  m.name = "landing"
    nx, nu = 6, 3
    x    = cs.MX.sym('x', nx)
    u    = cs.MX.sym('u', nu)
    xdot = cs.MX.sym('xdot', nx)
    p = cs.MX.sym("p", 3)          # p = [deltax, deltay, deltah]  (only deltax used)
    m.p = p                        # store in model 


    g = 9.81
    V, psi, gamma = x[3], x[4], x[5]
    n_x, n_z, mu  = u[0], u[1], u[2]

    f = cs.vertcat(
        V*cs.cos(psi)*cs.cos(gamma),
        V*cs.sin(psi)*cs.cos(gamma),
        V*cs.sin(gamma),
        g*(n_x - cs.sin(gamma)),
        g*n_z/V * cs.sin(mu)/cs.cos(gamma),
        g/V * (n_z*cs.cos(mu) - cs.cos(gamma))
    )

    m.sym_x, m.sym_u, m.sym_xdot = x, u, xdot
    m.f_expl_expr, m.f_impl_expr = f, xdot - f
    m.x, m.u, m.xdot = x, u, xdot      # legacy aliases
    return m

# ------------------------------------------------------------------
# 2)  OCP description (from MPC code)
# ------------------------------------------------------------------
def build_ocp(model: AcadosModel) -> AcadosOcp:
    ocp = AcadosOcp();       ocp.model = model
    nx, nu = 6, 3

    # ---------- horizon --------------------------------------------
    Tf, N = 1.0, 10
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
    Q = np.diag([0., 0.1, 0.01, 1.0, 20., 10.])
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
    ocp.dims.np = 3                # MUST match len(model.p)
    ocp.parameter_values = np.zeros((3, 1))   # or simply  np.zeros(3)
    
    # ---------- moving ellipsoids ----------------------------------
    p = model.p                    # ← fetch the parameter symbol here

    # Define base obstacle centers and radii (these are fixed, then shifted by 'p')
    # Use 6 obstacles as in the first script for consistency if desired
    # For simplicity, we'll keep the radii fixed and small for now
    global_radii   = np.array([10., 10., 10.])
    global_centers = np.array([[400, 2.5, 0], [600,-3,0], [500,12,0],
                               [200, 6 , 0], [100,-3,0], [500,-3,0]])

    phi = []
    for cx, cy, cz in global_centers:
        cxi = cx + p[0]            # shift in +x direction
        # Note: the MPC code only shifts in x. If you want y and z shifts, you'd modify this.
        phi.append(((model.x[0]-cxi)/global_radii[0])**2 +
                   ((model.x[1]-cy )/global_radii[1])**2 +
                   ((model.x[2]-cz )/global_radii[2])**2)

    expr_h = cs.vertcat(*phi)
    ocp.model.expr_h     = expr_h   # new name
    ocp.model.con_h_expr = expr_h   # legacy name
    n_h = expr_h.shape[0]
    ocp.dims.nh          = int(n_h)
    ocp.constraints.lh   = np.ones(n_h)
    ocp.constraints.uh   = 2e3*np.ones(n_h)

    ocp.constraints.idxsh = np.arange(n_h, dtype=int)

    # ---------- slack-penalty weights ---------------------------------
    slack_weight = 1e3                          # <-- tune! (larger => harder)
    ocp.cost.zl = np.zeros(n_h)                 # no L1 part on lower side
    ocp.cost.zu = np.zeros(n_h)                 # (upper side is unused)
    ocp.cost.Zl = slack_weight * np.ones(n_h)
    ocp.cost.Zu = np.zeros(n_h)                 # keep upper inactive


    ocp.code_export_directory = "c_generated_code_mpc_dataset"
    return ocp

# ------------------------------------------------------------------
# 3)  Build solver (from MPC code)
# ------------------------------------------------------------------
def build_solver(ocp: AcadosOcp) -> AcadosOcpSolver:
    json = "landing_ocp_mpc_dataset.json"
    ocp.solver_options.nlp_solver_max_iter = 400    # ← outer SQP iterations
    ocp.solver_options.qp_solver_iter_max  = 100    # ← HPIPM Riccati sweeps
    AcadosOcpSolver.generate(ocp, json_file=json, verbose=False)
    return AcadosOcpSolver(ocp, json_file=json)

# -----------------------------------------------------------------------------
# Collision helper (adapted for shifted obstacles)
# -----------------------------------------------------------------------------
def trajectory_collides(X, base_centers, base_radii, current_p):
    pos = X[:, :3]
    # Apply the shift to the base centers
    shifted_centers = base_centers + current_p[0] # Assuming only x-shift from p[0]
    
    for c_base, r_base in zip(shifted_centers, base_radii):
        # The radii are global_radii defined in build_ocp
        d2 = ((pos - c_base)/r_base)**2
        if np.any(np.sum(d2,axis=1) <= 1.0):
            return True
    return False

# ------------------------------------------------------------------
# Simulate one step forward using RK4 (from MPC code)
# ------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# MAIN: compile once, then just swap in new p = [centers,radii]
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)  # For reproducibility
    # 1) Build the model, OCP, and solver
    model  = build_model()
    ocp    = build_ocp(model) # OCP now uses the 'p' parameter for obstacle shifting
    solver = build_solver(ocp)

    nx, nu = 6, 3
    N = ocp.dims.N_horizon if hasattr(ocp.dims, "N_horizon") else ocp.dims.N
    Tf = ocp.solver_options.tf
    dt = Tf / N
    sim_time = 25.0 # Total simulation time for each trajectory
    sim_steps_per_run = int(sim_time / dt) # Number of MPC steps per run

    Nrun = 10 # Number of full trajectories to generate

    # pre‐allocate lists
    X_list, U_list, C_list, R_list, CarVel_list = [], [], [], [], [] # <--- Add CarVel_list here


    # Get the fixed obstacle parameters from the OCP build
    base_radii = np.array([10., 10., 10.])
    base_centers = np.array([[400, 2.5, 0], [600,-3,0], [500,12,0],
                             [200, 6 , 0], [100,-3,0], [500,-3,0]])


    x_min_initial = np.array([-500, -200,    100,  50, -np.pi/4, -np.deg2rad(20)])
    x_max_initial = np.array([ 100,  200, 500, 100,  np.pi/4,  np.deg2rad(20)])
    
    xdes = np.array([1000., 0., 0., 50., 0., 0.])
    car_vel_options = [0, 10, 20, 30, 40] # Different car velocities to vary scenarios

    for i in range(Nrun):
        # 1) random x0
        x0_initial = np.random.uniform(x_min_initial, x_max_initial)
        x0_current = x0_initial.copy()

        # Randomly choose a car velocity for this run
        car_vel = np.random.choice(car_vel_options)

        # Allocate logs for this specific run
        X_run_log = np.zeros((sim_steps_per_run + 1, nx))
        U_run_log = np.zeros((sim_steps_per_run, nu))
        X_run_log[0] = x0_current.copy()

        actual_sim_steps = 0

        print(f"\n--- Starting Run {i+1} with car_vel={car_vel} m/s ---")

        for sim_k in range(sim_steps_per_run):
            # Set current state as initial constraint
            solver.set(0, "lbx", x0_current)
            solver.set(0, "ubx", x0_current)

            # Update references (target remains fixed)
            for k in range(N):
                solver.set(k, "y_ref", np.hstack((xdes, np.zeros(nu))))
            solver.set(N, "y_ref", xdes)
            
            # Update obstacle parameter 'p' at each stage for MPC's prediction horizon
            for k in range(N):
                t_stage = k * dt
                # The 'p' parameter shifts the obstacles based on time and car velocity
                # We assume p[0] is the x-shift
                shift_x = car_vel * (sim_k * dt + t_stage) 
                solver.set(k, "p", np.array([shift_x, 0.0, 0.0]))
            
            # Set 'p' for the terminal stage (N) as well
            terminal_shift_x = car_vel * (sim_k * dt + Tf)
            solver.set(N, "p", np.array([terminal_shift_x, 0.0, 0.0]))


            # Warm start (optional, but good practice for MPC)
            for k in range(N):
                solver.set(k, "x", x0_current)
                solver.set(k, "u", np.zeros(nu)) # Or from previous solution if available
            solver.set(N, "x", x0_current)

            # Solve the OCP
            try:
                solver.solve()
            except Exception as e:
                print(f"Run {i+1}, MPC Step {sim_k}: solver failed with error: {e}, skipping to next run.")
                X_run_log = X_run_log[:actual_sim_steps + 1]
                U_run_log = U_run_log[:actual_sim_steps]
                break # Exit this inner simulation loop and go to the next Nrun iteration

            # if solver.get_status() != 0:
            #     print(f"Run {i+1}, MPC Step {sim_k}: Solver failed with status: {solver.get_status()}, skipping to next run.")
            #     X_run_log = X_run_log[:actual_sim_steps + 1]
            #     U_run_log = U_run_log[:actual_sim_steps]
            #     break # Exit this inner simulation loop and go to the next Nrun iteration

            # Extract optimal control and advance the state
            u0 = solver.get(0, "u")
            U_run_log[sim_k] = u0
            x0_current = rk4(x0_current, u0, dt)
            X_run_log[sim_k+1] = x0_current
            actual_sim_steps = sim_k + 1

            # Check if the aircraft has crashed (optional, but useful for filtering)
            if x0_current[2] < 0:
                print(f"Run {i+1}, MPC Step {sim_k}: Aircraft crashed (height < 0), stopping simulation for this run.")
                break

        # After a full simulation run, check the final conditions and collision
        X_simulated_trajectory = X_run_log[:actual_sim_steps + 1]
        U_simulated_trajectory = U_run_log[:actual_sim_steps]

        # Calculate the final p parameter for the entire trajectory's obstacle check
        # This considers the total shift of the obstacles over the full sim_time
        final_shift_x_for_collision_check = car_vel * (actual_sim_steps * dt)
        current_p_for_collision_check = np.array([final_shift_x_for_collision_check, 0.0, 0.0])


        yf_final = X_simulated_trajectory[-1, 1]
        h_final = X_simulated_trajectory[-1, 2]

        # Check for successful landing conditions and no collision
        # The collision check needs to consider the dynamic obstacle positions
        if abs(yf_final) <= 12.0 and h_final <= 0.5 and not(trajectory_collides(X_simulated_trajectory, base_centers, base_radii, current_p_for_collision_check)):
            print(f"Run {i+1}: SUCCESS! y_final = {yf_final:.2f} m, h_final = {h_final:.2f} m")
            X_list.append(X_simulated_trajectory)
            U_list.append(U_simulated_trajectory)
            C_list.append(base_centers) 
            R_list.append(base_radii)
            CarVel_list.append(car_vel) # <--- Add this line to save car_vel for successful runs
        else:
            print(f"Run {i+1}: FAILED (y_final={yf_final:.2f}, h_final={h_final:.2f}, collision={trajectory_collides(X_simulated_trajectory, base_centers, base_radii, current_p_for_collision_check)})")


    # 6) save full dataset
    if len(X_list) > 0:
        np.savez_compressed(
            "landing_mpc_dataset.npz",
            X         = np.array(X_list, dtype=object),
            U         = np.array(U_list, dtype=object),
            centers   = np.array(C_list, dtype=object),
            radii     = np.array(R_list, dtype=object),
            car_velocities = np.array(CarVel_list) # <--- Make sure this line is present
        )
        print(f"Saved {len(X_list)} successful scenarios → landing_mpc_dataset.npz")
    else:
        print("No successful scenarios generated to save.")