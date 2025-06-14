import numpy as np
import casadi as cs
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import os
import shutil # Import shutil for directory removal
import multiprocessing
from functools import partial # To pass fixed arguments to pool.map
import uuid # For generating unique directory names

# ------------------------------------------------------------------
# 1) Aircraft model (from MPC code)
# ------------------------------------------------------------------
def build_model(model_name="landing") -> AcadosModel:
    """
    Builds the Acados model for the fixed-wing aircraft dynamics.
    The model defines the state (x), control input (u), parameters (p),
    and the continuous-time dynamics (f_expl_expr).

    Args:
        model_name (str): The name for the AcadosModel.

    Returns:
        AcadosModel: The configured Acados model.
    """
    m = AcadosModel();  m.name = model_name
    nx, nu = 6, 3
    
    # Define symbolic state, control, and parameter variables
    x    = cs.MX.sym('x', nx)      # State vector: [x_pos, y_pos, alt, airspeed, heading, flight_path_angle]
    u    = cs.MX.sym('u', nu)      # Control input: [n_x, n_z, bank_angle]
    xdot = cs.MX.sym('xdot', nx)   # State derivatives
    p = cs.MX.sym("p", 3)          # Parameters: [obstacle_x_shift, obstacle_y_shift, obstacle_z_shift]
    m.p = p                        # Store parameters in the model

    g = 9.81 # Acceleration due to gravity
    
    # Extract state variables for readability
    V, psi, gamma = x[3], x[4], x[5] # Airspeed, heading, flight path angle
    
    # Extract control inputs for readability
    n_x, n_z, mu  = u[0], u[1], u[2] # Longitudinal load factor, vertical load factor, bank angle

    # Define the continuous-time nonlinear dynamics
    f = cs.vertcat(
        V*cs.cos(psi)*cs.cos(gamma),             # dx_pos/dt
        V*cs.sin(psi)*cs.cos(gamma),             # dy_pos/dt
        V*cs.sin(gamma),                         # d_alt/dt
        g*(n_x - cs.sin(gamma)),                 # d_airspeed/dt
        g*n_z/V * cs.sin(mu)/cs.cos(gamma),      # d_heading/dt
        g/V * (n_z*cs.cos(mu) - cs.cos(gamma))   # d_flight_path_angle/dt
    )

    # Set symbolic variables and explicit/implicit dynamics
    m.sym_x, m.sym_u, m.sym_xdot = x, u, xdot
    m.f_expl_expr, m.f_impl_expr = f, xdot - f
    
    # Legacy aliases (for compatibility with older Acados versions)
    m.x, m.u, m.xdot = x, u, xdot
    return m

# ------------------------------------------------------------------
# 2) OCP description (from MPC code)
# ------------------------------------------------------------------
def build_ocp(model: AcadosModel, current_base_centers: np.ndarray, base_radii: np.ndarray, highway_slope_a: float = 0.0, code_export_dir: str = "c_generated_code_mpc_dataset") -> AcadosOcp:
    """
    Builds the Acados Optimal Control Problem (OCP) for the aircraft landing.
    This includes defining the prediction horizon, cost function, constraints,
    and obstacle avoidance.

    Args:
        model (AcadosModel): The Acados model of the aircraft.
        current_base_centers (np.ndarray): Nx3 array of base [x,y,z] coordinates for obstacles.
        base_radii (np.ndarray): 1x3 array of [rx,ry,rz] radii for ellipsoidal obstacles.
        highway_slope_a (float): The 'a' parameter for the highway line y=ax.
        code_export_dir (str): The directory where Acados C code will be exported.

    Returns:
        AcadosOcp: The configured Acados OCP.
    """
    ocp = AcadosOcp();       ocp.model = model
    nx, nu = 6, 3

    # ---------- Horizon definition ---------------------------------
    Tf, N = 1.0, 10 # Prediction horizon duration and number of stages
    ocp.solver_options.tf = Tf
    ocp.dims.N = N

    # Integrator and QP solver options
    ocp.solver_options.integrator_type = "ERK" # Explicit Runge-Kutta
    ocp.solver_options.num_stages, ocp.solver_options.num_steps = 4, 5 # Integrator steps
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM" # QP solver
    ocp.solver_options.print_level = 0 # Suppress Acados console output

    # ---------- Cost function (Linear Least Squares) --------------
    # Calculate Q matrix considering highway slope for positional penalties
    alpha = np.arctan(highway_slope_a) # Highway orientation angle
    q_parallel = 0.0 # Weight for deviations parallel to the road
    q_normal = 0.1   # Weight for deviations perpendicular to the road

    q00 = q_parallel * np.cos(alpha)**2 + q_normal * np.sin(alpha)**2
    q01 = (q_parallel - q_normal) * np.cos(alpha) * np.sin(alpha)
    q10 = q01
    q11 = q_parallel * np.sin(alpha)**2 + q_normal * np.cos(alpha)**2

    Q = np.diag([0.0, 0.0, 0.01, 1.0, 20., 10.]) # Diagonal weights for states
    Q[0,0] = q00 # Override for x_pos penalty
    Q[0,1] = q01 # Override for x_pos - y_pos coupling penalty
    Q[1,0] = q10 # Override for y_pos - x_pos coupling penalty
    Q[1,1] = q11 # Override for y_pos penalty

    R = np.diag([0.1, 0.1, 1]) # Weights for control inputs
    ny, ny_e = nx + nu, nx # Dimension of combined state-control vector and terminal state vector

    ocp.cost.cost_type   = "LINEAR_LS"     # Running cost type
    ocp.cost.cost_type_e = "LINEAR_LS"     # Terminal cost type
    ocp.cost.W   = np.block([[Q,               np.zeros((nx, nu))],
                             [np.zeros((nu, nx)),         R     ]]) # Running cost weight matrix
    ocp.cost.W_e = Q                       # Terminal cost weight matrix
    
    # Selection matrices for state and control in combined vector
    ocp.cost.Vx  = np.vstack((np.eye(nx), np.zeros((nu, nx))))
    ocp.cost.Vu  = np.vstack((np.zeros((nx, nu)), np.eye(nu)))
    ocp.cost.Vx_e = np.eye(nx) # Selection matrix for terminal state
    
    ocp.cost.ny, ocp.cost.ny_e = ny, ny_e # Set dimensions
    ocp.cost.yref   = np.zeros(ny)     # Reference for running cost (set at runtime)
    ocp.cost.yref_e = np.zeros(ny_e)   # Reference for terminal cost (set at runtime)

    # ---------- Box constraints on control inputs ------------------
    u_min = np.array([-0.5, -3., -np.pi/4]);  u_max = -u_min # Min/Max for [n_x, n_z, bank_angle]
    ocp.constraints.lbu, ocp.constraints.ubu = u_min, u_max
    ocp.constraints.idxbu = np.arange(nu) # Indices of constrained control inputs

    # ---------- Box constraints on states --------------------------
    x_min = np.array([-2000, -2000, 0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000,  2000, 100,  np.pi,  np.deg2rad(20)])
    ocp.constraints.lbx, ocp.constraints.ubx = x_min, x_max
    ocp.constraints.idxbx = np.arange(nx) # Indices of constrained states

    # --- Initial state hard bounds (all states) --------------------
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = np.zeros(nx)   # Dummy values, overwritten at runtime
    ocp.constraints.ubx_0   = np.zeros(nx)

    # ---------- Ellipsoidal Obstacles (Nonlinear Constraints) ------
    ocp.dims.np = 3 # Number of parameters for obstacles (shifts in x,y,z)
    ocp.parameter_values = np.zeros(ocp.dims.np) # Initial parameter values (dummy)
    
    p = model.p # Fetch the parameter symbol from the model
    phi = [] # List to store obstacle avoidance expressions

    # Generate an obstacle avoidance constraint for each obstacle
    for cx_base, cy_base, cz_base in current_base_centers:
        # Obstacle center shifted by parameters p[0], p[1] (x, y shifts)
        # Note: assuming p[0] is x-shift, p[1] is y-shift. If p[0] is distance along highway,
        # these calculations would need to be adapted based on highway_slope_a.
        cxi = cx_base + p[0]
        cyi = cy_base + p[1] # p[1] is currently always 0 in the main loop, but included for completeness

        # Ellipsoid inequality constraint: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 >= 1
        phi.append(((model.x[0]-cxi)/base_radii[0])**2 +
                   ((model.x[1]-cyi)/base_radii[1])**2 +
                   ((model.x[2]-cz_base)/base_radii[2])**2)

    expr_h = cs.vertcat(*phi) # Vertically stack all obstacle expressions
    ocp.model.expr_h     = expr_h   # Nonlinear constraint expression
    ocp.model.con_h_expr = expr_h   # Legacy name

    n_h = expr_h.shape[0] # Number of nonlinear constraints (one per obstacle)
    ocp.dims.nh          = int(n_h)
    ocp.constraints.lh   = np.ones(n_h)     # Lower bound: must be >= 1 (outside ellipsoid)
    ocp.constraints.uh   = 2e3*np.ones(n_h) # Upper bound: large value to allow soft constraint

    ocp.constraints.idxsh = np.arange(n_h, dtype=int) # Indices for soft constraints

    # ---------- Slack penalty weights ------------------------------
    slack_weight = 1e3 # Weight for penalizing violations of soft constraints
    ocp.cost.zl = np.zeros(n_h) # L1 part for lower side slack (unused here)
    ocp.cost.zu = np.zeros(n_h) # L1 part for upper side slack (unused here)
    ocp.cost.Zl = slack_weight * np.ones(n_h) # Quadratic part for lower side slack
    ocp.cost.Zu = np.zeros(n_h) # Quadratic part for upper side slack (unused)

    ocp.code_export_directory = code_export_dir # Set the code export directory
    return ocp

# ------------------------------------------------------------------
# 3) Build solver (from MPC code)
# ------------------------------------------------------------------
def build_solver(ocp: AcadosOcp, json_file: str) -> AcadosOcpSolver:
    """
    Builds and compiles the Acados OCP solver.

    Args:
        ocp (AcadosOcp): The configured Acados OCP.
        json_file (str): The JSON file path for solver options.

    Returns:
        AcadosOcpSolver: The compiled Acados OCP solver.
    """
    ocp.solver_options.nlp_solver_max_iter = 400    # Max SQP iterations
    ocp.solver_options.qp_solver_iter_max  = 100    # Max HPIPM Riccati sweeps
    
    # Generate C code for this specific OCP instance in its dedicated directory
    AcadosOcpSolver.generate(ocp, json_file=json_file, verbose=False)
    
    # Create solver instance from the generated JSON/code
    return AcadosOcpSolver(ocp, json_file=json_file)

# -----------------------------------------------------------------------------
# Collision helper (adapted for shifted obstacles)
# -----------------------------------------------------------------------------
def trajectory_collides(X_trajectory, base_centers, base_radii, total_p_shift):
    """
    Checks if any point in the aircraft trajectory collides with any obstacle.

    Args:
        X_trajectory (np.ndarray): Nx6 array of aircraft states over time.
        base_centers (np.ndarray): Nx3 array of base [x,y,z] coordinates for obstacles.
        base_radii (np.ndarray): 1x3 array of [rx,ry,rz] radii for ellipsoidal obstacles.
        total_p_shift (np.ndarray): 1x3 array of total [x_shift, y_shift, z_shift] for obstacles.

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    pos = X_trajectory[:, :3] # Extract positions from trajectory
    
    # Expand total_p_shift to match the number of obstacles for component-wise addition
    p_expanded = np.tile(total_p_shift[:base_centers.shape[1]], (base_centers.shape[0], 1))
    
    # Calculate the current (shifted) centers of the obstacles
    shifted_centers = base_centers + p_expanded
    
    collision_detected = False
    for i in range(shifted_centers.shape[0]):
        c_shifted = shifted_centers[i]
        r_obstacle = base_radii # Radii are assumed uniform for all obstacles here
        
        # Calculate squared normalized distance for all trajectory points to current obstacle
        # This checks if any point is inside or on the boundary of the ellipsoid (<= 1.0)
        d2 = ((pos[:, 0] - c_shifted[0]) / r_obstacle[0])**2 + \
             ((pos[:, 1] - c_shifted[1]) / r_obstacle[1])**2 + \
             ((pos[:, 2] - c_shifted[2]) / r_obstacle[2])**2
        
        if np.any(d2 <= 1.0):
            collision_detected = True
            break # Collision detected, no need to check further obstacles
    return collision_detected

# ------------------------------------------------------------------
# Simulate one step forward using RK4 (from MPC code)
# ------------------------------------------------------------------
def aircraft_dynamics_np(x, u):
    """
    Calculates the derivatives of the aircraft state based on current state and control.
    Uses numpy for numerical computation.

    Args:
        x (np.ndarray): Current aircraft state vector.
        u (np.ndarray): Current control input vector.

    Returns:
        np.ndarray: State derivatives.
    """
    g = 9.81
    V, psi, gamma = x[3], x[4], x[5]
    n_x, n_z, mu = u
    dx = np.array([
        V*np.cos(psi)*np.cos(gamma),
        V*np.sin(psi)*np.cos(gamma),
        V*np.sin(gamma),
        g*(n_x - np.sin(gamma)),
        g*n_z/V * np.sin(mu)/np.cos(gamma),
        g/V * (n_z*np.cos(mu) - np.cos(gamma))
    ])
    return dx

def rk4(x, u, dt):
    """
    Performs one step of Runge-Kutta 4th order integration for aircraft dynamics.

    Args:
        x (np.ndarray): Current aircraft state.
        u (np.ndarray): Current control input.
        dt (float): Time step for integration.

    Returns:
        np.ndarray: Next aircraft state.
    """
    k1 = aircraft_dynamics_np(x, u)
    k2 = aircraft_dynamics_np(x + dt/2 * k1, u)
    k3 = aircraft_dynamics_np(x + dt/2 * k2, u)
    k4 = aircraft_dynamics_np(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)

# -----------------------------------------------------------------------------
# Worker function for parallel simulation
# -----------------------------------------------------------------------------
def simulate_single_run(run_idx, params):
    """
    Function to simulate a single aircraft landing trajectory.
    This function will be executed in parallel processes.

    Args:
        run_idx (int): Index of the current simulation run.
        params (dict): A dictionary containing all necessary constant parameters
                       for the simulation (ranges, dimensions, fixed values).

    Returns:
        tuple or None: A tuple containing (X_trajectory, U_trajectory, base_centers, base_radii, car_velocity)
                       if the run is successful, otherwise None.
    """
    # Set seed for reproducibility within each process (based on process ID and run index)
    np.random.seed(os.getpid() + run_idx)

    nx, nu = params['nx'], params['nu']
    sim_time = params['sim_time']
    Tf, N = params['Tf'], params['N']
    dt = Tf / N
    sim_steps_per_run = int(sim_time / dt)

    # --- Generate unique directory and JSON file for this process ---
    # Using uuid for uniqueness to avoid conflicts even if process IDs overlap across runs
    unique_id = str(uuid.uuid4())[:8] # Use a short unique ID
    current_code_export_dir = os.path.join(params['base_code_export_directory'], f"run_{run_idx}_{unique_id}")
    current_json_file_name = f"landing_ocp_mpc_dataset_run_{run_idx}_{unique_id}.json"
    current_json_file_path = os.path.join(current_code_export_dir, current_json_file_name) # JSON inside the code dir

    # Ensure the unique directory exists and is clean
    if os.path.exists(current_code_export_dir):
        shutil.rmtree(current_code_export_dir)
    os.makedirs(current_code_export_dir, exist_ok=True)


    # --- Build model, OCP, and solver for this specific run ---
    # The number of obstacles MUST match NUM_OBSTACLES_PER_RUN for proper code generation
    current_obstacle_centers = []
    for _ in range(params['NUM_OBSTACLES_PER_RUN']):
        obstacle_x = np.random.uniform(params['OBSTACLE_X_MIN'], params['OBSTACLE_X_MAX'])
        obstacle_y = np.random.uniform(params['OBSTACLE_Y_MIN'], params['OBSTACLE_Y_MAX'])
        current_obstacle_centers.append([obstacle_x, obstacle_y, 0])
    current_base_centers = np.array(current_obstacle_centers)

    model = build_model(params['model_name'])
    ocp = build_ocp(model, current_base_centers, params['base_radii'], 
                    params['highway_slope_a'], code_export_dir=current_code_export_dir)
    
    # Compile and get solver for this worker's unique directory
    solver = None
    try:
        # Build solver implicitly calls generate based on the ocp.code_export_directory
        solver = build_solver(ocp, current_json_file_path)
        # print(f"Run {run_idx}: Acados solver compiled in {current_code_export_dir}")
    except Exception as e:
        print(f"Run {run_idx}: Error during Acados compilation for this run: {e}. Skipping run.")
        # Cleanup on failure
        if os.path.exists(current_code_export_dir):
            shutil.rmtree(current_code_export_dir)
        return None


    # Randomly choose initial state for the aircraft
    x0_initial = np.random.uniform(params['x_min_initial'], params['x_max_initial'])
    x0_current = x0_initial.copy()

    # Randomly choose a car velocity for this run
    car_vel = np.random.choice(params['car_vel_options'])

    # Allocate logs for this specific run's trajectory
    X_run_log = np.zeros((sim_steps_per_run + 1, nx))
    U_run_log = np.zeros((sim_steps_per_run, nu))
    X_run_log[0] = x0_current.copy() # Store initial state

    actual_sim_steps = 0 # Counter for actual steps simulated

    # print(f"Run {run_idx}: Starting with car_vel={car_vel} m/s, initial x0={x0_initial[0]:.2f}, obstacles from {current_base_centers[0,0]:.2f}...")

    # Main simulation loop for a single trajectory
    try:
        for sim_k in range(sim_steps_per_run):
            # Set current state as initial constraint for the MPC problem
            solver.set(0, "lbx", x0_current)
            solver.set(0, "ubx", x0_current)

            # Update references (target landing state remains fixed)
            for k in range(N):
                # Target for state-control vector in running cost
                solver.set(k, "y_ref", np.hstack((params['xdes'], np.zeros(nu))))
            solver.set(N, "y_ref", params['xdes']) # Target for terminal state

            # Update obstacle parameter 'p' at each stage of MPC's prediction horizon
            for k in range(N):
                t_stage = k * dt # Time into the current MPC prediction horizon
                shift_x = car_vel * (sim_k * dt + t_stage)
                solver.set(k, "p", np.array([shift_x, 0.0, 0.0]))
            
            # Set 'p' for the terminal stage (N) as well
            terminal_shift_x = car_vel * (sim_k * dt + Tf)
            solver.set(N, "p", np.array([terminal_shift_x, 0.0, 0.0]))

            # Warm start the solver with current state and nominal controls
            for k in range(N):
                solver.set(k, "x", x0_current)
                solver.set(k, "u", np.zeros(nu))
            solver.set(N, "x", x0_current)

            # Solve the OCP for the current MPC step
            solver.solve()

            # Extract optimal control input for the first stage and apply it
            u0 = solver.get(0, "u")
            U_run_log[sim_k] = u0
            
            # Advance aircraft state using RK4 integration
            x0_current = rk4(x0_current, u0, dt)
            X_run_log[sim_k+1] = x0_current
            actual_sim_steps = sim_k + 1

            # Check if aircraft has crashed (altitude below zero)
            if x0_current[2] < 0:
                print(f"Run {run_idx}, MPC Step {sim_k}: Aircraft crashed (height < 0). Stopping simulation for this run.")
                break # Exit this inner simulation loop

        # After a full simulation run (or early termination), process results
        X_simulated_trajectory = X_run_log[:actual_sim_steps + 1]
        U_simulated_trajectory = U_run_log[:actual_sim_steps]

        # Calculate the final obstacle shift for the overall trajectory collision check
        total_shift_for_collision_check = car_vel * (actual_sim_steps * dt)
        current_p_for_collision_check = np.array([total_shift_for_collision_check, 0.0, 0.0])


        yf_final = X_simulated_trajectory[-1, 1]
        h_final = X_simulated_trajectory[-1, 2]

        # Check for successful landing conditions and absence of collision
        is_landing_successful = abs(yf_final) <= 12.0 and h_final <= 0.5
        has_collision = trajectory_collides(X_simulated_trajectory, current_base_centers, params['base_radii'], current_p_for_collision_check)

        if is_landing_successful and not has_collision:
            print(f"Run {run_idx}: SUCCESS! y_final = {yf_final:.2f} m, h_final = {h_final:.2f} m")
            return (X_simulated_trajectory, U_simulated_trajectory, current_base_centers, params['base_radii'], car_vel)
        else:
            print(f"Run {run_idx}: FAILED (y_final={yf_final:.2f}, h_final={h_final:.2f}, collision={has_collision}).")
            return None # Indicate failure
    finally:
        # --- Cleanup: remove the unique C code directory and JSON file ---
        if os.path.exists(current_code_export_dir):
            shutil.rmtree(current_code_export_dir)
        # The JSON file is inside the code_export_dir, so it's removed with the directory.
        # If it were in the parent directory, it would need a separate os.remove.


# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK: Orchestrates parallel simulations
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Define global constants and parameters
    nx, nu = 6, 3
    Nrun = 200 # Number of full trajectories to generate

    # Fixed parameters for the OCP/simulation
    sim_time = 25.0
    Tf, N_horizon = 1.0, 10
    
    # Obstacle distribution parameters
    OBSTACLE_X_MIN = -500
    OBSTACLE_X_MAX = 1000
    OBSTACLE_Y_MIN = -15
    OBSTACLE_Y_MAX = 15
    NUM_OBSTACLES_PER_RUN = 6 # This value is crucial for consistent C code generation
    base_radii = np.array([10., 10., 10.]) # Uniform radii for all obstacles
    highway_slope_a = 0.0 # Assuming a straight highway y=0x for this example

    # Aircraft initial state ranges
    x_min_initial = np.array([-500, -500,    100,  50, -np.pi/4, -np.deg2rad(20)])
    x_max_initial = np.array([ 100,  500, 500, 100,  np.pi/4,  np.deg2rad(20)])

    # Desired landing state
    xdes = np.array([1000., 0., 0., 50., 0., 0.])
    
    # Car velocity options for varying scenarios
    car_vel_options = [0, 10, 20, 30, 40]

    # Model name for Acados
    model_name = "landing"
    # Base directory for C code export. Each worker will create a sub-directory here.
    base_code_export_directory = "c_generated_code_mpc_dataset_runs" 

    # --- Step 1: Clean up the *base* C code directory if it exists ---
    # This cleans up previous runs' subdirectories, but individual workers will manage their own.
    print(f"Checking for existing base Acados C code directory '{base_code_export_directory}'...")
    if os.path.exists(base_code_export_directory):
        print(f"Removing existing base directory: {base_code_export_directory}")
        shutil.rmtree(base_code_export_directory)
    os.makedirs(base_code_export_directory, exist_ok=True) # Create the base directory


    # --- Step 2: Prepare parameters dictionary for worker processes ---
    params = {
        'nx': nx, 'nu': nu,
        'sim_time': sim_time, 'Tf': Tf, 'N': N_horizon,
        'OBSTACLE_X_MIN': OBSTACLE_X_MIN, 'OBSTACLE_X_MAX': OBSTACLE_X_MAX,
        'OBSTACLE_Y_MIN': OBSTACLE_Y_MIN, 'OBSTACLE_Y_MAX': OBSTACLE_Y_MAX,
        'NUM_OBSTACLES_PER_RUN': NUM_OBSTACLES_PER_RUN,
        'x_min_initial': x_min_initial, 'x_max_initial': x_max_initial,
        'xdes': xdes, 'car_vel_options': car_vel_options,
        'base_radii': base_radii,
        'model_name': model_name,
        'base_code_export_directory': base_code_export_directory, # Pass the base directory
        'highway_slope_a': highway_slope_a
    }

    # --- Step 3: Run simulations in parallel ---
    print(f"Starting {Nrun} simulation runs in parallel using {os.cpu_count()} processes...")
    
    num_processes = max(1, os.cpu_count() - 1) 
    
    results = []
    # Using Pool.imap_unordered for better progress reporting and memory management
    with multiprocessing.Pool(processes=num_processes) as pool:
        # functools.partial is used to pass the constant 'params' dictionary
        # to each call of simulate_single_run
        for result in pool.imap_unordered(partial(simulate_single_run, params=params), range(Nrun)):
            if result is not None:
                results.append(result)
            
    # Filter successful runs
    X_list, U_list, C_list, R_list, CarVel_list = [], [], [], [], []
    for X_traj, U_traj, centers, radii, car_vel in results:
        X_list.append(X_traj)
        U_list.append(U_traj)
        C_list.append(centers)
        R_list.append(radii)
        CarVel_list.append(car_vel)

    # --- Step 4: Save aggregated data ---
    if len(X_list) > 0:
        np.savez_compressed(
            "landing_mpc_dataset.npz",
            X         = np.array(X_list, dtype=object),
            U         = np.array(U_list, dtype=object),
            centers   = np.array(C_list, dtype=object),
            radii     = np.array(R_list, dtype=object),
            car_velocities = np.array(CarVel_list)
        )
        print(f"\nSaved {len(X_list)} successful scenarios to landing_mpc_dataset.npz")
    else:
        print("\nNo successful scenarios generated to save.")

