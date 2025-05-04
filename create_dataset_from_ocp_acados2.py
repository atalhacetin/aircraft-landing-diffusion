#!/usr/bin/env python3
import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# -----------------------------------------------------------------------------
# 1) Build your fixed-wing model (unchanged)
# -----------------------------------------------------------------------------
def build_model() -> AcadosModel:
    m = AcadosModel();  m.name = "landing2"
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
    m.x, m.u, m.xdot = x, u, xdot
    return m

# -----------------------------------------------------------------------------
# 2) Build OCP, **parameterizing** obstacle centers & radii
# -----------------------------------------------------------------------------
def build_ocp(model: AcadosModel) -> AcadosOcp:
    ocp = AcadosOcp()
    ocp.model = model
    nx, nu = 6, 3
    Tf, N = 20.0, 40

    # --- 1) Pack all 6 centers and 6 radii into one parameter vector p ∈ ℝ³⁶
    M = 6
    p = ca.MX.sym('p', M*6)       # [cx1,cy1,cz1, … , cx6,cy6,cz6, rx1,ry1,rz1, … , rx6,ry6,rz6]
    model.sym_p = p              # for your own reference
    model.p     = p              # **this** makes p an input to all acados functions
    ocp.dims.np = p.numel()
    ocp.parameter_values = np.zeros(ocp.dims.np) # initial guess for p

    # --- 2) Build ellipsoidal‐avoidance in terms of p
    #    split p into centers and radii
    C = ca.reshape(p[:3*M],   3, M).T   # shape (M,3)
    R = ca.reshape(p[3*M:],    3, M).T   # shape (M,3)
    xi, yi, zi = model.x[0], model.x[1], model.x[2]
    phi_terms = []
    for i in range(M):
        cx, cy, cz = C[i,0], C[i,1], C[i,2]
        rx, ry, rz = R[i,0], R[i,1], R[i,2]
        phi_terms.append(((xi-cx)/rx)**2 +
                         ((yi-cy)/ry)**2 +
                         ((zi-cz)/rz)**2)
    expr_h = ca.vertcat(*phi_terms)

    ocp.model.expr_h     = expr_h
    ocp.model.con_h_expr = expr_h
    ocp.dims.nh          = M
    ocp.constraints.lh   = np.ones(M)
    ocp.constraints.uh   = 1e6*np.ones(M)

    # --- 3) The rest of your OCP (cost, bounds, horizon) stays exactly the same:
    ocp.solver_options.tf = Tf
    if hasattr(ocp.dims, "N_horizon"):
        ocp.dims.N_horizon = N
    else:
        ocp.dims.N        = N
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver        = "FULL_CONDENSING_HPIPM"

    Q = np.diag([0.,0.01,0.01,0.001,20.,10.])
    Rm= np.diag([0.1,0.1,1])
    ny, ny_e = nx+nu, nx
    ocp.cost.cost_type    = "LINEAR_LS"
    ocp.cost.cost_type_e  = "LINEAR_LS"
    ocp.cost.W            = np.block([[Q, np.zeros((nx,nu))],
                                      [np.zeros((nu,nx)), Rm       ]])
    ocp.cost.W_e          = Q
    ocp.cost.Vx           = np.vstack((np.eye(nx), np.zeros((nu,nx))))
    ocp.cost.Vu           = np.vstack((np.zeros((nx,nu)), np.eye(nu)))
    ocp.cost.Vx_e         = np.eye(nx)
    ocp.cost.ny, ocp.cost.ny_e = ny, ny_e
    ocp.cost.yref         = np.zeros(ny)
    ocp.cost.yref_e       = np.zeros(ny_e)

    u_min = np.array([-0.5, -3., -np.pi/4])
    u_max = -u_min
    ocp.constraints.lbu, ocp.constraints.ubu = u_min, u_max
    ocp.constraints.idxbu = np.arange(nu)

    x_min = np.array([-2000, -2000,    0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000, 2000,100,  np.pi,  np.deg2rad(20)])
    ocp.constraints.lbx, ocp.constraints.ubx = x_min, x_max
    ocp.constraints.idxbx = np.arange(nx)

    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = np.zeros(nx)
    ocp.constraints.ubx_0   = np.zeros(nx)

    ocp.code_export_directory = "c_generated_code2"
    return ocp


def build_solver(ocp: AcadosOcp) -> AcadosOcpSolver:
    AcadosOcpSolver.generate(ocp, json_file="landing_ocp2.json", verbose=False)
    return AcadosOcpSolver(ocp, json_file="landing_ocp2.json")


# -----------------------------------------------------------------------------
# 3) Collision helper (unchanged)
# -----------------------------------------------------------------------------
def trajectory_collides(X, centers, radii):
    pos = X[:, :3]
    for c, r in zip(centers, radii):
        d2 = ((pos - c)/r)**2
        if np.any(np.sum(d2,axis=1) <= 1.0):
            return True
    return False


# -----------------------------------------------------------------------------
# 4) MAIN: compile once, then just swap in new p = [centers,radii]
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    model  = build_model()
    ocp    = build_ocp(model)
    solver = build_solver(ocp)

    x_min = np.array([-2000, -2000,    0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000, 2000, 100,  np.pi,  np.deg2rad(20)])
    Nrun  = 1000

    # pre‐allocate lists
    X_list, U_list, C_list, R_list = [], [], [], []
    Col_list = []

    # solver dimensions
    N = ocp.dims.N_horizon if hasattr(ocp.dims, "N_horizon") else ocp.dims.N

    for i in range(Nrun):
        # → 1) random x0
        x0 = np.random.uniform(np.array([-500, -200,    100,  50, -np.pi/4, -np.deg2rad(20)]), 
                               np.array([ 100,  200, 500, 100,  np.pi/4,  np.deg2rad(20)]))

        # → 2) random obstacles
        M = 6
        centers = np.vstack([
            [np.random.uniform(200,800),
             np.random.uniform(-20,20),
             0.0]
            for _ in range(M)
        ])
        radii = np.ones((M,3))*10.0

        # pack into parameter vector p
        p_num = np.hstack([centers.ravel(), radii.ravel()])

        # set p at every stage
        for k in range(N+1):
            solver.set(k, "p", p_num)

        # 3) set initial‐state and references
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)
        xdes = np.array([1000.,0.,0.,0.,0.,0.])
        for k in range(N):
            solver.set(k, "y_ref", np.hstack((xdes, np.zeros(3))))
            solver.set(k, "x",     x0)
            solver.set(k, "u",     np.zeros(3))
        solver.set(N, "y_ref", xdes)
        solver.set(N, "x",     x0)

        # 4) solve
        try:
            solver.solve()
        except:
            print(f"Run {i}: solver failed, skip")
            continue

        # 5) extract & filter
        X = np.vstack([solver.get(k,"x") for k in range(N+1)])
        U = np.vstack([solver.get(k,"u") for k in range(N)])
        yf = X[-1,1] 
        h_final = X[-1,2]
        if abs(yf) <= 12.0  and h_final <= 0.5 and not(trajectory_collides(X, centers, radii)):
            
            print(f"Run {i}: y_final = {yf:.2f} m")
            print(f"Run {i}: y_final = {yf:.2f} m")
            print(f"Run {i}: y_final = {yf:.2f} m")
            print(f"Run {i}: y_final = {yf:.2f} m")
            print(f"Run {i}: y_final = {yf:.2f} m")
            X_list.append(X)
            U_list.append(U)
            C_list.append(centers)
            R_list.append(radii)

    # 6) save full dataset
    np.savez_compressed(
        "landing_param_dataset.npz",
        X         = np.array(X_list),
        U         = np.array(U_list),
        centers   = np.array(C_list),
        radii     = np.array(R_list),
    )
    print(f"Saved {len(X_list)} scenarios → landing_param_dataset.npz")
