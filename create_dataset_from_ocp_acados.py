#!/usr/bin/env python3
import numpy as np
import casadi as ca
import time
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

# ------------------------------------------------------------------------------
# 1)  model builder (unchanged)
# ------------------------------------------------------------------------------
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
    m.x, m.u, m.xdot = x, u, xdot
    return m

# ------------------------------------------------------------------------------
# 2)  ocp‐builder now takes obstacle params
# ------------------------------------------------------------------------------
def build_ocp(model: AcadosModel,
              centers: np.ndarray,   # shape (M,3)
              radii:   np.ndarray    # shape (M,3)
             ) -> AcadosOcp:

    ocp = AcadosOcp(); ocp.model = model
    nx, nu = 6, 3
    # horizon
    Tf, N = 20.0, 40
    ocp.solver_options.tf = Tf
    if hasattr(ocp.dims, "N_horizon"):
        ocp.dims.N_horizon = N
    else:
        ocp.dims.N = N
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"

    # cost
    Q = np.diag([0.,0.01,0.01,0.001,20.,10.])
    R = np.diag([0.1,0.1,1])
    ny, ny_e = nx+nu, nx
    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W   = np.block([[Q,np.zeros((nx,nu))],
                             [np.zeros((nu,nx)),R]])
    ocp.cost.W_e = Q
    ocp.cost.Vx  = np.vstack((np.eye(nx), np.zeros((nu,nx))))
    ocp.cost.Vu  = np.vstack((np.zeros((nx,nu)), np.eye(nu)))
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.ny, ocp.cost.ny_e = ny, ny_e
    ocp.cost.yref   = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # control bounds
    u_min = np.array([-0.5, -3., -np.pi/4])
    u_max = -u_min
    ocp.constraints.lbu, ocp.constraints.ubu = u_min, u_max
    ocp.constraints.idxbu = np.arange(nu)

    # state bounds
    x_min = np.array([-2000, -2000,    0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000, 2000, 100,  np.pi,  np.deg2rad(20)])
    ocp.constraints.lbx, ocp.constraints.ubx = x_min, x_max
    ocp.constraints.idxbx = np.arange(nx)

    # initial‐state hard bounds (all states)
    ocp.constraints.idxbx_0 = np.arange(nx)
    ocp.constraints.lbx_0   = np.zeros(nx)
    ocp.constraints.ubx_0   = np.zeros(nx)

    # ellipsoidal obstacles
    phi_terms = []
    for (cx,cy,cz), (rx,ry,rz) in zip(centers, radii):
        phi_terms.append(
            ((model.sym_x[0]-cx)/rx)**2 +
            ((model.sym_x[1]-cy)/ry)**2 +
            ((model.sym_x[2]-cz)/rz)**2
        )
    expr_h        = ca.vertcat(*phi_terms)
    ocp.model.expr_h     = expr_h
    ocp.model.con_h_expr = expr_h
    n_h = expr_h.shape[0]
    ocp.dims.nh          = n_h
    ocp.constraints.lh   = np.ones(n_h)
    ocp.constraints.uh   = 1e6*np.ones(n_h)

    ocp.code_export_directory = "c_generated_code"
    return ocp

def build_solver(ocp: AcadosOcp) -> AcadosOcpSolver:
    AcadosOcpSolver.generate(ocp, json_file="landing_ocp.json", verbose=False)
    return AcadosOcpSolver(ocp, json_file="landing_ocp.json")


# ------------------------------------------------------------------------------
# 3)  collision‐check helper (now per‐ellipsoid radii)
# ------------------------------------------------------------------------------
def trajectory_collides(X, centers, radii):
    pos = X[:, :3]
    for i, (c, r) in enumerate(zip(centers, radii)):
        d2 = ((pos - c)/r)**2
        if np.any(np.sum(d2, axis=1) <= 1.0):
            return True
    return False


# ------------------------------------------------------------------------------
# 4)  main: randomize x0 ∈ [x_min, x_max], obstacles; solve; filter; save
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    x_min = np.array([-2000, -2000,    0,  50, -np.pi, -np.deg2rad(20)])
    x_max = np.array([ 2000,  2000, 2000, 100,  np.pi,  np.deg2rad(20)])
    Nrun = 500

    X_list       = []
    U_list       = []
    centers_list = []
    radii_list   = []
    coll_list    = []

    for i in range(Nrun):
        # 1) sample random x0
        x0 = np.random.uniform(x_min, x_max)

        # 2) sample random ellipsoids
        M = 6
        centers = np.vstack([
            [np.random.uniform(200,800),
             np.random.uniform(-12,12),
             0]
            for _ in range(M)
        ])
        radii = np.ones((M,3)) * 10.0

        # 3) build & solve
        model  = build_model()
        ocp    = build_ocp(model, centers, radii)
        solver = build_solver(ocp)

        solver.set(0, "lbx", x0); solver.set(0, "ubx", x0)
        xdes = np.array([1000.,0.,0.,0.,0.,0.])
        N    = ocp.dims.N_horizon if hasattr(ocp.dims,"N_horizon") else ocp.dims.N
        for k in range(N):
            solver.set(k, "y_ref", np.hstack((xdes, np.zeros(3))))
            solver.set(k, "x",    x0)
            solver.set(k, "u",    np.zeros(3))
        solver.set(N, "y_ref", xdes); solver.set(N, "x", x0)

        try:
            solver.solve()
        except:
            print(f"Sample {i}: solver failed")
            continue

        # 4) extract trajectories
        X = np.vstack([solver.get(k,"x") for k in range(N+1)])   # (N+1)x6
        U = np.vstack([solver.get(k,"u") for k in range(N)])     # Nx3

        y_final = X[-1,1]
        h_final = X[-1,2]
        print(f"Sample {i}: y_final = {y_final:.2f} m")
        print(f"Sample {i}: y_final = {y_final:.2f} m")
        print(f"Sample {i}: y_final = {y_final:.2f} m")
        print(f"Sample {i}: y_final = {y_final:.2f} m")
        print(f"Sample {i}: y_final = {y_final:.2f} m")
        if abs(y_final) <= 12.0 and h_final <= 0.5:
            collided = trajectory_collides(X, centers, radii)

            X_list.append(X)
            U_list.append(U)
            centers_list.append(centers)
            radii_list.append(radii)
            coll_list.append(int(collided))

    # 5) save everything in one .npz
    np.savez_compressed(
        "landing_dataset_acados2.npz",
        X       = np.array(X_list),        # shape (n_samp, N+1, 6)
        U       = np.array(U_list),        # shape (n_samp, N,   3)
        centers = np.array(centers_list),  # shape (n_samp, 6, 3)
        radii   = np.array(radii_list),    # shape (n_samp, 6, 3)
        collision = np.array(coll_list)    # shape (n_samp,)
    )
    print(f"Saved {len(X_list)} scenarios to landing_dataset.npz")