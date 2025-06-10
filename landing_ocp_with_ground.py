#!/usr/bin/env python3
# ------------------------------------------------------------------
# NMPC for aircraft landing with altitude‑dependent Vmin
# and moving ellipsoidal obstacles – pure CasADi/IPOPT
# ------------------------------------------------------------------
#%%

import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------------------------------------------------------------
# 0) helpers
# ------------------------------------------------------------------
def aircraft_dynamics(x, u):
    g = 9.81
    V, psi, gamma = x[3], x[4], x[5]
    n_x, n_z, mu  = u[0], u[1], u[2]
    dx     = V * cs.cos(psi) * cs.cos(gamma)
    dy     = V * cs.sin(psi) * cs.cos(gamma)
    dh     = V * cs.sin(gamma)
    dV     = g * (n_x - cs.sin(gamma))
    dpsi   = g * n_z / V * cs.sin(mu) / cs.cos(gamma)
    dgamma = g / V * (n_z * cs.cos(mu) - cs.cos(gamma))
    return cs.vertcat(dx, dy, dh, dV, dpsi, dgamma)

def rk4_step(fun, x, u, dt):
    k1 = fun(x,u); k2 = fun(x+dt/2*k1,u)
    k3 = fun(x+dt/2*k2,u); k4 = fun(x+dt*k3,u)
    return x + dt/6*(k1+2*k2+2*k3+k4)

# altitude‑dependent minimum speed [m/s]
def Vmin_of_h(h):
    # 30 m/s on the ground, 50 m/s at ≥100 m, linear in between
    return 1 + 50*cs.fmin(h,1)/1

def mu_max_of_h(h):
    """
    45° bank allowed above 100 m,
    linearly reduced to 5° on the runway (h = 0).
    Returns [rad] and works with CasADi SX/MX.
    """
    mu_low  = np.deg2rad(5)   # at h = 0
    mu_high = np.deg2rad(45)  # at h ≥ 100
    return mu_low + (mu_high - mu_low) * cs.fmin(h, 10) / 10

def nx_min_of_h(h):
    """
    En fazla -0.6 g fren (≈ -6 m/s²) pistte (h = 0),
    yükseklikte lineer olarak -0.05 g’ye kadar azalır
    ve 100 m’den sonra kısıt pasif olur.
    Dönüş [yük‑faktörü] birimsizdir.
    """
    nx_brake = -0.6         # yerdeki max negatif
    nx_air   = -0.05        # 100 m ve üstünde
    return nx_air + (nx_brake - nx_air) * (1 - cs.fmin(h,1)/1)

# ------------------------------------------------------------------
# 1) NMPC class
# ------------------------------------------------------------------
class LandingNMPC:
    def __init__(self, T, N, M, q_diag, r_diag, u_bounds):
        self.T, self.N, self.M = T, N, M
        self.dt = T/(N*M)
        self.Q = cs.diag(q_diag)
        self.R = cs.diag(r_diag)
        self.u_min = (-u_bounds).tolist()
        self.u_max =  u_bounds.tolist()

        self.u_min = [-0.6,  self.u_min[1], self.u_min[2]]   # -1 g 
        self.u_max = [ 1.0,  self.u_max[1], self.u_max[2]]

        # symbols
        self.x = cs.MX.sym("x",6)
        self.u = cs.MX.sym("u",3)
        self.f = cs.Function("f",[self.x,self.u],
                             [aircraft_dynamics(self.x,self.u)])

        # RK4 integrator over dt
        X = self.x
        for _ in range(M):
            k1 = self.f(X,self.u)
            k2 = self.f(X + self.dt/2*k1, self.u)
            k3 = self.f(X + self.dt/2*k2, self.u)
            k4 = self.f(X + self.dt   *k3, self.u)
            X  = X + self.dt/6*(k1+2*k2+2*k3+k4)
        self.F = cs.Function("F",[self.x,self.u],[X])

        self.stage_cost = cs.Function(
            "L",[self.x,self.u],
            [ self.x.T@self.Q@self.x + self.u.T@self.R@self.u ])

        # moving ellipsoids
        size = np.array([10.,10.,10.])
        self.car_velocity = np.array([80/3.6,0,0])  # 80 km/h in +x
        self.ellipsoids = [
            dict(center=np.array([400,  2.5, 0]), axes=size),
            dict(center=np.array([600, -3. , 0]), axes=size),
            dict(center=np.array([500, 12. , 0]), axes=size),
            dict(center=np.array([200,  6. , 0]), axes=size),
            dict(center=np.array([100, -3. , 0]), axes=size),
            dict(center=np.array([1000,-3. , 0]), axes=size),
        ]

    # --------------------------------------------------------------
    def solve(self, x0, x_target):
        w, lbw, ubw, w0 = [], [], [], []
        g, lbg, ubg    = [], [], []
        J = 0

        Xk = cs.MX.sym("X0",6)
        w  += [Xk];  lbw += x0.tolist();  ubw += x0.tolist();  w0 += x0.tolist()

        for k in range(self.N):
            # control
            Uk = cs.MX.sym(f"U{k}",3)
            w += [Uk];  w0 += [0,0,0]
            lbw += self.u_min;  ubw += self.u_max

            # integrate
            Xk_end = self.F(Xk,Uk)
            J += self.stage_cost(Xk,Uk)

            # new state variable
            Xk = cs.MX.sym(f"X{k+1}",6)
            w += [Xk];  w0 += [0,0,0,70,0,0]
            lbw += [-2e3,-2e3,0,  0, -np.pi, -np.deg2rad(20)]
            ubw += [ 2e3, 2e3,2e3,100,  np.pi,  np.deg2rad(20)]

            # dynamics equality
            g.append(Xk_end - Xk);  lbg += [0]*6;  ubg += [0]*6

            # ----- altitude‑dependent V ≥ Vmin(h) -----------------
            Vmin_expr = Xk[3] - Vmin_of_h(Xk[2])
            g.append(Vmin_expr);  lbg.append(0.0);  ubg.append(cs.inf)

            # ----- moving ellipsoids -----------------------------
            for obs in self.ellipsoids:
                cx,cy,cz = obs["center"] + self.car_velocity*k*self.dt
                a,b,c    = obs["axes"]
                phi = ((Xk[0]-cx)/a)**2 + ((Xk[1]-cy)/b)**2 + ((Xk[2]-cz)/c)**2
                g.append(phi);  lbg.append(1.0);  ubg.append(cs.inf)

        # soft terminal target (optional)
        # J += cs.mtimes((Xk - x_target).T, (Xk - x_target))*10
        mu_max = mu_max_of_h(Xk[2])              # depends on current altitude
        g.append(Xk[5] - mu_max)                 #  μ − μ_max(h) ≤ 0
        lbg.append(-cs.inf);  ubg.append(0.0)

        g.append(-Xk[5] - mu_max)                # -μ − μ_max(h) ≤ 0
        lbg.append(-cs.inf);  ubg.append(0.0)

        nx_min = nx_min_of_h(Xk[2])            # h = Xk[2]  (irtifa)
        g.append(Uk[0] - nx_min)               # n_x - nx_min(h) ≥ 0  ➜  lbg=0
        lbg.append(0.0)
        ubg.append(cs.inf)

        nlp = dict(f=J, x=cs.vertcat(*w), g=cs.vertcat(*g))
        solver = cs.nlpsol("ipopt","ipopt",nlp,
                           {"print_time":0,"ipopt.print_level":0})
        sol = solver(x0=w0,lbx=lbw,ubx=ubw,lbg=lbg,ubg=ubg)
        return sol["x"].full().flatten()


# ------------------------------------------------------------------
# 2) main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # horizon & weights
    T, N, M = 20.0, 40, 4
    Qd = [0.,0.01,0.01,0.,20.,10.]
    Rd = [0.1,0.1,1.0]
    u_lim = np.array([0.1,3.0,np.pi/4])
    nmpc = LandingNMPC(T,N,M,Qd,Rd,u_lim)

    x0  = np.array([-200,-500,200,70, np.pi/4, -0.05])
    xT  = np.array([1000,0,0,1,0,0])
    sol = nmpc.solve(x0,xT)

    # unpack for plotting
    n_state, n_ctrl = 6,3
    per_step = n_state + n_ctrl
    x_opt = sol[0::per_step]
    y_opt = sol[1::per_step]
    h_opt = sol[2::per_step]
    V_opt = sol[3::per_step]
    psi_opt = sol[4::per_step]
    gamma_opt = sol[5::per_step]
    a1_opt = sol[n_state::per_step]
    a2_opt = sol[n_state+1::per_step]
    a3_opt = sol[n_state+2::per_step]
    tgrid = np.linspace(0,T,N)

    # -------------- plots (same as before) ------------------------
    fig = plt.figure(figsize=(9,6)); ax = fig.add_subplot(111,projection='3d')
    ax.plot(x_opt,y_opt,h_opt,marker='o'); ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_zlabel('h'); ax.set_title('Optimal trajectory')

    fig2 = plt.figure(); plt.plot(x_opt,h_opt,marker='o')
    plt.title('Altitude vs x'); plt.xlabel('x'); plt.ylabel('h'); plt.grid()

    fig3 = plt.figure(); plt.step(tgrid,a1_opt,label='n_x')
    plt.step(tgrid,a2_opt,label='n_z'); plt.step(tgrid,a3_opt,label='mu')
    plt.legend(); plt.grid(); plt.xlabel('t'); plt.title('controls')

    # obstacles on 3‑D plot
    u = np.linspace(0,2*np.pi,24); v = np.linspace(0,np.pi,12)
    uu,vv = np.meshgrid(u,v)
    for obs in nmpc.ellipsoids:
        a,b,c = obs["axes"]; cx,cy,cz = obs["center"]
        xs = a*np.cos(uu)*np.sin(vv)+cx
        ys = b*np.sin(uu)*np.sin(vv)+cy
        zs = c*np.cos(vv)+cz
        ax.plot_surface(xs,ys,zs,color='r',alpha=0.3,linewidth=0)
    plt.show()
