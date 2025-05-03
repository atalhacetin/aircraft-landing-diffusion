import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Aircraft dynamics in control-affine form ===
def aircraft_dynamics(x, u):
    g = 9.81
    V, psi, gamma = x[3], x[4], x[5]
    nx, nz, mu = u[0], u[1], u[2]
    dx  = V * cs.cos(psi) * cs.cos(gamma)
    dy  = V * cs.sin(psi) * cs.cos(gamma)
    dh  = V * cs.sin(gamma)
    dV  = g * (nx - cs.sin(gamma))
    dpsi   = g * nz / V * cs.sin(mu) / cs.cos(gamma)
    dgamma = g / V * (nz * cs.cos(mu) - cs.cos(gamma))
    return cs.vertcat(dx, dy, dh, dV, dpsi, dgamma)

# RK4 integrator step
def rk4_step(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt   * k3, u)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# === NMPC for landing with curved highway constraints ===
class LandingNMPC:
    def __init__(self, T, N, m_steps, q_diag, r_diag, u_lim, road_poly, half_width):
        # horizon and discretization
        self.T = T; self.N = N; self.m = m_steps
        self.dt = T/(N*m_steps)
        # cost weights
        self.Q = cs.diag(q_diag)
        self.R = cs.diag(r_diag)
        # input limits
        self.u_min = (-u_lim).tolist(); self.u_max = u_lim.tolist()
        # curved road description: polynomial coeffs [a,b,c] for centerline y=a x^2+b x + c
        self.a, self.b, self.c = road_poly
        self.half_w = half_width
        # symbols
        self.x = cs.MX.sym('x',6)   # [x,y,h,V,psi,gamma]
        self.u = cs.MX.sym('u',3)   # [nx,nz,mu]
        # dynamics function
        xdot = aircraft_dynamics(self.x, self.u)
        self.f_dyn = cs.Function('f', [self.x, self.u], [xdot])
        # stage cost
        L = self.x.T@self.Q@self.x + self.u.T@self.R@self.u
        self.f_cost = cs.Function('L', [self.x, self.u],[L])
        # integrator builder
        self.F = self._build_integrator()

    def _build_integrator(self):
        x = self.x; u = self.u; dt = self.dt
        f = self.f_dyn; qf = self.f_cost
        X = x; Q = 0
        for _ in range(self.m):
            k1 = f(X,u); k2 = f(X+dt/2*k1,u)
            k3 = f(X+dt/2*k2,u); k4 = f(X+dt*k3,u)
            X = X + dt/6*(k1+2*k2+2*k3+k4)
            Q = Q + qf(X,u)
        return cs.Function('F', [x,u],[X,Q])

    def solve(self, x0, x_target):
        # create decision variables
        w, lbw, ubw, w0 = [], [], [], []
        g, lbg, ubg = [], [], []
        J = 0
        # initial state
        Xk = cs.MX.sym('X0',6); w += [Xk]
        lbw += x0.tolist(); ubw += x0.tolist(); w0 += x0.tolist()
        # loop stages
        for k in range(self.N):
            # control
            Uk = cs.MX.sym(f'U_{k}',3); w += [Uk]
            lbw += self.u_min; ubw += self.u_max; w0 += [0,0,0]
            # dynamics
            Fk = self.F(Xk,Uk)
            Xk_end = Fk[0]; J += Fk[1]
            # next state
            Xk = cs.MX.sym(f'X_{k+1}',6); w += [Xk]
            # state bounds: velocity [50,100], angles [-pi,pi], gamma [-20°,20°]
            lbw += [-cs.inf, -cs.inf, -cs.inf, 50, -cs.pi, -0.35]
            ubw += [ cs.inf,  cs.inf,  cs.inf,100,  cs.pi,  0.35]
            w0 += [0,0,0,70,0,0]
            # collocation constraint
            g += [Xk_end - Xk]; lbg += [0]*6; ubg += [0]*6
            # curved road constraints y ∈ [yR(x), yL(x)]
            x_sym = Xk[0]; y_sym = Xk[1]
            # centerline y_c = a x^2 + b x + c
            y_c = self.a*x_sym**2 + self.b*x_sym + self.c
            yL = y_c + self.half_w; yR = y_c - self.half_w
            g += [ y_sym - yR,  yL - y_sym ]
            lbg += [0,0]; ubg += [cs.inf, cs.inf]
        # terminal cost / target can be added similarly
        # pack NLP
        w_all = cs.vertcat(*w); g_all = cs.vertcat(*g)
        prob = {'f': J, 'x': w_all, 'g': g_all}
        opts = {'ipopt.print_level':0,'print_time':0}
        solver = cs.nlpsol('solver','ipopt',prob,opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        return sol['x'].full().flatten()

if __name__ == '__main__':
    # parameters
    T=20; N=40; m=4
    qd=[1,10,10,1,1,1]; rd=[0.1,0.1,0.1]
    ulim=np.array([2.0,3.0,np.pi/6])
    # curved road: y = 0.001 x^2 - 0.05 x + 2.0
    road_poly=(0.001,-0.05,2.0); half_w=5.0
    nmpc = LandingNMPC(T,N,m,qd,rd,ulim,road_poly,half_w)
    # init and target
    x0 = np.array([0,0,100,70,0,-0.05],float)
    xT = np.array([1000,0,0,0,0,0],float)
    sol = nmpc.solve(x0,xT)
    # unpack and plot
    n_x, n_u = 6,3
    step = n_x+n_u; w=sol
    xs = [w[i]   for i in range(0,len(w),step)]
    ys = [w[i+1] for i in range(0,len(w),step)]
    hs = [w[i+2] for i in range(0,len(w),step)]
    # 3D trajectory
    fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
    ax.plot(xs,ys,hs,'-o'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('h')
    plt.figure(); plt.plot(xs,hs,'-o'); plt.xlabel('x'); plt.ylabel('h'); plt.grid()
    plt.show()
