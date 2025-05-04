#!/usr/bin/env python3
# ------------------------------------------------------------------
# NMPC for aircraft landing
# ------------------------------------------------------------------
#%%
import numpy as np, casadi as cs, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ------------- helper functions -----------------------------------
def aircraft_dynamics(x,u):
    g=9.81; V,psi,gamma=x[3],x[4],x[5]; n_x,n_z,mu=u[0], u[1],u[2]
    return cs.vertcat(
        V*cs.cos(psi)*cs.cos(gamma),
        V*cs.sin(psi)*cs.cos(gamma),
        V*cs.sin(gamma),
        g*(n_x-cs.sin(gamma)),
        g*n_z/V*cs.sin(mu)/cs.cos(gamma),
        g/V*(n_z*cs.cos(mu)-cs.cos(gamma)))

def Vmin_of_h(h):        return 1+50*cs.fmin(h,1)/1
def mu_max_of_h(h):      return np.deg2rad(5)+np.deg2rad(40)*cs.fmin(h,10)/10
def nx_min_of_h(h):      return -0.05+(-0.55)*(1-cs.fmin(h,1)/1)

# ------------- NMPC class -----------------------------------------
class LandingNMPC:
    def __init__(self,T,N,M,q_diag,r_diag,u_bounds,
                 ellipsoids,car_velocity):
        self.T,Tf,self.N,self.M = T,T,N,M
        self.dt = T/(N*M)
        self.Q = cs.diag(q_diag); self.R = cs.diag(r_diag)
        self.u_min = [-0.6, -u_bounds[1], -u_bounds[2]]
        self.u_max = [ 1.0,  u_bounds[1],  u_bounds[2]]
        self.ellipsoids = ellipsoids
        self.car_velocity = car_velocity

        x=cs.MX.sym('x',6); u=cs.MX.sym('u',3)
        f=cs.Function('f',[x,u],[aircraft_dynamics(x,u)])
        X=x
        for _ in range(M):
            k1=f(X,u); k2=f(X+self.dt/2*k1,u)
            k3=f(X+self.dt/2*k2,u); k4=f(X+self.dt*k3,u)
            X = X + self.dt/6*(k1+2*k2+2*k3+k4)
        self.F = cs.Function('F',[x,u],[X])
        self.L = cs.Function('L',[x,u],[x.T@self.Q@x+u.T@self.R@u])

    # ----------------------------------------------------------------
    def solve(self,x0,xT):
        w,lbw,ubw,w0=[],[],[],[]
        g,lbg,ubg=[],[],[]
        Xk=cs.MX.sym('X0',6); w+=[Xk]; lbw+=x0.tolist(); ubw+=x0.tolist(); w0+=x0.tolist()
        J=0
        for k in range(self.N):
            Uk=cs.MX.sym(f'U{k}',3)
            w+=[Uk]; w0+=[0,0,0]; lbw+=self.u_min; ubw+=self.u_max
            Xk_end=self.F(Xk,Uk); J+=self.L(Xk,Uk)

            Xk=cs.MX.sym(f'X{k+1}',6)
            w+=[Xk]; w0+=[0,0,0,70,0,0]
            lbw+=[-2e3,-2e3,0,0,-np.pi,-np.deg2rad(20)]
            ubw+=[ 2e3, 2e3,2e3,100,np.pi, np.deg2rad(20)]
            g.append(Xk_end-Xk); lbg+=[0]*6; ubg+=[0]*6

            # speed floor
            g.append(Xk[3]-Vmin_of_h(Xk[2])); lbg.append(0); ubg.append(cs.inf)
            # μ bound
            mu_max=mu_max_of_h(Xk[2])
            g.extend([Xk[5]-mu_max,-Xk[5]-mu_max]); lbg.extend([-cs.inf,-cs.inf]); ubg.extend([0,0])
            # n_x braking
            g.append(Uk[0]-nx_min_of_h(Xk[2])); lbg.append(0); ubg.append(cs.inf)
            # ellipsoids
            for obs in self.ellipsoids:
                cx,cy,cz = obs['center']+self.car_velocity*k*self.dt
                a,b,c = obs['axes']
                phi=((Xk[0]-cx)/a)**2+((Xk[1]-cy)/b)**2+((Xk[2]-cz)/c)**2
                g.append(phi); lbg.append(1); ubg.append(cs.inf)

        J+=10*cs.sumsqr(Xk-xT)
        nlp=dict(f=J,x=cs.vertcat(*w),g=cs.vertcat(*g))
        sol=cs.nlpsol('ipopt','ipopt',nlp,{'print_time':0,'ipopt.print_level':0})
        return sol(x0=w0,lbx=lbw,ubx=ubw,lbg=lbg,ubg=ubg)['x'].full().flatten()

# ------------------------------------------------------------------
# 2) main – generate ellipsoids then build controller
# ------------------------------------------------------------------
if __name__=="__main__":
    # create ellipsoids in this scope
    size=np.array([10.,10.,10.])
    centers=np.array([[400,2.5,0],[600,-3,0],[500,12,0],
                      [200,6,0],[100,-3,0],[1000,-3,0]])
    ellipsoids=[dict(center=c,axes=size) for c in centers]
    car_vel=np.array([80/3.6,0,0])            # 80 km/h forward

    # build MPC
    T,N,M=20.0,40,4
    Qd=[0,0.01,0.01,0,20,10]; Rd=[0.1,0.1,1]
    u_bounds=np.array([0.1,3.0,np.pi/4])
    nmpc=LandingNMPC(T,N,M,Qd,Rd,u_bounds,ellipsoids,car_vel)

    # initial & target
    x0=np.array([-200,-500,100,70,np.pi/4,-0.05])
    xT=np.array([1000,0,0,1,0,0])
    sol=nmpc.solve(x0,xT)

    # ---------- plotting (unchanged) ------------------------------
    n_state,n_ctrl=6,3; step=n_state+n_ctrl
    x_opt=sol[0::step]; y_opt=sol[1::step]; h_opt=sol[2::step]
    V_opt=sol[3::step]; psi_opt=sol[4::step]; gamma_opt=sol[5::step]
    a1=sol[n_state::step]; a2=sol[n_state+1::step]; a3=sol[n_state+2::step]
    t=np.linspace(0,T,N)

    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111,projection='3d')
    ax.plot(x_opt,y_opt,h_opt,marker='o'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('h')
    ax.set_title('Optimal trajectory')
    u_mesh=np.linspace(0,2*np.pi,24); v_mesh=np.linspace(0,np.pi,12)
    uu,vv=np.meshgrid(u_mesh,v_mesh)
    for obs in ellipsoids:
        a,b,c=obs['axes']; cx,cy,cz=obs['center']
        xs=a*np.cos(uu)*np.sin(vv)+cx
        ys=b*np.sin(uu)*np.sin(vv)+cy
        zs=c*np.cos(vv)+cz
        ax.plot_surface(xs,ys,zs,color='r',alpha=0.3,linewidth=0)

    plt.figure(); plt.step(t,a1,label='n_x'); plt.step(t,a2,label='n_z'); plt.step(t,a3,label='μ')
    plt.legend(); plt.grid(); plt.xlabel('time [s]'); plt.title('controls')
    plt.show()
