#!/usr/bin/env python3
# ---------------------------------------------------------------
# DATASET GENERATOR  –  Landing NMPC  +  moving ellipsoids
# ---------------------------------------------------------------
#%%
from landing_ocp_with_ground2 import LandingNMPC
import numpy as np, casadi as cs, tqdm
import matplotlib.pyplot as plt

# --------- random initial aircraft state ------------------------
def random_initial_state():
    x  = np.random.uniform(-300,  300) - 500
    y  = np.random.uniform(-200,  200)
    h  = np.random.uniform(  300,  500)
    V  = np.random.uniform(  60,   90)
    psi= np.random.uniform(-np.pi/4, np.pi/4)
    gam= np.random.uniform(-0.10, 0.05)
    return np.array([x,y,h,V,psi,gam], dtype=float)
    #return np.array([-200,500,100,70,np.pi/4,-0.05])

# --------- random ellipsoid set ---------------------------------
LANES_Y  = np.array([ 2.5, -3.0, 12.0, 6.0, -3.0, -3.0])   # m
LANES_Z  = np.zeros(6)                                     # on ground
X_E_RANGE= (100, 1000)                                     # m
ELL_SIZE = np.array([10., 10., 10.])                       # same axes

def build_random_ellipsoids():
    
    x0s = np.random.uniform(*X_E_RANGE, size=6)
    centers = np.vstack((x0s, LANES_Y, LANES_Z)).T
    return [dict(center=centers[i].copy(), axes=ELL_SIZE.copy())
            for i in range(6)], centers         # list + ndarray(6,3)

# --------- OCP / simulation parameters --------------------------
T, N, M = 20.0, 40, 4
Qd=[0,0.01,0.01,0.001,20,10]; Rd=[0.1,0.1,1]
u_bounds = np.array([0.1,3.0,np.pi/4])
TARGET = np.array([1000,0,0,1,0,0])
NUM_SAMPLES = 100

# --------- allocate dataset arrays ------------------------------
X_ds   = np.zeros((NUM_SAMPLES, N+1, 6), dtype=np.float32)
U_ds   = np.zeros((NUM_SAMPLES, N,   3), dtype=np.float32)
SPEED  = np.zeros(NUM_SAMPLES,        dtype=np.float32)
X0_ds  = np.zeros((NUM_SAMPLES, 6),   dtype=np.float32)
E0_ds  = np.zeros((NUM_SAMPLES, 6, 3),dtype=np.float32)   # ellipsoid centres

keep = 0
# --------- generate ---------------------------------------------
for m in tqdm.tqdm(range(NUM_SAMPLES), desc="dataset"):
    x0 = random_initial_state()
    v_car = 80/3.6          # 10–33 m/s
    ellipsoids, centres0 = build_random_ellipsoids()

    nmpc = LandingNMPC(T,N,M,Qd,Rd,u_bounds,
                       ellipsoids, np.array([v_car,0,0]))

    try:
        sol = nmpc.solve(x0, TARGET)
    except RuntimeError:
        print(f"sample {m} infeasible, skip");  continue

    n_state, n_ctrl = 6,3
    per_step = n_state + n_ctrl
    x_opt = sol[0::per_step]
    y_opt = sol[1::per_step]
    h_opt = sol[2::per_step]
    V_opt = sol[3::per_step]
    psi_opt = sol[4::per_step]
    gamma_opt = sol[5::per_step]

    # unpack decision vector ------------------------------------------------
    states   = np.zeros((N+1,6))
    controls = np.zeros((N,3))
    for i in range(N):
        base = i*(6+3)
        states[i,:]   = sol[base:base+6]
        controls[i,:] = sol[base+6:base+9]
    states[N,:] = sol[N*(6+3):N*(6+3)+6]

    # store -----------------------------------------------------------------
    y_final = states[-1, 1]                               # X_N[1]
    if np.abs(y_final) > 12.0:
        print(f"sample {m} rejected (y_final = {y_final:.1f})")
        continue                                          # do NOT store

    # ---------- store accepted sample ----------------------------
    X_ds[keep,:,:]  = states
    U_ds[keep,:,:]  = controls
    SPEED[keep]     = v_car
    X0_ds[keep,:]   = x0
    E0_ds[keep,:,:] = centres0
    keep += 1


# --------- save --------------------------------------------------
np.savez_compressed(
    "landing_dataset.npz",
    X   = X_ds[:keep],
    U   = U_ds[:keep],
    car_speed = SPEED[:keep],
    x0  = X0_ds[:keep],
    ellipsoid_centres = E0_ds[:keep]
)
print("Saved landing_dataset.npz",
      f"({NUM_SAMPLES} samples, horizon N={N})")

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
# %%
