import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import scipy.io


def quat_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return [qx, qy, qz, qw]


class HighwayLanding3DEnv(gym.Env):
    """
    A 3D PyBullet Gym env with fixed-wing dynamics per AircraftTFRBot.
    External forces computed via PD altitude and lateral controllers,
    then applied to the PyBullet body for integration.

    Action: [phiCmd, nGCmd]
    State: [x,y,h,V,psi,gamma,phi]
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, mesh_path='f16.mat', dt=0.02, max_steps=1000,
                 vehicle_rate=0.02, num_lanes=3, lane_width=4.0):
        super().__init__()
        # time and physics
        self.dt = dt
        self.g = 9.81
        # Single GUI connection
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.g, physicsClientId=self.client)
        p.setTimeStep(self.dt, physicsClientId=self.client)
        self.max_steps = max_steps
        self.vehicle_rate = vehicle_rate
        # road dims
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.half_width = (num_lanes * lane_width) / 2.0
        self.road_half_length = 50.0
        # aircraft mesh
        mat = scipy.io.loadmat(mesh_path)
        self.faces = (mat['F'] - 1).astype(np.int32)
        self.vertices = mat['V'].astype(np.float32)
        # create spaces
        self.action_space = spaces.Box(
            low=np.array([-np.pi/3, -2.0], dtype=np.float32),
            high=np.array([ np.pi/3,  5.0], dtype=np.float32)
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        # build scene
        self._create_highway()
        self.reset()

    def _create_highway(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -self.g, physicsClientId=self.client)
        # road surface
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.road_half_length, self.half_width, 0.01],
            physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_half_length, self.half_width, 0.01],
            rgbaColor=[0, 0, 0, 1],
            physicsClientId=self.client
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, -0.01],
            physicsClientId=self.client
        )
        # lane stripes
        stripe_w = self.lane_width * 0.2
        for i in range(1, self.num_lanes):
            y = -self.half_width + i * self.lane_width
            col_s = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.road_half_length, stripe_w/2, 0.01],
                physicsClientId=self.client
            )
            vis_s = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.road_half_length, stripe_w/2, 0.01],
                rgbaColor=[1, 1, 1, 1],
                physicsClientId=self.client
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_s,
                baseVisualShapeIndex=vis_s,
                basePosition=[0, y, 0.01],
                physicsClientId=self.client
            )

    def reset(self):
        self._create_highway()
        self.step_count = 0
        self.vehicles = []
        # load aircraft
        col = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=self.vertices.tolist(),
            indices=self.faces.flatten().tolist(),
            physicsClientId=self.client
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            vertices=self.vertices.tolist(),
            indices=self.faces.flatten().tolist(),
            rgbaColor=[0.8, 0.8, 0.8, 1],
            physicsClientId=self.client
        )
        ori = quat_from_euler(0, 0, 0)
        self.ac_id = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0, 0, 10],
            baseOrientation=ori,
            physicsClientId=self.client
        )
        self.States = np.array([0.0, 0.0, 10.0, 50.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.States.copy()

    def step(self, action):
        phiCmd, nGCmd = action
        x, y, h, V, psi, gamma, phi = self.States
        # compute body forces
        Fyb = 10.0 * self.g * np.tan(phiCmd)
        Fxb = 0.0
        Fzb = 10.0 * self.g * nGCmd
        force_body = np.array([Fxb, Fyb, Fzb], dtype=np.float32)
        # transform to world
        _, orn = p.getBasePositionAndOrientation(self.ac_id, physicsClientId=self.client)
        R = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        force_world = R.dot(force_body)
        p.applyExternalForce(
            self.ac_id, -1, force_world.tolist(), [0, 0, 0], p.WORLD_FRAME,
            physicsClientId=self.client
        )
        # physics step
        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1
        # update state
        pos, ori = p.getBasePositionAndOrientation(self.ac_id, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.ac_id, physicsClientId=self.client)
        phi = phiCmd
        gamma = np.arcsin(vel[2] / np.linalg.norm(vel[:3]))
        psi = np.arctan2(vel[1], vel[0])
        V = np.linalg.norm(vel[:3])
        x, y, h = pos
        self.States = np.array([x, y, h, V, psi, gamma, phi], dtype=np.float32)
        done = bool(h <= 1.0 or self.step_count >= self.max_steps)
        reward = 100.0 if h <= 1.0 else -1.0
        return self.States.copy(), reward, done, {}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            pos, orn = p.getBasePositionAndOrientation(self.ac_id, physicsClientId=self.client)
            yaw = p.getEulerFromQuaternion(orn)[2]
            eye = [
                pos[0] - 10 * np.cos(yaw),
                pos[1] - 10 * np.sin(yaw),
                pos[2] + 5
            ]
            view = p.computeViewMatrix(
                cameraEyePosition=eye,
                cameraTargetPosition=pos,
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.client
            )
            proj = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0,
                nearVal=0.1, farVal=100,
                physicsClientId=self.client
            )
            _, _, px, _, _ = p.getCameraImage(
                width=512, height=512,
                viewMatrix=view,
                projectionMatrix=proj,
                physicsClientId=self.client
            )
            return np.reshape(px, (512, 512, 4))[:, :, :3]

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    import time, imageio
    env = HighwayLanding3DEnv()
    env.reset()
    frames = []
    for _ in range(500):
        _, _, done, _ = env.step(env.action_space.sample())
        frames.append(env.render('rgb_array'))
        time.sleep(env.dt)
        if done:
            break
    env.close()
    imageio.mimsave('demo_forces.mp4', frames, fps=int(1/env.dt))
