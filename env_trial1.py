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
    A 3D PyBullet-based Gym environment where a fixed-wing aircraft (F-16 mesh)
    must land on a highway with moving vehicles. The highway lanes and road surface
    are visualized as a black box with white lane markings.

    Mesh defined in 'f16.mat' with variables 'F' (faces, 1-based) and 'V' (vertices).

    Observation:
        Box(6): [x, y, z, vx, vy, vz]
    Action:
        Box(3): [ax, ay, az]  # accelerations scaled to environment limits
    Reward:
        -1 per step, +100 for safe landing, -100 for collision or crash.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 mesh_path: str = 'f16.mat',
                 time_step: float = 1/20,
                 max_steps: int = 2000,
                 vehicle_rate: float = 0.05,
                 num_lanes: int = 3,
                 lane_width: float = 4.0):
        super().__init__()
        self.time_step = time_step
        self.max_steps = max_steps
        self.vehicle_rate = vehicle_rate
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        # compute half road width (y-direction) and road length
        self.half_width = (self.num_lanes * self.lane_width) / 2.0
        self.road_half_length = 50.0  # x-direction half-length
        # load mesh
        mat = scipy.io.loadmat(mesh_path)
        F_mat = mat['F']
        V_mat = mat['V']
        self.faces = (F_mat - 1).astype(np.int32)
        self.vertices = V_mat.astype(np.float32)
        # connect physics
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        # spaces
        obs_low = -np.inf * np.ones(6, dtype=np.float32)
        obs_high = np.inf * np.ones(6, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # build environment
        self._create_highway()
        self.reset()

    def _create_highway(self):
        # clear simulation
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(self.time_step, physicsClientId=self.client)
        # road surface as black box
        road_thickness = 0.01
        col_road = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.road_half_length, self.half_width, road_thickness],
            physicsClientId=self.client)
        vis_road = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.road_half_length, self.half_width, road_thickness],
            rgbaColor=[0, 0, 0, 1],
            physicsClientId=self.client)
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_road,
            baseVisualShapeIndex=vis_road,
            basePosition=[0, 0, -road_thickness],
            physicsClientId=self.client)
        # lane markings
        for i in range(1, self.num_lanes):
            y = -self.half_width + i * self.lane_width
            start = [-self.road_half_length, y, 0.01]
            end = [self.road_half_length, y, 0.01]
            p.addUserDebugLine(
                start, end,
                lineColorRGB=[1, 1, 1],
                lineWidth=2,
                lifeTime=0,
                physicsClientId=self.client)

    def reset(self):
        self._create_highway()
        self.step_count = 0
        self.vehicles = []
        # load F-16 mesh
        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            vertices=self.vertices.tolist(),
            indices=self.faces.flatten().tolist(),
            physicsClientId=self.client)
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            vertices=self.vertices.tolist(),
            indices=self.faces.flatten().tolist(),
            rgbaColor=[0.8, 0.8, 0.8, 1],
            physicsClientId=self.client)
        # spawn aircraft
        start_pos = [0, 0, 10]
        start_ori = quat_from_euler(0, 0, 0)
        self.ac_id = p.createMultiBody(
            baseMass=10.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=start_pos,
            baseOrientation=start_ori,
            physicsClientId=self.client)
        return self._get_obs()

    def _get_obs(self):
        pos, _ = p.getBasePositionAndOrientation(self.ac_id, physicsClientId=self.client)
        vel, _ = p.getBaseVelocity(self.ac_id, physicsClientId=self.client)
        return np.array([*pos, *vel], dtype=np.float32)

    def step(self, action):
        ax, ay, az = np.clip(action, -1, 1)
        force = [ax * 200.0, ay * 200.0, az * 200.0]
        p.applyExternalForce(
            self.ac_id, -1, forceObj=force, posObj=[0, 0, 0],
            flags=p.LINK_FRAME, physicsClientId=self.client)
        # spawn vehicles
        if np.random.rand() < self.vehicle_rate:
            car_size = [1, 0.5, 0.5]
            col_car = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=car_size, physicsClientId=self.client)
            vis_car = p.createVisualShape(
                p.GEOM_BOX, halfExtents=car_size,
                rgbaColor=[1, 0, 0, 1], physicsClientId=self.client)
            y = np.random.uniform(-self.half_width, self.half_width)
            car_id = p.createMultiBody(
                baseMass=2,
                baseCollisionShapeIndex=col_car,
                baseVisualShapeIndex=vis_car,
                basePosition=[-self.road_half_length, y, car_size[2]],
                physicsClientId=self.client)
            speed = np.random.uniform(8, 12)
            self.vehicles.append((car_id, speed))
        # step sim
        p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1
        # update vehicles
        to_del = []
        for cid, speed in self.vehicles:
            pos, _ = p.getBasePositionAndOrientation(cid, physicsClientId=self.client)
            new_x = pos[0] + speed * self.time_step
            p.resetBasePositionAndOrientation(
                cid, [new_x, pos[1], pos[2]], [0, 0, 0, 1], physicsClientId=self.client)
            if new_x > self.road_half_length:
                to_del.append((cid, speed))
        for rem in to_del:
            self.vehicles.remove(rem)
            p.removeBody(rem[0], physicsClientId=self.client)
        # collisions & landing
        obs = self._get_obs()
        reward = -1.0; done = False
        for cid, _ in self.vehicles:
            if p.getContactPoints(self.ac_id, cid, physicsClientId=self.client):
                reward, done = -100.0, True; break
        if not done and obs[2] <= 1.0 and abs(obs[5]) < 1.0:
            reward, done = 100.0, True
        if self.step_count >= self.max_steps: done = True
        return obs, reward, done, {}

    def render(self, mode='human'):
        if mode == 'rgb_array':
            pos, orn = p.getBasePositionAndOrientation(self.ac_id, physicsClientId=self.client)
            yaw = p.getEulerFromQuaternion(orn)[2]
            eye = [pos[0] - 10*np.cos(yaw), pos[1] - 10*np.sin(yaw), pos[2] + 5]
            view = p.computeViewMatrix(cameraEyePosition=eye, cameraTargetPosition=pos,
                                       cameraUpVector=[0,0,1], physicsClientId=self.client)
            proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0,
                                               physicsClientId=self.client)
            _,_,px,_,_ = p.getCameraImage(width=512, height=512,
                                          viewMatrix=view, projectionMatrix=proj,
                                          physicsClientId=self.client)
            return np.reshape(px,(512,512,4))[:,:,:3]
    def close(self): p.disconnect(self.client)


if __name__ == '__main__':
    import time, imageio
    env = HighwayLanding3DEnv(mesh_path='f16.mat')
    env.reset(); frames=[]
    for _ in range(800):
        a=env.action_space.sample()
        obs,rew,done,_=env.step(a)
        frames.append(env.render(mode='rgb_array'))
        time.sleep(env.time_step)
        if done: break
    env.close()
    imageio.mimsave('highway3d_f16_lanes.mp4', frames, fps=int(1/env.time_step))
