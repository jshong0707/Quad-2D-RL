#!/usr/bin/env python
import os
os.environ["MUJOCO_GL"] = "glfw"  # GLFW 백엔드 사용

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from gymnasium.utils import seeding
class Walker2DEnv(gym.Env):

    # metadata = {"render_modes": ["human"]}

    def __init__(self, xml_file="xml/2D_Quad_SPINE.xml"):
        super(Walker2DEnv, self).__init__()

        

        # MuJoCo 모델 로드 및 데이터 초기화
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)


        # Action Space: 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)

        # Observation Space: 8개 qpos (rootx, rootz, rootyaw, front_hip, front_knee, waist, back_hip, back_knee)
        #                + 8개 qvel  => 총 16차원
        pi = np.pi
        obs_high_constraint = [500, 0.8, pi/9, pi, pi, pi/10, pi, pi, 10, 5, 3, 4*pi, 4*pi, 4*pi, 4*pi, 4*pi, 50, 50, 50, 50, 50]
        obs_low_constraint = [-3, 0, -pi/9, 0, 0, -pi/10, 0, 0, -6, -5, -3,   -4*pi, -4*pi, -4*pi, -4*pi, -4*pi, -50, -50, -50, -50, -50]
        # obs_high_constraint = np.ones(21)*np.inf
        # obs_low_constraint = -np.ones(21)*np.inf
        obs_high = np.array(obs_high_constraint, dtype=np.float32)
        obs_low = np.array(obs_low_constraint, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # 시뮬레이션 파라미터
        self.dt = self.model.opt.timestep  # XML에 설정된 타임스텝
        self.max_time = 10                # 최대 에피소드 시간 (초)
        self.elapsed_time = 0.0
        self.done_flag = False
        self.noise_scale = 1.0

        # Observation space normalization
        obs_dim = self.observation_space.shape[0]
        self.obs_mean = np.zeros(obs_dim)
        self.obs_var = np.ones(obs_dim)
        self.alpha = 0.001

    # Observation space normalization
    def normalize_obs(self, obs):
        self.obs_mean = (1 - self.alpha) * self.obs_mean + self.alpha * obs
        self.obs_var = (1 - self.alpha) * self.obs_var + self.alpha * (obs - self.obs_mean)**2
        return (obs - self.obs_mean) / (np.sqrt(self.obs_var) + 1e-8)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = np.random.RandomState(), seed
        return [seed]

    def step(self, action):

        # 1) Actuate

        # print(action, '\n')
        scaled_action = 50*action
        # print(scaled_action)

        # 학습 초기에 action에 noise를 추가
        scaled_action = scaled_action # + np.random.normal(0, self.noise_scale)
        self.noise_scale *= 0.995  # 점진적으로 줄임
 

        # Body_pos = self.data.sensordata[6:6+3]
        # Body_vel = self.data.sensordata[9:9+3]
        # print(self.data.site_xpos[1], self.data.site_xpos[2])
        # F_Fz = F_Force[2]
        # F_Fx = F_Force[0]
        # R_Fz = R_Force[2]
        # R_Fx = R_Force[0]
        

        self.data.ctrl[0] = scaled_action[0]  # front_hip
        self.data.ctrl[1] = scaled_action[1]  # front_knee
        self.data.ctrl[2] = scaled_action[2]  # Spine (waist)
        self.data.ctrl[3] = scaled_action[3]  # back_hip
        self.data.ctrl[4] = scaled_action[4]  # back_knee
        
        mujoco.mj_step(self.model, self.data)
        self.elapsed_time += self.dt


    # 2) Observation
        obs = self._get_obs()
        # Observation space normalization
        obs_norm = self.normalize_obs(obs)  # 여기서 normalization 수행
        
        vx = obs[8]  # qvel[0] => x방향 속도
        pitch = obs[2]  # rootyaw가 아니라 root_pitch 값(가정)
        joint_vel = obs[11:11+5]
        t = self.data.time
        # ---- 보상 설계 ----
        # 전방속도 보상 (속도가 양수일 때만; 음수일 땐 0)
        desired_vel = 4
        forward_vel = max(vx, 0.0)

        # 높이 유지 보상(또는 페널티)
        # => 목표 높이보다 얼마나 차이 나는지 제곱 페널티
        z_now = obs[1]
        z_target = 0.35 + 0.1 * np.sin(t)

        height_cost = (z_target - z_now)**2

        # 제어 비용
        control_cost = np.sum(np.square(action))

        # 살아있음 보너스
        alive_bonus = 0.5  # 크게 주면 가만히 있으려 하므로 조금만 줌

        # 허리 과도한 모션 제한
        spine_ang = obs[5]
        reward_posture = (spine_ang**2)

        # 과도한 joint velocity 제한
        penalty_joint_vel = np.sum(np.square(joint_vel))

        # Energy
        joint_vel = obs[11:11+5]
       
        # 종합 보상 (가중치 활용)
        # w_vel = 4.0
        # w_control = 0.005
        # w_spine = 0.1
        # w_joint_vel = 0.002
        # w_z = 2

        # reward = (- w_vel * (desired_vel - forward_vel)
        #         + alive_bonus 
        #         - w_z * height_cost
        #         - w_control * control_cost
        #         - w_spine * reward_posture
        #         - w_joint_vel * penalty_joint_vel
        #         )
        
        reward = ( - 2 * (z_target - z_now)**2
                  - 2 * (0 - pitch)**2
                  + alive_bonus)

        # ---- 종료 조건 ----
        done = False
        # (1) 루트 높이가 너무 낮으면 넘어짐
        if z_now < 0.15:
            done = True
            reward -= 50.0

        if 0.4 < pitch or pitch < -0.4:         # 이걸 넣었더니 좀 뛰는 느낌 남
            # done = True
            reward -= 50.0

        return obs_norm, reward, done, False, {}




    def reset(self, **kwargs):
        mujoco.mj_resetData(self.model, self.data)
        # 초기 상태 설정: qpos와 qvel 초기화
        init_qpos = np.zeros(self.model.nq)
        init_qvel = np.zeros(self.model.nv)
        init_qpos[0] = 0 # np.random.uniform(-0.1, 0.1)     # rootx
        init_qpos[1] = 0.38 # + np.random.uniform(0, 0.05) # rootz (살짝 띄워서 시작)
        init_qpos[2] = 0                                   # rootyaw
        init_qpos[3] = np.pi/4 + np.random.uniform(-0.05, 0.05) # Front Hip
        init_qpos[4] = np.pi/2 + np.random.uniform(-0.05, 0.05) # Front Knee
        init_qpos[5] = np.random.uniform(-0.05, 0.05)     # Spine
        init_qpos[6] = np.pi/4 + np.random.uniform(-0.05, 0.05) # Rear Hip
        init_qpos[7] = np.pi/2 + np.random.uniform(-0.05, 0.05) # Rear Knee

        self.data.qpos[:] = init_qpos
        self.data.qvel[:] = init_qvel
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.elapsed_time = 0.0
        self.done_flag = False

        obs = self._get_obs()
        
        return obs, {}

    def _get_obs(self):
        """
        관측값 구성:
        - qpos: [rootx, rootz, rootyaw, front_hip, front_knee, Spine, rear_hip, rear_knee] (8개)
        - qvel: 8개 (각 관절 속도)
        => 총 16차원 관측 벡터
        """
        qpos = self.data.qpos
        qvel = self.data.qvel
        torque = self.data.ctrl
        F_sensor = self.data.sensordata[6:6+6]
        obs = np.concatenate([qpos, qvel, torque]).astype(np.float32)
        return obs

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    env = Walker2DEnv()
    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
    # Random action
        action = env.action_space.sample()
        obs, reward, done, A, info = env.step(action)
        
        if done:
            obs = env.reset()