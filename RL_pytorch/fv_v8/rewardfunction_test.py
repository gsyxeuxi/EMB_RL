import gymnasium as gym
import os
from typing import Optional
import numpy as np
import time
import torch
import math
import random
from EMB_model_fv_v2 import FI_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
'''
EMB_env_fv:
measure value of x1 and x2 in obs
sample fv and fv in obs
h(x) = [x1, x2]
force back to zero
''' 

def draw(position_buffers, velocity_buffers):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('Position and Velocity vs Time for 6 Tests', fontsize=16)
    axes = axes.flatten()
    for i, (position_buffer, velocity_buffer) in enumerate(zip(position_buffers, velocity_buffers)):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax1.set_ylim(-20, 110)
        ax1.set_yticks(range(-20, 111, 10))
        ax2.set_yticks(range(-600, 601, 100))
        ax2.set_ylim(-600, 600)
        ax1.set_xlabel('Time / ms')
        ax1.set_ylabel('Position (Rad)', color=color)
        ax1.plot(position_buffer, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=100, color=color, linestyle='--', linewidth=1)
        ax1.axhline(y=-10, color=color, linestyle='--', linewidth=1)
        color = 'tab:red'
        ax2.set_ylabel('Velocity (Rad/s)', color=color)
        ax2.plot(velocity_buffer, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=500, color=color, linestyle='--', linewidth=1)
        ax2.axhline(y=-500, color=color, linestyle='--', linewidth=1)
        ax1.set_title(f'Experiment {i + 1}')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', 'position_velocity_6_tests.jpg'), dpi=300)
    plt.close()

def draw_action_reward(action_buffers, reward_buffers):
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('Action and Reward vs Time for 6 Tests', fontsize=16)
    axes = axes.flatten()
    for i, (action_buffer, reward_buffer) in enumerate(zip(action_buffers, reward_buffers)):
        ax1 = axes[i]
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax1.set_ylim(-7, 7)
        # ax2.set_ylim(-5e4, 5e4)
        ax1.set_xlabel('Time / ms')
        ax1.set_ylabel('Action (V)', color=color)
        ax1.plot(action_buffer, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=6, color=color, linestyle='--', linewidth=1)
        ax1.axhline(y=-6, color=color, linestyle='--', linewidth=1)
        color = 'tab:red'
        ax2.set_ylabel('Total Reward', color=color)
        ax2.plot(reward_buffer, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f'Experiment {i + 1}')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', 'action_reward_6_tests.jpg'), dpi=300)
    plt.close()



class EMB_All_info_Env(gym.Env):
    metadata = {"render.modes": ["human"]}  # metadata attribute is used to define render model and the framrate
    
    def __init__(self) -> None:
        self.fi_matrix = FI_matrix()
        self._dt = 0.001
        self.max_env_steps = 500
        self.count = 0 
        self.reward = None
        self.current = None # input u
        self.max_current = 6.0
        self.T_current = 0
        self.T_last = 0
        self.max_action = 1 # action space normalizaton
        self.action_fact = 6 # restore action space to (-6, 6)
        self.position_range = (-10, 100) #(-10, 100)
        self.velocity_range = (-500, 500)
        self.fv_range_high = 5e-5
        self.fv_range_low = 1e-5
        self.dangerous_position = 75
        self.reset_num = 0
        self.pos_std = 0.001
        self.vel_std = 1.0
        # *************************************************** Define the Observation Space ***************************************************
        """
        An 13-Dim Space: [motor position theta, time setp index k, fv, FIM element]
        """
        high = np.array([100, 500, 400, 5e-5, 1e10], dtype=np.float64)
        low = np.array([-10, -500, 0, 1e-5, 0], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(5,), dtype=np.float64)   

        # *************************************************** Define the Action Space ***************************************************
        """
        A 1-Dim Space: Control the voltage of motor
        """
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float64)  
        self.total_reward_scale = 0      
        
    @property
    def terminated(self):
        terminated = not self.is_safe
        return terminated
    
    @property
    def is_safe(self):
        min_x, max_x = self.position_range
        min_v, max_v = self.velocity_range
        x = self.state[0]
        v = self.state[1]
        return min_x < x < max_x and min_v < v < max_v
        # return min_v < v < max_v

    @property
    def is_dangerous(self):
        return self.state[0] > self.dangerous_position
    
    def _get_obs(self):
        return self.state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        random.seed(seed)
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) #add fv
        self.obs_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # if sample the fv
        self.state[3] = random.uniform(self.fv_range_low, self.fv_range_high)

        self.obs_state[3] = self.state[3]
        
        self.count = 0
        observation = self._get_obs()

        self.chi = torch.zeros((2, 1), dtype=torch.float64)
        self.scale_factor = 1
        self.scale_factor_previous = 1
        self.fi_info = torch.zeros((1,1), dtype=torch.float64)
        self.det_init = torch.det(self.fi_info)
 
        self.position_buffer = [0.0]
        self.velocity_buffer = [0.0]
        return observation, {}

    def step(self, action):
        # ************get observation space:************
        x0, x1, k, theta, sv = self._get_obs() # real state from last lime
        x = torch.tensor([x0, x1], dtype=torch.float64)
        u = self.action_fact * action.item() #input current
        dx = self.fi_matrix.f(x, u, torch.tensor([theta], dtype=torch.float64))
        x = x + self._dt * dx
        x0_new, x1_new = x[0].item(), x[1].item() #for plot

        x0_noise = x0_new + np.random.normal(0, self.pos_std)
        x1_noise = x1_new + np.random.normal(0, self.vel_std)

        jacobian = self.fi_matrix.jacobian(x, u, torch.tensor([theta], dtype=torch.float64))
        J_f = jacobian[0]
        J_h = self.fi_matrix.jacobian_h(x)
        df_theta = jacobian[1]
        self.chi = self.fi_matrix.sensitivity_x(J_f, df_theta, self.chi)
        # print(self.chi[0].item(), self.chi[1].item())
        dh_theta = self.fi_matrix.sensitivity_y(self.chi, J_h)
        fi_info_new = self.fi_matrix.fisher_info_matrix(dh_theta)
        step_reward = fi_info_new.item()
        self.fi_info += fi_info_new

        k_new = k + 1
        #update the state
        self.state[0] = x0_new
        self.state[1] = x1_new
        self.state[2] = k_new
        self.state[-1] = self.fi_info
        self.obs_state[0] = x0_noise
        self.obs_state[1] = x1_noise
        self.obs_state[2] = k_new
        self.obs_state[-1] = self.fi_info
        # ************calculate the rewards************
        if not self.is_safe:
            self.reward = -4e4
        elif self.is_dangerous:
            self.reward = (1 - (self.count/self.max_env_steps) ** 2) * step_reward - 200 * (x0_new - self.dangerous_position) ** 2 - 1 * (self.count/self.max_env_steps) ** 2 * (4 * x0_new ** 2 + x1_new ** 2)
            # self.reward = step_reward - 200 * (x0_new - self.dangerous_position) ** 2
        else:
            self.reward = (1 - (self.count/self.max_env_steps) ** 2) * step_reward - 1 * (self.count/self.max_env_steps) ** 2 * (4 * x0_new ** 2 + x1_new ** 2)
            # self.reward = step_reward
        self.count += 1
        terminated = self.terminated

        if self.count == self.max_env_steps:
            # # sprase reward
            # self.reward -= 1000 * (4 * x0_new ** 2 + x1_new ** 2)
            truncated = True
        else: truncated = False

        # for drawing
        self.position_buffer.append(x0_new)
        self.velocity_buffer.append(x1_new)
        return self.obs_state, self.reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None

env = EMB_All_info_Env()
env.reset()
total_reward = 0
episodic_returns = []
cont = 1
position_buffer = [0.0]
velocity_buffer = [0.0]
action_buffer = [0.0]
reward_buffer = [0.0]
position_buffers = []
velocity_buffers = []
action_buffers = []
reward_buffers = []
for i in range(1):
    for k in range(500):
        # u = torch.Tensor([0.2])
        u = torch.tensor(0.2 + 0.2 * torch.math.sin(2*math.pi*k/100 - math.pi/2), dtype=torch.float64)
        next_obs, reward, terminations, truncations, infos = env.step(u)
        total_reward += reward
        position_buffer.append(next_obs[0])
        velocity_buffer.append(next_obs[1]) 
        action_buffer.append(6 * np.clip(u.item(), -1, 1))
        reward_buffer.append(reward)   
         
    position_buffers.append(position_buffer)
    velocity_buffers.append(velocity_buffer)
    action_buffers.append(action_buffer)
    reward_buffers.append(reward_buffer)
    cont += 1
    total_reward = 0
    position_buffer = [0.0]
    velocity_buffer = [0.0]
    action_buffer = [0.0]
    reward_buffer = [0.0]

draw(position_buffers, velocity_buffers)
draw_action_reward(action_buffers, reward_buffers)





