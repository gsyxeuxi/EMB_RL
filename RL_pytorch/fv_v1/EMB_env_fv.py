import gymnasium as gym
import os
from typing import Optional
import numpy as np
import time
import torch
from EMB_model_fv import FI_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
'''
EMB_env_fv:
real value of x1 and x2 in obs
fixed fv and no fv in obs

'''

class EMB_All_info_Env(gym.Env):
    metadata = {"render.modes": ["human"]}  # metadata attribute is used to define render model and the framrate
    
    def __init__(self) -> None:
        self.fi_matrix = FI_matrix()
        self._dt = 0.001
        self.max_env_steps = 300
        self.count = 0 
        self.reward = None
        self.current = None # input u
        self.max_current = 6.0
        self.T_current = 0
        self.T_last = 0
        self.max_action = 1 # action space normalizaton
        self.action_fact = 3 # restore action space to (0, 6)

        
        self.theta = torch.tensor([2.16e-5], dtype=torch.float64) #[self.fv]
        self.position_range = (-100, 100)
        self.velocity_range = (-500, 500)
        self.dangerous_position = 75
        # *************************************************** Define the Observation Space ***************************************************
        """
        An 13-Dim Space: [motor position theta, motor velocity omega, time setp index k, FIM element]
        """
        high = np.array([100, 500, 400, 1e10], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-high, high=high, shape=(4,), dtype=np.float64)   

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
    
    @property
    def is_dangerous(self):
        return self.state[0] > self.dangerous_position
    
    def _get_obs(self):
        return self.state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) #add fv
        self.count = 0
        observation = self._get_obs()

        self.chi = torch.zeros((2, 1), dtype=torch.float64)
        self.scale_factor = 1
        self.scale_factor_previous = 1
        self.fi_info = torch.zeros((1,1), dtype=torch.float64)
        self.det_init = torch.det(self.fi_info)
        self.fi_info_scale = self.fi_info * self.scale_factor
        self.fi_info_previous_scale = self.fi_info_scale
        det_previous_scale = torch.det(self.fi_info_previous_scale)
        self.log_det_previous_scale = torch.log(det_previous_scale)
        self.total_reward_scale = self.log_det_previous_scale

        self.position_buffer = [0.0]
        self.velocity_buffer = [0.0]
        return observation, {}

    def step(self, action):
        # ************get observation space:************
        # x0, x1, k, s1, s2, s3, s4 = self._get_obs() #state from last lime
        x0, x1, k, sv = self._get_obs() #state from last lime
        x = torch.tensor([x0, x1], dtype=torch.float64)
        u = self.action_fact * (action.item() + self.max_action) #input current
        dx = self.fi_matrix.f(x, u, self.theta)
        x = x + self._dt * dx
        x0_new, x1_new = x[0].item(), x[1].item() #for plot
        jacobian = self.fi_matrix.jacobian(x, u)
        J_f = jacobian[0]
        J_h = self.fi_matrix.jacobian_h(x)
        df_theta = jacobian[1]
        self.chi = self.fi_matrix.sensitivity_x(J_f, df_theta, self.chi)
        dh_theta = self.fi_matrix.sensitivity_y(self.chi, J_h)
        fi_info_new = self.fi_matrix.fisher_info_matrix(dh_theta)
        step_reward = fi_info_new.item()
        self.fi_info += fi_info_new
        fi_info_new_scale = fi_info_new * self.scale_factor
        self.fi_info_scale = self.fi_info_previous_scale + fi_info_new_scale
        
        # FIM_upper_triangle = []
        # for i in range(np.array(self.fi_info_scale).shape[0]):
        #     for j in range(i, np.array(self.fi_info_scale).shape[1]):
        #         FIM_upper_triangle.append(np.array(self.fi_info_scale)[i, j])
        
        # s1_new, s2_new, s3_new, s4_new = upper_triangle_elements[:]

        # self.fi_matrix.symmetrie_test(self.fi_info_scale)
        det_fi_scale = torch.det(self.fi_info_scale)
        log_det_scale = torch.log(det_fi_scale)
        step_reward_scale = log_det_scale - self.log_det_previous_scale
        self.scale_factor = (self.det_init / det_fi_scale) ** (1/4)
        self.fi_info_previous_scale = (self.fi_info_scale / self.scale_factor_previous) * self.scale_factor
        self.log_det_previous_scale = torch.slogdet(self.fi_info_previous_scale)[1]
        self.scale_factor_previous = self.scale_factor

        k_new = k + 1
        #update the state
        self.state[:2] = x0_new, x1_new
        self.state[2] = k_new
        self.state[-1] = self.fi_info
        # self.state = np.array([x0_new, x1_new, k_new, s1_new, s2_new, s3_new, s4_new], dtype=np.float64)
        # ************calculate the rewards************
        if not self.is_safe:
            self.reward = -4e4
        elif self.is_dangerous:
            self.reward = step_reward - 50 * (x0_new - self.dangerous_position) ** 2
        else:
            self.reward = step_reward
        self.count += 1
        terminated = self.terminated

        if self.count == self.max_env_steps:
            truncated = True
        else: truncated = False

        # for drawing
        self.position_buffer.append(x0_new)
        self.velocity_buffer.append(x1_new)
        return self._get_obs(), self.reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
    
    def draw(self):
        fig, ax1 = plt.subplots()
        color = 'tab:blue'
        ax1.set_xlabel('Time / ms')
        ax1.set_ylabel('Position / Rad', color=color)
        ax1.plot(self.position_buffer, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('Velocity / Rad/s', color=color)
        ax2.plot(self.velocity_buffer, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('Position and Velocity vs Time')
        plt.savefig(os.path.join('image', 'position_velocity_vs_time.png'))
        plt.close()


# env = EMB_All_info_Env()
# env.reset()
# for k in range(3):
#     u = [-1/3]
#     next_obs, reward, terminations, truncations, infos = env.step(u)

