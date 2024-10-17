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
EMB_env_fv_v11:
the right model we want
for actor: measured x1 and x2
for critic: real x1 and x2 and k and fv and FIM
sample fv and fv in obs
h(x) = [x1, x2]
force back to zero with reward function v3: quintic polynomial
x1 x2 reset around 0
''' 
def quintic_polynomial(t, coeff):
    return coeff[0] + coeff[1]*t + coeff[2]*t**2 + coeff[3]*t**3 + coeff[4]*t**4 + coeff[5]*t**5

def quintic_polynomial_dt(t, coeff):
    return coeff[1] + 2*coeff[2]*t + 3* coeff[3]*t**2 + 4*coeff[4]*t**3 + 5*coeff[5]*t**4


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
        self.pos_reset_range_high = 0.5
        self.vel_reset_range_high= 2.5
        self.dangerous_position = 75
        self.reset_num = 0
        self.pos_std = 0.001
        self.vel_std = 1.0
        # *************************************************** Define the Observation Space ***************************************************
        """
        An 7-Dim Space: [pos, vel, pos_noise, vel_noise, time setp index k, fv, FIM element]
        """
        high = np.array([100, 500, 100, 500, 400, 5e-5, 1e10], dtype=np.float64)
        low = np.array([-10, -500, -10, -500, 0, 1e-5, 0], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(7,), dtype=np.float64)   

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
        super().reset(seed=seed)
        self.state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.state[0] = self.state[2] = random.uniform(-self.pos_reset_range_high, self.pos_reset_range_high)
        self.state[1] = self.state[3] = random.uniform(-self.vel_reset_range_high, self.vel_reset_range_high)

        # if sample the fv
        self.state[5] = random.uniform(self.fv_range_low, self.fv_range_high)
        
        # # if the fv increase continiues
        # self.state[5] = self.fv_range_low + self.reset_num * 1e-6
        # self.reset_num += 1
 
        self.count = 0
        observation = self._get_obs()

        self.chi = torch.zeros((2, 1), dtype=torch.float64)
        self.scale_factor = 1
        self.scale_factor_previous = 1
        self.fi_info = torch.zeros((1,1), dtype=torch.float64)
        self.det_init = torch.det(self.fi_info)
        # self.fi_info_scale = self.fi_info * self.scale_factor
        # self.fi_info_previous_scale = self.fi_info_scale
        # det_previous_scale = torch.det(self.fi_info_previous_scale)
        # self.log_det_previous_scale = torch.log(det_previous_scale)
        # self.total_reward_scale = self.log_det_previous_scale
        self.back_reward = 0
        self.minus_reward = 0
        self.find_polynomial = True
        return observation, {}

    def step(self, action):
        # ************get observation space:************
        x0, x1, _, _, k, theta, sv = self._get_obs() # real state from last lime
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
        dh_theta = self.fi_matrix.sensitivity_y(self.chi, J_h)
        fi_info_new = self.fi_matrix.fisher_info_matrix(dh_theta)
        step_reward = fi_info_new.item()
        self.fi_info += fi_info_new
        # fi_info_new_scale = fi_info_new * self.scale_factor
        # self.fi_info_scale = self.fi_info_previous_scale + fi_info_new_scale
        
        # FIM_upper_triangle = []
        # for i in range(np.array(self.fi_info_scale).shape[0]):
        #     for j in range(i, np.array(self.fi_info_scale).shape[1]):
        #         FIM_upper_triangle.append(np.array(self.fi_info_scale)[i, j])
        
        # s1_new, s2_new, s3_new, s4_new = upper_triangle_elements[:]

        # self.fi_matrix.symmetrie_test(self.fi_info_scale)
        # det_fi_scale = torch.det(self.fi_info_scale)
        # log_det_scale = torch.log(det_fi_scale)
        # step_reward_scale = log_det_scale - self.log_det_previous_scale
        # self.scale_factor = (self.det_init / det_fi_scale) ** (1/4)
        # self.fi_info_previous_scale = (self.fi_info_scale / self.scale_factor_previous) * self.scale_factor
        # self.log_det_previous_scale = torch.slogdet(self.fi_info_previous_scale)[1]
        # self.scale_factor_previous = self.scale_factor

        k_new = k + 1
        #update the state
        self.state[0] = x0_new
        self.state[1] = x1_new
        self.state[2] = x0_noise
        self.state[3] = x1_noise
        self.state[4] = k_new
        self.state[-1] = self.fi_info

        # ************calculate the polynomial************
        if self.count >= 300 and self.find_polynomial:
            pos_start = x0_new
            vel_start = x1_new
            pos_end = 0
            vel_end = 0
            acc_start = 0
            acc_end = 0
            t_start = 0.3
            t_end = 0.5

            A = np.array([
                [1, t_start, t_start**2, t_start**3, t_start**4, t_start**5],
                [0, 1, 2*t_start, 3*t_start**2, 4*t_start**3, 5*t_start**4],
                [0, 0, 2, 6*t_start, 12*t_start**2, 20*t_start**3],
                [1, t_end, t_end**2, t_end**3, t_end**4, t_end**5],
                [0, 1, 2*t_end, 3*t_end**2, 4*t_end**3, 5*t_end**4],
                [0, 0, 2, 6*t_end, 12*t_end**2, 20*t_end**3]
                ])
            b = np.array([pos_start, vel_start, acc_start, pos_end, vel_end, acc_end])
            coeff = np.linalg.solve(A, b)
            t_vals = np.linspace(0.3, 0.5, 200)
            self.theta_vals = [quintic_polynomial(t, coeff) for t in t_vals]
            self.theta_dt = [quintic_polynomial_dt(t, coeff) for t in t_vals]
            self.find_polynomial = False
        
        # ************calculate the rewards************
        if not self.is_safe:
            self.reward = -1e7 #4e4

        # # for test
        # elif self.is_dangerous:
        #     self.reward = step_reward - 200 * (x0_new - self.dangerous_position) ** 2
        # else:
        #     self.reward = step_reward

        # variant 3
        elif self.is_dangerous:
            if self.count >= 300:
                self.reward =  - 20 * (x0_new - self.dangerous_position) ** 2 - \
                    25 * (x0_new - self.theta_vals[self.count-300]) ** 2 - 1 * (x1_new - self.theta_dt[self.count-300]) ** 2
                self.back_reward += step_reward
                self.minus_reward += self.reward 
            else:
                self.reward = step_reward - 300 * (x0_new - self.dangerous_position) ** 2
        else:
            if self.count >= 300:
                self.reward = - 25 * (x0_new - self.theta_vals[self.count-300]) ** 2 - 1 * (x1_new - self.theta_dt[self.count-300]) ** 2
                self.back_reward += step_reward
                self.minus_reward += self.reward 
            else:
                self.reward = step_reward
                # print(step_reward)
                
        self.count += 1
        terminated = self.terminated

        if self.count == self.max_env_steps:
            # print('back_reward', self.back_reward)
            # print('minus_reward', self.minus_reward )
            # print('difference', self.minus_reward+self.back_reward)
            # sparse reward
            if abs(x0_new) <= 1.2 and abs(x1_new) <= 6:
                self.reward += self.back_reward
                print('************pos and vel back to zero***********')
            truncated = True
        else: truncated = False
      
        return self.state, self.reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        return None
        
    def close(self):
        return None

# env = EMB_All_info_Env()
# env.reset()
# total_reward = 0
# for k in range(500):
#     u = torch.Tensor([0.1])
#     next_obs, reward, terminations, truncations, infos = env.step(u)
#     total_reward += reward
# print(total_reward)