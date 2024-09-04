import gym
import os
import numpy as np
import math
import time
import tensorflow as tf
from EMB_model_v1 import FI_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
'''
EMB_env_v3:
Use scaled FIM for obs



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
        self.theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5], dtype=tf.float64) #[self.km, self.k1, self.fc, self.fv]
        self.position_range = (-100, 100)
        self.velocity_range = (-500, 500)
        self.dangerous_position = 75
        # *************************************************** Define the Observation Space ***************************************************
        """
        An 13-Dim Space: [motor position theta, motor velocity omega, time setp index k, FIM element]
        """
        high = np.array([100, 500, 400, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-high, high=high, shape=(13,), dtype=np.float64)   

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

    def reset(self):
        # self.state = np.array([0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float64)
        self.state = np.array([0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 0.0, 1e-6, 0.0, 0.0, 1e-6, 0.0, 1e-6], dtype=np.float64)
        self.count = 0
        observation = self._get_obs()
        self.chi = tf.convert_to_tensor(np.zeros((2,4)), dtype=tf.float64)
        self.scale_factor = 1
        self.scale_factor_previous = 1
        self.fi_info = tf.convert_to_tensor(np.eye(4) * 1e-6, dtype=tf.float64)
        self.det_init = tf.linalg.det(self.fi_info)
        self.fi_info_scale = self.fi_info * self.scale_factor
        self.fi_info_previous_scale = self.fi_info_scale
        det_previous_scale = tf.linalg.det(self.fi_info_previous_scale)
        self.log_det_previous_scale = tf.math.log(det_previous_scale)
        self.total_reward_scale = tf.cast(self.log_det_previous_scale, dtype=tf.float32)

        self.position_buffer = [0.0]
        self.velocity_buffer = [0.0]
        return observation, {}

    def step(self, action):
        # ************get observation space:************
        # x0, x1, k, s1, s2, s3, s4 = self._get_obs() #state from last lime
        x0, x1, k, e11, e12, e13, e14, e22, e23, e24, e33, e34, e44 = self._get_obs() #state from last lime
        x = tf.Variable([x0, x1], dtype=tf.float64)
        action = np.clip(action, -1, 1)
        u = self.action_fact * (action + self.max_action) #input current
        
        dx = self.fi_matrix.f(x, u, self.theta)
        x = x + self._dt * dx
        x0_new, x1_new = np.array(x)[:]
        jacobian = self.fi_matrix.jacobian(x, u)
        J_f = jacobian[0]
        J_h = self.fi_matrix.jacobian_h(x)
        df_theta = jacobian[1]
        self.chi = self.fi_matrix.sensitivity_x(J_f, df_theta, self.chi)
        dh_theta = self.fi_matrix.sensitivity_y(self.chi, J_h)
        fi_info_new = self.fi_matrix.fisher_info_matrix(dh_theta)
        self.fi_info += fi_info_new
        fi_info_new_scale = fi_info_new * self.scale_factor
        self.fi_info_scale = self.fi_info_previous_scale + fi_info_new_scale
        
        FIM_upper_triangle = []
        for i in range(np.array(self.fi_info_scale).shape[0]):
            for j in range(i, np.array(self.fi_info_scale).shape[1]):
                FIM_upper_triangle.append(np.array(self.fi_info_scale)[i, j])
        
        # s1_new, s2_new, s3_new, s4_new = upper_triangle_elements[:]

        self.fi_matrix.symmetrie_test(self.fi_info_scale)
        det_fi_scale = tf.linalg.det(self.fi_info_scale)
        log_det_scale = tf.math.log(det_fi_scale)
        step_reward_scale = log_det_scale - self.log_det_previous_scale
        self.scale_factor = (self.det_init / det_fi_scale) ** (1/4)
        self.fi_info_previous_scale = (self.fi_info_scale / self.scale_factor_previous) * self.scale_factor
        self.log_det_previous_scale = np.linalg.slogdet(self.fi_info_previous_scale)[1]
        self.scale_factor_previous = self.scale_factor

        k_new = k + 1
        #update the state
        self.state[:2] = x0_new, x1_new
        self.state[2] = k_new
        self.state[-10:] = FIM_upper_triangle[:]
        # self.state = np.array([x0_new, x1_new, k_new, s1_new, s2_new, s3_new, s4_new], dtype=np.float64)
        # ************calculate the rewards************
        if not self.is_safe:
            self.reward = -50
        elif self.is_dangerous:
            self.reward = tf.cast(step_reward_scale, dtype=tf.float32) - 0.01 * (x0_new - self.dangerous_position) ** 2
        else:
            self.reward = tf.cast(step_reward_scale, dtype=tf.float32)
        
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