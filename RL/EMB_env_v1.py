import gym
import numpy as np
import math
import csv
import time
import tensorflow as tf
from EMB_model_v1 import FI_matrix


class EMB_All_info_Env(gym.Env):
    metadata = {"render.modes": ["human"]}  # metadata attribute is used to define render model and the framrate
    
    def __init__(self) -> None:
        self.fi_matrix = FI_matrix()
        self._dt = 0.001
        self.max_env_steps = 25000
        self.count = 0 
        self.reward = None
        self.current = None # input u
        self.max_current = 6.0
        self.T_current = 0
        self.T_last = 0
        self.max_action = 1 # action space normalizaton
        self.action_fact = 3 # restore action space to (0, 6)
        # *************************************************** Define the Observation Space ***************************************************
        """
        An 7-Dim Space: [motor position theta, motor velocity omega, time setp index k, sensitivity y]
        """
        high = np.array([150 ,200, 400, 200, 200, 200, 200], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-high, high=high, shape=(7,), dtype=np.float64)   

        # *************************************************** Define the Action Space ***************************************************
        """
        A 1-Dim Space: Control the voltage of motor
        """
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float64)        
        
        with open("data.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            #columns_name
            writer.writerow(["x_current","y_current","d_x","d_y","c_0","c_1","angle_x","angle_y","reward"])

    def _get_obs(self):
        return self.state

    # def get_logs(self):
    #     logs = np.array([
    #         self.position[0],
    #         self.position[1],
    #         self.x_target,
    #         self.y_target,
    #         self.p_value,
    #         self.d_value,
    #         self.reward,
    #         self.angle_x,
    #         self.angle_y
    #     ])
    #     with open("data.csv","a+") as csvfile: 
    #         writer = csv.writer(csvfile)
    #         writer.writerow([logs[0],logs[1],logs[0]-logs[2],logs[1]-logs[3],logs[4],logs[5],logs[7],logs[8],logs[6]])
    #     return logs

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 1e-3, 1e-3, 1e-3, 1e-3], dtype=np.float64)
        observation = self._get_obs()
        self.chi = tf.convert_to_tensor(np.zeros((2,4)), dtype=tf.float64)

        self.scale_factor = 1
        self.scale_factor_previous = 1
        fi_info = tf.convert_to_tensor(np.eye(4) * 1e-6, dtype=tf.float64)
        self.det_init = tf.linalg.det(fi_info)
        self.fi_info_scale = fi_info * self.scale_factor
        self.fi_info_previous_scale = self.fi_info_scale
        det_previous_scale = tf.linalg.det(self.fi_info_previous_scale)
        self.log_det_previous_scale = tf.math.log(det_previous_scale)
        # total_reward_scale = log_det_previous_scale

        return observation, {}

    def step(self, action):
        theta = tf.constant([21.7e-03, 23.04, 10.37e-3, 2.16e-5], dtype=tf.float64) #[self.km, self.k1, self.fc, self.fv]
        # ************get observation space:************
        x0, x1, k, s1, s2, s3, s4 = self._get_obs() #state from last lime
        x = tf.Variable([x0, x1], dtype=tf.float64)

        action = np.clip(action, -1, 1)
        u = self.action_fact * (action + self.max_action) #input current

        dx = self.fi_matrix.f(x, u, theta)
        x = x + self._dt * dx
        # clip the speed an velocity in setting aera
        # x = tf.clip_by_value(x, clip_value_min=[5,2], clip_value_max=[20,30])
        x0_new, x1_new = np.array(x)[:]

        jacobian = self.fi_matrix.jacobian(x, u)
        J_f = jacobian[0]
        J_h = self.fi_matrix.jacobian_h(x)
        df_theta = jacobian[1]
        self.chi = self.fi_matrix.sensitivity_x(J_f, df_theta, self.chi)
        dh_theta = self.fi_matrix.sensitivity_y(self.chi, J_h)
        s1_new, s2_new, s3_new, s4_new = np.array(dh_theta)[0][:]

        fi_info_new = self.fi_matrix.fisher_info_matrix(dh_theta)
        self.fi_info += fi_info_new
        fi_info_new_scale = fi_info_new * self.scale_factor
        self.fi_info_scale = self.fi_info_previous_scale + fi_info_new_scale

        self.fi_matrix.symmetrie_test(self.fi_info_scale)
        det_fi_scale = tf.linalg.det(self.fi_info_scale)
        log_det_scale = tf.math.log(det_fi_scale)
        step_reward_scale = log_det_scale - self.log_det_previous_scale
        # total_reward_scale = total_reward_scale + step_reward_scale
        self.scale_factor = (self.det_init / det_fi_scale) ** (1/4)
        self.fi_info_previous_scale = (self.fi_info_scale / self.scale_factor_previous) * self.scale_factor
        self.log_det_previous_scale = np.linalg.slogdet(self.fi_info_previous_scale)[1]
        self.scale_factor_previous = self.scale_factor

        self.count += 1
        k_new = k + 1
        self.state = np.array([x0_new, x1_new, k_new, s1_new, s2_new, s3_new, s4_new], dtype=np.float64)
        done = False
        # ************calculate the rewards************
        self.reward = step_reward_scale

        return self._get_obs(), step_reward_scale, done, {}

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None


env = EMB_All_info_Env()
env.reset()
for k in range(5):
    u = -1/3
    next_state, reward, done, _ = env.step(u)
    print(reward)

env.reset()
print("*******************")
for k in range(5):
    u = -1/3
    next_state, reward, done, _ = env.step(u)
    print(reward)