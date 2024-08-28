import argparse
import os
import time
import datetime
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tensorlayer as tl
import EMB_env_v2

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--random', dest='random', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'EMB_v1'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = False  # render while training

ALG_NAME = 'PPO'
NUM_ENVS = 1 # the number of parallel running envs
TOTAL_TIMESETPS = 25000  # total timesteps for training 1000
TEST_EPISODES = 1  # total number of episodes for testing
NUM_STEPS = 512 # total number of steps N*M 512
GAMMA = 0.99  # reward discount 0.99 recommanded
LR_A = 0.0002  # learning rate for actor
LR_C = 0.0002  # learning rate for critic 0.0003 is a safe default
MINIBATCH_SIZE = 64  # update minibatch size, Recommend equating this to the width of the neural network
BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_EPOCHS = 10 # number of epochs K for each minibatch
ACTOR_UPDATE_STEPS = 10  # actor update steps
CRITIC_UPDATE_STEPS = 10  # critic update steps

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters, 0.25 recommanded
EPSILON = 0.2



def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
###############################  PPO  ####################################


class PPO(object):
    """
    PPO class
    """
    def __init__(self, state_dim, action_dim):
        # critic
        with tf.name_scope('critic'):
            input = tf.keras.Input(shape = (None, state_dim), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            v = tf.keras.layers.Dense(1,  kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                      bias_initializer=tf.keras.initializers.Constant(1.0))(layer)
        self.critic = tf.keras.Model(input, v, name="critic")              

        # actor
        with tf.name_scope('actor'):
            input = tf.keras.Input(shape = (None, state_dim), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            a_mean = tf.keras.layers.Dense(action_dim, activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                           bias_initializer=tf.keras.initializers.Constant(0.1))(layer)
            logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32), trainable=True, name='logstd') # This have no input, so it is state independent
        self.actor = tf.keras.Model(input, a_mean, name='actor')
        self.actor.logstd = logstd

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)


        # create tensorboard logs
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        self.log_dir = 'logs/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []


    def train_actor(self, state, action, adv, old_pi):
        """
        Update policy network
        :param state: state batch
        :param action: action batch
        :param adv: advantage batch
        :param old_pi: old pi distribution
        :return: kl_mean or None
        """
        with tf.GradientTape() as tape:
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)

            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            if self.method == 'penalty':  # ppo penalty
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:  # ppo clip
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        a_gard = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.trainable_weights))

        if self.method == 'kl_pen':
            return kl_mean

    def train_critic(self, reward, state):
        """
        Update actor network
        :param reward: cumulative reward batch
        :param state: state batch
        :return: None
        """
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grad = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grad, self.critic.trainable_weights))

    def update(self):
        """
        Update parameter with the constraint of KL divergent
        :return: None
        """
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        mean, std = self.actor(s), tf.exp(self.actor.logstd)
        pi = tfp.distributions.Normal(mean, std)
        adv = r - self.critic(s)

        # update actor
        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                self.train_actor(s, a, adv, pi)

        # update critic
        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def get_action(self, state, greedy=False):
        """
        Choose action
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        state = state[np.newaxis, :].astype(np.float32)
        mean, std = self.actor(state), tf.exp(self.actor.logstd)
        if greedy:
            action = mean[0]
        else:
            pi = tfp.distributions.Normal(mean, std) #this function can do the broadcast automatically
            action = tf.squeeze(pi.sample(1), axis=0)[0]  # choosing action
        return np.clip(action, -self.action_bound, self.action_bound)
    
    def get_action_and_value(self, state, action=None): 
        """
        Choose action and get value
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        state = state[np.newaxis, :].astype(np.float32) # from [,,,] to [[,,,]]
        mean, std = self.actor(state), tf.exp(self.actor.logstd)
        probs = tfp.distributions.Normal(mean, std) #this function can do the broadcast automatically
        if action is None:
            action = tf.squeeze(probs.sample(1), axis=0)[0] # choosing action
        log_prob = tf.reduce_sum(probs.log_prob(action), axis=1)
        entropy = tf.reduce_sum(probs.entropy(), axis=1)
        value = self.critic(state)
        return action, log_prob, entropy, value

    def get_value(self, state):
        state = state[np.newaxis, :].astype(np.float32) # from [,,,] to [[,,,]]
        return self.critic(state)
    
    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)

    def store_transition(self, state, action, reward):
        """
        Store state, action, reward at each step
        :param state:
        :param action:
        :param reward:
        :return: None
        """
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        """
        Calculate cumulative reward
        :param next_state:
        :return: None
        """
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()

if __name__ == '__main__':
    env = EMB_env_v2.EMB_All_info_Env()
    # env = NormalizedEnv(env)

    # reproducible
    # env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    states = np.zeros((NUM_STEPS, NUM_ENVS) + env.observation_space.shape, dtype=np.float32)
    # shape=(512, 1, 13)
    actions = np.zeros((NUM_STEPS, NUM_ENVS) + env.action_space.shape, dtype=np.float32)
    # shape=(512, 1, 1)
    logprobs = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    rewards = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    dones = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    values = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)

    t0 = time.time()
    if args.train:
        total_step = 0
        next_state, _ = env.reset()
        next_done = tf.zeros(NUM_ENVS)
        # episode_reward = env.total_reward_scale # this need to be reconsider
        num_updates = TOTAL_TIMESETPS // BATCH_SIZE 

        all_rollout_reward = []
        for updates in range(num_updates):
            for step in range(NUM_STEPS):  # N envs with M steps
                total_step += NUM_ENVS * 1
                states[step] = next_state
                dones[step] = next_done
                # get action and value in same function
                action, logprob, _, value = agent.get_action_and_value(next_state)
                values[step] = value[0] #tf.([x])
                actions[step] = action #tf.([x])
                logprobs[step] = logprob #tf.([x])
                next_state, reward, done, info = env.step(action[0]) 
                #next_state = [], reward = tf.([x]), done = bool, action[0] = []
                rewards[step] = reward
                next_done = done

            # bootstrap value if not done
            next_value = agent.get_value(next_state)[0] #tf.([x])
            # if args.gae:
            #     advantages = torch.zeros_like(rewards).to(device)
            #     lastgaelam = 0
            #     for t in reversed(range(args.num_steps)):
            #         if t == args.num_steps - 1:
            #             nextnonterminal = 1.0 - next_done
            #             nextvalues = next_value
            #         else:
            #             nextnonterminal = 1.0 - dones[t + 1]
            #             nextvalues = values[t + 1]
            #         delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            #         advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            #     returns = advantages + values
            # if not gae:
            returns = np.zeros_like(rewards)
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + GAMMA * nextnonterminal * next_return
            advantages = returns - values

            # flatten the batch
            b_states = states.reshape((-1,) + env.observation_space.shape)# (N*M, 13)
            b_logprobs = logprobs.reshape(-1)# (N*M,)
            b_actions = actions.reshape((-1,) + env.action_space.shape)# (N*M, 1)
            b_advantages = advantages.reshape(-1)# (N*M,)
            b_returns = returns.reshape(-1)# (N*M,)
            b_values = values.reshape(-1)# (N*M,)
            b_rewards = rewards.reshape(-1)
            rollout_reward = np.sum(b_rewards)
            
            # update ppo
            b_inds = np.arange(BATCH_SIZE)
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            for epoch in range(NUM_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    mb_advantages = b_advantages[mb_inds]
                    # advantage nomalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    with tf.GradientTape() as tape:
                        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = tf.exp(logratio)
                        # Policy loss
                        surr = mb_advantages * ratio
                        p_loss = -tf.reduce_mean(
                        tf.minimum(surr,
                                tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * mb_advantages)
                                )
                        
                        # Value loss
                        v_loss = tf.reduce_mean(tf.square(newvalue - b_returns[mb_inds]))
                        # Entropy loss
                        e_loss = tf.reduce_mean(entropy)

                    # minimal policy loss and value loss, but maximal entropy loss, maximal entropy will let the agent explore more
                    print('p_loss', p_loss)
                    print('v_loss', v_loss)
                    print('e_loss', e_loss)
                    a_gard = tape.gradient(p_loss, agent.actor.trainable_weights)
                    agent.actor_opt.apply_gradients(zip(a_gard, agent.actor.trainable_weights))
                    c_grad = tape.gradient(v_loss, agent.critic.trainable_weights)
                    agent.critic_opt.apply_gradients(zip(c_grad, agent.critic.trainable_weights))

                    actor_losses.append(p_loss.numpy())
                    critic_losses.append(v_loss.numpy())
                    entropy_losses.append(e_loss.numpy())

            with agent.summary_writer.as_default():
                tf.summary.scalar('actor_loss', np.mean(actor_losses), step=total_step)
                tf.summary.scalar('critic_loss', np.mean(critic_losses), step=total_step)
                tf.summary.scalar('entropy', np.mean(entropy_losses), step=total_step)

            actor_loss_avg = np.mean(actor_losses)
            critic_loss_avg = np.mean(critic_losses)
            print(f"| Avg. Loss (Act / Crit): {actor_loss_avg:.3f} / {critic_loss_avg:.3f} ")
            print( "'-----------------------------------------")
            
        all_rollout_reward.append(rollout_reward)
        print(
            'Training  | Update: {}/{}  | Rollout Reward: {:.4f}  | Running Time: {:.4f}'.format(
                updates + 1, num_updates, rollout_reward, time.time() - t0)
        )
        agent.save()

        plt.plot(all_rollout_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # test
        agent.load()
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            episode_reward = env.total_reward_scale
            for step in range(MAX_STEPS):
            # for step in range(5):
                env.render()
                action = agent.get_action(state, greedy=True)[0]
                state, reward, done, info = env.step(action)
                print(3 * action + 3)
                print(reward)

                episode_reward += reward
                if done:
                    break
            print(state)
        
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0))
            
    if args.random:
        # random
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            episode_reward = env.total_reward_scale
            for step in range(MAX_STEPS):
                env.render()
                np.random.seed(42)
                action = np.random.uniform(low=-1, high=1)
                state, reward, done, info = env.step(action)
                print(reward)

                episode_reward += reward
                if done:
                    break
            print(state)
        
            print(
                'Random Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0))
            
# for _ in range(int(replay_len / batch_size)):
#     data = RelplayBuffer.random_sample(batch_size)
# ReplayBuffer_size = 4096
# batch_size = 512
# repeat_time = 8
