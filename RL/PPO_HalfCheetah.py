import argparse
import os
import time
import datetime
import gym
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mujoco

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--random', dest='random', action='store_true', default=False)
args = parser.parse_args()

'''
Use scaled FIM for training
'''

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#####################  hyper parameters  ####################

ENV_ID = 'HalfCheetah-v2'  # environment id
RANDOM_SEED = 1  # random seed
RENDER = True  # render while training
ALG_NAME = 'PPO'
NUM_ENVS = 1 # the number of parallel running envs
TOTAL_TIMESETPS = 500000  # total timesteps for training 25000=832s
TEST_EPISODES = 1  # total number of episodes for testing
NUM_STEPS = 2048 # total number of steps N*M 1024
GAMMA = 0.99  # reward discount 0.99 recommanded
LR_A = 0.0003  # learning rate for actor
LR_C = 0.0003  # learning rate for critic 0.0003 is a safe default
MINIBATCH_SIZE = 64  # update minibatch size, Recommend equating this to the width of the neural network
BATCH_SIZE = NUM_ENVS * NUM_STEPS
NUM_EPOCHS = 10 # number of epochs K for each minibatch
EPSILON = 0.2 # ppo-clip parameters
GAE_LAMBDA = 0.95
entropy_coef = 0.0 # 0.01 
max_grad_norm = 0.5 # maximal gradient, to prevent form gradient explosion

###############################  PPO  ####################################
class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Constant(0.0))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Constant(0.0))
        self.a_mean = tf.keras.layers.Dense(action_dim, activation='tanh',
                                            kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)),
                                            bias_initializer=tf.keras.initializers.Constant(0.01))
        self.logstd = tf.Variable(initial_value=np.zeros(action_dim, dtype=np.float32), trainable=True, name='logstd')

    def build(self, input_shape):
        super(Actor, self).build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        mean = self.a_mean(x)
        std = tf.exp(self.logstd)
        return mean, std
    
class PPO(object):
    """
    PPO class
    """
    def __init__(self, envs):
        # critic
        with tf.name_scope('critic'):
            input = tf.keras.Input(shape = (None, np.array(envs.single_observation_space.shape).prod()), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            v = tf.keras.layers.Dense(1,  kernel_initializer=tf.keras.initializers.Orthogonal(gain=1.0), 
                                      bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
        self.critic = tf.keras.Model(input, v, name="critic")  
         
        # actor
        with tf.name_scope('actor'):
            input = tf.keras.Input(shape = (None, np.array(envs.single_observation_space.shape).prod()), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            a_mean = tf.keras.layers.Dense(np.prod(envs.single_action_space.shape), kernel_initializer=tf.keras.initializers.Orthogonal(gain=0.01), 
                                           bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            logstd = tf.Variable(np.zeros(np.prod(envs.single_action_space.shape), dtype=np.float32), trainable=True, name='logstd') # This have no input, so it is state independent
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

    def get_action_and_value(self, state, action=None, greedy=False): 
        """
        Choose action and get value
        :param state: state
        :param greedy: choose action greedy or not
        :return: clipped action
        """
        state = state[np.newaxis, :].astype(np.float32) # from [,,,] to [[,,,]]
        mean, std = self.actor(state), tf.exp(self.actor.logstd)
        probs = tfp.distributions.Normal(mean, std) #this function can do the broadcast automatically
        if greedy:
            action = mean[0]
        if action is None:
            action = tf.squeeze(probs.sample(1), axis=0)[0] # choosing action
        
        log_prob = tf.reduce_sum(probs.log_prob(action), axis=-1)
        entropy = tf.reduce_sum(probs.entropy(), axis=-1)
        value = self.critic(state)
        return action, log_prob, entropy, value

    def get_value(self, state):
        state = state[np.newaxis, :].astype(np.float32) # from [,,,] to [[,,,]]
        return self.critic(state)

    def save_weight(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        self.actor.save_weights(os.path.join(path, 'actor.h5'))
        self.critic.save_weights(os.path.join(path, 'critic.h5'))

    def load_weight(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        self.critic.load_weights(os.path.join(path, 'critic.h5'))
        self.actor.load_weights(os.path.join(path, 'actor.h5'))
        

if __name__ == '__main__':
    env = gym.make(ENV_ID)
    # env = gym.make(ENV_ID, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)
    # env.reset(seed=RANDOM_SEED)
    envs = gym.vector.SyncVectorEnv([lambda: env for i in range(NUM_ENVS)])

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    agent = PPO(envs)
    states = np.zeros((NUM_STEPS, NUM_ENVS) + envs.observation_space.shape, dtype=np.float32)
    # shape=(512, 1, 13)
    actions = np.zeros((NUM_STEPS, NUM_ENVS) + envs.action_space.shape, dtype=np.float32)
    # shape=(512, 1, 1)
    logprobs = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    rewards = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    dones = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)
    values = np.zeros((NUM_STEPS, NUM_ENVS), dtype=np.float32)

    t0 = time.time()
    if args.train:
        total_step = 0
        next_state, _ = envs.reset(seed=RANDOM_SEED)
        next_done = tf.zeros(NUM_ENVS)
        # episode_reward = env.total_reward_scale # this need to be reconsider
        num_updates = TOTAL_TIMESETPS // BATCH_SIZE 

        all_rollout_reward = []
        for updates in range(num_updates):
            for step in range(NUM_STEPS):  # N envs with M steps
                if RENDER:
                    env.render()
                total_step += NUM_ENVS * 1
                states[step] = next_state
                dones[step] = next_done
                # get action and value in same function
                action, logprob, _, value = agent.get_action_and_value(next_state)
                values[step] = value[0] #tf.([x])
                actions[step] = action #tf.([x])
                logprobs[step] = logprob #tf.([x])
                next_state, reward, terminated, truncated, info = envs.step(action[0]) 
                #next_state = [], reward = tf.([x]), done = bool, action[0] = []
                rewards[step] = reward
                next_done = terminated or truncated

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={total_step}, episodic_return={item['episode']['r']}")
                        # writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        # writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
            
            # bootstrap value if not done
            next_value = agent.get_value(next_state)[0] #tf.([x])

            # if args.gae:
            advantages = np.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

            # flatten the batch
            b_states = states.reshape((-1,) + envs.single_observation_space.shape)# (N*M, 13)
            b_logprobs = logprobs.reshape(-1)# (N*M,)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)# (N*M, 1)
            b_advantages = advantages.reshape(-1)# (N*M,)
            b_returns = returns.reshape(-1)# (N*M,)
            b_values = values.reshape(-1)# (N*M,)
            b_rewards = rewards.reshape(-1)
            rollout_reward = np.sum(b_rewards)
            all_rollout_reward.append(rollout_reward)
            
            # update ppo
            b_inds = np.arange(BATCH_SIZE)
            actor_losses = []
            critic_losses = []
            entropy_losses = []
            old_approx_kl = []
            approx_kl = []
            clipfracs = []
            for epoch in range(NUM_EPOCHS):
                np.random.shuffle(b_inds)
                for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                    end = start + MINIBATCH_SIZE
                    mb_inds = b_inds[start:end]
                    with tf.GradientTape() as a_tape:
                        _, newlogprob, entropy, _ = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = tf.exp(logratio)
                        old_app_kl = np.mean(-logratio)
                        app_kl = np.mean((ratio - 1) - logratio)
                        clipfracs += [np.mean(tf.abs(ratio - 1.0) > EPSILON)]

                        mb_advantages = b_advantages[mb_inds]
                        # advantage nomalization
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        # print('adv mean', mb_advantages.mean())
                        # print('adv std', mb_advantages.std())

                        # Policy loss
                        surr = mb_advantages * ratio
                        # print(mb_advantages)
                        p_loss = -tf.reduce_mean(
                        tf.minimum(surr,
                                tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * mb_advantages)
                                )
                        # Entropy loss
                        e_loss = tf.reduce_mean(entropy)
                        # loss = p_loss - entropy_coef * e_loss
                    # minimal policy loss and value loss, but maximal entropy loss, maximal entropy will let the agent explore more
                    a_grad = a_tape.gradient(p_loss, agent.actor.trainable_weights)
                    a_grad_clipped, _ = tf.clip_by_global_norm(a_grad, max_grad_norm)
                    agent.actor_opt.apply_gradients(zip(a_grad, agent.actor.trainable_weights))

                    with tf.GradientTape() as c_tape:
                        _, _, _, newvalue = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                        # Value loss
                        v_loss = 0.5 * tf.reduce_mean(tf.square(newvalue - b_returns[mb_inds]))
                    c_grad = c_tape.gradient(v_loss, agent.critic.trainable_weights)
                    # print('c_grad', c_grad) #c_grad seems very big
                    c_grad_clipped, _ = tf.clip_by_global_norm(c_grad, max_grad_norm)
                    agent.critic_opt.apply_gradients(zip(c_grad, agent.critic.trainable_weights))
                    # print('v_loss', v_loss)

                    actor_losses.append(p_loss.numpy())
                    critic_losses.append(v_loss.numpy())
                    entropy_losses.append(e_loss.numpy())
                    old_approx_kl.append(old_app_kl)
                    approx_kl.append(app_kl)

            with agent.summary_writer.as_default():
                tf.summary.scalar('actor_loss', np.mean(actor_losses), step=total_step)
                tf.summary.scalar('critic_loss', np.mean(critic_losses), step=total_step)
                tf.summary.scalar('reward', rollout_reward, step=total_step)
                tf.summary.scalar('old_approx_kl', np.mean(old_approx_kl), step=total_step)
                tf.summary.scalar('approx_kl', np.mean(approx_kl), step=total_step)
                tf.summary.scalar('clipfracs', np.mean(clipfracs), step=total_step )
                tf.summary.scalar('entropy', np.mean(entropy_losses), step=total_step)
            actor_loss_avg = np.mean(actor_losses)
            critic_loss_avg = np.mean(critic_losses)
            print(f"| Avg. Loss (Act / Crit): {actor_loss_avg:.3f} / {critic_loss_avg:.3f} ")
            print(
                'Training  | Update: {}/{}  | Rollout Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    updates + 1, num_updates, rollout_reward, time.time() - t0)
            )
            print( "'-----------------------------------------")
        agent.save_weight()

        plt.plot(all_rollout_reward)
        plt.xlabel('Updates')  
        plt.ylabel('Reward')       
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

        envs.close()

    if args.test:
        # test
        print('____________________test__________________')
        agent.load_weight()
        state_mean = 000
        state_var = 000
        epsilon = 1e-8
        epsiode_action = []
        for episode in range(TEST_EPISODES):
            state, _ = env.reset(seed=3)
            print('bigging state', state)
            # episode_reward = env.total_reward_scale
            episode_reward = 0
            for step in range(400):
                if RENDER:
                    env.render()
                # state = (state - state_mean) / np.sqrt(state_var + epsilon)
                action, logprob, _, value = agent.get_action_and_value(state, greedy=True)
                # print(action)
                state, reward, terminated, truncated, info = env.step(action[0])
                # print(state, reward)
                epsiode_action.append(action)
                episode_reward += reward
                if terminated:
                    print('terminated')
                    break

            plt.plot(epsiode_action)
            plt.ylim(-1, 1)
            plt.xlabel('Time / ms')  
            plt.ylabel('Current / A')       
            if not os.path.exists('image'):
                os.makedirs('image')
            plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID, 'test'])))

            # env.draw()
            print('Reward: ', episode_reward)
        env.close()
            
    if args.random:
        # random
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            episode_reward = env.total_reward_scale
            for step in range(MAX_STEPS):
                env.render()
                np.random.seed(42)
                # action = np.random.uniform(low=-1, high=1)
                action = env.action_space.sample()
                state, reward, done, _, info = env.step(action)
                print(reward)

                episode_reward += reward
                if done:
                    break
            print(state)
        
            print(
                'Random Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - t0))