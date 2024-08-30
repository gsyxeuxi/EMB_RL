import argparse
import os
import time
import datetime
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
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
entropy_coef = 0.0 # 0.01 
max_grad_norm = 0.5 # maximal gradient, to prevent form gradient explosion
# ppo-penalty parameters
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
    def __init__(self, envs):
        # critic
        with tf.name_scope('critic'):
            input = tf.keras.Input(shape = (None, np.array(envs.single_observation_space.shape).prod()), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            v = tf.keras.layers.Dense(1,  kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                      bias_initializer=tf.keras.initializers.Constant(1.0))(layer)
        self.critic = tf.keras.Model(input, v, name="critic")  
        self.critic.summary()            

        # actor
        with tf.name_scope('actor'):
            input = tf.keras.Input(shape = (None, np.array(envs.single_observation_space.shape).prod()), dtype=tf.float32, name='state')
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(input)
            layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                          bias_initializer=tf.keras.initializers.Constant(0.0))(layer)
            a_mean = tf.keras.layers.Dense(np.prod(envs.single_action_space.shape), activation='tanh', kernel_initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2)), 
                                           bias_initializer=tf.keras.initializers.Constant(0.01))(layer)
            logstd = tf.Variable(np.zeros(np.prod(envs.single_action_space.shape), dtype=np.float32), trainable=True, name='logstd') # This have no input, so it is state independent
        self.actor = tf.keras.Model(input, a_mean, name='actor')
        self.actor.logstd = logstd
        self.actor.summary()

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        # create tensorboard logs
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists('logs/'):
            os.makedirs('logs/')
        self.log_dir = 'logs/' + self.current_time
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)

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
        self.actor.save(os.path.join(path, 'actor.keras'))
        self.critic.save(os.path.join(path, 'critic.keras'))

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        self.actor = tf.keras.models.load_model(os.path.join(path, 'actor.keras'))
        self.critic = tf.keras.models.load_model(os.path.join(path, 'critic.keras'))

if __name__ == '__main__':
    env = EMB_env_v2.EMB_All_info_Env()
    env = gym.wrappers.NormalizeObservation(env)
    envs = gym.vector.SyncVectorEnv([lambda: env for i in range(NUM_ENVS)])
    # reproducible
    # env.seed(RANDOM_SEED)
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
        next_state, _ = envs.reset()
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
                next_state, reward, terminated, truncated, info = envs.step(action[0]) 
                #next_state = [], reward = tf.([x]), done = bool, action[0] = []
                rewards[step] = reward
                next_done = terminated or truncated
            
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
            b_states = states.reshape((-1,) + envs.single_observation_space.shape)# (N*M, 13)
            b_logprobs = logprobs.reshape(-1)# (N*M,)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)# (N*M, 1)
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

                    with tf.GradientTape() as a_tape:
                        _, newlogprob, entropy, _ = agent.get_action_and_value(b_states[mb_inds], b_actions[mb_inds])
                        logratio = newlogprob - b_logprobs[mb_inds]
                        ratio = tf.exp(logratio)
                        # Policy loss
                        surr = mb_advantages * ratio
                        p_loss = -tf.reduce_mean(
                        tf.minimum(surr,
                                tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * mb_advantages)
                                )
                        # Entropy loss
                        e_loss = tf.reduce_mean(entropy)
                        loss = p_loss - entropy_coef * e_loss
                    # minimal policy loss and value loss, but maximal entropy loss, maximal entropy will let the agent explore more
                    # print('p_loss', p_loss)
                    # print('e_loss', e_loss)
                    a_grad = a_tape.gradient(p_loss, agent.actor.trainable_weights)
                    # print('a_grad', a_grad) #a_grad seems 0
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
                state, reward, done, _, info = env.step(action)
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
            
