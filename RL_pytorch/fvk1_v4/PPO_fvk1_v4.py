import argparse
import os
import random
import time
import datetime
import math
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import EMB_env_fvk1_v4
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-id", type=str, default="EMB-fvk1-v4",
        help="the id of the environment")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="emb-ppo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--train-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to train the model")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to save the model")
    parser.add_argument("--test-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="weather to test the model")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, run_name):
    def thunk():
        env = EMB_env_fvk1_v4.EMB_All_info_Env()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

def make_env_test(env_id, seed, idx, run_name):
    def thunk():
        env = EMB_env_fvk1_v4.EMB_All_info_Env()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def save_model(num):
    model_path = f"runs/{run_name}/{args.exp_name}_{num}.pth"
    # torch.save(agent.state_dict(), model_path)
    print(f"model saved to {model_path}")

    obs_rms_list = []
    rew_rms_list = []
    for i in range(len(envs.envs)):
        env = envs.envs[i]
        obs_rms = env.get_wrapper_attr('obs_rms')
        # rew_rms = getattr(env, 'rew_rms', None)
        if obs_rms is not None:
            obs_rms_list.append({
                'mean': obs_rms.mean,
                'var': obs_rms.var
            })

        # if rew_rms is not None:
        #     rew_rms_list.append({
        #         'mean': rew_rms.mean,
        #         'var': rew_rms.var
        #     })

    torch.save({
        'model_state_dict': agent.state_dict(),
        'obs_rms_list': obs_rms_list
        # 'rew_rms_list': rew_rms_list
    }, model_path)

    return model_path

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
        # ax2.set_ylim(-1e8, 1e8)
        ax1.set_xlabel('Time / ms')
        ax1.set_ylabel('Action (V)', color=color)
        ax1.plot(action_buffer, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=6, color=color, linestyle='--', linewidth=1)
        ax1.axhline(y=-6, color=color, linestyle='--', linewidth=1)
        color = 'tab:red'
        ax2.set_ylabel('Step Reward', color=color)
        ax2.plot(reward_buffer, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f'Experiment {i + 1}')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) 
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', 'action_reward_6_tests.jpg'), dpi=300)
    plt.close()

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(7, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(3, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(torch.cat((x[:, :2], x[:, -5:]), dim=1))

    def get_action_and_value(self, x, action=None):
        actor_obs = x[:,2:5]
        critic_obs =  torch.cat((x[:, :2], x[:, -5:]), dim=1)
        action_mean = self.actor_mean(actor_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(critic_obs)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    if args.train_model:
        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed, i, run_name) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        num_updates = args.total_timesteps // args.batch_size

        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if update!=0 and update%100==0 and args.save_model:
                print('savedddddddddddddddddddddddddddddddddd')
                save_model(update)

    if args.save_model:
        model_path = save_model(num_updates)

    if args.test_model:
        # model_path = f"runs/EMB-fvk1-v4__ppo_fvk1_v4__4654__20241025-noback/ppo_fvk1_v4_244.pth"
        epsilon = 1e-8
        eval_episodes = 6
        # use the rms in the first env
        # env = envs.envs[0] 
        # obs_rms = env.get_wrapper_attr('obs_rms')

        checkpoint = torch.load(model_path, map_location=device)
        
        obs_rms_list = checkpoint.get('obs_rms_list', [])
        # rew_rms_list = checkpoint.get('rew_rms_list', [])
        
        env_test = gym.vector.SyncVectorEnv([make_env_test(args.env_id, args.seed, 0, run_name=f"{run_name}-eval")])
        # the rms will not  be recorded at here
        means = [item['mean'] for item in obs_rms_list]
        vars = [item['var'] for item in obs_rms_list]
        mean_avg = np.mean(means, axis=0)
        var_avg = np.mean(vars, axis=0)

        agent = Agent(env_test).to(device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        agent.eval()

        next_obs, _ = env_test.reset(seed=args.seed+4)
        next_obs_norm = (next_obs - mean_avg) / np.sqrt(var_avg + epsilon)
        episodic_returns = []
        cont = 1
        step = 0
        total_reward_test = 0
        position_buffer = [0.0]
        velocity_buffer = [0.0]
        action_buffer = [0.0]
        reward_buffer = [0.0]
        position_buffers = []
        velocity_buffers = []
        action_buffers = []
        reward_buffers = []

        while len(episodic_returns) < eval_episodes:
            next_obs_norm_actor =  next_obs_norm[:,2:5]
            with torch.no_grad():
                actions = agent.actor_mean(torch.Tensor(next_obs_norm_actor).to(device)).detach()

            # if step <= 300:
            #     actions = torch.Tensor([[1.0]]) if (step // 20) % 2 == 0 else torch.Tensor([[-1.0]])
            # else:
            #     actions = torch.Tensor([[0.0]])
            # step += 1

            next_obs, reward_test, _, _, infos = env_test.step(actions.cpu().numpy())
            total_reward_test += reward_test[0]
            # if not "final_info" in infos:
            position_buffer.append(next_obs[0][2])
            velocity_buffer.append(next_obs[0][3])  #len(pos) = len(act) - 1
            action_buffer.append(6 * np.clip(actions.item(), -1, 1))
            # reward_buffer.append(total_reward_test)    
            reward_buffer.append(reward_test[0]) 

            next_obs_norm = (next_obs - mean_avg) / np.sqrt(var_avg + epsilon)
   
            if "final_info" in infos:
                position_buffer.pop()
                velocity_buffer.pop()
                for obs in infos["final_observation"]:
                    position_buffer.append(obs[2])
                    velocity_buffer.append(obs[3])
                position_buffers.append(position_buffer)
                velocity_buffers.append(velocity_buffer)
                action_buffers.append(action_buffer)
                reward_buffers.append(reward_buffer)
                for idx, action in enumerate(action_buffer):
                    writer.add_scalar(f"eval/action_each_step_{cont}", action, idx)
                for idx, total_reward_test in enumerate(reward_buffer):
                    writer.add_scalar(f"eval/step_reward_{cont}", total_reward_test, idx)
                for idx, position in enumerate(position_buffer):
                    writer.add_scalar(f"eval/position_each_step_{cont}", position, idx)
                for idx, velocity in enumerate(velocity_buffer):
                    writer.add_scalar(f"eval/velocity_each_step_{cont}", velocity, idx)

                cont += 1
                step = 0
                total_reward_test = 0
                position_buffer = [0.0]
                velocity_buffer = [0.0]
                action_buffer = [0.0]
                reward_buffer = [0.0]
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns += [info["episode"]["r"]]

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        draw(position_buffers, velocity_buffers)
        draw_action_reward(action_buffers, reward_buffers)

    envs.close()
    writer.close()