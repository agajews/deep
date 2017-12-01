import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from multi_env import MultiEnv


def train(model, create_env, num_envs, optimizer, gamma, num_updates,
          max_episode_length, steps_per_update):
    # torch.manual_seed(args.seed)

    # env.seed(args.seed)

    model.train()

    env = MultiEnv(create_env, num_envs)
    state = env.reset()  # list of states for each concurrent env
    state = torch.from_numpy(state)
    episode_done = True

    episode_length = 0
    update = 0
    while update < num_updates:
        episode_length += 1

        values = []
        log_action_probs = []
        rewards = []
        entropies = []

        for step in range(steps_per_update):
            # list of values and action logits for each concurrent env
            value, action_logit = model(Variable(state))
            action_prob = F.softmax(action_logit)
            log_action_prob = F.log_softmax(action_logit)
            entropy = -(log_action_prob * action_prob).sum(1)
            entropies.append(entropy)

            action = action_prob.multinomial().data
            log_action_prob = log_action_prob.gather(1, Variable(action))

            state, reward, episode_done, _ = env.step(action.numpy())
            if episode_length >= max_episode_length:
                episode_done = True
            reward = max(min(reward, 1), -1)

            state = torch.from_numpy(state)
            values.append(value)
            log_action_probs.append(log_action_prob)
            rewards.append(reward)

            if episode_done:
                episode_length = 0
                state = env.reset()
                break

        R = torch.zeros(1, 1)
        if not episode_done:
            value, _ = model(Variable(state))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        advantage = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            advantage = Variable(advantage * gamma + rewards[i] +
                                 gamma * values[i + 1].data - values[i].data)

            policy_loss = policy_loss - log_action_probs[i] * advantage - 0.01 * entropies[i]

        loss = policy_loss + 0.5 * value_loss

        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 40)

        optimizer.step()
        update += 1
