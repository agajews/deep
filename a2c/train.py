import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from multi_env import MultiEnv


def train(model,
          optim,
          env_fn,
          num_envs,
          num_stack,
          num_steps,
          num_updates,
          gamma,
          value_loss_coef,
          entropy_coef,
          max_grad_norm,
          log_freq=10):
    envs = MultiEnv(env_fn, num_envs)

    model.cuda()

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, obs_shape[1], obs_shape[2])

    states = torch.zeros(num_steps + 1, num_envs, *obs_shape)
    current_state = torch.zeros(num_envs, *obs_shape)

    def update_current_state(state):
        state = torch.from_numpy(np.stack(state)).float()
        current_state[:, :-1] = current_state[:, 1:]
        current_state[:, -1] = state

    state = envs.reset()
    update_current_state(state)

    rewards = torch.zeros(num_steps, num_envs, 1)
    value_preds = torch.zeros(num_steps + 1, num_envs, 1)
    old_log_probs = torch.zeros(num_steps, num_envs, envs.action_space.n)
    returns = torch.zeros(num_steps + 1, num_envs, 1)

    actions = torch.LongTensor(num_steps, num_envs)
    masks = torch.zeros(num_steps, num_envs, 1)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_envs, 1])
    final_rewards = torch.zeros([num_envs, 1])

    states = states.cuda()
    current_state = current_state.cuda()
    rewards = rewards.cuda()
    value_preds = value_preds.cuda()
    old_log_probs = old_log_probs.cuda()
    returns = returns.cuda()
    actions = actions.cuda()
    masks = masks.cuda()

    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            value, logits = model(Variable(states[step], volatile=True))
            probs = F.softmax(logits)
            log_probs = F.log_softmax(logits).data
            actions[step] = probs.multinomial().data

            cpu_actions = actions[step].cpu()
            cpu_actions = cpu_actions.numpy()

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward),
                                                     1)).float()
            episode_rewards += reward

            np_masks = np.array([0.0 if done_ else 1.0 for done_ in done])

            # If done then clean the history of observations.
            pt_masks = torch.from_numpy(
                np_masks.reshape(np_masks.shape[0], 1, 1, 1)).float()
            pt_masks = pt_masks.cuda()
            current_state *= pt_masks

            update_current_state(state)
            states[step + 1].copy_(current_state)
            value_preds[step].copy_(value.data)
            old_log_probs[step].copy_(log_probs)
            rewards[step].copy_(reward)
            masks[step].copy_(torch.from_numpy(np_masks).unsqueeze(1))

            final_rewards *= masks[step].cpu()
            final_rewards += (1 - masks[step].cpu()) * episode_rewards

            episode_rewards *= masks[step].cpu()

        returns[-1] = model(Variable(states[-1], volatile=True))[0].data
        for step in reversed(range(num_steps)):
            returns[step] = returns[step + 1] * \
                gamma * masks[step] + rewards[step]

        # Reshape to do in a single forward pass for all steps
        values, logits = model(
            Variable(states[:-1].view(-1, *states.size()[-3:])))
        log_probs = F.log_softmax(logits)

        # Unreshape
        logits_size = (num_steps, num_envs, logits.size(-1))

        log_probs = F.log_softmax(logits).view(logits_size)
        probs = F.softmax(logits).view(logits_size)

        values = values.view(num_steps, num_envs, 1)
        logits = logits.view(logits_size)

        action_log_probs = log_probs.gather(2, Variable(actions.unsqueeze(2)))

        dist_entropy = -(log_probs * probs).sum(-1).mean()

        advantages = Variable(returns[:-1]) - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(Variable(advantages.data) * action_log_probs).mean()

        optim.zero_grad()
        (value_loss * value_loss_coef + action_loss -
         dist_entropy * entropy_coef).backward()

        nn.utils.clip_grad_norm(model.parameters(), max_grad_norm)
        optim.step()

        states[0].copy_(states[-1])

        if j % log_freq == 0:
            print(
                "Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, j * num_envs * num_steps,
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), -dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]))
