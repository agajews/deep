from env import get_env
from model import ActorCritic
from train import train
import torch.optim as optim

env_name = 'PongNoFrameskip-v4'
num_stack = 4
num_envs = 16

num_steps = 5
num_updates = int(10e6)

lr = 7e-4
eps = 1e-5
alpha = 0.99

max_grad_norm = 0.5

gamma = 0.99

value_loss_coef = 0.5
entropy_coef = 0.01


def run():
    dummy_env = get_env(env_name)
    model = ActorCritic(dummy_env.observation_space.shape[0] * num_stack,
                        dummy_env.action_space)
    del dummy_env
    optimizer = optimizer = optim.RMSprop(
        model.parameters(), lr, eps=eps, alpha=alpha)
    train(model, optimizer, lambda: get_env(env_name), num_envs, num_stack,
          num_steps, num_updates, gamma, value_loss_coef, entropy_coef,
          max_grad_norm)


if __name__ == '__main__':
    run()
