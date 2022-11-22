import copy
import math

import numpy as np
import torch
from torch.nn import Linear

from factored_rl import configs
from factored_rl.models.nnutils import extract, Sequential, Module
from .replaymemory import ReplayMemory

class DQNAgent():
    def __init__(self, action_space, model: Module, cfg: configs.Config):
        self.action_space = action_space
        self.cfg = cfg

        on_retrieve = {
            '*': lambda x: torch.as_tensor(np.asarray(x)).to(cfg.model.device),
            'ob': lambda x: x.float(),
            'reward': lambda x: x.float(),
            'terminal': lambda x: x.bool(),
            'truncated': lambda x: x.bool(),
            'next_ob': lambda x: x.float(),
            'action': lambda x: x.long()
        }
        self.replay = ReplayMemory(cfg.agent.replay_buffer_size, on_retrieve=on_retrieve)

        self.n_training_steps = 0
        n_actions = self.action_space.n

        q_net_template = Sequential(model.encoder, Linear(cfg.model.n_latent_dims, n_actions))
        self.q = q_net_template.to(cfg.model.device)
        self.q_target = copy.deepcopy(q_net_template).to(cfg.model.device)
        self.q_target.hard_copy_from(self.q)
        self.q.print_summary()
        self.replay.reset()
        params = list(self.q.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.trainer.rl_learning_rate)

    def save(self, fname, dir, is_best):
        self.q.save(fname, dir, is_best)

    def act(self, observation, testing=False):
        if ((len(self.replay) < self.cfg.agent.replay_warmup_steps
             or np.random.uniform() < self._get_epsilon(testing=testing))):
            action = self.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self._get_q_values_for_observation(observation)
                action = torch.argmax(q_values.squeeze())
        return int(action)

    def store(self, experience: dict):
        self.replay.push(experience)

    def update(self):
        if len(self.replay) < self.cfg.agent.replay_warmup_steps:
            return 0.0

        if len(self.replay) >= self.cfg.trainer.batch_size:
            fields = ['ob', 'action', 'reward', 'terminal', 'next_ob']
            batch = self.replay.sample(self.cfg.trainer.batch_size, fields)
            obs, actions, rewards, terminals, next_obs = batch
        else:
            return 0.0

        self.q.train()
        self.optimizer.zero_grad()
        q_values = self._get_q_predictions(obs, actions)
        q_targets = self._get_q_targets(rewards, terminals, next_obs)
        loss = torch.nn.functional.smooth_l1_loss(input=q_values, target=q_targets)
        loss.backward()
        self.optimizer.step()
        self.n_training_steps += 1

        if self.cfg.agent.target_copy_mode == 'soft':
            self.q_target.soft_copy_from(self.q, self.cfg.agent.target_copy_alpha)
        else:
            if self.n_training_steps % self.cfg.agent.target_copy_period == 0:
                self.q_target.hard_copy_from(self.q)

        return loss.detach().item()

    def _get_epsilon(self, testing=False):
        if testing:
            epsilon = self.cfg.agent.epsilon_final
        else:
            n_fractional_half_lives = self.n_training_steps / self.cfg.agent.epsilon_half_life_steps
            scale = self.cfg.agent.epsilon_initial - self.cfg.agent.epsilon_final
            epsilon = self.cfg.agent.epsilon_final + scale * math.pow(0.5, n_fractional_half_lives)
        return epsilon

    def _get_q_targets(self, rewards, terminals, next_obs):
        with torch.no_grad():
            # Compute Double-Q targets
            next_action = torch.argmax(self.q(next_obs), dim=-1) # (-1, )
            next_q = self.q_target(next_obs) # (-1, A)
            next_v = next_q.gather(-1, next_action.unsqueeze(-1)).squeeze(-1) # (-1, )
            non_terminal_idx = ~terminals # (-1, )
            q_targets = rewards + non_terminal_idx * self.cfg.agent.discount_factor * next_v # (-1, )
        return q_targets

    def _get_q_predictions(self, obs, action):
        q_values = self.q(obs) # (-1, A)
        q_acted = extract(q_values, idx=action, idx_dim=-1) #(-1, )
        return q_acted

    def _get_q_values_for_observation(self, obs):
        return self.q(torch.as_tensor(obs).float().to(self.cfg.model.device))
