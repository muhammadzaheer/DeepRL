#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
from ..component.trace import Trace

from torch.autograd import Variable

class SarsaAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.network = config.network_fn()
        self.network.share_memory()
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.state = None
        self.ts = None
        self.action = None
        self.next_state = None
        self.tns = None
        self.next_action = None
        self.reset = True

        self.task = config.task_fn()
        self.total_steps = 0

        self.tiles_rep = TileCoder(config)

        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        pass

    def start(self):
        config = self.config

        self.state = self.task.reset()
        state = config.state_normalizer(self.state)
        self.ts = self.tiles_rep.get_representation(state[0])
        q_values = self.get_values(self.ts)

        self.action = self.select_action(q_values)

        self.ep_steps = 0
        self.reset = False

    def step(self):
        if self.reset:
            self.start()

        self.next_state, reward, done, _ = self.task.step([self.action])

        reward = reward[0]
        next_state = self.config.state_normalizer(self.next_state)
        self.tns = self.tiles_rep.get_representation(next_state[0])
        q_next = self.get_values(self.tns)
        self.next_action = self.select_action(q_next)

        # UPDATE
        if self.total_steps > self.config.exploration_steps:
            q_next = tensor(q_next)[self.next_action]
            q_next = self.config.discount * q_next * (1 - tensor(int(done)))
            q_next.add_(tensor(reward))
            q = self.network(self.ts)
            q = q[self.action]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.state = self.next_state
        self.action = self.next_action
        self.ts = self.tns

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.reset = True
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0

    def select_action(self, q_values):
        if self.total_steps < self.config.exploration_steps or np.random.rand() < self.config.random_action_prob():
            return np.random.randint(0, len(q_values))
        else:
            return np.argmax(q_values)

    def get_values(self, s):
        q_values = self.network(s)
        q_values = to_np(q_values).flatten()
        return q_values

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        state =  self.tiles_rep.get_representation(state[0])
        q = self.network(state)
        if np.random.rand() < self.config.random_action_prob():
            action = np.random.randint(0, len(q))
        else:
            action = np.argmax(to_np(q))

        self.config.state_normalizer.unset_read_only()
        return action


class SarsaLmbdaAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config

        self.network = config.network_fn()
        self.network.share_memory()

        self.trace = Trace(self.network)
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0

        self.state = None
        self.ts = None
        self.action = None
        self.next_state = None
        self.tns = None
        self.next_action = None
        self.reset = True

        self.task = config.task_fn()
        self.total_steps = 0

        self.tiles_rep = TileCoder(config)

        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        pass

    def start(self):
        config = self.config

        self.state = self.task.reset()
        state = config.state_normalizer(self.state)
        self.ts = self.tiles_rep.get_representation(state[0])
        q_values = self.get_values(self.ts)

        self.action = self.select_action(q_values)

        self.ep_steps = 0
        self.reset = False

        # Resetting trace
        self.trace.reset()

    def step(self):
        if self.reset:
            self.start()

        self.next_state, reward, done, _ = self.task.step([self.action])

        reward = reward[0]
        next_state = self.config.state_normalizer(self.next_state)
        self.tns = self.tiles_rep.get_representation(next_state[0])
        q_next = self.get_values(self.tns)
        self.next_action = self.select_action(q_next)

        # UPDATE
        q_next = tensor(q_next)[self.next_action]
        q_next = self.config.discount * q_next * (1 - tensor(int(done)))
        q_next.add_(tensor(reward))
        q = self.network(self.ts)
        q = q[self.action]

        delta = (q_next - q).item()

        self.optimizer.zero_grad()
        q.backward()
        self.trace.mul(self.config.discount * self.config.lmbda)
        self.trace.add_network_gradient(self.network)
        self.trace.copy_to_network(self.network)
        self.mul_network_grad_by_const(-delta)
        self.optimizer.step()

        self.state = self.next_state
        self.action = self.next_action
        self.ts = self.tns

        self.episode_reward += reward
        self.total_steps += 1
        self.ep_steps += 1
        if done or self.ep_steps == self.timeout:
            self.reset = True
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0

    def select_action(self, q_values):
        if np.random.rand() < self.config.random_action_prob():
            return np.random.randint(0, len(q_values))
        else:
            return np.argmax(q_values)

    def get_values(self, s):
        q_values = self.network(s)
        q_values = to_np(q_values).flatten()
        return q_values

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        state =  self.tiles_rep.get_representation(state[0])
        q = self.network(state)
        if np.random.rand() < self.config.random_action_prob():
            action = np.random.randint(0, len(q))
        else:
            action = np.argmax(to_np(q))

        self.config.state_normalizer.unset_read_only()
        return action

    def mul_network_grad_by_const(self, const):
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad.mul_(float(const))