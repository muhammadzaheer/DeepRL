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

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.reset = True

    def _transition(self):
        if self.reset is True:
            self._state = self._task.reset()
            self.reset = False
        config = self.config
        with config.lock:
            if config.l2:
                q_values, phi = self._network(config.state_normalizer(self._state))
            elif config.drop:
                dropout_mask = config.drop_dist.sample()
                q_values = self._network(config.state_normalizer(self._state), dropout_mask)
            else:
                q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state

        return entry


class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        self.reset = True
        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1

            self.ep_steps += 1
            reward = config.reward_normalizer(reward)
            if done or self.ep_steps == self.timeout:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.ep_steps = 0
                self.actor.reset = True
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            for _ in range(self.config.num_replay):
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network(next_states) if self.config.use_target_network else self.network(next_states)
                q_next = q_next.detach()
                if self.config.double_q:
                    best_actions = torch.argmax(self.network(next_states), dim=-1)
                    q_next = q_next[self.batch_indices, best_actions]
                else:
                    q_next = q_next.max(1)[0]
                terminals = tensor(terminals)
                rewards = tensor(rewards)

                # discount = self.config.discount_schedule()
                # q_next = discount * q_next * (1 - terminals)
                # print(discount); exit()
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).long()
                q = self.network(states)
                q = q[self.batch_indices, actions]
                loss = (q_next - q).pow(2).mul(0.5).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.optimizer.step()

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())


class DQNExpectedSarsaAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        self.reset = True
        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1

            self.ep_steps += 1
            reward = config.reward_normalizer(reward)
            if done or self.ep_steps == self.timeout:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.ep_steps = 0
                self.actor.reset = True
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            for _ in range(self.config.num_replay):
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.target_network(next_states) if self.config.use_target_network else self.network(next_states)
                q_next = q_next.detach()
                if self.config.double_q:
                    best_actions = torch.argmax(self.network(next_states), dim=-1)
                    q_next = q_next[self.batch_indices, best_actions]
                else:
                    if self.config.target_epsilon is None:
                        probs = torch.ones_like(q_next) * self.config.epsilon_end/self.config.action_dim
                        batch_indices = torch.arange(probs.size(0)).long()
                        probs[batch_indices, q_next.max(1)[1].type(torch.LongTensor)] += 1 - self.config.epsilon_end
                    else:
                        probs = torch.ones_like(q_next) * self.config.target_epsilon / self.config.action_dim
                        batch_indices = torch.arange(probs.size(0)).long()
                        probs[batch_indices, q_next.max(1)[1].type(torch.LongTensor)] += 1-self.config.target_epsilon


                    q_next = q_next * probs
                    q_next = torch.sum(q_next, 1)
                terminals = tensor(terminals)
                rewards = tensor(rewards)

                # discount = self.config.discount_schedule()
                # q_next = discount * q_next * (1 - terminals)
                # print(discount); exit()
                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).long()
                q = self.network(states)
                q = q[self.batch_indices, actions]
                loss = (q_next - q).pow(2).mul(0.5).mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.optimizer.step()

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())


class DQNTileActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.reset = True
        self.tiles_rep = config.tiles_rep
        self.ts = None
        self.tns = None

    def _transition(self):
        config = self.config

        if self.reset is True:
            self._state = self._task.reset()
            state = config.state_normalizer(self._state)
            self.ts = self.tiles_rep.get_representation(state[0])

            self.reset = False

        with config.lock:
            q_values = self._network(self.ts)
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        self.tns = self.tiles_rep.get_representation(next_state[0])
        entry = [self.ts, action, reward[0], self.tns, int(done[0]), info]
        self.ts = self.tns
        self._total_steps += 1
        self._state = next_state

        return entry


class DQNTileAgent(BaseAgent):
    # TODO: I'm tiling three times, whereas I could easily get away with tiling ones
    # if I keep track of tilings

    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.tiles_rep = TileCoder(config)

        self.replay = config.replay_fn()

        config.tiles_rep = self.tiles_rep
        self.actor = DQNTileActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        self.reset = True
        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        ts = self.tiles_rep.get_representation(state[0])
        q = self.network(ts)
        if np.random.rand() < self.config.random_action_prob():
            action = np.random.randint(0, len(q))
        else:
            action = np.argmax(to_np(q))

        self.config.state_normalizer.unset_read_only()
        return action


    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1

            self.ep_steps += 1
            reward = config.reward_normalizer(reward)
            if done or self.ep_steps == self.timeout:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.ep_steps = 0
                self.actor.reset = True
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            ts, actions, rewards, tns, terminals = experiences
            assert ts.dtype == np.float64
            q_next = self.target_network(tns) if self.config.use_target_network else self.network(tns)
            q_next = q_next.detach()
            q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(ts)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def get_values(self, s):
        q_values = self.network(s)
        q_values = to_np(q_values).flatten()
        return q_values


class DQNAgentL2(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        self.reset = True
        self.ep_steps = 0
        self.timeout = config.timeout

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q, phi = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1

            self.ep_steps += 1
            reward = config.reward_normalizer(reward)
            if done or self.ep_steps == self.timeout:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.ep_steps = 0
                self.actor.reset = True
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next, phi_next = self.target_network(next_states) if self.config.use_target_network else self.network(next_states)
            q_next = q_next.detach()
            q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q, phi = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            l2_penalty = torch.mean(torch.bmm(phi.unsqueeze(2), phi.unsqueeze(1)).sum(2).sum(1))
            loss += self.config.lmbda*l2_penalty
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())


class DQNAgentDrop(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.episode_reward = 0
        self.episode_rewards = []

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)


        self.reset = True
        self.ep_steps = 0
        self.timeout = config.timeout

        self.config.drop_dist = torch.distributions.Bernoulli(torch.ones(self.config.hidden_units) * (1-self.config.p))


    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)
        action = np.argmax(to_np(q))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, _ in transitions:
            self.episode_reward += reward
            self.total_steps += 1

            self.ep_steps += 1
            reward = config.reward_normalizer(reward)
            if done or self.ep_steps == self.timeout:
                self.episode_rewards.append(self.episode_reward)
                self.episode_reward = 0
                self.ep_steps = 0
                self.actor.reset = True
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        if self.total_steps > self.config.exploration_steps:
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)


            drop_mask = self.config.drop_dist.sample()


            q_next = self.target_network(next_states, drop_mask) if self.config.use_target_network else self.network(next_states, drop_mask)
            q_next = q_next.detach()
            q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)

            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()


            q = self.network(states, drop_mask)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())