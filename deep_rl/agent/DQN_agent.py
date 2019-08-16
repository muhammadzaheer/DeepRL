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

from numpy import linalg as LA


class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.reset = True
        # self._task.env.envs[0].render()

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
        # self._task.env.envs[0].render()
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


        self.config.ep_count = 0
        self.config.render = False

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
                self.config.ep_count += 1

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
                # nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
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


class DQNSoftmaxActor(BaseActor):
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
        tensor_q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(tensor_q_values).flatten()

        beta_q_values = self.config.beta * q_values

        b = np.max(beta_q_values)
        exp_q_values = np.exp(beta_q_values - b)
        action_probs = exp_q_values/np.sum(exp_q_values)
        action = np.random.choice(self.config.action_dim, 1, p=action_probs)[0]
        # try:
        #     action = np.random.choice(self.config.action_dim, 1, p=action_probs)[0]
        # except ValueError:
        #     print(torch.isnan(tensor_q_values))
        #     print(tensor_q_values)
        #     print(q_values)
        #     print(exp_q_values)
        #     print(action_probs)
        #     quit()
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state

        return entry


class DQNExpectedSarsaSoftmaxAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNSoftmaxActor(config)

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
        self.config.beta = 1024.0

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

                beta_q_next = self.config.beta * q_next

                b = torch.max(beta_q_next, dim=1)[0]
                b = b.unsqueeze(1).repeat(1, 4)
                # exp_q_next = torch.exp(beta_q_next - b) + 1e-10
                exp_q_next = torch.exp(beta_q_next - b)
                sums = torch.sum(exp_q_next, dim=1)
                sums = sums.unsqueeze(1)
                sums = sums.repeat(1, 4)

                probs = exp_q_next / sums

                q_next = q_next * probs

                if torch.any(torch.isnan(q_next)).item():
                    print(beta_q_next)
                    print(exp_q_next)
                    print(q_next)
                    print(probs)


                q_next = torch.sum(q_next, 1)

                terminals = tensor(terminals)
                rewards = tensor(rewards)

                q_next = self.config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                actions = tensor(actions).long()
                q = self.network(states)
                q = q[self.batch_indices, actions]
                loss = (q_next - q).pow(2).mul(0.5).mean()
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
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


class DQNNumpyActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.reset = True
        self.params_np = None

    def _transition(self):
        if self.reset is True:
            self._state = self._task.reset()
            self.reset = False
        config = self.config
        np_q_values = self.forward(tensor(config.state_normalizer(self._state)).numpy())
        np_q_values = np_q_values.flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(np_q_values))
        else:
            action = np.argmax(np_q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state

        return entry

    def set_params_np(self, params_np):
        self.params_np = params_np

    def forward(self, x):
        W1, b1 = self.params_np[2].T, np.expand_dims(self.params_np[3], 0)
        z1 = np.dot(x, W1) + b1
        h1 = np.maximum(z1, 0)

        W2, b2 = self.params_np[0].T, np.expand_dims(self.params_np[1], 0)
        z2 = np.dot(h1, W2) + b2
        return z2


class DQNNumpyAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNNumpyActor(config)

        self.network = config.network_fn()
        self.network.share_memory()

        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.network_cp = config.network_fn()
        self.network_cp.share_memory()
        self.network_cp.load_state_dict(self.network.state_dict())
        self.optimizer_cp = config.optimizer_fn(self.network_cp.parameters())

        self.params = list(self.network.parameters())
        self.params_np = []
        self.params_np_tn = []
        self.rms_np_opt_state = {}
        for k, param in enumerate(self.network.parameters()):
            self.params_np.append(param.detach().cpu().numpy().copy())
            self.params_np_tn.append(param.detach().cpu().numpy().copy())
            self.rms_np_opt_state[k] = {'lr': config.learning_rate,
                                        'alpha': 0.99, 'eps': 1e-08,
                                        'step': 0, 'square_avg': np.zeros_like(param.data.numpy())}

        self.actor.set_network(self.network)
        self.actor.set_params_np(self.params_np)

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

    def forward(self, x):
        W1, b1 = self.params_np[2].T, np.expand_dims(self.params_np[3], 0)
        z1 = np.dot(x, W1) + b1
        h1 = np.maximum(z1, 0)

        W2, b2 = self.params_np[0].T, np.expand_dims(self.params_np[1], 0)
        z2 = np.dot(h1, W2) + b2
        return z2

    def forward_tn(self, x):
        W1, b1 = self.params_np_tn[2].T, np.expand_dims(self.params_np_tn[3], 0)
        z1 = np.dot(x, W1) + b1
        h1 = np.maximum(z1, 0)

        W2, b2 = self.params_np_tn[0].T, np.expand_dims(self.params_np_tn[1], 0)
        z2 = np.dot(h1, W2) + b2
        return z2

    def get_gradients(self, x, delta_mat):
        W1, b1 = self.params_np[2], np.expand_dims(self.params_np[3], 0)
        W2, b2 = self.params_np[0], np.expand_dims(self.params_np[1], 0)
        """
        Given input X, return the gradient of Q with respect to weights
        """
        X = x.T.astype(np.float32)
        # W1: (N_HIDDEN, OBS_SIZE), X: (OBS_SIZE, BATCH) b1: (N_HIDDEN, 1) => (N_HIDDEN, BATCH)
        hidden = np.matmul(W1, X) + np.repeat(b1.T, 32, axis=1)
        # A: (N_HIDDEN, BATCH)
        activation = np.maximum(hidden, 0)
        # dZ: (N_HIDDEN, BATCH)
        dactivation = (hidden > 0).astype(np.float32)
        # W2.T: (N_HIDDEN, N_ACTIONS), dZ: (N_HIDDEN, BATCH), X.T: (BATCH, OBS_SIZE)
        dw2 = (delta_mat.T @ activation.T) * 1. / X.shape[-1]
        dw1 = (delta_mat @ W2 * dactivation.T).T @ X.T * 1. / X.shape[-1]

        db1 = np.sum((delta_mat @ W2 * dactivation.T).T, axis=1) * 1. / X.shape[-1]
        db2 = np.sum(delta_mat.T, axis=1) * 1. / X.shape[-1]

        return [dw2, db2, dw1, db1]

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
                states, actions_, rewards_, next_states, terminals_ = experiences
                states = self.config.state_normalizer(states)
                next_states = self.config.state_normalizer(next_states)
                q_next = self.forward_tn(next_states) if self.config.use_target_network else self.forward(
                    next_states)
                q_next = np.max(q_next, axis=1)
                q_next = self.config.discount * q_next * (1 - terminals_)
                q_next += rewards_
                q = self.forward(states)
                delta_mat = np.zeros_like(q).astype(np.float32)
                q = q[self.batch_indices, actions_]
                delta = (q - q_next)
                delta_mat[self.batch_indices, actions_] = delta
                grads = self.get_gradients(states, delta_mat)
                self.clip_grads(grads, self.config.gradient_clip)
                self.rms_np_opt_np_step(grads)

        if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            for k in range(len(self.params_np)):
                self.params_np_tn[k] = self.params_np[k].copy()

    def clip_grads(self, parameters, max_norm):
        max_norm = float(max_norm)
        total_norm = 0
        for p in parameters:
            param_norm = LA.norm(p.flatten(), 2.0)
            total_norm += param_norm ** 2.0
        total_norm = total_norm ** (1. / 2.0)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for k, p in enumerate(parameters):
                parameters[k] = p * clip_coef
        return total_norm

    def rms_np_opt_np_step(self, grads):
        for k in range(len(self.params_np)):
            p = self.params_np[k]
            state = self.rms_np_opt_state[k]
            grad = grads[k]
            alpha = state['alpha']
            state['square_avg'] = state['square_avg'] * alpha + (1-alpha) * grad**2
            avg = np.sqrt(state['square_avg']) + state['eps']
            self.params_np[k] = p - state['lr'] * grad * 1.0 / avg
            state['step'] += 1

# class DQNNumpyAgent(BaseAgent):
#     def __init__(self, config):
#         BaseAgent.__init__(self, config)
#         self.config = config
#         config.lock = mp.Lock()
#
#         self.replay = config.replay_fn()
#         self.actor = DQNNumpyActor(config)
#
#         self.network = config.network_fn()
#         self.network.share_memory()
#
#         self.target_network = config.network_fn()
#         self.target_network.load_state_dict(self.network.state_dict())
#         self.optimizer = config.optimizer_fn(self.network.parameters())
#
#         self.network_cp = config.network_fn()
#         self.network_cp.share_memory()
#         self.network_cp.load_state_dict(self.network.state_dict())
#         self.optimizer_cp = config.optimizer_fn(self.network_cp.parameters())
#
#         self.params = list(self.network.parameters())
#         self.params_np = []
#         self.params_np_tn = []
#         self.rms_np_opt_state = {}
#         for k, param in enumerate(self.network.parameters()):
#             self.params_np.append(param.detach().cpu().numpy().copy())
#             self.params_np_tn.append(param.detach().cpu().numpy().copy())
#             # self.rms_np_opt_state[param] = {'lr': config.learning_rate,
#             #                                 'alpha': 0.99, 'eps': 1e-08,
#             #                                 'step': 0, 'square_avg': torch.zeros_like(param.data)}
#             self.rms_np_opt_state[k] = {'lr': config.learning_rate,
#                                         'alpha': 0.99, 'eps': 1e-08,
#                                         'step': 0, 'square_avg': np.zeros_like(param.data.numpy())}
#
#         self.actor.set_network(self.network)
#         self.actor.set_params_np(self.params_np)
#
#         self.episode_reward = 0
#         self.episode_rewards = []
#
#         self.total_steps = 0
#         self.batch_indices = range_tensor(self.replay.batch_size)
#
#         self.reset = True
#         self.ep_steps = 0
#         self.timeout = config.timeout
#
#     def close(self):
#         close_obj(self.replay)
#         close_obj(self.actor)
#
#     def eval_step(self, state):
#         self.config.state_normalizer.set_read_only()
#         state = self.config.state_normalizer(state)
#         q = self.network(state)
#         action = np.argmax(to_np(q))
#         self.config.state_normalizer.unset_read_only()
#         return action
#
#     def forward(self, x):
#         W1, b1 = self.params_np[2].T, np.expand_dims(self.params_np[3], 0)
#         z1 = np.dot(x, W1) + b1
#         h1 = np.maximum(z1, 0)
#
#         W2, b2 = self.params_np[0].T, np.expand_dims(self.params_np[1], 0)
#         z2 = np.dot(h1, W2) + b2
#         return z2
#
#     def forward_tn(self, x):
#         W1, b1 = self.params_np_tn[2].T, np.expand_dims(self.params_np_tn[3], 0)
#         z1 = np.dot(x, W1) + b1
#         h1 = np.maximum(z1, 0)
#
#         W2, b2 = self.params_np_tn[0].T, np.expand_dims(self.params_np_tn[1], 0)
#         z2 = np.dot(h1, W2) + b2
#         return z2
#
#
#     def get_gradients(self, x, delta_mat):
#         W1, b1 = self.params_np[2], np.expand_dims(self.params_np[3], 0)
#         W2, b2 = self.params_np[0], np.expand_dims(self.params_np[1], 0)
#         """
#         Given input X, return the gradient of Q with respect to weights
#         """
#         X = x.T.astype(np.float32)
#         # W1: (N_HIDDEN, OBS_SIZE), X: (OBS_SIZE, BATCH) b1: (N_HIDDEN, 1) => (N_HIDDEN, BATCH)
#         hidden = np.matmul(W1, X) + np.repeat(b1.T, 32, axis=1)
#         # A: (N_HIDDEN, BATCH)
#         activation = np.maximum(hidden, 0)
#         # dZ: (N_HIDDEN, BATCH)
#         dactivation = (hidden > 0).astype(np.float32)
#         # W2.T: (N_HIDDEN, N_ACTIONS), dZ: (N_HIDDEN, BATCH), X.T: (BATCH, OBS_SIZE)
#         dw2 = (delta_mat.T @ activation.T) * 1. / X.shape[-1]
#         dw1 = (delta_mat @ W2 * dactivation.T).T @ X.T * 1. / X.shape[-1]
#
#         db1 = np.sum((delta_mat @ W2 * dactivation.T).T, axis=1) * 1. / X.shape[-1]
#         db2 = np.sum(delta_mat.T, axis=1) * 1. / X.shape[-1]
#
#         return [dw2, db2, dw1, db1]
#
#     def step(self):
#         config = self.config
#         transitions = self.actor.step()
#         experiences = []
#         for state, action, reward, next_state, done, _ in transitions:
#             self.episode_reward += reward
#             self.total_steps += 1
#
#             self.ep_steps += 1
#             reward = config.reward_normalizer(reward)
#             if done or self.ep_steps == self.timeout:
#                 self.episode_rewards.append(self.episode_reward)
#                 self.episode_reward = 0
#                 self.ep_steps = 0
#                 self.actor.reset = True
#             experiences.append([state, action, reward, next_state, done])
#         self.replay.feed_batch(experiences)
#
#         if self.total_steps > self.config.exploration_steps:
#             for _ in range(self.config.num_replay):
#                 experiences = self.replay.sample()
#
#                 states, actions_, rewards_, next_states, terminals_ = experiences
#                 states = self.config.state_normalizer(states)
#                 next_states = self.config.state_normalizer(next_states)
#
#                 # Copying parameters of the network to numpy arrays
#                 # for k, param in enumerate(self.network.parameters()):
#                 #     self.params_np[k] = param.detach().cpu().numpy().copy()
#                 #     assert np.allclose(self.params_np[k], param.detach().numpy(), rtol=1e-5)
#
#                 ### PYTORCH train step
#                 # q_next = self.target_network(next_states) if self.config.use_target_network else self.network(next_states)
#                 # q_next = q_next.detach()
#                 # q_next = q_next.max(1)[0]
#                 # terminals = tensor(terminals_)
#                 # rewards = tensor(rewards_)
#                 # q_next = self.config.discount * q_next * (1 - terminals)
#                 # q_next.add_(rewards)
#                 # actions = tensor(actions_).long()
#                 # q = self.network(states)
#                 # q = q[self.batch_indices, actions]
#                 # loss = (q_next - q).pow(2).mul(0.5).mean()
#                 # self.optimizer.zero_grad()
#                 # loss.backward()
#                 # nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
#                 # self.optimizer.step()
#                 # print('TO loss: {}'.format(loss))
#
#
#                 ### NUMPY train step
#                 # q_next = self.target_network(next_states) if self.config.use_target_network else self.network_cp(
#                 #     next_states)
#                 # q_next = q_next.detach()
#                 # q_next = q_next.max(1)[0]
#                 # terminals = tensor(terminals)
#                 # rewards = tensor(rewards)
#                 # q_next = self.config.discount * q_next * (1 - terminals)
#                 # q_next.add_(rewards)
#                 # actions = tensor(actions).long()
#                 # q = self.network_cp(states)
#                 # q = q[self.batch_indices, actions]
#                 # loss = (q_next - q).pow(2).mul(0.5).mean()
#                 # self.optimizer_cp.zero_grad()
#                 # loss.backward()
#                 # nn.utils.clip_grad_norm_(self.network_cp.parameters(), self.config.gradient_clip)
#                 # with config.lock:
#                 #     self.rms_np_opt_step()
#
#
#                 q_next = self.forward_tn(next_states) if self.config.use_target_network else self.forward(
#                     next_states)
#                 q_next = np.max(q_next, axis=1)
#                 q_next = self.config.discount * q_next * (1 - terminals_)
#                 q_next += rewards_
#                 q = self.forward(states)
#                 delta_mat = np.zeros_like(q).astype(np.float32)
#                 q = q[self.batch_indices, actions_]
#                 delta = (q - q_next)
#                 # print('NP loss: {}'.format( (((q-q_next)**2)*0.5).mean()   ))
#                 delta_mat[self.batch_indices, actions_] = delta
#                 grads = self.get_gradients(states, delta_mat)
#                 self.clip_grads(grads, self.config.gradient_clip)
#                 self.rms_np_opt_np_step(grads)
#
#                 # loss = (q_next - q).pow(2).mul(0.5).mean()
#                 # self.optimizer_cp.zero_grad()
#                 # loss.backward()
#                 # nn.utils.clip_grad_norm_(self.network_cp.parameters(), self.config.gradient_clip)
#                 # with config.lock:
#                 #     self.rms_np_opt_step()
#
#                 # # COPY updated weights to numpy_param list
#                 # for k, param in enumerate(self.network.parameters()):
#                 #     self.params_np[k] = param.detach().cpu().numpy().copy()
#
#         if self.config.use_target_network and self.total_steps / self.config.sgd_update_frequency % \
#                 self.config.target_network_update_freq == 0:
#             self.target_network.load_state_dict(self.network.state_dict())
#             for k in range(len(self.params_np)):
#                 self.params_np_tn[k] = self.params_np[k].copy()
#
#     def rms_np_opt_step(self):
#         wt_sums = []
#         gd_sums = []
#         sq_sums = []
#         for k, p in enumerate(self.network_cp.parameters()):
#         # for k, p in enumerate(self.network.parameters()):
#             state = self.rms_np_opt_state[p]
#
#             # square_avg = state['square_avg']
#             alpha = state['alpha']
#             state['step'] += 1
#             grad = p.grad.data
#             # if k == 0:
#             #     xyz=0
#             state['square_avg'] = state['square_avg'] * alpha + (1-alpha) * grad**2
#             avg = state['square_avg'].sqrt().add_(state['eps'])
#             p.data = p.data - state['lr'] * grad / avg
#             wt_sums.append(torch.sum(p))
#             gd_sums.append(torch.sum(grad))
#             sq_sums.append(torch.sum(state['square_avg']))
#         print(wt_sums, gd_sums, sq_sums)
#         xtz=0
#
#     def clip_grads(self, parameters, max_norm):
#         max_norm = float(max_norm)
#         total_norm = 0
#         for p in parameters:
#             param_norm = LA.norm(p.flatten(), 2.0)
#             total_norm += param_norm ** 2.0
#         total_norm = total_norm ** (1. / 2.0)
#         clip_coef = max_norm / (total_norm + 1e-6)
#         if clip_coef < 1:
#             for k, p in enumerate(parameters):
#                 parameters[k] = p * clip_coef
#         return total_norm
#
#     def rms_np_opt_np_step(self, grads):
#         wt_sums = []
#         gd_sums = []
#         sq_sums = []
#         params = list(self.network.parameters())
#         for k in range(len(self.params_np)):
#         # for k, p in enumerate(self.network.parameters()):
#             p = self.params_np[k]
#             state = self.rms_np_opt_state[k]
#             alpha = state['alpha']
#             state['step'] += 1
#             grad = grads[k]
#             state['square_avg'] = state['square_avg'] * alpha + (1-alpha) * grad**2
#             avg = np.sqrt(state['square_avg']) + state['eps']
#             wt_sums.append(np.sum(p))
#             self.params_np[k] = p - state['lr'] * grad * 1.0 / avg
#             param = params[k]
#             # assert np.allclose(self.params_np[k], param.detach().numpy(), rtol=1e-4)
#             # assert np.allclose(grad, param.grad.numpy(), rtol=1e-4)
#             gd_sums.append(np.sum(grad))
#             sq_sums.append(np.sum(state['square_avg']))
#         # wt_sums = [float("{0:0.8f}".format(i)) for i in wt_sums]
#         # gd_sums = [float("{0:0.8f}".format(i)) for i in gd_sums]
#         # sq_sums = [float("{0:0.8f}".format(i)) for i in sq_sums]
#         # print('np', wt_sums, gd_sums, sq_sums)
#         # xyz = 0
#         # print(gd_sums)