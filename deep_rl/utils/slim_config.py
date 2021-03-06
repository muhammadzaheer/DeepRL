#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from .normalizer import *
from .misc import *
import argparse
import torch

import os
from datetime import datetime

class SlimConfig:
    DEVICE = torch.device('cpu')
    def __init__(self):
        self.run = 0
        self.param_setting = 0
        self.exp_name = 'test'
        self.timeout = None
        self.task_fn = None
        self.logger = None
        self.tag = 'vanilla'
        self.max_steps = 0
        self.__eval_env = None
        self.state_dim = None
        self.action_dim = None
        self.task_name = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.eval_interval = 0
        self.eval_episodes = 10

        # project_root = os.getcwd()
        # self.data_root = os.path.join(project_root, 'data', 'output')
        self.data_root = None

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim

    # Helper functions
    def get_logdir(self):
        d = os.path.join(self.data_root, self.exp_name,
                            "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting))
        ensure_dir(d)
        return d

    def get_logdir_format(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run",
                            "{}_param_setting".format(self.param_setting))

    def get_tflogdir(self):
        return os.path.join(self.data_root, self.exp_name,
                            "{}_run".format(self.run),
                            "{}_param_setting".format(self.param_setting),
                            "tf_logs", datetime.now().strftime('%D-%T'))

    # Helper functions
    def get_modeldir(self):
        d = os.path.join(self.data_root, self.exp_name,
                        "{}_run".format(self.run),
                        "{}_param_setting".format(self.param_setting),
                        "model")
        ensure_dir(d)
        return d

    def log_config(self, logger):
        # Serializes the configure into a log file
        attrs = self.get_print_attrs()
        for param, value in sorted(attrs.items(), key=lambda x: x[0]):
            logger.info('{}: {}'.format(param, value))


class DQNAgentConfig(SlimConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(DQNAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()


        self.tile_coding = False
        self.use_target_network = True

        self.lift_project = False
        self.flp = False
        self.radius = 8.0

        self.l2 = False
        self.lmbda = -1.0

        self.num_replay = 1

        self.drop = False
        self.p = 0.0

        self.target_epsilon = None


    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class DQNSoftmaxAgentConfig(SlimConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(DQNSoftmaxAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer(coef=1./250)


        self.tile_coding = False
        self.use_target_network = True

        self.lift_project = False
        self.flp = False
        self.radius = 8.0

        self.l2 = False
        self.lmbda = -1.0

        self.num_replay = 1

        self.drop = False
        self.p = 0.0

        self.target_epsilon = None


    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs






class DQNTileAgentConfig(DQNAgentConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(DQNAgentConfig, self).__init__()
        self.agent = 'DQNTileAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.tiles_memsize = None
        self.tiles = None
        self.num_tilings = None
        self.tile_separate = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()


        self.tile_coding = False

    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class DQNLPAgentConfig(DQNAgentConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(DQNLPAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_type = 'RMSProp'
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.tiles_memsize = None
        self.tiles = None
        self.num_tilings = None
        self.tile_separate = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = True
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = False
        self.double_q = False

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()


        self.tile_coding = False

    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class SarsaAgentConfig(SlimConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(SarsaAgentConfig, self).__init__()
        self.agent = 'SarsaAgent'

        self.epsilon_start = None
        self.epsilon_end = None
        self.epsilon_schedule_steps = None
        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.optimizer_type = 'SGD'
        self.optimizer_fn = None
        self.gradient_clip = None

        self.tiles_memsize = None
        self.tiles = None
        self.num_tilings = None
        self.tile_separate = None

        self.hidden_units = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay = False
        self.replay_fn = None
        self.memory_size = None

        self.async_actor = True
        self.double_q = False

        self.state_normalizer = MinMaxNormalizer()
        self.reward_normalizer = MinMaxNormalizer()


    def __str__(self):
        attrs = self.get_print_attrs()
        s = ""
        for param, value in attrs.items():
            s += "{}: {}\n".format(param, value)
        return s

    def get_print_attrs(self):
        attrs = dict(self.__dict__)
        for k in ['state_normalizer', 'reward_normalizer', 'task_fn',
                  'logger', '_SlimConfig__eval_env', 'random_action_prob',
                  'optimizer_fn', 'network_fn', 'replay_fn', 'data_root']:
            del attrs[k]
        return attrs


class SarsaLmbdaAgentConfig(SarsaAgentConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(SarsaLmbdaAgentConfig, self).__init__()
        self.agent = 'SarsaLmbdaAgent'
        self.lmbda = 0.0




