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

        project_root = os.getcwd()
        self.data_root = os.path.join(project_root, 'data', 'output')

    @property
    def eval_env(self):
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env):
        self.__eval_env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim


class DQNAgentConfig(SlimConfig):
    DEVICE = torch.device('cpu')
    def __init__(self):
        super(DQNAgentConfig, self).__init__()
        self.agent = 'DQNAgent'

        self.random_action_prob = None
        self.exploration_steps = None

        self.discount = None

        self.batch_size = None
        self.optimizer_fn = None
        self.gradient_clip = None
        self.sgd_update_frequency = None

        self.network_fn = None
        self.target_network_update_freq = None

        self.replay_fn = None
        self.memory_size = None

        self.async_actor = True
        self.double_q = False

        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()

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




