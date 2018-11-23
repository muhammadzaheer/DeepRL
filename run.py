from deep_rl import *

from experiment.sweeper import Sweeper
import argparse

# # DQN
# def dqn_lunar_lander(config):
#     # config = Config()
#     config.task_fn = lambda: Task(config.task_name)
#     config.eval_env = config.task_fn()
#     config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
#     config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
#     config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=config.batch_size)
#     config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
#
#
#     logdir = cfg.get_logdir()
#     log_path = os.path.join(logdir, 'log')
#     config.logger = setup_logger(log_path, stdout=False)
#     config.log_config(config.logger)
#     run_steps(DQNAgent(config))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file', default='experiment/config_files/lunar_lander/sarsa_lambda/test.json')
    parser.add_argument('--run', default=0,type=int)

    args = parser.parse_args()

    sweeper = Sweeper(args.config_file)
    cfg = sweeper.parse(args.id)

    set_one_thread()
    random_seed()
    select_device(-1)

    # Setting up the config
    cfg.task_fn = lambda: Task(cfg.task_name)
    cfg.eval_env = cfg.task_fn()
    cfg.optimizer_fn = lambda params: torch.optim.SGD(params, cfg.learning_rate)
    cfg.network_fn = lambda: VanillaNet(cfg.action_dim, FCBody(cfg.tiles_memsize * cfg.state_dim, hidden_units=(cfg.hidden_units,)))
    cfg.random_action_prob = LinearSchedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_schedule_steps)

    # Setting up the logger
    logdir = cfg.get_logdir()
    log_path = os.path.join(logdir, 'log')
    cfg.logger = setup_logger(log_path, stdout=False)
    cfg.log_config(cfg.logger)

    # Initializing the agent and running the experiment
    agent_class = getattr(agent, cfg.agent)
    run_steps(agent_class(cfg))
