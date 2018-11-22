from deep_rl import *

from experiment.sweeper import Sweeper
import argparse

# DQN
def dqn_lunar_lander(config):
    # config = Config()
    config.task_fn = lambda: Task(config.task_name)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=config.batch_size)
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)


    logdir = cfg.get_logdir()
    log_path = os.path.join(logdir, 'log')
    config.logger = setup_logger(log_path, stdout=False)
    config.log_config(config.logger)
    run_steps(DQNAgent(config))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file', default='experiment/config_files/lunar_lander/test.json')
    parser.add_argument('--run', default=0,type=int)

    args = parser.parse_args()

    sweeper = Sweeper(args.config_file)
    cfg = sweeper.parse(args.id)
    logdir = cfg.get_logdir()
    set_one_thread()
    random_seed()
    select_device(-1)
    dqn_lunar_lander(cfg)

