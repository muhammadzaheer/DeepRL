from deep_rl import *

from experiment.sweeper import Sweeper
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="run_file")
    parser.add_argument('--id', default=0, type=int, help='identifies run number and configuration')
    parser.add_argument('--config-file', default='experiment/config_files/lunar_lander/sarsa_lambda/test.json')
    parser.add_argument('--run', default=0,type=int)

    args = parser.parse_args()
    project_root = os.path.abspath(os.path.dirname(__file__))
    sweeper = Sweeper(os.path.join(project_root, args.config_file))

    cfg = sweeper.parse(args.id)
    cfg.data_root = os.path.join(project_root, 'data', 'output')
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
    cfg.logger = setup_logger(log_path, stdout=True)
    cfg.log_config(cfg.logger)

    # Initializing the agent and running the experiment
    agent_class = getattr(agent, cfg.agent)
    run_steps(agent_class(cfg))
