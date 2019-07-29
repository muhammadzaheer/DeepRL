from deep_rl import *

from experiment.sweeper import Sweeper
import argparse


def set_optimizer_fn(cfg):
    if cfg.optimizer_type == 'SGD':
        cfg.optimizer_fn = lambda params: SGD(params, cfg.learning_rate)
    elif cfg.optimizer_type == 'RMSProp':
        cfg.optimizer_fn = lambda params: torch.optim.RMSprop(params, cfg.learning_rate)
    else:
        raise NotImplementedError

def set_network_fn(cfg):
    if cfg.tile_coding:
        cfg.network_fn = lambda: VanillaNet(cfg.action_dim, FCBody(cfg.tiles_memsize * cfg.state_dim,
                                                                   hidden_units=tuple(cfg.hidden_units)))
    elif cfg.lift_project:
        cfg.network_fn = lambda: VanillaLPNet(cfg.action_dim, FCLPBody(cfg.state_dim,
                                                                   hidden_units=tuple(cfg.hidden_units), radius=cfg.radius),
                                              flp=cfg.flp, radius=cfg.radius)
    elif cfg.l2:
        cfg.network_fn = lambda: VanillaNetL2(cfg.action_dim, FCBody(cfg.state_dim,
                                                                   hidden_units=tuple(cfg.hidden_units)))
    elif cfg.drop:
        cfg.network_fn = lambda: VanillaNetDrop(cfg.action_dim, FCBody(cfg.state_dim,
                                                                   hidden_units=tuple(cfg.hidden_units)))
    else:
        cfg.network_fn = lambda: VanillaNet(cfg.action_dim, FCBody(cfg.state_dim,
                                                                   hidden_units=tuple(cfg.hidden_units)))


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

    # Setting up the optimizer
    set_optimizer_fn(cfg)

    if cfg.replay:
        cfg.replay_fn = lambda: Replay(memory_size=int(cfg.memory_size), batch_size=cfg.batch_size)

    set_network_fn(cfg)

    cfg.random_action_prob = LinearSchedule(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_schedule_steps)
    cfg.discount_schedule = LinearScheduleAdam(cfg.epsilon_start, cfg.epsilon_end, cfg.epsilon_schedule_steps)

    # Setting up the logger
    logdir = cfg.get_logdir()
    log_path = os.path.join(logdir, 'log')
    cfg.logger = setup_logger(log_path, stdout=True)
    cfg.log_config(cfg.logger)

    # Initializing the agent and running the experiment
    agent_class = getattr(agent, cfg.agent)
    run_steps(agent_class(cfg))
