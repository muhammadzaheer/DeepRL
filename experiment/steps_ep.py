import os
import numpy as np


from sweeper import Sweeper
from visualizer import RunLines, RunLinesIndividual


def parse_steps_log(log_path, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    print(log_path)
    with open(log_path, "r") as f:
        lines = f.readlines()
    rewards_over_time = np.zeros(max_steps//interval)
    try:
        num_steps = get_max_steps(lines)
        if num_steps < max_steps:
            return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1])
            rewards_over_time[int(num_steps//interval)-1] = reward
        return rewards_over_time
    except:
        return None

def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1

def draw_lunar_lander(save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa/sweep_3.json", 1, 5, 2000000, "Sarsa(0) [sweep 3]"),
                ("experiment/config_files/lunar_lander/sarsa/sweep_2.json", 19, 2, 2000000, "Sarsa(0)b [sweep 2]")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps (in 10000)", ylabel="Episodes Completed",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces_h512(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 1, 5, max_steps, "Sarsa(0)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 4, 5, max_steps, "Sarsa(0.5)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 7, 5, max_steps, "Sarsa(0.7)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 11, 5, max_steps, "Sarsa(0.85)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 14, 5, max_steps, "Sarsa(0.9)")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces_h1024(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 1, 5, max_steps, "Sarsa(0)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 4, 5, max_steps, "Sarsa(0.5)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 7, 5, max_steps, "Sarsa(0.7)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 11, 5, max_steps, "Sarsa(0.85)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 14, 5, max_steps, "Sarsa(0.9)")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces_h2048(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h2048.json", 1, 5, max_steps, "Sarsa(0.5)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h2048.json", 4, 5, max_steps, "Sarsa(0.7)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h2048.json", 8, 5, max_steps, "Sarsa(0.85)")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces_h4096(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h4096.json", 1, 5, max_steps, "Sarsa(0.5)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h4096.json", 4, 5, max_steps, "Sarsa(0.7)"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h4096.json", 7, 5, max_steps, "Sarsa(0.85)")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()


def draw_lunar_lander_individual(save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []


    settings = [("experiment/config_files/lunar_lander/sarsa/sweep_3.json", 1, 5, 2000000, "Sarsa(0) [sweep 3]"),
                ("experiment/config_files/lunar_lander/sarsa/sweep_2.json", 19, 2, 2000000, "Sarsa(0)b [sweep 2]")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLinesIndividual(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 4, 5, max_steps, "Sarsa(0.5) - 512h"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 4, 5, max_steps, "Sarsa(0.5) - 1024h"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h2048.json", 1, 5, max_steps, "Sarsa(0.5) - 2048h"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h4096.json", 1, 5, max_steps, "Sarsa(0.5) - 4096h")]
    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()

def draw_mountain_car(save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/mountain_car/sarsa/sweep.json", 12, 3, 300000, "Sarsa(0)"),]

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()


def draw_lunar_lander_dqn(settings, save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for cfg, param_setting, num_runs, num_steps, label in settings:
        config_files.append((cfg, param_setting))
        runs.append(num_runs)
        num_datapoints.append(num_steps)
        labels.append(label)

    for cf, best_setting in config_files:
        swp = Sweeper(os.path.join(project_root, cf))
        cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        logdir_format = cfg.get_logdir_format()
        path_format = os.path.join(logdir_format, log_name
                                   )
        path_formatters.append(path_format)

    v = RunLines(path_formatters, runs, num_datapoints,
                 labels, parser_func=parser_func,
                 save_path=save_path, xlabel="Number of steps", ylabel="Reward",
                 interval=10000)
    v.draw()


# def draw_lunar_lander_dqn_online(save_path, max_steps=2000000):
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
#     log_name = "log"
#     parser_func = parse_steps_log
#     path_formatters = []
#
#     settings = [("experiment/config_files/lunar_lander/dqn/sweep_online.json", 10, 4, max_steps, "DQN - Online - LR 0.0001 - TF Update 8192 - Gamma 0.999")]
#
#     config_files, runs, num_datapoints, labels = [], [], [], []
#     for cfg, param_setting, num_runs, num_steps, label in settings:
#         config_files.append((cfg, param_setting))
#         runs.append(num_runs)
#         num_datapoints.append(num_steps)
#         labels.append(label)
#
#     for cf, best_setting in config_files:
#         swp = Sweeper(os.path.join(project_root, cf))
#         cfg = swp.parse(best_setting)      # Creating a cfg with an arbitrary run id
#         cfg.data_root = os.path.join(project_root, 'data', 'output')
#         logdir_format = cfg.get_logdir_format()
#         path_format = os.path.join(logdir_format, log_name
#                                    )
#         path_formatters.append(path_format)
#
#     v = RunLines(path_formatters, runs, num_datapoints,
#                  labels, parser_func=parser_func,
#                  save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
#                  interval=10000)
#     v.draw()




if __name__ == '__main__':
    # draw_lunar_lander(save_path="plots/lunar_lander/ll_results.png")
    # draw_lunar_lander_individual(save_path="plots/lunar_lander/ll_results_individual.png")
    # draw_mountain_car(save_path="plots/mc_results.png")

    # draw_lunar_lander_traces_h512(save_path="plots/lunar_lander_traces/ll_h512_results.png", max_steps=1800000)

    # draw_lunar_lander_traces_h1024(save_path="plots/lunar_lander_traces/ll_h1024_results.png", max_steps=1500000)
    # draw_lunar_lander_traces(save_path="plots/lunar_lander_traces/ll_results_2.png", max_steps=1000000)
    # draw_lunar_lander_traces_h2048(save_path="plots/lunar_lander_traces/ll_h2048_results.png", max_steps=2000000)

    # draw_lunar_lander_traces_h4096(save_path="plots/lunar_lander_traces/ll_h4096_results.png", max_steps=1400000)

    # draw_lunar_lander_traces(save_path="plots/lunar_lander_traces/ll_results.png", max_steps=1400000)

    # draw_lunar_lander_dqn(save_path="plots/lunar_lander_dqn/sweep_h1_128_h2_64/TF2048_individuals.png", max_steps=1000000)

    # draw_lunar_lander_dqn_online(save_path="plots/lunar_lander_dqn/sweep_online/best.png", max_steps=1000000)

    # settings = [("experiment/config_files/lunar_lander/dqn/sweep_h512.json", 0, 4, 1000000, "DQN - h512 - LR 0.001 - TNU 512 - G 0.99"),
    #             ("experiment/config_files/lunar_lander/dqn/sweep_h128.json", 9, 4, 1000000, "DQN - h128 - LR 0.001 - TNU 8192 - G 0.999"),
    #             ("experiment/config_files/lunar_lander/dqn/sweep_online.json", 10, 4, 1000000, "DQN Online - h512 - LR 0.0001 - TF Update 8192 - Gamma 0.999 [ER size 1, batch 1]")]
    #
    # settings = [("experiment/config_files/lunar_lander/dqn/sweep_h1_128_h2_64.json", 6, 7, 1000000, "TNUF 8192 - BS 8"),
    #             ("experiment/config_files/lunar_lander/dqn/sweep_h1_128_h2_64.json", 16, 7, 1000000, "TNUF 8192 - BS 32")]
    #

    # settings = [("experiment/config_files/lunar_lander/dqn/sweep_tc_online_h128.json", 7, 4, 1000000, "TNUF 8192 - Discount 0.999 - h128"),
    #             ("experiment/config_files/lunar_lander/dqn/sweep_tc_online_h512.json", 7, 4, 1000000, "TNUF 512 - Discount 0.999 - h512"),
    #             ("experiment/config_files/lunar_lander/dqn/sweep_online.json", 10, 4, 1000000, "Without TC; TNUF Update 8192 - Discount 0.999 - h512")]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/lunar_lander_dqn/sweep_tc_online/best.png", max_steps=1000000)

    # settings = [("experiment/config_files/stable_q/sw_dqn_h512.json", 0, 5, 800000, "DQN"),
    #             ("experiment/config_files/stable_q/sw_dqn_h512_notarget.json", 2, 4, 800000, "DQN - No Target Net")
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/stable_q/p1_one_hidden.pdf", max_steps=800000)
    #
    # settings = [("experiment/config_files/stable_q/sw_dqn_h512.json", 0, 3, 1000000, "DQN"),
    #             ("experiment/config_files/stable_q/sw_lp_dqn_h512.json", 1, 3, 1000000, "DQN LP"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/stable_q/lp_1.pdf", max_steps=1000000)


    # settings = [("experiment/config_files/stable_q/sw_dqn.json", 1, 3, 1000000, "DQN"),
    #             ("experiment/config_files/stable_q/sw_lp_dqn_h128_h64.json", 2, 3, 1000000, "DQN LP"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/stable_q/lp_h2.pdf", max_steps=1000000)


    # settings = [("experiment/config_files/stable_q/sw_dqn_h512.json", 0, 3, 500000, "DQN"),
    #             ("experiment/config_files/stable_q/sw_tc_dqn_h512.json", 0, 3, 500000, "DQN TC"),
    #             ("experiment/config_files/stable_q/sw_tc_dqn_h512_notarget.json", 2, 3, 500000, "DQN TC - No Target Net")
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/stable_q/p3_tc_h512.pdf", max_steps=500000)

    # settings = [("experiment/config_files/stable_q/sw_dqn.json", 1, 5, 800000, "DQN"),
    #             ("experiment/config_files/stable_q/sw_dqn_notarget.json", 2, 4, 800000, "DQN - No Target Net")
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/stable_q/p2_two_hidden.pdf", max_steps=800000)

    # settings = [("experiment/config_files/l2/sw_dqn.json", 0, 3, 500000, "DQN"),
    #             ("experiment/config_files/l2/sw_dqn_l2.json", 0, 3, 500000, "DQN L2"),
    #             ("experiment/config_files/l2/sw_dqn_l2_notarget.json", 1, 3, 500000, "DQN L2 I seeNTN"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/l2.pdf", max_steps=500000)

    # settings = [("experiment/config_files/dropout/sw_dqn_drop.json", 6, 3, 1000000, "DQN Drop"),
    #             ("experiment/config_files/dropout/sw_dqn_drop_notarget.json", 14, 3, 1000000, "DQN Drop NTN"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/drop.pdf", max_steps=1000000)

    # settings = [("experiment/config_files/dropout/sw_dqn_drop.json", 6, 3, 1000000, "DQN Drop"),
    #             ("experiment/config_files/dropout/sw_dqn_drop_notarget.json", 14, 3, 1000000, "DQN Drop NTN"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/drop.pdf", max_steps=1000000)

    # settings = [("experiment/config_files/replay/sw_dqn_replay.json", 10, 3, 1000000, "DQN Replay (26 replay steps)"),
    #             ("experiment/config_files/replay/sw_dqn_replay.json", 7, 3, 1000000, "DQN Replay (16 replay steps)"),
    #             ("experiment/config_files/replay/sw_dqn_replay.json", 4, 3, 1000000, "DQN Replay (8 replay steps)"),
    #             ("experiment/config_files/replay/sw_dqn_replay.json", 1, 3, 1000000, "DQN Replay (4 replay steps)"),
    #             ("experiment/config_files/l2/sw_dqn.json", 0, 3, 1000000, "DQN Replay (1 replay steps)"),
    #             # ("experiment/config_files/replay/sw_dqn_replay_notarget.json", 2, 3, 1000000, "DQN Replay NTN"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/replay_steps.pdf", max_steps=1000000)

    # settings = [# ("experiment/config_files/fitted_q/sw_dqn_fittedq.json", 7, 3, 1000000, "DQN (16 replay steps, gamma 0.99)"),
    #             # ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget.json", 9, 3, 1000000, "DQN w/o t-net (16 replay steps, gamma 0.99)"),
    #             ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_noreplay.json", 0, 3, 1000000, "DQN w/o t-net  (1 replay step, gamma 0.99)"),
    #             ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2.json", 7, 3, 800000, "DQN w/o t-net  (16 replay steps, gamma 1.0)"),
    #
    #
    #             ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2_schedule.json", 1, 3, 1000000, "DQN w/o t-net  (1 replay steps, gamma schedule)"),
    #             # ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2_schedule.json", 0, 2, 1000000, "DQN w/o t-net  (1 replay steps, gamma schedule) v2"),
    #             ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2_schedule.json", 13, 3, 1000000, "DQN w/o t-net  (8 replay steps, gamma schedule)"),
    #
    #             # ("experiment/config_files/replay/sw_dqn_replay_notarget.json", 2, 3, 1000000, "DQN Replay NTN"),
    #             ]
    #
    # draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/fitted_q.pdf", max_steps=1000000)

    settings = [("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_noreplay.json", 0, 3, 1000000, "DQN (1 replay step, gamma 0.99)"),
                ("experiment/config_files/fitted_q/sw_dqn_fittedq_notarget.json", 1, 3, 1000000, "DQN (4 replay steps, gamma 0.99)"),
                ("experiment/config_files/fitted_q/exp_sarsa_target_epsilon.json", 7, 3, 1000000, "Expected Sarsa (4 replay steps, gamma 0.99, target_pi eps 0.01"),
                # ("experiment/config_files/fitted_q/exp_sarsa_target_epsilon.json", 4, 3, 1000000, "Expected Sarsa (4 replay steps, gamma 0.99, target pi 0.05"),
                #("experiment/config_files/fitted_q/exp_sarsa_target_epsilon.json", 10, 3, 1000000, "Expected Sarsa (4 replay steps, gamma 0.99, target_pi eps 0.001"),
                # ("experiment/config_files/fitted_q/exp_sarsa_target_epsilon.json", 1, 3, 1000000, "Expected Sarsa (4 replay steps, gamma 0.99, targ_pi eps 0.1"),

                ]

    draw_lunar_lander_dqn(settings, save_path="plots/rl_mooc/plot.pdf", max_steps=1000000)


