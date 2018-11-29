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
    step = 0
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
            rewards_over_time[step] = reward
            step += 1
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
                 save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
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
                 save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
                 interval=10000)
    v.draw()


def draw_lunar_lander_traces(save_path, max_steps=2000000):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    settings = [("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json", 4, 5, max_steps, "Sarsa(0.5) - 512h"),
                ("experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json", 4, 5, max_steps, "Sarsa(0.5) - 1024h")]
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
                 save_path=save_path, xlabel="Number of steps", ylabel="Episodes Completed",
                 interval=10000, ylim=(-400, -100))
    v.draw()




if __name__ == '__main__':
    # draw_lunar_lander(save_path="plots/lunar_lander/ll_results.png")
    # draw_lunar_lander_individual(save_path="plots/lunar_lander/ll_results_individual.png")
    # draw_mountain_car(save_path="plots/mc_results.png")

    # draw_lunar_lander_traces_h512(save_path="plots/lunar_lander_traces/ll_h512_results.png", max_steps=1800000)

    # draw_lunar_lander_traces_h1024(save_path="plots/lunar_lander_traces/ll_h1024_results.png", max_steps=1500000)

    draw_lunar_lander_traces(save_path="plots/lunar_lander_traces/ll_results.png", max_steps=1500000)




