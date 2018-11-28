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
        num_steps = float(lines[-2].split("|")[1].split(",")[0].split(" ")[-1])
        # if max_steps != num_steps:
        #     return None
        for line in lines:
            if 'total steps' not in line:
                continue
            num_steps = float(lines[-2].split("|")[1].split(",")[0].split(" ")[-1])
            if num_steps > max_steps:
                break
            reward = float(line.split("|")[1].split(",")[2].split("/")[0].split(" ")[-1])
            rewards_over_time[step] = reward
            step += 1
        return rewards_over_time
    except:
        return None


def draw_lunar_lander(save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for k in [70, 19, 28, 106, 52, 34, 100, 61, 94, 103]:
        config_files.append(("experiment/config_files/lunar_lander/sarsa/sweep_2.json", k))
        runs.append(3)
        num_datapoints.append(2000000)
        labels.append(k)

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

    config_files, runs, num_datapoints, labels = [], [], [], []
    for k in [70, 19, 28]:
        config_files.append(("experiment/config_files/lunar_lander/sarsa/sweep_2.json", k))
        runs.append(3)
        num_datapoints.append(2000000)
        labels.append(k)

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


def draw_mountain_car(save_path):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    log_name = "log"
    parser_func = parse_steps_log
    path_formatters = []

    config_files, runs, num_datapoints, labels = [], [], [], []
    for k in [12]:
        config_files.append(("experiment/config_files/mountain_car/sarsa/sweep.json", k))
        runs.append(3)
        num_datapoints.append(300000)
        labels.append(k)

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
                 interval=10000, ylim=(-400, 0))
    v.draw()




if __name__ == '__main__':
    # draw_lunar_lander(save_path="plots/results.png")
    draw_mountain_car(save_path="plots/mc_results.png")





