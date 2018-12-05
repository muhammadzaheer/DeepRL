import os
import numpy as np
import matplotlib.pyplot as plt

from experiment.sweeper import Sweeper


def extract_line(lines, max_steps, interval=0):
    """
    Retrieves num_steps at which episodes were completed
    and returns an array where the value at index i represents
    the number of episodes completed until step i
    """
    step = 0
    rewards_over_time = np.zeros(max_steps//interval)
    try:
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
        raise


def get_max_steps(lines):
    for line in lines[::-1]:
        if 'total steps' in line:
            max_steps = float(line.split("|")[1].split(",")[0].split(" ")[-1])
            return max_steps
    return -1




def _eval_lines(config_file, start_idx, end_idx, max_steps):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    sweeper = Sweeper(os.path.join(project_root, config_file))
    eval = []
    eval_lines = []
    for k in range(sweeper.total_combinations):
        eval.append([])
        eval_lines.append([])

    for idx in range(start_idx, end_idx):
        cfg = sweeper.parse(idx)
        cfg.data_root = os.path.join(project_root, 'data', 'output')
        log_dir = cfg.get_logdir()
        log_path = os.path.join(log_dir, 'log')
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            continue


        if len(lines) == 0:
            continue
        # ugly parse based on the log_file format
        try:
            num_steps = get_max_steps(lines)
            if num_steps >= max_steps:
                assert idx % sweeper.total_combinations == cfg.param_setting
                avg_eval_steps = extract_line(lines, max_steps, interval=10000)
                eval[idx % sweeper.total_combinations].append(np.mean(avg_eval_steps))

        except IndexError:
            print(idx)
            raise
    summary = list(map(lambda x: (x[0], np.mean(x[1]), np.std(x[1]), len(x[1])), enumerate(eval)))
    summary = sorted(summary, key=lambda s: s[1], reverse=False)

    for idx, mean, std, num_runs in summary:
        print("Param Setting # {:>3d} | Average num of episodes: {:>10.2f} +/- {:>5.2f} ({:>2d} runs) {} | ".format(
            idx, mean, std, num_runs, sweeper.param_setting_from_id(idx)))


if __name__ == '__main__':
    np.set_printoptions(precision=0)
    # _eval_lines(config_file='experiment/config_files/mountain_car/sarsa/sweep.json', start_idx=0,
    #       end_idx=107, max_steps=300000)
    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa/sweep_2.json', start_idx=0,
    #       end_idx=324, max_steps=2000000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa/sweep_3.json', start_idx=0,
    #       end_idx=60, max_steps=2000000)

    # _eval_lines(config_file='experiment/config_files/mountain_car/sarsa_lmbda/sweep.json', start_idx=0,
    #       end_idx=162, max_steps=300000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json', start_idx=0,
    #       end_idx=75, max_steps=2000000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h512.json', start_idx=0,
    #       end_idx=75, max_steps=1500000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h1024.json', start_idx=0,
    #       end_idx=75, max_steps=1500000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h2048.json', start_idx=0,
    #       end_idx=45, max_steps=2000000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/sarsa_lmbda/sweep_h4096.json', start_idx=0,
    #       end_idx=45, max_steps=1400000)

    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_h1_128_h2_64.json', start_idx=0,
    #       end_idx=135, max_steps=1000000)

    _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_online.json', start_idx=0,
          end_idx=48, max_steps=1000000)


