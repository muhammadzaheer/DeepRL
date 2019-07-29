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
        print("Param Setting # {:>3d} | Average reward: {:>10.2f} +/- {:>5.2f} ({:>2d} runs) {} | ".format(
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

    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_tc_online_h128.json', start_idx=0,
    #       end_idx=48, max_steps=1000000)
    #
    # print("")
    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_tc_online_h512.json', start_idx=0,
    #       end_idx=48, max_steps=1000000)

    # print("sweep_h128")
    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_h128.json', start_idx=0,
    #       end_idx=100, max_steps=1000000)
    #
    # print("sweep_h128_h64")
    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_h1_128_h2_64.json', start_idx=0,
    #       end_idx=135, max_steps=1000000)


    # print("sw_h512")
    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_h128.json', start_idx=0,
    #       end_idx=100, max_steps=1000000)
    #
    # print("sw_h512_notarget")
    # _eval_lines(config_file='experiment/config_files/lunar_lander/dqn/sweep_h1_128_h2_64.json', start_idx=0,
    #       end_idx=135, max_steps=1000000)

    # print("sw_h512")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn_h512.json', start_idx=0,
    #       end_idx=15, max_steps=850000)
    #
    # print("sw_h512_notarget")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn_h512_notarget.json', start_idx=0,
    #       end_idx=15, max_steps=850000)
    # print("sw_dqb")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn.json', start_idx=0,
    #       end_idx=15, max_steps=850000)
    #
    # print("sw_dqn_notarget")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn_notarget.json', start_idx=0,
    #       end_idx=15, max_steps=850000)


    # print("sw_h512")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn_h512.json', start_idx=0,
    #       end_idx=15, max_steps=1000000)
    #
    # print("sw_lp_h512")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_lp_dqn_h512.json', start_idx=0,
    #       end_idx=15, max_steps=1000000)

    # print("sw_h1024")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_lp_dqn_h128_h64.json', start_idx=0,
    #       end_idx=15, max_steps=1000000)

    # print("sw_lp_h1024")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_dqn_h512.json', start_idx=0,
    #       end_idx=15, max_steps=1000000)



    #
    # print("sw_h512")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_tc_dqn_h512.json', start_idx=0,
    #       end_idx=12, max_steps=500000)
    #
    # print("sw_h512_notarget")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_tc_dqn_h512_notarget.json', start_idx=0,
    #       end_idx=12, max_steps=500000)
    # print("sw_dqb")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_tc_dqn_h1024.json', start_idx=0,
    #       end_idx=12, max_steps=350000)
    #
    # print("sw_dqn_notarget")
    # _eval_lines(config_file='experiment/config_files/stable_q/sw_tc_dqn_h1024_notarget.json', start_idx=0,
    #       end_idx=12, max_steps=350000)


    # print("sw_dqn")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn.json', start_idx=0,
    #       end_idx=9, max_steps=550000)
    #
    # print("sw_dqn_l2")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn_l2.json', start_idx=0,
    #       end_idx=36, max_steps=550000)
    #
    # print("sw_dqn_l2_notarget")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn_l2_notarget.json', start_idx=0,
    #       end_idx=36, max_steps=550000)

    # print("sw_dqn")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn.json', start_idx=0,
    #       end_idx=9, max_steps=1000000)
    #
    # print("sw_dqn_l2")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn_l2.json', start_idx=0,
    #       end_idx=36, max_steps=550000)
    #
    # print("sw_dqn_l2_notarget")
    # _eval_lines(config_file='experiment/config_files/l2/sw_dqn_l2_notarget.json', start_idx=0,
    #       end_idx=36, max_steps=550000)

    # print("sw_dqn_dropout")
    # _eval_lines(config_file='experiment/config_files/dropout/sw_dqn_drop.json', start_idx=0,
    #       end_idx=45, max_steps=1000000)
    #
    # print("sw_dqn_dropout_notarget")
    # _eval_lines(config_file='experiment/config_files/dropout/sw_dqn_drop_notarget.json', start_idx=0,
    #       end_idx=45, max_steps=1000000)

    # print("sw_dqn_replay")
    # _eval_lines(config_file='experiment/config_files/replay/sw_dqn_replay.json', start_idx=0,
    #       end_idx=45, max_steps=1000000)
    #
    # print("\n\nsw_dqn_replay_notarget")
    # _eval_lines(config_file='experiment/config_files/replay/sw_dqn_replay_notarget.json', start_idx=0,
    #       end_idx=45, max_steps=1000000)

    # print("sw_dqn")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget.json', start_idx=0,
    #       end_idx=36, max_steps=1000000)
    #
    # print("sw_dqn_exp_sarsa")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_expectedsarsa.json', start_idx=0,
    #       end_idx=36, max_steps=1000000)

    # print("sw_dqn_exp_sarsa_tn512")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_expectedsarsa_tn512.json', start_idx=0,
    #       end_idx=32, max_steps=1000000)

    # print("sw_dqn_exp_sarsa")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_expectedsarsa_notarget.json', start_idx=0,
    #       end_idx=36, max_steps=1000000)

    # print("sw_dqn notarget")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget.json', start_idx=0,
    #       end_idx=36, max_steps=1000000)
    # #
    # print("sw_dqn notarget_noreplay")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_noreplay.json', start_idx=0,
    #       end_idx=12, max_steps=1000000)
    #
    # print("sw_dqn notarget_noreplay_v2")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2.json', start_idx=0,
    #       end_idx=36, max_steps=500000)

    # print("sw_dqn notarget_noreplay_v2_schedule")
    # _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget_v2_schedule.json', start_idx=0,
    #       end_idx=48, max_steps=1000000)

    print("sw_dqn")
    _eval_lines(config_file='experiment/config_files/fitted_q/exp_sarsa_target_epsilon.json', start_idx=0,
          end_idx=45, max_steps=1000000)

    print("sw_dqn")
    _eval_lines(config_file='experiment/config_files/fitted_q/sw_dqn_fittedq_notarget.json', start_idx=0,
          end_idx=36, max_steps=1000000)