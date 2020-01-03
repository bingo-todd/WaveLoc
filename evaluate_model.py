import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mtick
import os
import sys
from WaveLoc import WaveLoc

my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa:402

from utils import file_reader  # noqa:402


data_dir_base = '../../WaveLoc/Data'

reverb_room_all = ['A', 'B', 'C', 'D']
n_reverb_room = 4
chunk_size = 25
n_test = 3


def evaluate_mct(model_dir_base):
    rmse_all = np.zeros((n_test, n_reverb_room))
    for room_i, room in enumerate(reverb_room_all):
        model_dir = os.path.join(model_dir_base, room)
        model_config_fpath = os.path.join(model_dir, 'config.cfg')
        model = WaveLoc(file_reader, config_fpath=model_config_fpath,
                        gpu_index=0)
        model.load_model(model_dir)

        for test_i in range(n_test):
            dataset_dir_test = os.path.join(
                                    data_dir_base,
                                    f'v{test_i+1}/test/reverb/{room}')
            rmse_all[test_i, room_i] = model.evaluate_chunk_rmse(
                                        dataset_dir_test,
                                        chunk_size=chunk_size)
    return rmse_all


def evaluate_act(model_dir):

    model_config_fpath = os.path.join(model_dir, 'config.cfg')
    model = WaveLoc(file_reader, config_fpath=model_config_fpath,
                    gpu_index=0)
    model.load_model(model_dir)
    rmse_all = np.zeros((n_test, n_reverb_room))
    for test_i in range(n_test):
        for room_i, room in enumerate(reverb_room_all):
            dataset_dir_test = os.path.join(
                                    data_dir_base,
                                    f'v{test_i+1}/test/reverb/{room}')

            rmse_all[test_i, room_i] = model.evaluate_chunk_rmse(
                                            dataset_dir_test,
                                            chunk_size=chunk_size)
    return rmse_all


def plot_result():
    result_fpath_all = ('Result/rmse_mct.npy', 'Result/rmse_all_room.npy')
    label_all = ['mct', 'all_room']

    mean_std_all = []
    for result_fpath in result_fpath_all:
        rmse_all = np.load(result_fpath)
        rmse_mean = np.mean(rmse_all, axis=0)
        rmse_std = np.std(rmse_all, axis=0)
        mean_std_all.append([rmse_mean, rmse_std])

    fig = plot_tools.plot_bar(*mean_std_all, legend=label_all,
                              xticklabels=reverb_room_all,
                              xlabel='Room', ylabel='RMSE($^o$)',
                              ylim=[0, 4])
    plot_tools.savefig(fig, name='rmse_result')


if __name__ == '__main__':

    train_strategy = sys.argv[1]
    model_dir = sys.argv[2]

    if train_strategy == 'mct':
        rmse_all = evaluate_mct(model_dir)
    elif train_strategy == 'act':
        rmse_all = evaluate_act(model_dir)
    else:
        raise Exception()

    if len(sys.argv) >= 4:
        result_fpath = sys.argv[3]
        np.save(result_fpath, rmse_all)

    print(train_strategy)
    print(rmse_all)
    print('mean:', np.mean(rmse_all, axis=0))
    print('std:', np.std(rmse_all, axis=0))
