import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": "12"})  # noqa: 402

import copy

import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
from get_fpath import get_fpath  # noqa: E402
import wav_tools  # noqa: E402
import plot_tools  # noqa: 402


reverb_room_all = ['A', 'B', 'C', 'D']


def file_reader(record_set_dir, batch_size=-1, is_shuffle=True):
    """ read wav files in given directies, one file per time
    Args:
        record_set_dir: directory or list of directories where recordings exist
    Returns:
        samples generator, [samples, label_all]
    """
    if isinstance(record_set_dir, list):
        dirs = record_set_dir
    else:
        dirs = [record_set_dir]
    #
    fpath_all = []
    for sub_set_dir in dirs:
        fpath_all_sub = get_fpath(sub_set_dir, '.wav', is_absolute=True)
        fpath_all.extend(fpath_all_sub)

    if is_shuffle:
        np.random.shuffle(fpath_all)

    # print('#file',len(fpath_all))
    # raise Exception()

    if len(fpath_all) < 1:
        raise Exception('empty folder:{}'.format(record_set_dir))

    frame_len = 320
    shift_len = 160
    n_azi = 37

    if batch_size > 1:
        x_all = np.zeros((0, frame_len, 2, 1))
        y_all = np.zeros((0, n_azi))

    for fpath in fpath_all:
        record, fs = wav_tools.wav_read(fpath)
        x_file_all = wav_tools.frame_data(record, frame_len, shift_len)
        x_file_all = np.expand_dims(x_file_all, axis=-1)

        # onehot azi label
        n_sample_file = x_file_all.shape[0]
        fname = os.path.basename(fpath)
        azi = np.int16(fname.split('_')[0])
        y_file_all = np.zeros((n_sample_file, n_azi))
        y_file_all[:, azi] = 1

        if batch_size > 0:
            x_all = np.concatenate((x_all, x_file_all), axis=0)
            y_all = np.concatenate((y_all, y_file_all), axis=0)

            while x_all.shape[0] > batch_size:
                x_batch = copy.deepcopy(x_all[:batch_size])
                y_batch = copy.deepcopy(y_all[:batch_size])

                x_all = x_all[batch_size:]
                y_all = y_all[batch_size:]

                yield [x_batch, y_batch]
        else:
            yield [x_file_all, y_file_all]


def plot_train_process(record_fpath, ax=None, label=None):

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=[6, 4],
                               sharex=True, tight_layout=True)
    else:
        fig = None

    record_info = np.load(record_fpath)

    #
    cost_record_valid = record_info['cost_record_valid']
    rmse_record_valid = record_info['azi_rmse_record_valid']
    n_epoch = np.nonzero(cost_record_valid)[0][-1] + 1

    plot_settings = {'label': label, 'linewidth': 2}
    ax[0].plot(cost_record_valid[:n_epoch], **plot_settings)
    ax[1].plot(rmse_record_valid[:n_epoch], **plot_settings)

    ax[1].legend()

    ax[0].set_ylabel('Cross entrophy')
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('Epoch(n)')
    ax[1].set_xlabel('Epoch(n)')

    return fig


def plot_mct_train_process(model_dir):

    reverb_room_all = ['A', 'B', 'C', 'D']
    fig, ax = plt.subplots(1, 2, figsize=[6, 4], sharex=True,
                           tight_layout=True)

    for room_i, room in enumerate(reverb_room_all):
        record_fpath = os.path.join(model_dir, room, 'train_record.npz')
        plot_train_process(record_fpath, ax, label=room)
    return fig


def plot_evaluate_result(result_fpath_all, label_all):

    mean_std_all = []
    for result_fpath in result_fpath_all:
        rmse_multi_test = np.load(result_fpath)
        rmse_mean = np.mean(rmse_multi_test, axis=0)
        rmse_std = np.std(rmse_multi_test, axis=0)
        mean_std_all.append([rmse_mean, rmse_std])

    fig = plot_tools.plot_bar(*mean_std_all, legend=label_all,
                              xticklabels=reverb_room_all,
                              xlabel='Room', ylabel='RMSE($^o$)',
                              ylim=[0, 4])
    return fig


def plot_evaluation(result_fpath_all, legend_all):
    mean_std_all = []
    for result_fpath in result_fpath_all:
        rmse_multi_test = np.load(result_fpath)

        rmse_mean = np.mean(rmse_multi_test, axis=0)
        rmse_std = np.std(rmse_multi_test, axis=0)
        mean_std_all.append([rmse_mean, rmse_std])

    fig = plot_tools.plot_bar(*mean_std_all, legend=legend_all,
                              xticklabels=reverb_room_all,
                              xlabel='Room', ylabel='RMSE($^o$)',
                              ylim=[0, 4])
    return fig


if __name__ == '__main__':

    if False:
        model_dir = 'models/mct'
        plot_mct_train_process(model_dir)
