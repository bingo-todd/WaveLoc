import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from BasicTools import get_fpath
from BasicTools import wav_tools

plt.rcParams.update({"font.size": "12"})
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
        record, fs = wav_tools.read_wav(fpath)
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
