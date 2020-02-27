"""
frame raw wavefrom and save in batches
"""


import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from BasicTools import get_fpath, wav_tools, ProcessBar

plt.rcParams.update({"font.size": "12"})
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']


def wav2npy(reverb_set_dir, npy_dir, is_anechoic):
    """ read wav files in given directies, one file per time
    Args:
        record_set_dir: directory or list of directories where recordings exist
        batch_size:
        is_shuffle:
    Returns:
        samples generator, [samples, label_all]
    """

    frame_len = 320
    shift_len = 160
    n_azi = 37
    batch_size = 128

    os.makedirs(npy_dir, exist_ok=True)

    #
    fpath_reverb_all = get_fpath(reverb_set_dir, '.wav', is_absolute=True)
    if len(fpath_reverb_all) < 1:
        raise Exception('empty folder:{}'.format(reverb_set_dir))

    pb = ProcessBar(len(fpath_reverb_all))

    batch_count = 0
    x_r = np.zeros((0, frame_len, 2, 1))
    x_d = np.zeros((0, frame_len, 2, 1))
    y_loc = np.zeros((0, n_azi))

    for fpath_reverb in fpath_reverb_all:
        pb.update()
        # reverb signal
        record, fs = wav_tools.read_wav(fpath_reverb)
        x_r_file = np.expand_dims(
                        wav_tools.frame_data(record, frame_len, shift_len),
                        axis=-1)
        # direct signal
        fpath_direct = fpath_reverb.replace('reverb', 'direct')
        direct, fs = wav_tools.read_wav(fpath_direct)
        x_d_file = np.expand_dims(
                        wav_tools.frame_data(direct, frame_len, shift_len),
                        axis=-1)

        # onehot azi label
        n_sample_file = x_d_file.shape[0]
        if x_r_file.shape[0] != n_sample_file:
            raise Exception('sample number do not consist')

        fname = os.path.basename(fpath_reverb)
        azi = np.int16(fname.split('_')[0])
        y_loc_file = np.zeros((n_sample_file, n_azi))
        y_loc_file[:, azi] = 1

        x_r = np.concatenate((x_r, x_r_file), axis=0)
        x_d = np.concatenate((x_d, x_d_file), axis=0)
        y_loc = np.concatenate((y_loc, y_loc_file), axis=0)

        while x_d.shape[0] > batch_size:
            x_r_batch = x_r[:batch_size]
            x_d_batch = x_d[:batch_size]
            y_loc_batch = y_loc[:batch_size]

            npy_fpath = os.path.join(npy_dir, '{}.npy'.format(batch_count))
            np.save(npy_fpath,[x_d_batch, x_r_batch, y_loc_batch, is_anechoic])
            batch_count = batch_count + 1

            x_r = x_r[batch_size:]
            x_d = x_d[batch_size:]
            y_loc = y_loc[batch_size:]




if __name__ == '__main__':
    
    for set_type in ['train', 'valid']:
        for room in room_all:
            print(room)
            wav_set_dir = '../Data/v1/{}/reverb/{}'.format(set_type, room)
            npy_set_dir = '../Data/v1/npy/{}/{}'.format(set_type, room)
            wav2npy(wav_set_dir, npy_set_dir, False)


    for test_i in range(1,5):
        for room in room_all:
            print(room)
            wav_set_dir = f'../Data/v{test_i}/test/reverb/{room}'
            npy_set_dir = f'../Data/v{test_i}/npy/test/{room}'
            wav2npy(wav_set_dir, npy_set_dir, False)