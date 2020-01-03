import numpy as np
import os
import scipy.io as sio
import shutil
import sys

my_module_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_module_dir, 'basic_tools/basic_tools'))
from ProcessBar import ProcessBar  # noqa: E402
import wav_tools  # noqa: E402
from Filter_GPU import Filter_GPU  # noqa: E402
from get_fpath import get_fpath  # noqa: E402

TIMIT_dir = '../../../Data/TIMIT'
data_dir = '../../WaveLoc/Data'
room_all = ['Anechoic', 'A', 'B', 'C', 'D']
n_room = 5
n_azi = 37  # -90~90 in step of 5
n_wav_per_azi_all = {'train': 24,
                     'valid': 6,
                     'test': 15}


def load_brirs(room):
    """load brirs of given room
    Args:
        room: room name from ['Anechoic','A','B','C','D']
    """
    brirs_dir = '/mnt/hd6t/songtao/Data/RealRoomBRIRs-master/16kHz/'
    # time aligned to corresponding brir of anechoic room
    brirs_dict = sio.loadmat(os.path.join(
                                    brirs_dir, f'BRIR_{room}_aligned.mat'))
    brirs = brirs_dict['hrirs']
    return brirs


def truncate_silence(x):
    fs = 16e3
    frame_len = int(20e-3*fs)
    shift_len = int(10e-3*fs)
    vad_flag = wav_tools.vad(x, frame_len, shift_len)
    if not vad_flag[0]:  # silence in the first frame
        x = x[frame_len:]
    if not vad_flag[-1]:  # silence in the last frame
        x = x[:-frame_len]
    return x


def gen_dataset(dir, set_type_all):

    TIMIT_train_dir = os.path.join(TIMIT_dir, 'TIMIT/TRAIN')
    src_fpath_train_all = get_fpath(TIMIT_train_dir, '.wav')
    TIMIT_test_dir = os.path.join(TIMIT_dir, 'TIMIT/TEST')
    src_fpath_test_all = get_fpath(TIMIT_test_dir, '.wav')

    n_wav_train_valid = (24+6)*n_azi*n_room
    n_wav = len(src_fpath_train_all)
    src_fpath_train_all.extend(np.random.choice(src_fpath_train_all,
                                                n_wav_train_valid-n_wav,
                                                replace=False))
    np.random.shuffle(src_fpath_train_all)

    # test set
    n_wav_test = 15*n_azi*n_room
    n_wav = len(src_fpath_test_all)
    src_fpath_test_all.extend(np.random.choice(src_fpath_test_all,
                                               n_wav_test-n_wav,
                                               replace=False))
    np.random.shuffle(src_fpath_test_all)

    np.save(os.path.join(dir, 'src_fpath_all.npy'),
            [src_fpath_train_all, src_fpath_test_all])

    brirs_direct = load_brirs('Anechoic')
    filter_gpu = Filter_GPU(gpu_index=0)

    n_wav_per_azi = np.sum([n_wav_per_azi_all[type] for type in set_type_all])
    pb = ProcessBar(n_azi*n_room*n_wav_per_azi)

    wav_count_train_valid = 0
    wav_count_test = 0

    for set_type in set_type_all:
        set_dir = os.path.join(dir, set_type)
        for room in room_all:
            brirs_room = load_brirs(room)
            os.makedirs(os.path.join(set_dir, 'source', room))
            os.makedirs(os.path.join(set_dir, 'reverb', room))
            os.makedirs(os.path.join(set_dir, 'direct', room))

            for azi_i in range(n_azi):
                for i in range(n_wav_per_azi_all[set_type]):
                    src_fpath = os.path.join(set_dir, 'source', room,
                                             '{}_{}.wav'.format(azi_i, i))
                    if set_type == 'test':
                        src_fpath_TIMIT = os.path.join(
                                    TIMIT_test_dir,
                                    src_fpath_test_all[wav_count_test])
                        wav_count_test = wav_count_test+1
                    else:
                        src_fpath_TIMIT = os.path.join(
                                    TIMIT_train_dir,
                                    src_fpath_train_all[wav_count_train_valid])
                        wav_count_train_valid = wav_count_train_valid+1

                    shutil.copyfile(src_fpath_TIMIT, src_fpath)

                    src, fs = wav_tools.wav_read(src_fpath)
                    src = truncate_silence(src)

                    reverb = filter_gpu.brir_filter(src, brirs_room[azi_i])
                    reverb_fpath = os.path.join(set_dir, 'reverb', room,
                                                '{}_{}.wav'.format(azi_i, i))
                    wav_tools.wav_write(reverb, fs, reverb_fpath)

                    direct = filter_gpu.brir_filter(src, brirs_direct[azi_i])
                    direct_fpath = os.path.join(set_dir, 'direct', room,
                                                '{}_{}.wav'.format(azi_i, i))
                    wav_tools.wav_write(direct, fs, direct_fpath)

                    pb.update()


if __name__ == '__main__':

    # train dataset as well as validation dataset
    dataset_dir = os.path.join(data_dir, 'v1')
    os.makedirs(dataset_dir)
    gen_dataset(dir=dataset_dir, set_type_all=['train', 'valid', 'test'])

    for i in range(2, 4):
        dataset_dir = os.path.join(data_dir, f'v{i}')
        os.makedirs(dataset_dir)
        gen_dataset(dir=dataset_dir, set_type_all=['test'])
