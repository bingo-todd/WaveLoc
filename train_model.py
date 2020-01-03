import configparser
from WaveLoc import WaveLoc

import os
import sys
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa: E402
from get_fpath import get_fpath  # noqa: E402
import wav_tools  # noqa: E402

from utils import file_reader  # noqa:402


data_dir = '../../WaveLoc/Data'
train_set_dir_base = os.path.join(data_dir, 'v1/train/reverb')
valid_set_dir_base = os.path.join(data_dir, 'v1/valid/reverb')

room_all = ['Anechoic', 'A', 'B', 'C', 'D']
reverb_room_all = ['A', 'B', 'C', 'D']

model_basic_settings = {'fs': 16000,
                        'n_band': 32,
                        'cf_low': 70,
                        'cf_high': 7000,
                        'frame_len': 320,
                        'shift_len': 160,
                        'filter_len': 320,
                        'azi_num': 37,
                        'is_use_gtf': False,
                        'is_padd': False}
gpu_index = 0


def train_mct(room_tar, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # filter out room_tar from room_all
    mct_room_all = [room for room in room_all if room != room_tar]
    config = configparser.ConfigParser()
    config['model'] = {**model_basic_settings}
    config['train'] = {'batch_size': 128,
                       'max_epoch': 50,
                       'is_print_log': True,
                       'train_set_dir': ';'.join(
                           [os.path.join(train_set_dir_base, room)
                            for room in mct_room_all]),
                       'valid_set_dir': ';'.join(
                           [os.path.join(valid_set_dir_base, room)
                            for room in mct_room_all])}

    config_fpath = os.path.join(model_dir, 'config.cfg')
    with open(config_fpath, 'w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    model = WaveLoc(file_reader, config_fpath=config_fpath,
                    gpu_index=gpu_index)
    model.train_model(model_dir)


def train_act(model_dir):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config = configparser.ConfigParser()
    config['model'] = {**model_basic_settings}

    config['train'] = {'batch_size': 128,
                       'max_epoch': 50,
                       'is_print_log': True,
                       'train_set_dir': ';'.join(
                           [os.path.join(train_set_dir_base, room)
                            for room in room_all]),
                       'valid_set_dir': ';'.join(
                           [os.path.join(valid_set_dir_base, room)
                            for room in room_all])}

    config_fpath = os.path.join(model_dir, 'config.cfg')
    with open(config_fpath, 'w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    model = WaveLoc(file_reader, config_fpath=config_fpath,
                    gpu_index=gpu_index)
    model.train_model()


if __name__ == '__main__':

    train_strategy = sys.argv[1]

    if train_strategy == 'mct':
        room_tar = sys.argv[2]
        model_dir = sys.argv[3]
        train_mct(room_tar, model_dir)
    elif train_strategy == 'act':
        model_dir = sys.argv[2]
        train_act(model_dir)
