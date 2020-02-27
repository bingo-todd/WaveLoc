import sys
import os
import configparser
from multiprocessing import Process
from WaveLoc import WaveLoc
from utils import file_reader_v2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_dir = 'Data'
train_set_dir_base = os.path.join(data_dir, 'v1/npy/train')
valid_set_dir_base = os.path.join(data_dir, 'v1/npy/valid')

room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']

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
gpu_index = 1


def train_mct(room_tar, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # filter out room_tar from room_all
    print('tar_room', room_tar)
    mct_room_all = [room for room in room_all if room != room_tar]
    config = configparser.ConfigParser()
    config['model'] = {**model_basic_settings}
    config['train'] = {'batch_size': 128,
                       'max_epoch': 50,
                       'is_print_log': False,
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

    model = WaveLoc(file_reader_v2.file_reader, config_fpath=config_fpath,
                    gpu_index=gpu_index)
    model.train_model(model_dir)


if __name__ == '__main__':
    thread_all = []
    for room_tar in reverb_room_all:
        model_dir = 'models/mct/{}'.format(room_tar)
        thread = Process(target=train_mct,args=(room_tar,model_dir))
        thread.start()
        thread_all.append(thread)

    [thread.join() for thread in thread_all]