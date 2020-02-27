import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
from WaveLoc import WaveLoc
from BasicTools import plot_tools
from utils import file_reader


room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']
n_reverb_room = 4
chunk_size = 25
n_test = 4


def evaluate_mct(model_dir_base):
    rmse_all = np.zeros((n_test, n_reverb_room))
    for room_i, room in enumerate(reverb_room_all):
        model_dir = os.path.join(model_dir_base, room)
        model_config_fpath = os.path.join(model_dir, 'config.cfg')
        model = WaveLoc(file_reader.file_reader,
                        model_config_fpath, gpu_index=0)
        model.load_model(model_dir)

        for test_i in range(n_test):
            dataset_dir_test = os.path.join(
                                    '/home/st/Work_Space/Localize/WaveLoc/Data',
                                    f'v{test_i+1}/test/reverb/{room[-1]}')
            rmse_all[test_i, room_i] = model.evaluate_chunk_rmse(
                                        dataset_dir_test,
                                        chunk_size=chunk_size)
    return rmse_all


if __name__ == '__main__':
    model_dir = sys.argv[1]  #'models/mct'
    rmse_all = evaluate_mct(model_dir)

    with open(os.path.join(model_dir, 'result.txt'), 'w') as result_file:
        result_file.write(f'{rmse_all}')
        result_file.write('mean: {}\n'.format(np.mean(rmse_all, axis=0)))
        result_file.write('std: {}\n'.format(np.std(rmse_all, axis=0)))

    print(rmse_all)
    print('mean:', np.mean(rmse_all, axis=0))
    print('std:', np.std(rmse_all, axis=0))
