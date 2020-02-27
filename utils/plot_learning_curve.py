import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from BasicTools import plot_tools
plt.rcParams.update({"font.size": "12"})
room_all = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']
reverb_room_all = ['Room_A', 'Room_B', 'Room_C', 'Room_D']


def plot_train_process(model_dir):

    fig, ax = plt.subplots(1, 1)
    plot_settings = {'linewidth': 4}
    n_epoch_max = 0
    for room_i, room in enumerate(reverb_room_all):
        record_fpath = os.path.join(model_dir, room, 'train_record.npz')
        record_info = np.load(record_fpath)
        cost_record_valid = record_info['cost_record_valid']
        # rmse_record_valid = record_info['azi_rmse_record_valid']
        n_epoch = np.nonzero(cost_record_valid)[0][-1] + 1
        if n_epoch > n_epoch_max:
            n_epoch_max = n_epoch
        ax.plot(cost_record_valid[:n_epoch], **plot_settings, label=room[-1])

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.set_ylabel('Cross entrophy')
    ax.set_xlabel('Epoch(n)')
    
    plot_tools.savefig(fig, 'learning_curve_all.png', '../images')    


if __name__ == '__main__':

    model_dir = '../models/mct'
    plot_train_process(model_dir)
