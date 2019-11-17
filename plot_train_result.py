import numpy as np
import matplotlib.pyplot as plt
import sys
import os

home_dir = os.path.expanduser('~')
sys.path.append(os.path.join(home_dir,'Work_Space/my_module/basic_tools/basic_tools'))
import plot_tools

def plot_train_result(model_dirs):

    rooms = ['A','B','C','D']
    fig_train,ax_train = plt.subplots(1,4,figsize=[8,3])
    fig_valid,ax_valid = plt.subplots(1,4,figsize=[8,3])

    for model_i,model_dir in enumerate(model_dirs):
        for room_i,room in enumerate(rooms):
            record_info=np.load(os.path.join(model_dir,'mct_not_{}'.format(room),
                                'train_record.npz'))
            cost_record_train=record_info['cost_record_train']
            azi_rmse_record_train=record_info['azi_rmse_record_train']
            cost_record_valid=record_info['cost_record_valid']
            azi_rmse_record_valid=record_info['azi_rmse_record_valid']
            n_epoch = np.nonzero(cost_record_train)[0][-1]+1

            ax_train[room_i].plot(cost_record_train[:n_epoch],label=str(model_i),linewidth=2)
            ax_valid[room_i].plot(cost_record_valid[:n_epoch],label=str(model_i),linewidth=2)

    for i in range(4):
        ax_train[i].set_ylim([0,3])
        ax_train[i].set_xlabel('epoch')
        ax_train[i].set_ylabel('cost')
        ax_train[i].set_title(rooms[i])

        ax_valid[i].set_ylim([0,3])
        ax_valid[i].set_xlabel('epoch')
        ax_valid[i].set_ylabel('cost')
        ax_valid [i].set_title(rooms[i])

    ax_train[3].legend()
    ax_valid[3].legend()

    fig_train.tight_layout()
    fig_valid.tight_layout()

    return [fig_train,fig_valid]


if __name__ == '__main__':
    model_dirs = ['models/basic_{}'.format(i) for i in range(3)]
    fig_train,fig_valid = plot_train_result(model_dirs)
    plot_tools.savefig(fig_train,name='basic_model_train_record_multi_run_train.png')
    plot_tools.savefig(fig_valid,name='basic_model_train_record_multi_run_valid.png')
