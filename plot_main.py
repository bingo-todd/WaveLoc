import os
import sys
my_modules_dir = os.path.join(os.path.expanduser('~'), 'my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import plot_tools  # noqa:402

import utils  # noqa:402


if __name__ == '__main__':
    # plot traininig curves
    if True:
        fig = utils.plot_train_process('models/act/train_record.npz')
        plot_tools.savefig(fig, fig_name='act_train_curve.png',
                           fig_dir='images/train')

        fig = utils.plot_mct_train_process('models/mct')
        plot_tools.savefig(fig, fig_name='mct_train_curve.png',
                           fig_dir='images/train')

        fig = utils.plot_evaluation(
                            ('Result/rmse_act.npy', 'Result/rmse_mct.npy'),
                            ('act', 'mct'))
        plot_tools.savefig(fig, fig_name='rmse_result_all.png',
                           fig_dir='images/evaluate')
