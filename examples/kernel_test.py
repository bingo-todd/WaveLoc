import matplotlib.pyplot as plt
import configparser
import tensorflow as tf
import numpy as np
import os
import sys

import WavLoc
import WavLoc_kernel_no_padd


def main():

    config = configparser.ConfigParser()

    config['model']={'model_dir':'test',
                    'frame_len':'320',
                    'filter_len':'320',
                    'overlap_len':'160',
                    'azi_num':'37'}

    config['train']={'batch_size':'128',
                     'max_epoch':'10',
                     'train_set_dir':'../../Data/Record_v4/train/Anechoic/;\
                     ../../Data/Record_v4/train/B/;\
                     ../../Data/Record_v4/train/C/;\
                     ../../Data/Record_v4/train/D/',
                     'valid_set_dir':'../../Data/Record_v4/valid/Anechoic/;\
                     ../../Data/Record_v4/valid/B/;\
                     ../../Data/Record_v4/valid/C/;\
                     ../../Data/Record_v4/valid/D/'}

    with open('test/test.cfg','w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    model = WavLoc.WavLoc('test/test.cfg')
    fig_dir = 'images/kernel_test'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # kernel initial value of different length
    model.filter_len = 320
    filter_ir_320 = model.get_gtf_kernel_old()
    model.filter_len = 640
    filter_ir_640 = model.get_gtf_kernel_old()
    model.filter_len = 960
    filter_ir_960 = model.get_gtf_kernel_old()

    fig = plt.figure(figsize=[12,3])
    axes = fig.subplots(1,3)
    axes[0].plot(filter_ir_320); axes[0].set_title('filter_len: 320')
    axes[1].plot(filter_ir_640); axes[1].set_title('filter_len: 640')
    axes[2].plot(filter_ir_960); axes[2].set_title('filter_len: 960')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,'gtf_kernels_diff_len.png'))


    model.filter_len = 320
    model.build_model()
    frame_i = 60
    with model._graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        model._sess.run(init)
        # kernel value that net actually use
        filter_kernel = model._sess.run(model.layer1_conv.weights)[0]
        # normalized gtf layer output
        batch_generator = model.file_reader(model.train_set_dir)
        for batch_value in batch_generator:
            layer1_norm_output = model._sess.run(model.layer1_conv_output_norm,
                                         feed_dict={model._x:batch_value[0][frame_i:]})


    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.plot(np.squeeze(filter_kernel))
    ax.set_title('conv2d_kernel')
    fig.savefig(os.path.join(fig_dir,'gtf_kernel_in_net.png'))

    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.plot(np.squeeze(np.squeeze(batch_value[0][frame_i])))
    ax.set_title('input')
    fig.savefig(os.path.join(fig_dir,'input.png'))

    fig = plt.figure(figsize=[6,10])
    axes = fig.subplots(4,2)
    for i in range(8):
        band_i = i*4
        sub_axes = axes[np.int8(i/2),np.mod(i,2)]
        sub_axes.plot(layer1_norm_output[0,:,:,band_i])
        sub_axes.set_ylim([-1,1])
        sub_axes.set_title('cf= {:.0f}'.format(model.cfs[band_i]))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir,'layer1_norm_output.png'))


    model2 = WavLoc_kernel_no_padd.WavLoc('test/test.cfg')
    model2.filter_len = 320
    model2.build_model()
    with model2._graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        model2._sess.run(init)

        filter_kernel = model2._sess.run(model2.layer1_conv.weights)[0]

        batch_generator = model2.file_reader(model2.train_set_dir)
        for batch_value in batch_generator:
            layer1_norm_output = model2._sess.run(model2.layer1_conv_output_norm,
                                         feed_dict={model2._x:batch_value[0][frame_i:]})

    fig = plt.figure()
    ax = fig.subplots(1,1)
    ax.plot(np.squeeze(filter_kernel))
    ax.set_title('conv2d_kernel')
    fig.savefig(os.path.join(fig_dir,'gtf_kernel_in_net_no_padd.png'))

    fig = plt.figure(figsize=[6,10])
    axes = fig.subplots(4,2)
    for i in range(8):
        band_i = i*4
        sub_axes = axes[np.int8(i/2),np.mod(i,2)]
        sub_axes.plot(layer1_norm_output[0,:,:,band_i])
        sub_axes.set_ylim([-1,1])
        sub_axes.set_title('cf= {:.0f}'.format(model2.cfs[band_i]))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir,'layer1_norm_output_no_padd.png'))

if __name__ == '__main__':
    main()
