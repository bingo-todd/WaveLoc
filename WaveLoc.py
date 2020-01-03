# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import configparser
import time
import gammatone.filters as gt_filters
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#
my_modules_dir = os.path.expanduser('~/my_modules')
sys.path.append(os.path.join(my_modules_dir, 'basic_tools/basic_tools'))
import wav_tools  # noqa:402
import TFData  # noqa:402
from get_fpath import get_fpath  # noqa:402

sys.path.append(os.path.join(my_modules_dir, 'GTF'))
from GTF import GTF  # noqa:402


class WaveLoc(object):
    """
    """
    def __init__(self, file_reader, reader_args={}, config_fpath=None,
                 gpu_index=0):
        """
        """

        # constant settings
        self.epsilon = 1e-20
        self._file_reader = file_reader
        self._reader_args = reader_args

        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph, config=config)

        self._load_cfg(config_fpath)
        self._build_model()

    def _add_log(self, log_info):
        self._log_file.write(log_info)
        self._log_file.write('\n')
        self._log_file.flush()
        if self.is_print_log:
            print(log_info)

    def _load_cfg(self, config_fpath):
        if config_fpath is not None and os.path.exists(config_fpath):
            config = configparser.ConfigParser()
            config.read(config_fpath)

            # settings for model
            self.fs = np.int16(config['model']['fs'])
            self.n_band = np.int16(config['model']['n_band'])
            self.cf_low = np.int16(config['model']['cf_low'])
            self.cf_high = np.int16(config['model']['cf_high'])
            self.frame_len = np.int16(config['model']['frame_len'])
            self.shift_len = np.int16(config['model']['shift_len'])
            self.filter_len = np.int16(config['model']['filter_len'])
            self.is_padd = config['model']['is_padd'] == 'True'
            self.n_azi = np.int16(config['model']['azi_num'])

            # settings for training
            self.batch_size = np.int16(config['train']['batch_size'])
            self.max_epoch = np.int16(config['train']['max_epoch'])
            self.is_print_log = config['train']['is_print_log'] == 'True'
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            if self.valid_set_dir[0] == '':
                self.valid_set_dir = None

            # print('Train set:')
            # [print('\t{}'.format(item)) for item in self.train_set_dir]

            # print('Valid set:')
            # [print('\t{}'.format(item)) for item in self.valid_set_dir]

        else:
            print(config_fpath)
            raise OSError

    def get_gtf_kernel(self):
        """
        """
        cfs = gt_filters.erb_space(self.cf_low, self.cf_high, self.n_band)
        self.cfs = cfs

        sample_times = np.arange(0, self.filter_len, 1)/self.fs
        irs = np.zeros((self.filter_len, self.n_band), dtype=np.float32)

        EarQ = 9.26449
        minBW = 24.7
        order = 1
        N = 4
        for band_i in range(self.n_band):
            ERB = ((cfs[band_i]/EarQ)**order+minBW**order)**(1/order)
            b = 1.019*ERB
            numerator = np.multiply(sample_times**(N-1),
                                    np.cos(2*np.pi*cfs[band_i]*sample_times))
            denominator = np.exp(2*np.pi*b*sample_times)
            irs[:, band_i] = np.divide(numerator, denominator)

        gain = np.max(np.abs(np.fft.fft(irs, axis=0)), axis=0)
        irs_gain_norm = np.divide(np.flipud(irs), gain)
        if self.is_padd:
            kernel = np.concatenate((irs_gain_norm,
                                     np.zeros((self.filter_len, self.n_band))),
                                    axis=0)
        else:
            kernel = irs_gain_norm

        # fig = plt.figure()
        # axes = fig.subplots(1,2)
        # axes[0].plot(irs)
        # axes[1].plot(kernel)
        # fig.savefig('irs.png')
        return kernel

    def _fcn_layers(self, input, *layers_setting):
        for setting in layers_setting:
            fcn_size = setting['fcn_size']
            activation = setting['activation']
            rate = setting['rate']

            layer_fcn = tf.keras.layers.Dense(units=fcn_size,
                                              activation=activation)
            if rate > 0:
                layer_drop = tf.keras.layers.Dropout(rate=rate)
                output = layer_fcn(layer_drop(input))
            elif rate == 0:
                output = layer_fcn(input)
            else:
                raise Exception('illegal dropout rate')
            input = output
        return output

    def _build_model_subband(self, input):
        """
        """
        layer1_conv = tf.keras.layers.Conv2D(filters=6,
                                             kernel_size=[18, 2],
                                             strides=[1, 1],
                                             activation=tf.nn.relu)
        layer1_pool = tf.keras.layers.MaxPool2D([4, 1], [4, 1])
        layer1_out = layer1_pool(layer1_conv(input))

        layer2_conv = tf.keras.layers.Conv2D(filters=12,
                                             kernel_size=[6, 1],
                                             strides=[1, 1],
                                             activation=tf.nn.relu)
        layer2_pool = tf.keras.layers.MaxPool2D([4, 1], [4, 1])
        layer2_out = layer2_pool(layer2_conv(layer1_out))

        flatten_len = np.prod(layer2_out.get_shape().as_list()[1:])
        out = tf.reshape(layer2_out, [-1, flatten_len])  # flatten
        return out

    def _build_model(self):
        """Build graph
        """
        # gammatone layer kernel initalizer
        with self._graph.as_default():
            kernel_initializer = tf.constant_initializer(
                                                self.get_gtf_kernel())

            if self.is_padd:
                gtf_kernel_len = 2*self.filter_len
            else:
                gtf_kernel_len = self.filter_len

            x = tf.compat.v1.placeholder(shape=[None, self.frame_len, 2, 1],
                                         dtype=tf.float32,
                                         name='x')  #

            layer1_conv = tf.keras.layers.Conv2D(
                                        filters=self.n_band,
                                        kernel_size=[gtf_kernel_len, 1],
                                        strides=[1, 1],
                                        padding='same',
                                        kernel_initializer=kernel_initializer,
                                        trainable=False, use_bias=False)

            # add to model for test
            self.layer1_conv = layer1_conv

            # amplitude normalization across frequency channs
            # problem: silence ?
            layer1_conv_output = layer1_conv(x)
            amp_max = tf.reduce_max(
                        tf.reduce_max(
                            tf.reduce_max(
                                tf.abs(layer1_conv_output),
                                axis=1, keepdims=True),
                            axis=2, keepdims=True),
                        axis=3, keepdims=True)
            layer1_conv_output_norm = tf.divide(layer1_conv_output, amp_max)

            # layer1_pool
            layer1_pool = tf.keras.layers.MaxPool2D([2, 1], [2, 1])
            layer1_output = layer1_pool(layer1_conv_output_norm)

            band_out_list = []
            for band_i in range(self.n_band):
                band_output = self._build_model_subband(
                                tf.expand_dims(layer1_output[:, :, :, band_i],
                                               axis=-1))
                band_out_list.append(band_output)
            band_out = tf.concat(band_out_list, axis=1)

            layer1 = {'fcn_size': 1024,
                      'activation': tf.nn.relu,
                      'rate': 0.5}
            layer2 = {'fcn_size': 1024,
                      'activation': tf.nn.relu,
                      'rate': 0.5}
            output_layer = {'fcn_size': self.n_azi,
                            'activation': tf.nn.softmax,
                            'rate': 0}

            y_est = self._fcn_layers(band_out, layer1, layer2, output_layer)

            # groundtruth of two tasks
            y = tf.compat.v1.placeholder(shape=[None, self.n_azi],
                                         dtype=tf.float32)
            # cost function
            cost = self._cal_cross_entropy(y_est, y)
            # additional measurement of localization
            azi_rmse = self._cal_azi_rmse(y_est, y)

            #
            lr = tf.compat.v1.placeholder(tf.float32, shape=[])
            opt_step = tf.compat.v1.train.AdamOptimizer(
                                            learning_rate=lr).minimize(cost)

            # initialize of model
            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)

            # input and output
            self._x = x
            self._y_est = y_est
            # groundtruth
            self._y = y
            # cost function and optimizer
            self._cost = cost
            self._azi_rmse = azi_rmse
            self._lr = lr
            self._opt_step = opt_step

    def _cal_cross_entropy(self, y_est, y):
        cross_entropy = -tf.reduce_mean(
                            tf.reduce_sum(
                                tf.multiply(
                                    y, tf.math.log(y_est+self.epsilon)),
                                axis=1))
        return cross_entropy

    def _cal_mse(self, y_est, y):
        rmse = tf.reduce_mean(tf.reduce_sum((y-y_est)**2, axis=1))
        return rmse

    def _cal_azi_rmse(self, y_est, y):
        azi_est = tf.argmax(y_est, axis=1)
        azi = tf.argmax(y, axis=1)
        diff = tf.cast(azi_est - azi, tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.pow(diff, 2)))

    def _cal_cp(self, y_est, y):
        equality = tf.equal(tf.argmax(y_est, axis=1), tf.argmax(y, axis=1))
        cp = tf.reduce_mean(tf.cast(equality, tf.float32))
        return cp

    def load_model(self, model_dir):
        """load model"""
        if not os.path.exists(model_dir):
            raise Exception('no model exists in {}'.format(model_dir))

        with self._graph.as_default():
            # restore model
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess, ckpt.model_checkpoint_path)

            print(f'load model from {model_dir}')

    def _train_record_init(self, model_dir, is_load_model):
        """
        """
        if is_load_model:
            record_info = np.load(os.path.join(model_dir,
                                               'train_record.npz'))
            cost_record_valid = record_info['cost_record_valid']
            azi_rmse_record_valid = record_info['azi_rmse_record_valid']
            lr_value = record_info['lr']
            best_epoch = record_info['best_epoch']
            min_valid_cost = record_info['min_valid_cost']
            last_epoch = np.nonzero(cost_record_valid)[0][-1]
        else:
            cost_record_valid = np.zeros(self.max_epoch)
            azi_rmse_record_valid = np.zeros(self.max_epoch)
            lr_value = 1e-3
            min_valid_cost = np.infty
            best_epoch = 0
            last_epoch = -1
        return [cost_record_valid, azi_rmse_record_valid,
                lr_value, min_valid_cost, best_epoch, last_epoch]

    def train_model(self, model_dir, is_load_model=False):
        """Train model either from initial state(self._build_model()) or
        already existed model
        """
        if is_load_model:
            self.load_model(model_dir)

        if not os.path.exists(model_dir):
            os.makedirs(self.model_dir)

        # open text file for logging
        self._log_file = open(os.path.join(model_dir, 'log.txt'), 'a')

        with self._graph.as_default():

            [cost_record_valid, azi_rmse_record_valid,
             lr_value, min_valid_cost,
             best_epoch, last_epoch] = self._train_record_init(model_dir,
                                                               is_load_model)

            saver = tf.compat.v1.train.Saver()
            print('start training')
            for epoch in range(last_epoch+1, self.max_epoch):
                t_start = time.time()
                print(f'epoch {epoch}')
                for x, y in self._file_reader(self.train_set_dir,
                                              **self._reader_args):
                    self._sess.run(self._opt_step,
                                   feed_dict={self._x: x,
                                              self._y: y,
                                              self._lr: lr_value})
                # model test
                [cost_record_valid[epoch],
                 azi_rmse_record_valid[epoch]] = self.evaluate(
                                                        self.valid_set_dir)

                # write to log
                iter_time = time.time()-t_start
                self._add_log(' '.join((f'epoch:{epoch}',
                                        f'lr:{lr_value}',
                                        f'time:{iter_time:.2f}\n')))

                log_template = '\t cost:{:.2f} azi_rmse:{:.2f}\n'
                self._add_log('\t valid ')
                self._add_log(log_template.format(
                                                cost_record_valid[epoch],
                                                azi_rmse_record_valid[epoch]))

                #
                if min_valid_cost > cost_record_valid[epoch]:
                    self._add_log('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost = cost_record_valid[epoch]
                    saver.save(self._sess, os.path.join(model_dir,
                                                        'model'),
                               global_step=epoch)

                    # save record info
                    np.savez(os.path.join(model_dir, 'train_record'),
                             cost_record_valid=cost_record_valid,
                             azi_rmse_record_valid=azi_rmse_record_valid,
                             lr=lr_value,
                             best_epoch=best_epoch,
                             min_valid_cost=min_valid_cost)

                # early stop
                if epoch-best_epoch > 5:
                    print('early stop\n', min_valid_cost)
                    self._add_log('early stop{}\n'.format(min_valid_cost))
                    break

                # learning rate decay
                if epoch > 2:  # no better performance in 2 epoches
                    min_valid_cost_local = np.min(
                                        cost_record_valid[epoch-1:epoch+1])
                    if cost_record_valid[epoch-2] < min_valid_cost_local:
                        lr_value = lr_value*.2

            self._log_file.close()

            if True:
                fig, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
                ax[0].plot(cost_record_valid)
                ax[0].set_ylabel('cross entrophy')

                ax[1].plot(azi_rmse_record_valid)
                ax[1].set_ylabel('rmse(deg)')
                #
                fig_path = os.path.join(model_dir, 'train_curve.png')
                plt.savefig(fig_path)

    def predict(self, x):
        """Model output of x
        """
        y_est = self._sess.run(self._y_est, feed_dict={self._x: x})
        return y_est

    def evaluate(self, set_dir):
        cost_all = 0.
        rmse_all = 0.
        n_sample_all = 0
        for x, y in self._file_reader(set_dir, is_shuffle=False):
            n_sample_tmp = x.shape[0]
            [cost_tmp, rmse_tmp] = self._sess.run([self._cost, self._azi_rmse],
                                                  feed_dict={self._x: x,
                                                             self._y: y})
            #
            n_sample_all = n_sample_all+n_sample_tmp
            cost_all = cost_all+n_sample_tmp*cost_tmp
            rmse_all = rmse_all+n_sample_tmp*(rmse_tmp**2)

        # average across all set
        cost_all = cost_all/n_sample_all
        rmse_all = np.sqrt(rmse_all/n_sample_all)
        return [cost_all, rmse_all]

    def evaluate_chunk_rmse(self, record_set_dir, chunk_size=25):
        """ Evaluate model on given data_set, only for loc
        Args:
            data_set_dir:
        Returns:
            [rmse_chunk,cp_chunk,rmse_frame,cp_frame]
        """
        rmse_chunk = 0.
        n_chunk = 0

        for x, y in self._file_reader(record_set_dir, is_shuffle=False):
            sample_num = x.shape[0]
            azi_true = np.argmax(y[0])

            y_est = self.predict(x)
            for sample_i in range(0, sample_num-chunk_size+1):
                azi_est_chunk = np.argmax(
                                    np.mean(
                                        y_est[sample_i:sample_i+chunk_size],
                                        axis=0))
                rmse_chunk = rmse_chunk+(azi_est_chunk-azi_true)**2
                n_chunk = n_chunk+1

        rmse_chunk = np.sqrt(rmse_chunk/n_chunk)*5
        return rmse_chunk
