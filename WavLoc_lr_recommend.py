# -*- coding:utf-8 -*-
import numpy as np
import os
import glob # check if file exists using regex
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import configparser
import sys
import time
import gammatone.filters as gt_filters
#
workspace_dir = os.path.join(os.path.expanduser('~'),'Work_Space')
sys.path.append(os.path.join(workspace_dir,'my_module/basic-toolbox/basic_tools'))
import wav_tools
import TFData

sys.path.append(os.path.join(workspace_dir,'my_module/auditory'))
from auditory.gtf import gtf


class WavLoc(object):
    """
    """
    def __init__(self,config_fpath=None,gpu_index=0,is_load_model=False):

        if config_fpath is None and meta_fpath is None:
            raise Exception('neither config_fpath or meta_fpath is given')

        # constant settings
        self.fs=16e3
        self.n_band = 32  # frequency bands number
        self.epsilon = 1e-20
        # previous settings
        self.cf_low=70
        self.cf_high=7e3
        # using gtf module
        self.freq_low = 70 #
        self.freq_high = 7e3

        self._graph = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '{}'.format(gpu_index)
        self._sess = tf.compat.v1.Session(graph=self._graph,config=config)

        self._load_cfg(config_fpath)
        self._build_model()

        # whether to load existed model
        self._is_load_model = is_load_model
        if is_load_model:
            self._load_model(self.model_dir)
        else:
            existed_meta_fnames = glob.glob(''.join([self.model_dir,'*.meta']))
            if len(existed_meta_fnames) > 0:
                if input('existed models are found\
                      {}\
                      if overwrite[y/n]'.format(existed_meta_fnames)) == 'y':
                    os.removedirs(self.model_dir)
                else:
                    return

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # open text file for logging
        self._log_file = open(os.path.join(self.model_dir,'log.txt'),'a')


    def _add_log(self,log_info):
        self._log_file.write(log_info)
        self._log_file.write('\n')
        self._log_file.flush()
        if self.is_print_log:
            print(log_info)


    def _load_cfg(self,config_fpath):
        if config_fpath is not None and os.path.exists(config_fpath):
            config = configparser.ConfigParser()
            config.read(config_fpath)

            # settings for model
            self.model_dir = config['model']['model_dir']
            self.frame_len = np.int16(config['model']['frame_len'])
            self.overlap_len = np.int16(config['model']['overlap_len'])
            self.filter_len = np.int16(config['model']['filter_len'])
            self.is_padd = config['model']['is_padd'] == 'True'
            self.n_azi = np.int16(config['model']['azi_num'])

            # settings for training
            self.batch_size= np.int16(config['train']['batch_size'])
            self.max_epoch = np.int16(config['train']['max_epoch'])
            # settings for AdamOptimizer
            self.lr_init,self.beta1,self.beta2,self.epsilon = np.fromstring(config['train']['adam_params'],sep=',')
            self.is_print_log = config['train']['is_print_log']=='True'
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            print('train_set{} \n valid_set{}'.format(self.train_set_dir,
                                                      self.valid_set_dir))

            # temp settings, for test
            # TODO: 是否使用gtf
            if not config.has_option('model','is_use_gtf'):
                self.is_use_gtf = False
            else:
                self.is_use_gtf = (config['model']['is_use_gtf'] == 'True')

            if not config.has_option('model','is_gtf_align'):
                self.is_gtf_align = False
            else:
                self.is_gtf_align = (config['model']['is_gtf_align'] == 'True')
        else:
            print(config_fpath)
            raise OSError


    def _load_model(self,model_dir):
        """load model"""
        if not os.path.exists(model_dir):
            raise Exception('no model exists in {}'.format(model_dir))

        with self._graph.as_default():
            # restore model
            saver = tf.compat.v1.train.Saver()
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(self._sess,ckpt.model_checkpoint_path)


    def get_gtf_kernel_old(self):
        """
        """
        cfs = gt_filters.erb_space(self.cf_low,self.cf_high,self.n_band)
        self.cfs = cfs

        sample_times = np.arange(0,self.filter_len,1)/self.fs
        irs = np.zeros((self.filter_len,self.n_band),dtype=np.float32)

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
            irs[:,band_i] = np.divide(numerator,denominator)

        gain = np.max(np.abs(np.fft.fft(irs,axis=0)),axis=0)
        irs_gain_norm = np.divide(np.flipud(irs),gain)
        if self.is_padd:
            kernel = np.concatenate((irs_gain_norm,
                                     np.zeros((self.filter_len,self.n_band))),
                                    axis=0)
        else:
            kernel = irs_gain_norm

        # fig = plt.figure()
        # axes = fig.subplots(1,2)
        # axes[0].plot(irs)
        # axes[1].plot(kernel)
        # fig.savefig('irs.png')
        return kernel


    def get_gtf_kernel(self,fs=16e3):
        """get gammatone filter impulse response as kernel weight of Gammatone convolution
        """
        gt_filter = gtf(fs,freq_low=self.freq_low,freq_high=self.freq_high,
                        n_band=self.n_band)
        self.cfs = gt_filter.cfs

        # whether to align env & fine structure
        if self.is_gtf_align:
            filter_ir = (gt_filter.get_ir(is_env_aligned=True,
                                          is_fine_aligned=True)).T
        else:
            filter_ir = (gt_filter.get_ir()).T

        # truncate zeros in the begining
        filter_ir = wav_tools.truncate_data(filter_ir,
                                            truncate_type='begin',
                                            eps=1e-3)

        if self.is_padd:
            # padd 0
            kernel = np.concatenate((np.flipud(filter_ir[:self.filter_len,]),
                                     np.zeros((self.filter_len,self.n_band))),
                                    axis=0)
        else:
            kernel = np.flipud(filter_ir[:self.filter_len,])
        return kernel


    def _fcn_layers(self,input,*layers_setting):
        for setting in layers_setting:
            fcn_size = setting['fcn_size']
            activation = setting['activation']
            rate = setting['rate']

            layer_fcn = tf.keras.layers.Dense(units=fcn_size,activation=activation)
            if rate > 0:
                layer_drop = tf.keras.layers.Dropout(rate=rate)
                output = layer_fcn(layer_drop(input))
            elif rate == 0:
                output = layer_fcn(input)
            else:
                raise Exception('illegal dropout rate')
            input = output
        return output


    def _build_model_subband(self,input):
        """
        """
        layer1_conv = tf.keras.layers.Conv2D(filters=6,
                                           kernel_size=[18,2],
                                           strides=[1,1],
                                           activation=tf.nn.relu)
        layer1_pool = tf.keras.layers.MaxPool2D([4,1],[4,1])
        layer1_out = layer1_pool(layer1_conv(input))

        layer2_conv = tf.keras.layers.Conv2D(filters=12,
                                           kernel_size=[6,1],
                                           strides=[1,1],
                                           activation=tf.nn.relu)
        layer2_pool = tf.keras.layers.MaxPool2D([4,1],[4,1])
        layer2_out = layer2_pool(layer2_conv(layer1_out))

        flatten_len = np.prod(layer2_out.get_shape().as_list()[1:])
        out = tf.reshape(layer2_out,[-1,flatten_len])# flatten
        return out


    def _build_model(self):
        """Build graph
        """
        # gammatone layer kernel initalizer
        # TODO: 待确定使用那一种kernel
        with self._graph.as_default():
            if self.is_use_gtf:
                kernel_initializer = tf.constant_initializer(self.get_gtf_kernel())
            else:
                kernel_initializer = tf.constant_initializer(self.get_gtf_kernel_old())

            if self.is_padd:
                gtf_kernel_len = 2*self.filter_len
            else:
                gtf_kernel_len = self.filter_len

            x = tf.compat.v1.placeholder(shape=[None,self.frame_len,2,1],
                                         dtype=tf.float32,
                                         name='x')#


            layer1_conv = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=[gtf_kernel_len,1],
                                        strides=[1,1],
                                        padding='same',
                                        kernel_initializer=kernel_initializer,
                                        trainable=False,use_bias=False)

            # add to model for test
            self.layer1_conv = layer1_conv

            # amplitude normalization across frequency channs
            # problem: silence ?
            layer1_conv_output = layer1_conv(x)
            amp_max = tf.reduce_max(\
                          tf.reduce_max(\
                            tf.reduce_max(tf.abs(layer1_conv_output),
                                          axis=1,keepdims=True),
                            axis=2,keepdims=True),
                        axis=3,keepdims=True)
            layer1_conv_output_norm = tf.divide(layer1_conv_output,amp_max)

            # layer1_pool
            layer1_pool = tf.keras.layers.MaxPool2D([2,1],[2,1])
            layer1_output = layer1_pool(layer1_conv_output_norm)

            band_out_list = []
            for band_i in range(self.n_band):
                band_output = self._build_model_subband(\
                                    tf.expand_dims(layer1_output[:,:,:,band_i],
                                                   axis=-1))
                band_out_list.append(band_output)
            band_out = tf.concat(band_out_list,axis=1)

            layer1 = {'fcn_size':1024,
                      'activation':tf.nn.relu,
                      'rate':0.5}
            layer2 = {'fcn_size':1024,
                      'activation':tf.nn.relu,
                      'rate':0.5}
            output_layer = {'fcn_size':self.n_azi,
                            'activation':tf.nn.softmax,
                            'rate':0}

            y_est = self._fcn_layers(band_out,
                                     layer1,
                                     layer2,
                                     output_layer)

            # groundtruth of two tasks
            y = tf.compat.v1.placeholder(shape=[None,self.n_azi],
                                         dtype=tf.float32)#
            # cost function
            cost = self._cal_cross_entropy(y_est,y)
            # cost = self._cal_cross_entropy(y_est,y)
            # additional measurement of localization
            azi_rmse = self._cal_azi_rmse(y_est,y)

            #
            # lr = tf.compat.v1.placeholder(tf.float32, shape=[])
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr_init,
                                                        beta1=self.beta1,
                                                        beta2=self.beta2,
                                                        epsilon=self.epsilon)
            opt_step = optimizer.minimize(cost)

            # train data pipeline
            n_batch_queue = 20
            coord = tf.train.Coordinator()
            train_tfdata = TFData.TFData([None,self.frame_len,2,1],
                                         [None,self.n_azi],
                                         sess=self._sess,
                                         batch_size=self.batch_size,
                                         n_batch_queue=n_batch_queue,
                                         coord=coord,
                                         file_reader=self._file_reader)

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
            self._opt_step = opt_step
            # data pipline
            self._coord = coord
            self._train_tfdata = train_tfdata


    def _cal_cross_entropy(self,y_est,y):
        cross_entropy = -tf.reduce_mean(tf.reduce_sum(\
                                    tf.multiply(y,tf.math.log(y_est+self.epsilon)),
                                    axis=1))
        return cross_entropy


    def _cal_mse(self,y_est,y):
        rmse = tf.reduce_mean(tf.reduce_sum((y-y_est)**2,axis=1))
        return rmse


    def _cal_azi_rmse(self,y_est,y):
        azi_est = tf.argmax(y_est,axis=1)
        azi = tf.argmax(y,axis=1)
        diff = tf.cast(azi_est-azi,tf.float32)
        return tf.sqrt(tf.reduce_mean(tf.pow(diff,2)))


    def _cal_cp(self,y_est,y):
        equality = tf.equal(tf.argmax(y_est,axis=1),tf.argmax(y,axis=1))
        cp = tf.reduce_mean(tf.cast(equality,tf.float32))
        return cp


    def _train_record_init(self):
        """
        """
        try:
            record_info=np.load(os.path.join(saved_model_dir,'train_record.npz'))
            cost_record_train=record_info['cost_record_train']
            azi_rmse_record_train=record_info['azi_rmse_record_train']
            cost_record_valid=record_info['cost_record_valid']
            azi_rmse_record_valid=record_info['azi_rmse_record_valid']
            best_epoch=record_info['best_epoch']
            min_valid_cost=record_info['min_valid_cost']
            last_epoch = np.nonzero(cost_record_train)[0][-1]
        except Exception as e:
            cost_record_train = np.zeros(self.max_epoch)
            azi_rmse_record_train = np.zeros(self.max_epoch)
            cost_record_valid = np.zeros(self.max_epoch)
            azi_rmse_record_valid = np.zeros(self.max_epoch)
            min_valid_cost=np.infty
            best_epoch = 0
            last_epoch = -1
        return [cost_record_train,azi_rmse_record_train,
                cost_record_valid,azi_rmse_record_valid,
                min_valid_cost,best_epoch,last_epoch]


    def _file_reader(self,record_set_dir):
        """ read wav files in given directies, one file per time
        Args:
            record_set_dir: directory or list of directories where recordings exist
        Returns:
            samples generator, [samples, onehot_labels]
        """
        if isinstance(record_set_dir,list):
            dir_list = record_set_dir
        else:
            dir_list = [record_set_dir]

        #
        wav_file_filter = lambda fname:(len(fname)>3 and \
                                        (fname[-3:]=='wav') and \
                                        (fname[0]!='.'))
        fpath_list = []
        for sub_set_dir in dir_list:
            for root,dirs,fnames in os.walk(sub_set_dir):
                for fname in fnames:
                    if wav_file_filter(fname):
                        fpath_list.append(os.path.join(root,fname))
        np.random.shuffle(fpath_list)

        if len(fpath_list) <1:
            raise Exception('empty folder:{}'.format(record_set_dir))

        for fpath in fpath_list:
            record,fs_record = wav_tools.wav_read(fpath)
            if fs_record != self.fs:
                raise Exception('fs is not {}'.format(self.fs))

            frames = wav_tools.frame_data(record,self.frame_len,
                                          self.overlap_len)
            # onehot azi label
            azi = np.int16(fpath.split('/')[-2])
            # print(fpath,azi)
            onehot_labels=np.zeros((frames.shape[0],self.n_azi))
            onehot_labels[:,azi]=1

            yield [np.expand_dims(frames,axis=-1),
                   onehot_labels]


    def train_model(self):
        """Train model either from initial state(self._build_model()) or
        already existed model
        """
        with self._graph.as_default():

            [cost_record_train,azi_rmse_record_train,
             cost_record_valid,azi_rmse_record_valid,
             min_valid_cost,best_epoch,last_epoch] = self._train_record_init()

            saver = tf.compat.v1.train.Saver()
            print('start training')
            for epoch in range(last_epoch+1,self.max_epoch):
                t_start = time.time()
                self._train_tfdata.start(file_dir=self.train_set_dir,
                                         n_thread=1,
                                         is_repeat=False)
                print('{} start'.format(epoch))
                while not self._train_tfdata.query_if_finish():
                    x,y = self._sess.run(self._train_tfdata.var_batch)
                    self._sess.run(self._opt_step,
                                   feed_dict={self._x:x,
                                              self._y:y})
                # model test
                # test on train data
                [cost_record_train[epoch],
                 azi_rmse_record_train[epoch]] = self.evaluate(self.train_set_dir)

                # test on valid data
                [cost_record_valid[epoch],
                 azi_rmse_record_valid[epoch]] = self.evaluate(self.valid_set_dir)

                # write to log
                iter_time = time.time()-t_start
                self._add_log('epoch:{} time:{:.2f}\n'.format(epoch,iter_time))

                log_template = '\t cost:{:.2f} azi_rmse:{:.2f}\n'
                self._add_log('\t train ')
                self._add_log(log_template.format(cost_record_train[epoch],
                                                  azi_rmse_record_train[epoch]))
                self._add_log('\t valid ')
                self._add_log(log_template.format(cost_record_valid[epoch],
                                                  azi_rmse_record_valid[epoch]))

                #
                if min_valid_cost>cost_record_valid[epoch]:
                    self._add_log('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost=cost_record_valid[epoch]
                    saver.save(self._sess,os.path.join(self.model_dir,'model'),
                               global_step=epoch)

                    ## save record info
                    np.savez(os.path.join(self.model_dir,'train_record'),
                             cost_record_valid=cost_record_valid,
                             azi_rmse_record_valid=azi_rmse_record_valid,
                             cost_record_train=cost_record_train,
                             azi_rmse_record_train=azi_rmse_record_train,
                             best_epoch=best_epoch,
                             min_valid_cost=min_valid_cost)

                 # early stop
                if epoch-best_epoch>5:
                    print('early stop\n',min_valid_cost)
                    self._add_log('early stop{}\n'.format(min_valid_cost))
                    break

            self._log_file.close()

            if True:
                fig,axs = plt.subplots(1,2)
                axs[0].plot(cost_record_train,label='train')
                axs[0].plot(cost_record_valid,label='valid')
                axs[0].set_title('cost')

                axs[1].plot(azi_rmse_record_train,label='train')
                axs[1].plot(azi_rmse_record_valid,label='valid')
                axs[1].set_title('rmse')
                axs[1].legend()
                #
                plt.tight_layout()
                fig_path = os.path.join(self.model_dir,'train_curve.png')
                plt.savefig(fig_path)


    def predict(self,x):
        """Model output of x
        """
        y_est = self._sess.run(self._y_est,feed_dict={self._x:x})
        return y_est


    def evaluate(self,set_dir):
        cost_all = 0.
        azi_rmse_all = 0.
        n_sample_all = 0

        self._train_tfdata.start(file_dir=set_dir,
                                 n_thread=1,
                                 is_repeat=False)
        while not self._train_tfdata.query_if_finish():
            x,y = self._sess.run(self._train_tfdata.var_batch)
            n_sample_tmp = x.shape[0]

            [cost_tmp,azi_rmse_tmp] = self._sess.run([self._cost,self._azi_rmse],
                                                    feed_dict={self._x:x,
                                                               self._y:y})
            #
            n_sample_all = n_sample_all+n_sample_tmp
            cost_all = cost_all+n_sample_tmp*cost_tmp
            azi_rmse_all = azi_rmse_all+n_sample_tmp*(azi_rmse_tmp**2)

        # average across all set
        cost_all = cost_all/n_sample_all
        azi_rmse_all = np.sqrt(azi_rmse_all/n_sample_all)
        return [cost_all,azi_rmse_all]



    def evaluate_chunk(self,record_set_dir,chunk_size=25):
        """ Evaluate model on given data_set, only for loc
        Args:
            data_set_dir:
        Returns:
            [rmse_chunk,cp_chunk,rmse_frame,cp_frame]
        """

        rmse_chunk = 0.
        n_chunk = 0

        for x,y in self._file_reader(record_set_dir=record_set_dir):
            sample_num = x.shape[0]
            azi_true = np.argmax(y[0])

            y_est = self.predict(x)
            for sample_i in range(0,sample_num-chunk_size+1):
                azi_est_chunk = np.argmax(np.mean(y_est[sample_i:sample_i+chunk_size],
                                                  axis=0))
                rmse_chunk = rmse_chunk + (azi_est_chunk-azi_true)**2
                n_chunk = n_chunk+1

        rmse_chunk = np.sqrt(rmse_chunk/n_chunk)*5
        return rmse_chunk
