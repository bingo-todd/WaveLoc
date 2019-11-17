import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import matplotlib.pyplot as plt
import configparser
import soundfile as sf
import sys
import time
import gammatone.filters as gt_filters

#
sys.path.append('/home/st/Work_Space/module_st/basic-toolbox')
sys.path.append('/home/st/Work_Space/module_st/Gammatone-filters')
from wav_tools import wav_tools
from TFData import TFData
from gtf import gtf


class WavLoc(object):
    """
"""
    def __init__(self,config_fpath=None):

        if config_fpath is not None and os.path.exists(config_fpath):

            config = configparser.ConfigParser()
            config.read(config_fpath)

            self.model_dir = config['model']['model_dir']
            self.frame_len = np.int16(config['model']['frame_len'])
            self.overlap_len = np.int16(config['model']['overlap_len'])
            self.filter_len = np.int16(config['model']['filter_len'])
            self.n_azi = np.int16(config['model']['azi_num'])

            self.batch_size= np.int16(config['train']['batch_size'])
            self.max_epoch = np.int16(config['train']['max_epoch'])
            self.train_set_dir = config['train']['train_set_dir'].split(';')
            self.valid_set_dir = config['train']['valid_set_dir'].split(';')
            print('train_set{} \n valid_set{}'.format(self.train_set_dir,
                                                      self.valid_set_dir))

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

        self.log_file = open(os.path.join(self.model_dir,'log.txt'),'a')

        # constant settings
        self.fs=16e3
        self.n_band = 32  # frequency bands number
        # previous settings
        self.cf_low=70
        self.cf_high=7e3
        # using gtf module
        self.freq_low = 70 #
        self.freq_high = 7e3

        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph,config=config)


    def file_reader(self,record_set_dir,fs=16e3):
        """ waveform signal generator
        Args:
            record_set_dir: directory or list of directories where recordings exist
        Returns:
            sample generator, [record_samples, onehot_label]
        """
        if isinstance(record_set_dir,list):
            dir_list = record_set_dir
        else:
            dir_list = [record_set_dir]

        wav_file_filter = lambda fname:(len(fname)>3 and \
                                        (fname[-3:]=='wav') and \
                                        (fname[0]!='.'))
        record_fpath_list = []
        for sub_set_dir in dir_list:
            for root,dirs,fnames in os.walk(sub_set_dir):
                for fname in fnames:
                    if wav_file_filter(fname):
                        record_fpath_list.append(os.path.join(root,fname))
        # np.random.shuffle(record_fpath_list)

        for fpath in record_fpath_list:
            wav,fs_record = wav_tools.wav_read(fpath)
            if fs_record != self.fs:
                raise Exception('fs is not {}'.format(self.fs))

            frames = wav_tools.frame_data(wav,self.frame_len,self.overlap_len)
            fea = np.expand_dims(frames,axis=-1) #to [n_frame,frame_szie,n_chann,1]

             # onehot azi label
            azi = np.int16(fpath.split('/')[-2])
            onehot_label=np.zeros((fea.shape[0],self.n_azi))
            onehot_label[:,azi]=1

            yield [fea,onehot_label]


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
        kernel = np.concatenate((irs_gain_norm,
                                 np.zeros((self.filter_len,self.n_band))),
                                axis=0)

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
        filter_ir = (gt_filter.get_ir()).T
        # truncate zeros in the begining
        filter_ir = wav_tools.truncate_data(filter_ir,
                                            truncate_type='begin',
                                            eps=1e-3)
        kernel = filter_ir[:self.filter_len,]
        # padd 0
        kernel = np.concatenate((np.flipud(kernel),
                                 np.zeros((self.filter_len,self.n_band))),
                                axis=0)
        return kernel[0,np.newaxis,np.newaxis,1]


    def build_model_band(self,input):
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


    def build_model(self):
        kernel_initializer = tf.constant_initializer(self.get_gtf_kernel_old())

        with self._graph.as_default():
            x = tf.compat.v1.placeholder(shape=[None,self.frame_len,2,1],
                                         dtype=tf.float32,
                                         name='x')#

            layer1_conv = tf.keras.layers.Conv2D(filters=32,
                                        kernel_size=[self.filter_len*2,1],
                                        strides=[1,1],
                                        padding='same',
                                        kernel_initializer=kernel_initializer,
                                        trainable=False,use_bias=False)

            self.layer1_conv = layer1_conv # for test

            # amplitude normalization across frequency channs
            layer1_conv_output = layer1_conv(x)
            amp_max = tf.reduce_max(\
                          tf.reduce_max(\
                            tf.reduce_max(tf.abs(layer1_conv_output),
                                          axis=1,keepdims=True),
                            axis=2,keepdims=True),
                        axis=3,keepdims=True)
            layer1_conv_output_norm = tf.divide(layer1_conv_output,amp_max)

            self.layer1_conv_output_norm = layer1_conv_output_norm # for test

            # layer1_pool
            layer1_pool = tf.keras.layers.MaxPool2D([2,1],[2,1])
            layer1_output = layer1_pool(layer1_conv_output_norm)
            # self._x_conv1_norm = [x_conv1,x_conv1_norm,x_conv1_max]

            hidd_out_bands = []
            for band_i in range(self.n_band):
                band_output = self.build_model_band(\
                                    tf.expand_dims(layer1_output[:,:,:,band_i],
                                                   axis=-1))
                hidd_out_bands.append(band_output)
            concat_hidd_out = tf.concat(hidd_out_bands,axis=1)

            layer2_fcn = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu)
            layer2_dropout = tf.keras.layers.Dropout(rate=0.5)
            layer2_out = layer2_dropout(layer2_fcn(concat_hidd_out))

            layer3_fcn = tf.keras.layers.Dense(units=1024,activation=tf.nn.relu)
            layer3_dropout =  tf.keras.layers.Dropout(rate=0.5)
            fcn_layer2_out = layer3_dropout(layer3_fcn(layer2_out))

            output_layer = tf.keras.layers.Dense(units=37,activation=tf.nn.softmax)
            y_est = tf.identity(output_layer(fcn_layer2_out),name='y_est')

            self._x = x
            self._y_est = y_est


    def cal_cost(self,y):
        """cross entrophy"""
        cost = -tf.reduce_mean(\
                    tf.reduce_sum(\
                        tf.multiply(y,tf.math.log(self._y_est+1e-20)),
                        axis=1))
#         mse = tf.reduce_mean((self._y_est-y)**2)
        return cost


    def cal_accuracy(self,y):
        equality = tf.equal(tf.argmax(self._y_est,axis=1),tf.argmax(y,axis=1))
        accuracy = tf.reduce_mean(tf.cast(equality,tf.float32))
        return accuracy


    def cal_mae(self,y):
        """mean abs error"""
        azi_est = tf.argmax(self._y_est,axis=1)
        azi = tf.argmax(y,axis=1)
        mae = tf.reduce_mean(tf.abs(azi_est-azi))
        return mae


    def train_model(self,saved_model_dir=None):
        ##
        n_batch_queue = 10
        with self._graph.as_default():
            coord = tf.train.Coordinator()
            train_tfdata = TFData(self.train_set_dir,
                                  [None,self.frame_len,2,1],
                                  [None,self.n_azi],
                                  self.batch_size,n_batch_queue,
                                  coord,self.file_reader,False)
            train_x_batch,train_y_batch = train_tfdata.dequeue()


            y = tf.compat.v1.placeholder(shape=[None,self.n_azi],
                                    dtype=tf.float32)#
            cost = self.cal_cost(y)
            accuracy = self.cal_accuracy(y)

            learning_rate = tf.compat.v1.placeholder(tf.float32, shape=[])
            opt_step = tf.compat.v1.train.AdamOptimizer(\
                                    learning_rate=learning_rate).minimize(cost)

            init = tf.compat.v1.global_variables_initializer()
            self._sess.run(init)

            saver = tf.compat.v1.train.Saver()
            #

            if saved_model_dir is not None:
                ckpt = tf.train.get_checkpoint_state(saved_model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess,ckpt.model_checkpoint_path)

                    record_infor=np.load(os.path.join(saved_model_dir,
                                                      'train_record.npz'))
                    cost_record_valid=record_infor['cost_record_valid']
                    accuracy_record_valid=record_infor['accuracy_record_valid']
                    cost_record_train=record_infor['cost_record_train']
                    accuracy_record_train=record_infor['accuracy_record_train']
                    lr = record_infor['lr']
                    best_epoch=record_infor['best_epoch']
                    min_valid_cost=record_infor['min_valid_cost']
                    last_epoch = np.nonzero(cost_record_train)[0][-1]
                    print('continue trraining from eoch {}'.format(last_epoch))
                else:
                    raise Exception('not finished')
            else:
                ## training record
                cost_record_train = np.zeros(self.max_epoch)
                accuracy_record_train = np.zeros(self.max_epoch)
                cost_record_valid = np.zeros(self.max_epoch)
                accuracy_record_valid = np.zeros(self.max_epoch)
                #
                lr = 1e-3
                min_valid_cost=np.infty
                best_epoch = 0
                last_epoch = -1
                print('start training')


            for epoch in range(last_epoch+1,self.max_epoch):
                # update model
                threads = train_tfdata.start_thread(self._sess)
                while self._sess.run(train_tfdata.x_queue.size())<\
                                                            self.batch_size:
                    time.sleep(0.5)

                print('epoch {}'.format(epoch))
                while (not train_tfdata.is_epoch_finish or
                         self._sess.run(train_tfdata.x_queue.size())>self.batch_size):
                    batch_value_train = self._sess.run([train_x_batch,train_y_batch])
                    self._sess.run(opt_step,feed_dict={self._x:batch_value_train[0],
                                                       y:batch_value_train[1],
                                                       learning_rate:lr})
                train_tfdata.empty_queue(self._sess)

                # model test
                # test on train data
                cost_train = 0.
                accuracy_train = 0.
                n_train = 0
                batch_generator_train = self.file_reader(self.train_set_dir)
                for batch_value_train in batch_generator_train:
                    [cost_batch_train,
                     accuracy_batch_train] = self._sess.run([cost,accuracy],
                                    feed_dict={self._x:batch_value_train[0],
                                               y:batch_value_train[1],
                                               learning_rate:lr})
                    n_train_batch = batch_value_train[0].shape[0]
                    n_train = n_train+n_train_batch

                    cost_train = cost_train+n_train_batch*cost_batch_train
                    accuracy_train = accuracy_train+\
                                            n_train_batch*accuracy_batch_train
                # print('cost_train:{}  n_train:{}'.format(cost_train,n_train))

                cost_record_train[epoch] = cost_train/n_train
                accuracy_record_train[epoch] = accuracy_train/n_train

                # test on valid data
                cost_valid = 0.
                accuracy_valid = 0.
                n_valid = 0
                batch_generator_valid = self.file_reader(self.valid_set_dir)
                for batch_value_valid in batch_generator_valid:
                    [cost_batch_valid,
                    accuracy_batch_valid] = self._sess.run([cost,accuracy],
                                       feed_dict={self._x:batch_value_valid[0],
                                                  y:batch_value_valid[1]})
                    n_valid_batch = batch_value_valid[0].shape[0]
                    n_valid = n_valid+n_valid_batch

                    cost_valid = cost_valid + n_valid_batch*cost_batch_valid
                    accuracy_valid = accuracy_valid+\
                                        n_valid_batch*accuracy_batch_valid

                cost_record_valid[epoch] = cost_valid/n_valid
                accuracy_record_valid[epoch] = accuracy_valid/n_valid

                ###
                self.log_file.write('epoch:{}  lr:{} \n \t train cost:{}\
                                accuracy:{}\n'.format(epoch,lr,
                                                  cost_record_train[epoch],
                                                  accuracy_record_train[epoch]))
                self.log_file.write('\t valid cost_valid:{}  \
                                accuracy:{} \n'.format(cost_record_valid[epoch],
                                                 accuracy_record_valid[epoch]))

                print('epoch:{}  lr:{} \n \t train cost:{}\
                       accuracy:{}\n'.format(epoch,lr,
                                            cost_record_train[epoch],
                                            accuracy_record_train[epoch]))
                print('\t valid cost_valid:{}\
                      accuracy:{} \n'.format(cost_record_valid[epoch],
                                            accuracy_record_valid[epoch]))


                ### save record info
                np.savez(os.path.join(self.model_dir,'train_record'),\
                         cost_record_valid=cost_record_valid,
                         accuracy_record_valid=accuracy_record_valid,
                         cost_record_train=cost_record_train,
                         accuracy_record_train=accuracy_record_train,
                         lr=lr,
                         best_epoch=best_epoch,
                         min_valid_cost=min_valid_cost)
                ###

                #
                if min_valid_cost>cost_record_valid[epoch]:
                    print('find new optiml\n')
                    self.log_file.write('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost=cost_record_valid[epoch]
                    saver.save(self._sess,os.path.join(self.model_dir,'model'),
                               global_step=epoch)

                if epoch > 2:
                    if cost_record_train[epoch] > \
                                np.min(cost_record_train[epoch-2:epoch]):
                        lr = lr*.2

                # early stop
                if epoch-best_epoch>5:
                    print('early stop\n',min_valid_cost)
                    self.log_file.write('early stop{}\n'.format(min_valid_cost))
                    break
                self.log_file.flush()

            plt.figure()
            plt.subplot(121); plt.plot(cost_record_valid);
            plt.title('cost valid')
            plt.subplot(122); plt.plot(accuracy_record_valid);
            plt.title('accuracy valid')
            plt.tight_layout()
            fig_path = os.path.join(self.model_dir,'train_curve_valid.png')
            plt.savefig(fig_path)

            plt.figure()
            plt.subplot(121); plt.plot(cost_record_train);
            plt.title('cost train')
            plt.subplot(122); plt.plot(accuracy_record_train);
            plt.title('accuracy train')
            plt.tight_layout()
            fig_path = os.path.join(self.model_dir,'train_curve_train.png')
            plt.savefig(fig_path)

            coord.request_stop()

            self.log_file.close()


    def load_model(self,meta_filepath):
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self._graph.as_default():
            self._sess = tf.Session(config=config)
            saver = tf.train.import_meta_graph(meta_filepath);
            saver.restore(self._sess,
                    tf.train.latest_checkpoint(os.path.dirname(meta_filepath)))
            self._x = self._graph.get_tensor_by_name('x:0')
            self._y_est = self._graph.get_tensor_by_name('y_est:0')


    def model_test(self,set_dir):
        y = self._graph.get_tensor_by_name('y:0')
        accuracy = self._graph.get_tensor_by_name('accuracy:0')
        batch_generator=self.file_reader(set_dir)

        accuracy_all = 0
        N = 0
        for batch_value in batch_generator:
            accuracy_batch_value=self._sess.run(accuracy,
                                            feed_dict={self._x:batch_value[0],
                                                        y:batch_value[1]})
            N_batch = batch_value[0].shape[0]
            accuracy_all = accuracy_all+accuracy_batch_value*N_batch
            N = N+N_batch
        accuracy_all = accuracy_all/N
        return accuracy_all


    def predict(self,x):
        y_est = self._sess.run(self._y_est,feed_dict={self._x:x})
        return y_est

if __name__ == '__main__':

    config = configparser.ConfigParser()

    config['model']={'model_dir':'test',
                    'frame_len':'320',
                    'filter_len':'320',
                    'overlap_len':'160',
                    'azi_num':'37'}

    config['train']={'batch_size':'128',
                     'max_epoch':'10',
                     'train_set_dir':'../Data/Record_v4/train/Anechoic/;\
                     ../Data/Record_v4/train/B/;\
                     ../Data/Record_v4/train/C/;\
                     ../Data/Record_v4/train/D/',
                     'valid_set_dir':'../Data/Record_v4/valid/Anechoic/;\
                     ../Data/Record_v4/valid/B/;\
                     ../Data/Record_v4/valid/C/;\
                     ../Data/Record_v4/valid/D/'}

    with open('test/test.cfg','w') as config_file:
        if config_file is None:
            raise Exception('fail to create file')
        config.write(config_file)

    model = WavLoc('test/test.cfg')

    filter_ir = model.get_gtf_kernel_old()
    plt.plot(filter_ir)
    plt.savefig('images/gammatone_kernel_old.png')

    model.build_model()
    with model._graph.as_default():
        init = tf.compat.v1.global_variables_initializer()
        model._sess.run(init)

        filter_kernel = model._sess.run(model.layer1_conv.weights)[0]

        batch_generator = model.file_reader(model.train_set_dir)
        for batch_value in batch_generator:
            layer1_norm_output = model._sess.run(model.layer1_conv_output_norm,
                                                 feed_dict={model._x:batch_value[0]})
            break


    fig = plt.figure()
    axes = fig.subplots(1,1)
    axes.plot(np.squeeze(filter_kernel))
    axes.set_title('Gammatone_layer_kernel')
    fig.savefig('images/Gammatone_layer_kernel.png')

    fig = plt.figure()
    axes = fig.subplots(2,2)
    frame_i = 10
    for i in range(4):
        band_i = i*8
        sub_axes = axes[np.int8(i/2),np.mod(i,2)]
        sub_axes.plot(layer1_norm_output[frame_i,:,:,band_i])
        sub_axes.set_title('band {}'.format(band_i))
    plt.tight_layout()
    fig.savefig('images/layer1_norm_output')
    print(layer1_norm_output.shape)
