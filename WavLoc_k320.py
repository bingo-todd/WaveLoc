"""
使用默认的卷积操作
kernel length: 320
"""

import scipy.io as sio
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pickle as cPickle
import gammatone.filters as gt_filters
import time
import matplotlib.pyplot as plt
import configparser
import logging
from file_reader import file_reader
import sys
sys.path.append('/home/st/Work_Space/module_st/')
import TFData

class WavLoc(object):
    def __init__(self,config_path=None):
        
        if config_path is not None:
            config = configparser.ConfigParser()
            config.read(config_path)
            self.frame_len = np.int16(config['basic']['frame_len'])
            self.azi_num = np.int16(config['basic']['azi_num'])
            self.batch_size= np.int16(config['basic']['batch_size'])
            self.max_epoch = np.int16(config['basic']['max_epoch'])
            self.train_set_dir = config['basic']['train_set_dir'].split(';')
            self.valid_set_dir = config['basic']['valid_set_dir'].split(';')
            print(self.train_set_dir)
            
            self.model_dir = config['basic']['model_dir']
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        

        self.freq_chann_num = 32
        self.cf_low = 70
        self.cf_high = 7e3
        
        self._graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph,config=config)
    
    def GT_filter_kernel(self,kernel_len=320,fs=16e3):
        cfs = gt_filters.erb_space(self.cf_low,self.cf_high,self.freq_chann_num)
        sample_times = np.arange(0,kernel_len,1)/fs
        kernel = np.zeros((kernel_len,self.freq_chann_num),dtype=np.float32)
        
        EarQ = 9.26449
        minBW = 24.7
        order = 1
        N = 4
        for freq_chann_i in range(self.freq_chann_num):
            ERB = ((cfs[freq_chann_i]/EarQ)**order+minBW**order)**(1/order)
            b = 1.019*ERB
            kernel[:,freq_chann_i] =  np.multiply(np.multiply(sample_times**(N-1),
                                                           np.cos(2*np.pi*cfs[freq_chann_i]*sample_times)),
                                               np.exp(-2*np.pi*b*sample_times))
        gain = np.max(np.abs(np.fft.fft(kernel,axis=0)),axis=0)
        kernel = np.divide(np.flipud(kernel),gain)
#         kernel = np.concatenate((kernel,np.zeros((kernel_len,self.freq_chann_num))),axis=0)
        return kernel
                                                          
    def build_model_band(self,x_conv1,freq_chann_i):
           
        x_pool1 = tf.layers.max_pooling2d(x_conv1,[2,1],[2,1],name='%d_pool1'%freq_chann_i)
        #
        x_conv2 = tf.layers.conv2d(x_pool1,
                                   filters=6,
                                   kernel_size=[18,2],
                                   strides=[1,1],
                                   activation=tf.nn.relu,
                                   name='%d_conv_2'%freq_chann_i)
        x_pool2 = tf.layers.max_pooling2d(x_conv2,[4,1],[4,1],
                                          name='%d_pool2'%freq_chann_i)
        #
        x_conv3 = tf.layers.conv2d(x_pool2,
                                   filters=12,
                                   kernel_size=[6,1],
                                   strides=[1,1],
                                   activation=tf.nn.relu,
                                   name='%d_conv_3'%freq_chann_i)
        x_pool3 = tf.layers.max_pooling2d(x_conv3,[4,1],[4,1],
                                         name='%d_pool3'%freq_chann_i)
        out = tf.reshape(x_pool3,[-1,12*7])
        return out
        
    def build_model(self):
        kernels = self.GT_filter_kernel()
        # kernels = np.transpose(gt[:,:,np.newaxis,np.newaxis],[1,2,3,0])
        with self._graph.as_default():
            x = tf.placeholder(shape=[None,self.frame_len,2,1],
                                dtype=tf.float32,name='x')#

            x_conv1 = tf.layers.conv2d(x,filters=32,
                                       kernel_size=[320,1],strides=[1,1],
                                       padding='same',
                                       name='conv1',
                                       kernel_initializer=tf.constant_initializer(kernels),
                                       trainable=False,use_bias=False)

            x_conv1_max = tf.reduce_max(tf.reduce_max(tf.reduce_max(tf.abs(x_conv1),
                                                                    axis=1,keep_dims=True),
                                                      axis=2,keep_dims=True),
                                        axis=3,keep_dims=True)
            x_conv1_norm = tf.divide(x_conv1,x_conv1_max)
            # self._x_conv1_norm = [x_conv1,x_conv1_norm,x_conv1_max]

            hidd_out_bands = []
            for freq_chann_i in range(self.freq_chann_num): 
                hidd_out_bands.append(self.build_model_band(tf.expand_dims(x_conv1_norm[:,:,:,freq_chann_i],
                                                                           axis=-1),
                                                            freq_chann_i))
            concat_hidd_out = tf.concat(hidd_out_bands,axis=1)
            fcn1 = tf.layers.dense(concat_hidd_out,units=1024,activation=tf.nn.relu)
            drop1 = tf.layers.dropout(fcn1)
            fcn2 = tf.layers.dense(drop1,units=1024,activation=tf.nn.relu)
            drop2 = tf.layers.dropout(fcn2) 
            y_est_tem =tf.layers.dense(drop2,units=37,activation=tf.nn.softmax)
            y_est = tf.identity(y_est_tem, name='y_est')
            
            self._x = x
            self._y_est = y_est
           
    def cal_cost(self,y):
        cost = -tf.reduce_mean(tf.reduce_sum(tf.multiply(y,tf.log(tf.add(self._y_est,1e-20))),axis=1),name='cost')
#         mse = tf.reduce_mean((self._y_est-y)**2)
        return cost
    
    def cal_accuracy(self,y):
        equality = tf.equal(tf.argmax(self._y_est,axis=1),tf.argmax(y,axis=1))
        accuracy = tf.reduce_mean(tf.cast(equality,tf.float32),name='accuracy')
        return accuracy
    
    def cal_mae(self,y):
        """mean abs error"""
        azi_est = tf.argmax(self._y_est,axis=1)
        azi = tf.argmax(y,axis=1)
        mae = tf.reduce_mean(tf.abs(azi_est-azi),name='mae')
        return mae
    
    def train_model(self,saved_model_dir=None):
        ##
        with self._graph.as_default():
            coord = tf.train.Coordinator()
            train_tfdata = TFData.TFData(self.train_set_dir,[None,320,2,1],[None,37],
                                     self.batch_size,20,coord,file_reader,False)
            train_x_batch,train_y_batch = train_tfdata.dequeue()
    
        
            y = tf.placeholder(shape=[None,self.azi_num],
                                    dtype=tf.float32,name='y')#
            cost = self.cal_cost(y)
            accuracy = self.cal_accuracy(y)
        
            learning_rate = tf.placeholder(tf.float32, shape=[])
            opt_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)        
        
            init = tf.global_variables_initializer() 
            self._sess.run(init)
            
            saver = tf.train.Saver()  
            #

            if saved_model_dir is not None:
                ckpt = tf.train.get_checkpoint_state(saved_model_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(self._sess,ckpt.model_checkpoint_path)
                    record_infor=np.load(os.path.join(saved_model_dir,'train_record.npz'))
                    cost_record_valid=record_infor['cost_record_valid']
                    accuracy_record_valid=record_infor['accuracy_record_valid']
                    cost_record_train=record_infor['cost_record_train']
                    accuracy_record_train=record_infor['accuracy_record_train'] 
                    lr = record_infor['lr']
                    best_epoch=record_infor['best_epoch']
                    min_valid_cost=record_infor['min_valid_cost']
                    last_epoch = np.nonzero(cost_record_train)[0][-1]
                    log_file = open(os.path.join(self.model_dir,'log.txt'),'a')
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
                log_file = open(os.path.join(self.model_dir,'log.txt'),'w')
                print('start training')
            
            
            for epoch in range(last_epoch+1,self.max_epoch):
                # update model
                threads = train_tfdata.start_thread(self._sess)
                while self._sess.run(train_tfdata.x_queue.size())<self.batch_size:
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
                N_train = 0
                batch_generator_train = file_reader(self.train_set_dir)
                for batch_value_train in batch_generator_train:
                    [cost_batch_train,
                     accuracy_batch_train] = self._sess.run([cost,accuracy],
                                                            feed_dict={self._x:batch_value_train[0],
                                                                       y:batch_value_train[1],
                                                                       learning_rate:lr})
                    N_train_batch = batch_value_train[0].shape[0]
                    N_train = N_train+N_train_batch

                    cost_train = cost_train+N_train_batch*cost_batch_train
                    accuracy_train = accuracy_train+N_train_batch*accuracy_batch_train

                cost_record_train[epoch] = cost_train/N_train
                accuracy_record_train[epoch] = accuracy_train/N_train
                
                # test on valid data    
                cost_valid = 0.
                accuracy_valid = 0.
                N_valid = 0
                batch_generator_valid = file_reader(self.valid_set_dir)
                for batch_value_valid in batch_generator_valid:
                    [cost_batch_valid,
                    accuracy_batch_valid] = self._sess.run([cost,accuracy],
                                                           feed_dict={self._x:batch_value_valid[0],
                                                                      y:batch_value_valid[1]})
                    N_valid_batch = batch_value_valid[0].shape[0]
                    N_valid = N_valid+N_valid_batch
                    
                    cost_valid = cost_valid + N_valid_batch*cost_batch_valid
                    accuracy_valid = accuracy_valid+N_valid_batch*accuracy_batch_valid

                cost_record_valid[epoch] = cost_valid/N_valid
                accuracy_record_valid[epoch] = accuracy_valid/N_valid
                
                ###
                log_file.write('epoch:{}  lr:{} \n \t train cost:{}  accuracy:{}\n'.format(epoch,lr,
                                                                                cost_record_train[epoch],
                                                                                accuracy_record_train[epoch]))
                log_file.write('\t valid cost_valid:{}  accuracy:{} \n'.format(cost_record_valid[epoch],
                                                                                accuracy_record_valid[epoch]))
                
                print('epoch:{}  lr:{} \n \t train cost:{}  accuracy:{}\n'.format(epoch,lr,
                                                                                cost_record_train[epoch],
                                                                                accuracy_record_train[epoch]))
                print('\t valid cost_valid:{}  accuracy:{} \n'.format(cost_record_valid[epoch],
                                                                                accuracy_record_valid[epoch]))


                ### save record info
                np.savez(os.path.join(self.model_dir,'train_record'),cost_record_valid=cost_record_valid,
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
                    log_file.write('find new optimal\n')
                    best_epoch = epoch
                    min_valid_cost=cost_record_valid[epoch]
                    saver.save(self._sess,os.path.join(self.model_dir,'model'),global_step=epoch)
                
                if epoch > 2:
                    if cost_record_train[epoch] > np.min(cost_record_train[epoch-2:epoch]):
                        lr = lr*.2

                # early stop
                if epoch-best_epoch>5:
                    print('early stop\n',min_valid_cost)
                    log_file.write('early stop{}\n'.format(min_valid_cost))
                    break
                log_file.flush()

            plt.figure()
            plt.subplot(121); plt.plot(cost_record_valid); plt.title('cost valid')
            plt.subplot(122); plt.plot(accuracy_record_valid); plt.title('accuracy valid')
            plt.tight_layout()
            fig_path = os.path.join(self.model_dir,'train_curve_valid.png')
            plt.savefig(fig_path)

            plt.figure()
            plt.subplot(121); plt.plot(cost_record_train); plt.title('cost train')
            plt.subplot(122); plt.plot(accuracy_record_train); plt.title('accuracy train')
            plt.tight_layout()
            fig_path = os.path.join(self.model_dir,'train_curve_train.png')
            plt.savefig(fig_path)
            
            coord.request_stop()
            
            log_file.close()

            
    def load_model(self,meta_filepath):
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with self._graph.as_default():
            self._sess = tf.Session(config=config)
            saver = tf.train.import_meta_graph(meta_filepath);
            saver.restore(self._sess,tf.train.latest_checkpoint(os.path.dirname(meta_filepath)))
            self._x = self._graph.get_tensor_by_name('x:0')
            self._y_est = self._graph.get_tensor_by_name('y_est:0')
    
    def model_test(self,set_dir):
        y = self._graph.get_tensor_by_name('y:0')
        accuracy = self._graph.get_tensor_by_name('accuracy:0')
        batch_generator=file_reader(set_dir)

        accuracy_all = 0
        N = 0
        for batch_value in batch_generator:
            accuracy_batch_value=self._sess.run(accuracy,feed_dict={self._x:batch_value[0],
                                                                    y:batch_value[1]})
            N_batch = batch_value[0].shape[0]
            accuracy_all = accuracy_all+accuracy_batch_value*N_batch
            N = N+N_batch
        accuracy_all = accuracy_all/N
        return accuracy_all

    def predict(self,x):
        y_est = self._sess.run(self._y_est,feed_dict={self._x:x})
        return y_est
