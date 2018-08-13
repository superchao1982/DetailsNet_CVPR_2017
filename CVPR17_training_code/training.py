#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of our paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import re
import random
import numpy as np
import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt
from GuidedFilter import guided_filter



##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################

tf.reset_default_graph()

##################### Network parameters ###################################
num_feature = 16             # number of feature maps
num_channels = 3             # number of input's channels 
patch_size = 64              # patch size 
KernelSize = 3               # kernel size 
learning_rate = 0.1          # learning rate
iterations = int(6e4)        # iterations
batch_size = 20              # batch size
save_model_path = "./model/" # saved model's path
model_name = 'model-epoch'   # saved model's name
############################################################################


input_path = "./TrainData/input/"    # the path of rainy images
gt_path = "./TrainData/label/"       # the path of ground truth


# randomly select image patches
def read_data(input_path, gt_path, size_input, num_channel, batch_size):
    input_files= os.listdir(input_path)
    gt_files= os.listdir(gt_path)  
    
    Data  = np.zeros((batch_size, size_input, size_input, num_channel)) 
    Label = np.zeros((batch_size, size_input, size_input, num_channel)) 
  
    for i in range(batch_size):
  
        r_idx = random.randint(0,len(input_files)-1)
    
        rainy = img.imread(input_path + input_files[r_idx])
        if np.max(rainy) > 1:
           rainy = rainy/255.0

        label = img.imread(gt_path + gt_files[r_idx])
        if np.max(label) > 1:
           label = label/255.0

        x = random.randint(0,rainy.shape[0] - size_input)
        y = random.randint(0,rainy.shape[1] - size_input)

        subim_input = rainy[x : x+size_input, y : y+size_input, :]
        subim_label = label[x : x+size_input, y : y+size_input, :]

        Data[i,:,:,:] = subim_input
        Label[i,:,:,:] = subim_label
		
    return Data, Label




# network structure
def inference(images, is_training):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)
    initializer = tf.contrib.layers.xavier_initializer()
	
	
    base = guided_filter(images, images, 15, 1, nhwc=True) # using guided filter for obtaining base layer
    detail = images - base   # detail layer
	
   #  layer 1
    with tf.variable_scope('layer_1'):	
         output = tf.layers.conv2d(detail, num_feature, KernelSize, padding = 'same', kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_1')
         output = tf.layers.batch_normalization(output, training=is_training, name='bn_1')
         output_shortcut = tf.nn.relu(output, name='relu_1')
  
   #  layers 2 to 25
    for i in range(12):
        with tf.variable_scope('layer_%d'%(i*2+2)):	
             output = tf.layers.conv2d(output_shortcut, num_feature, KernelSize, padding='same', kernel_initializer = initializer, 
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+2)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+2)))	
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+2)))
		
		
        with tf.variable_scope('layer_%d'%(i*2+3)): 
             output = tf.layers.conv2d(output, num_feature, KernelSize, padding='same', kernel_initializer = initializer,
                                       kernel_regularizer = regularizer, name=('conv_%d'%(i*2+3)))
             output = tf.layers.batch_normalization(output, training=is_training, name=('bn_%d'%(i*2+3)))
             output = tf.nn.relu(output, name=('relu_%d'%(i*2+3)))
			 
        output_shortcut = tf.add(output_shortcut, output)	# shortcut
		
   # layer 26
    with tf.variable_scope('layer_26'):
         output = tf.layers.conv2d(output_shortcut, num_channels, KernelSize, padding='same',   kernel_initializer = initializer, 
                                   kernel_regularizer = regularizer, name='conv_26')
         neg_residual = tf.layers.batch_normalization(output, training=is_training, name='bn_26')
		 
    final_out = tf.add(images, neg_residual)
	
    return final_out

  


if __name__ == '__main__':
   images = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, num_channels), name='rainy_images')  # data
   labels = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, num_channels), name='ground_truth')  # label
   is_training = tf.placeholder(tf.bool, name='is_training')
   
   outputs = inference(images, is_training)
   loss = tf.reduce_mean(tf.square(labels - outputs))    # MSE loss

   
   lr_ = learning_rate
   lr = tf.placeholder(tf.float32 ,shape = [])  

   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
        train_op =  tf.train.MomentumOptimizer(lr, 0.9).minimize(loss) 

   
   var_list = tf.trainable_variables()   
   g_list = tf.global_variables()
   bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
   bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
   var_list += bn_moving_vars
   saver = tf.train.Saver(var_list=var_list, max_to_keep=5)
   
   
   config = tf.ConfigProto()
   config.gpu_options.per_process_gpu_memory_fraction = 0.8 # GPU setting
   config.gpu_options.allow_growth = True
  
    
   validation_data, validation_label =  read_data(input_path, gt_path, patch_size, num_channels, batch_size) #  data for validation
   print("check patch pair:")  
   plt.subplot(1,2,1)     
   plt.imshow(validation_data[0,:,:,:])
   plt.title('input')         
   plt.subplot(1,2,2)    
   plt.imshow(validation_label[0,:,:,:])
   plt.title('ground truth')        
   plt.show()   
   
   
   
   with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
		
        if tf.train.get_checkpoint_state('./model/'):   # load previous trained models
           ckpt = tf.train.latest_checkpoint('./model/')
           saver.restore(sess, ckpt)
           ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
           start_point = int(ckpt_num[0])    
           print("successfully load previous model")
   
        else:   # re-training if no previous trained models
           start_point = 0    
           print("re-training")

        for j in range(start_point,iterations):   #  iterations
            if j+1 > int(2e4):
                lr_ = learning_rate*0.1
            if j+1 > int(4e4):
                lr_ = learning_rate*0.01             
                
            train_data, train_label = read_data(input_path, gt_path, patch_size, num_channels, batch_size) 
			
            _,Training_Loss = sess.run([train_op,loss], feed_dict={images: train_data, labels: train_label, is_training: True, lr: lr_}) # training
      
            if np.mod(j+1,100) == 0 and j != 0: # save the model every 100 iterations
               Validation_Loss = sess.run(loss,  feed_dict={images: validation_data, labels: validation_label, is_training: False})  # validation loss
               
               print ('%d / %d iteraions, learning rate = %.3f, Training Loss = %.4f,  Validation Loss = %.4f' 
                      % (j+1, iterations, lr_, Training_Loss, Validation_Loss))   
               
               save_path_full = os.path.join(save_model_path, model_name) # save model
               saver.save(sess, save_path_full, global_step = j+1)