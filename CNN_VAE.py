'''
variant autoencoder based on convolution neural network

generate trajectory that exactly like the original one

require full trajectory from certain initial points.
'''
import os
import re
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import random
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import dropout
import matplotlib.pyplot as plt
from datetime import datetime
import sys

now = datetime.utcnow().strftime("%Y%m%d")
logdir = "/temp/run-{}/example".format(now)

with open("full_dictionary.pkl","rb") as f:
  full_batches = pickle.load(f)

def xyz_norm(batch_dictionary):
  key_list = list(batch_dictionary.keys())
  full_sample = np.zeros((1, 1100, 3))
  
  for key in key_list:
    cur_batch = batch_dictionary[key]
    input_max = np.array([60, 500, 300])
    input_min = np.array([30, 0, 0])
    
    input_norm = (cur_batch - input_min) / (input_max - input_min)
    input_reshape = np.reshape(input_norm, newshape = (1, 1100, 3))
    full_sample = np.concatenate([full_sample, input_reshape], axis = 0)
    
  return full_sample
  
def generator(real_state):  #None, 1100, 3
  """ Given input rank 3 matrix, run it over rnn model and return generated model output """
  with tf.variable_scope('cnn_vae', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    rand_layer0 = tf.expand_dims(tf.transpose(real_state, perm = [0, 2, 1]), 3) #-1, 3, 1100, 1
    batch_num = tf.shape(real_state)[0]
    
    rand_w1 = tf.get_variable('rand_w1', [1, 11, 1, 2])
    rand_b1 = tf.get_variable('rand_b1', [2])
    rand_layer1 = tf.nn.elu(tf.nn.conv2d(rand_layer0, rand_w1, strides = [1,1,11,1], padding = 'VALID') + rand_b1)#15, 3, 100, 1
    
    rand_w2 = tf.get_variable('rand_w2', [2, 10, 2, 5])
    rand_b2 = tf.get_variable('rand_b2', [5])
    rand_layer2 = tf.nn.elu(tf.nn.conv2d(rand_layer1, rand_w2, strides = [1,1,10,1], padding = 'VALID') + rand_b2)#15, 3, 10, 5
    
    rand_w3 = tf.get_variable('rand_w3', [2, 10, 5, 10])
    rand_b3 = tf.get_variable('rand_b3', [10])
    rand_layer3 = tf.nn.elu(tf.nn.conv2d(rand_layer2, rand_w3, strides = [1,1,10,1], padding = 'VALID') + rand_b3)#15, 3, 1, 10
    
    rand_layer4_mean = tf.contrib.layers.fully_connected(rand_layer3, 10, activation_fn = None, biases_initializer = tf.contrib.layers.xavier_initializer())
    rand_layer4_gamma = tf.contrib.layers.fully_connected(rand_layer3, 10, activation_fn = None, biases_initializer = tf.contrib.layers.xavier_initializer())
    Noise = tf.random_normal(tf.shape(rand_layer4_gamma), dtype = tf.float32)
    rand_layer4 = rand_layer4_mean + rand_layer4_gamma * Noise
    
    rand_layer4_reshape = tf.reshape(rand_layer4, shape = (batch_num, 1, 1, 10))
    rand_w5 = tf.get_variable('rand_w5', [1, 11, 10, 10])
    rand_b5 = tf.get_variable('rand_b5', [10])
    rand_layer5 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer4_reshape, rand_w5, output_shape =[batch_num, 1, 11, 10], strides = [1,1,11,1],  padding = 'VALID') + rand_b5)
    
    rand_w6 = tf.get_variable('rand_w6', [2, 5, 10, 10])
    rand_b6 = tf.get_variable('rand_b6', [10])
    rand_layer6 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer5, rand_w6, output_shape =[batch_num, 2, 55, 10], strides = [1,1,5,1],  padding = 'VALID') + rand_b6)
    
    rand_w7 = tf.get_variable('rand_w7', [2, 2, 10, 10])
    rand_b7 = tf.get_variable('rand_b7', [10])
    rand_layer7 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer6, rand_w7, output_shape =[batch_num, 3, 110, 10], strides = [1,1,2,1],  padding = 'VALID') + rand_b7)
    
    rand_layer8 = tf.reshape(rand_layer7, shape = (batch_num, 3, 1100, 1))
    rand_filt = tf.ones(shape = (1, 51, 1, 1)) / 51
    rand_layer8_pat = tf.concat([rand_layer8[:,:,:50,:], rand_layer8], axis = 2)
    
    rand_layer9 = tf.nn.conv2d(rand_layer8_pat, rand_filt, strides = [1,1,1,1], padding = 'VALID')
    output = tf.reshape(rand_layer9, shape = (batch_num, 3, 1100))

    latent_loss = 0.5 * tf.reduce_sum(tf.square(rand_layer4_mean) + tf.square(rand_layer4_gamma) - tf.log(tf.square(rand_layer4_gamma)) - 1)
    
  return tf.transpose(output, perm = [0, 2, 1]), rand_layer4_mean, rand_layer4_gamma, latent_loss
 
real_xyz = tf.placeholder("float", shape = [None, 1100, 3])
fake_xyz, latent_mean, latent_gamma, latent_loss= generator(real_xyz)

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'cnn_vae')

all_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels = real_xyz, predictions = fake_xyz))

g_loss = all_loss + 1e-2 * latent_loss

train_G = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(g_loss, var_list = parameter_g)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epoch = 5000
loss_matrix = np.zeros((num_epoch, 1))

with tf.Session() as sess:
  init.run()
  full_xyz = xyz_norm(full_batches)
  
  for epoch in range(num_epoch):
    print(epoch)
    np.random.shuffle(full_xyz)
    
    sess.run(train_G, feed_dict = {real_xyz: full_xyz})
    train_loss = sess.run(all_loss, feed_dict = {real_xyz: full_xyz})
    
    print(train_loss)
    loss_matrix[epoch, 0] = train_loss
    
    if (epoch + 1) % 500 == 0:
      true_traj = full_xyz
      fake_traj, MEAN, GAMMA = sess.run((fake_xyz, latent_mean, latent_gamma), feed_dict = {real_xyz: full_xyz})
      
      #save_path = saver.save(sess,"/tmp/ash_particle/CNN_VAE.ckpt")

#np.savetxt('CNN_VAE_training_curve.txt', loss_matrix, delimiter = '\t')

input_max = np.array([60, 500, 300])
input_min = np.array([30, 0, 0])

fake_traj = fake_traj * (input_max - input_min) + input_min
true_traj = true_traj * (input_max - input_min) + input_min

fake_traj_print = np.reshape(np.transpose(fake_traj[1:], (0, 2, 1)), newshape = (45, 1100))
np.savetxt('vae_1e-2.txt', fake_traj_print, delimiter = '\t')

'''
MEAN_mean = np.transpose(np.mean(MEAN, 0), (1, 0, 2))
MEAN_std = np.transpose(np.std(MEAN, 0), (1, 0, 2))
GAMMA_mean = np.transpose(np.mean(GAMMA, 0), (1, 0, 2))
GAMMA_std = np.transpose(np.std(GAMMA, 0), (1, 0, 2))

Simu_info = np.concatenate([MEAN_mean, MEAN_std, GAMMA_mean, GAMMA_std], axis = 0)
f = open("CNN_VAE_recover.pkl","wb")
pickle.dump(Simu_info,f)
f.close()

BATCH_NUM = 2
plt.subplot(211)
plt.plot(np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), true_traj[BATCH_NUM,:,2], 'g')
plt.subplot(212)
plt.plot(np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,0], 'r', np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,1], 'b', np.linspace(1, 1100, 1100), fake_traj[BATCH_NUM,:,2], 'g')
plt.show()

plt.subplot(311)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), MEAN[i,0,:])
plt.subplot(312)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), MEAN[i,1,:])
plt.subplot(313)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), MEAN[i,2,:])
plt.show()

plt.subplot(311)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), GAMMA[i,0,:])
plt.subplot(312)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), GAMMA[i,1,:])
plt.subplot(313)
for i in range(15):
  plt.plot(np.linspace(1, 10, 10), GAMMA[i,2,:])
plt.show()

for j in range(15):
  fake_traj[j,:,0] = scipy.signal.savgol_filter(fake_traj[j,:,0], 251, 2)
  fake_traj[j,:,1] = scipy.signal.savgol_filter(fake_traj[j,:,1], 251, 2)
  fake_traj[j,:,2] = scipy.signal.savgol_filter(fake_traj[j,:,2], 251, 2)
  true_traj[j,:,0] = scipy.signal.savgol_filter(true_traj[j,:,0], 251, 2)
  true_traj[j,:,1] = scipy.signal.savgol_filter(true_traj[j,:,1], 251, 2)
  true_traj[j,:,2] = scipy.signal.savgol_filter(true_traj[j,:,2], 251, 2)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_traj[BATCH_NUM, :, 0], fake_traj[BATCH_NUM,:, 1], fake_traj[BATCH_NUM,:, 2], label='generated curve')
ax.plot(true_traj[BATCH_NUM, :, 0], true_traj[BATCH_NUM,:, 1], true_traj[BATCH_NUM,:, 2], label='experiment curve')
ax.legend()
plt.show()
'''
