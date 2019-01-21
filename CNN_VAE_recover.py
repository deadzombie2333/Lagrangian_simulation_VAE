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
from scipy import interpolate

now = datetime.utcnow().strftime("%Y%m%d")
logdir = "/temp/run-{}/example".format(now)

with open("full_dictionary.pkl","rb") as f:
  full_batches = pickle.load(f)
  
with open("CNN_VAE_recover.pkl","rb") as f:
  Simu_info = pickle.load(f)

def xyz_norm(batch_dictionary):
  key_list = list(batch_dictionary.keys())
  full_sample = np.zeros((1, 1100, 3))
  
  for key in key_list:
    cur_batch = batch_dictionary[key]
    input_max = np.array([60, 500, 300])
    input_min = np.array([20, 0, 0])
    
    input_norm = (cur_batch - input_min) / (input_max - input_min)
    input_reshape = np.reshape(input_norm, newshape = (1, 1100, 3))
    full_sample = np.concatenate([full_sample, input_reshape], axis = 0)
    
  return full_sample[1:2], full_sample[2:]
  
def generator(simu_info):  #4, 3, 10
  """ Given input rank 3 matrix, run it over rnn model and return generated model output """
  with tf.variable_scope('cnn_init', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    rand_1 = tf.get_variable('rand_1', [15, 3, 10])
    rand_2 = tf.get_variable('rand_2', [15, 3, 10])
    rand_3 = tf.get_variable('rand_3', [15, 3, 10])
    
  with tf.variable_scope('cnn_vae', reuse = tf.AUTO_REUSE, initializer=tf.contrib.layers.xavier_initializer()) as scope:
    rand_layer4_mean = tf.tile(simu_info[0:1], (15, 1, 1)) + rand_1 * tf.tile(simu_info[1:2], (15, 1, 1))
    rand_layer4_gamma = tf.tile(simu_info[2:3], (15, 1, 1)) + rand_2 * tf.tile(simu_info[3:4], (15, 1, 1))
    
    rand_layer4 = rand_layer4_mean + rand_layer4_gamma * rand_3
    
    rand_layer4_reshape = tf.reshape(rand_layer4, shape = (-1, 3, 1, 10))
    rand_w5 = tf.get_variable('rand_w5', [1, 11, 10, 10])
    rand_b5 = tf.get_variable('rand_b5', [10])
    rand_layer5 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer4_reshape, rand_w5, output_shape =[15, 3, 11, 10], strides = [1,1,11,1],  padding = 'VALID') + rand_b5)
    
    rand_w6 = tf.get_variable('rand_w6', [1, 5, 10, 10])
    rand_b6 = tf.get_variable('rand_b6', [10])
    rand_layer6 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer5, rand_w6, output_shape =[15, 3, 55, 10], strides = [1,1,5,1],  padding = 'VALID') + rand_b6)
    
    rand_w7 = tf.get_variable('rand_w7', [1, 2, 10, 10])
    rand_b7 = tf.get_variable('rand_b7', [10])
    rand_layer7 = tf.nn.elu(tf.nn.conv2d_transpose(rand_layer6, rand_w7, output_shape =[15, 3, 110, 10], strides = [1,1,2,1],  padding = 'VALID') + rand_b7)
    
    rand_layer8 = tf.reshape(rand_layer7, shape = (15, 3, 1100, 1))
    rand_filt = tf.ones(shape = (1, 51, 1, 1)) / 51
    rand_layer8_pat = tf.concat([rand_layer8[:,:,:50,:], rand_layer8], axis = 2)
    
    rand_layer9 = tf.nn.conv2d(rand_layer8_pat, rand_filt, strides = [1,1,1,1], padding = 'VALID')
    output = tf.reshape(rand_layer9, shape = (15, 3, 1100))

  rand_sum = tf.reshape(tf.concat([rand_1, rand_2, rand_3], 1), shape = (-1,1))
    
  latent_loss = 0.5 * tf.reduce_sum(tf.square(rand_sum) - tf.log(tf.square(rand_sum)) - 1)
    
  return tf.transpose(output, perm = [0, 2, 1]), latent_loss
 
simu_info = tf.placeholder("float", shape = [4, 3, 10])
real_xyz = tf.placeholder('float', shape = [15, 1100, 3])
fake_xyz, latent_loss= generator(simu_info)

init_loss = tf.reduce_sum(tf.losses.mean_squared_error(labels = real_xyz[:,:40], predictions = fake_xyz[:,:40])) + 2e-5 * latent_loss
parameter_init = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'cnn_init')
train_INIT = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(init_loss, var_list = parameter_init)

parameter_g = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'cnn_vae')
reuse_vars_dict = dict([(var.op.name, var) for var in parameter_g])
pre_vars_dict = {**reuse_vars_dict}
pre_saver = tf.train.Saver(pre_vars_dict)

saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epoch = 1000

with tf.Session() as sess:
  init.run()
  test_xyz, full_xyz = xyz_norm(full_batches)
  pre_saver.restore(sess,"/tmp/ash_particle/CNN_VAE.ckpt")
  
  train_shuffle_xyz = np.concatenate([test_xyz, full_xyz], axis = 0)
  for epoch in range(num_epoch):
    sess.run(train_INIT, feed_dict = {simu_info: Simu_info, real_xyz: train_shuffle_xyz})
  
  true_train = train_shuffle_xyz[1:]
  train_shuffle_xyz_output = sess.run(fake_xyz, feed_dict = {simu_info: Simu_info, real_xyz: train_shuffle_xyz})

  true_test = train_shuffle_xyz[0:1]
  fake_train = train_shuffle_xyz_output[1:]
  fake_test = train_shuffle_xyz_output[0:1]
  
def smoothing_fn(traj):
  for j in range(traj.shape[0]):
    """
    tck, u = interpolate.splprep([traj[j,:,0], traj[j,:,1], traj[j,:,2]], s=0.6, ub = 5)
    u_fine = np.linspace(0,1,1100)
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    
    traj[j,:,0] = x_fine
    traj[j,:,1] = y_fine
    traj[j,:,2] = z_fine
    """
    traj[j,:,0] = scipy.signal.medfilt(traj[j,:,0], 21)
    traj[j,:,1] = scipy.signal.medfilt(traj[j,:,1], 21)
    traj[j,:,2] = scipy.signal.medfilt(traj[j,:,2], 21)
  return traj

fake_train = smoothing_fn(fake_train)
fake_test = smoothing_fn(fake_test)

input_max = np.array([60, 500, 300])
input_min = np.array([30, 0, 0])

fake_train = fake_train * (input_max - input_min) + input_min#14, 1100, 3
fake_test = fake_test * (input_max - input_min) + input_min#1, 1100, 3
true_train = true_train * (input_max - input_min) + input_min
true_test = true_test * (input_max - input_min) + input_min

def rz2xyz(rz_traj):
  r_traj = rz_traj[:,:,0:1]
  phi_traj = rz_traj[:,:,1:2]
  z_traj = rz_traj[:,:,2:3]
  
  x_traj = r_traj * np.cos(phi_traj/360*2*np.pi)
  y_traj = r_traj * np.sin(phi_traj/360*2*np.pi)
  
  xyz_traj = np.concatenate([x_traj, y_traj, z_traj], axis = 2)
  
  return xyz_traj

fake_train_xyz = rz2xyz(fake_train)
fake_test_xyz = rz2xyz(fake_test)
true_train_xyz = rz2xyz(true_train)
true_test_xyz = rz2xyz(true_test)

fake_traj = np.concatenate([fake_train, fake_test], axis = 0)
fake_traj_print = np.reshape(np.transpose(fake_traj, (0, 2, 1)), newshape = (45, 1100))
np.savetxt('VAE_generate_curve_constrained.txt', fake_traj_print, delimiter = '\t')

"""
fake_train_xyz_save = np.reshape(np.transpose(fake_xyz, (0, 2, 1)), newshape = (42, 1100))
true_train_xyz_save = np.reshape(np.transpose(true_xyz, (0, 2, 1)), newshape = (42, 1100))

np.savetxt('cnn_vae_fake_xyz.txt', fake_xyz_save, delimiter='\t')
np.savetxt('cnn_vae_true_xyz.txt', true_xyz_save, delimiter='\t')
"""
BATCH_NUM = 6
plt.subplot(211)
plt.plot(np.linspace(1, 1100, 1100), true_train[BATCH_NUM,:,0], 'k--', label='x')
plt.plot(np.linspace(1, 1100, 1100), true_train[BATCH_NUM,:,1], 'k:', label='y')
plt.plot(np.linspace(1, 1100, 1100), true_train[BATCH_NUM,:,2], 'k', label='z')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.xlim([0, 1100])
plt.ylim([0, 350])
plt.xticks([])
plt.yticks([0, 100, 200, 300])
text_str = 'experimental observation'
plt.text(500, 300, text_str, fontsize=14)

plt.subplot(212)
plt.plot(np.linspace(1, 1100, 1100), fake_train[BATCH_NUM,:,0], 'k--', np.linspace(1, 1100, 1100), fake_train[BATCH_NUM,:,1], 'k:', np.linspace(1, 1100, 1100), fake_train[BATCH_NUM,:,2], 'k')
plt.xlim([0, 1100])
plt.ylim([0, 350])
plt.yticks([0, 100, 200, 300])
plt.xticks([0, 500, 1000])
text_str = 'variant autoencoder generate'
plt.text(450, 300, text_str, fontsize=14)
plt.show()

BATCH_NUM = 0
plt.subplot(211)
plt.plot(np.linspace(1, 1100, 1100), true_test[BATCH_NUM,:,0], 'k--', label='x')
plt.plot(np.linspace(1, 1100, 1100), true_test[BATCH_NUM,:,1], 'k:', label='y')
plt.plot(np.linspace(1, 1100, 1100), true_test[BATCH_NUM,:,2], 'k', label='z')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.xlim([0, 1100])
plt.ylim([0, 150])
plt.xticks([])
plt.yticks([0, 50, 100, 150])
text_str = 'experimental observation'
plt.text(500, 130, text_str, fontsize=14)

plt.subplot(212)
plt.plot(np.linspace(1, 1100, 1100), fake_test[BATCH_NUM,:,0], 'k--', np.linspace(1, 1100, 1100), fake_test[BATCH_NUM,:,1], 'k:', np.linspace(1, 1100, 1100), fake_test[BATCH_NUM,:,2], 'k')
plt.xlim([0, 1100])
plt.ylim([0, 150])
plt.yticks([0, 50, 100, 150])
plt.xticks([0, 500, 1000])
text_str = 'variant autoencoder generate'
plt.text(450, 130, text_str, fontsize=14)
plt.show()

plt.subplot(211)
for j in range(14):
  plt.plot(true_train_xyz[j,:,0], true_train_xyz[j,:,1], 'k:')
plt.plot(true_test_xyz[0,:,0], true_test_xyz[0,:,1], 'k')
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.yticks([-50, -25, 0, 25, 50])
plt.xticks([-50, -25, 0, 25, 50])
text_str = 'experimental observation'
plt.text(-40, 60, text_str, fontsize=14)

plt.subplot(212)
for j in range(14):
  plt.plot(fake_train_xyz[j,:,0], fake_train_xyz[j,:,1], 'k:')
plt.plot(fake_test_xyz[0,:,0], fake_test_xyz[0,:,1], 'k')
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.yticks([-50, -25, 0, 25, 50])
plt.xticks([-50, -25, 0, 25, 50])
text_str = 'variant autoencoder generate'
plt.text(-40, 60, text_str, fontsize=14)
plt.show()

BATCH_NUM = 6
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_train_xyz[BATCH_NUM, :, 0], fake_train_xyz[BATCH_NUM,:, 1], fake_train_xyz[BATCH_NUM,:, 2], 'k--', label='variant autoencoder generate')
ax.plot(true_train_xyz[BATCH_NUM, :, 0], true_train_xyz[BATCH_NUM,:, 1], true_train_xyz[BATCH_NUM,:, 2], 'k', label='experimental observation')
plt.xlabel('x')
plt.ylabel('y')
ax.legend()
text_str = 'trajectory start'
ax.text(-30, -25, 70, text_str, fontsize=14)
plt.show()

BATCH_NUM = 0
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(fake_test_xyz[BATCH_NUM, :, 0], fake_test_xyz[BATCH_NUM,:, 1], fake_test_xyz[BATCH_NUM,:, 2], 'k--', label='variant autoencoder generate')
ax.plot(true_test_xyz[BATCH_NUM, :, 0], true_test_xyz[BATCH_NUM,:, 1], true_test_xyz[BATCH_NUM,:, 2], 'k', label='experimental observation')
plt.xlabel('x')
plt.ylabel('y')
ax.legend()
text_str = 'trajectory start'
ax.text(0, 40, 60, text_str, fontsize=14)
plt.show()


training_curve = np.loadtxt('CNN_VAE_training_curve.txt')
l1 = plt.scatter(np.linspace(1, 5000, 50), np.log(training_curve[range(0,4999,100),0]), c='k', marker = '<', alpha=0.5, label='training loss')
l2 = plt.scatter(np.linspace(1, 5000, 50), np.log(training_curve[range(0,4999,100),1]), c='k', marker = 'o', alpha=0.5, label='test loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.xlim([0, 5000])
plt.ylim([-6, -1])
plt.xticks([0, 1000, 2000, 3000, 4000, 5000])
plt.yticks([-6, -5, -4, -3, -2, -1])
plt.xlabel('epoches')
plt.ylabel('log loss')
plt.show()

