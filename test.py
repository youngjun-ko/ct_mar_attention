import os
import vgg16
import random
import numpy as np
import tensorflow as tf
from models import *


# Task Settings --------------------------------------------------------------------------------------------------------

task = 'FBCT_Teeth'  # FBCT_Teeth, CBCT_Teeth, Chest

test_file = np.load('./data/%s_test.npy' % task)


# Testing --------------------------------------------------------------------------------------------------------------

model_path = 'log/%s' % task
if not os.path.isdir(model_path):
    os.makedirs(model_path)

with tf.Session() as sess:

    # Placeholders
    input_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    target_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1])

    # Deblur with network
    deblur_output = network(input_ph)

    # L1 loss
    loss_l1 = 1e2 * tf.reduce_mean(abs(deblur_output - target_ph))  # L1 loss

    # VGG loss
    loss_vgg = tf.zeros(1, tf.float32)
    target_resize = convert_tensor(target_ph)
    vgg_t = vgg16.Vgg16()
    vgg_t.build(target_resize)

    target_feature = [vgg_t.conv3_3, vgg_t.conv4_3]
    output_resize = convert_tensor(deblur_output)
    vgg_o = vgg16.Vgg16()
    vgg_o.build(output_resize)

    output_feature = [vgg_o.conv3_3, vgg_o.conv4_3]
    for f, f_ in zip(output_feature, target_feature):
        loss_vgg += 5*1e-5 * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])  # Perceptual(vgg) loss

    # Total loss & Optimizer
    loss = loss_l1 + loss_vgg
    opt = tf.train.AdamOptimizer(learning_rate=5*1e-5).minimize(loss, var_list=tf.trainable_variables())

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='%s' % model_path)

    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('* Testing model loaded: ' + ckpt.model_checkpoint_path)

        test_npy = np.zeros([test_file.shape[0], test_file.shape[1], test_file.shape[2], test_file.shape[3]])
        test_npy_name = ('./%s_result.npy' % task)

        for test_index in range(test_file.shape[0]):
            test_input = test_file[test_index:test_index+1, :, :, :]
            test_output = sess.run([deblur_output], feed_dict={input_ph: test_input})
            test_result = test_output[0][0]
            test_npy[test_index, :, :, :] = test_result

        np.save(test_npy_name, test_npy)
        print('* Testing has been finished.')

    else:

        print('* Testing model does not exist or load failed.')
