import os
import vgg16
import random
import numpy as np
import tensorflow as tf
from models import *


# Task Settings --------------------------------------------------------------------------------------------------------

task = 'FBCT_Teeth'

train_file = np.load('./data/%s_train.npy' % task)
target_file = np.load('./data/%s_target.npy' % task)

start_epoch = 0  # starting epoch. [0]
end_Epoch = 100  # last epoch. [100]


# Training -------------------------------------------------------------------------------------------------------------

model_path = 'log/%s' % task
if not os.path.isdir(model_path):
    os.makedirs(model_path)

with tf.Session() as sess:

    # Training Specifications
    nTrain = np.size(train_file, 0)

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

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_path)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('* Training model loaded: ' + ckpt.model_checkpoint_path)
    else:
        print('* Training model does not exist or load failed.')

    # Training!
    print('* Start training...')

    epoch = start_epoch
    while epoch < end_Epoch:
        for train_index in range(nTrain):

            random_index = random.randint(1, nTrain) - 1
            input_image = train_file[random_index:random_index+1, :, :, :]
            target_image = target_file[random_index:random_index+1, :, :, :]
            loss_, l1_, vgg_, _ = sess.run([loss, loss_l1, loss_vgg, opt], feed_dict={input_ph: input_image, target_ph: target_image})

            if not (train_index + 1) % (nTrain/10):
                print("Epoch:[%3d/%d] Batch:[%5d/%5d] - Loss:[%4.4f] L1:[%4.4f] VGG:[%4.4f]"
                      % (epoch, end_Epoch, (train_index+1), nTrain, loss_, l1_, vgg_))

        epoch += 1
        saver.save(sess, "%s/model.ckpt" % model_path)

        if epoch == end_Epoch:
            break

    print('* Training Finished!')
