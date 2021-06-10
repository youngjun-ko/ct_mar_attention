import os
import cv2
import vgg16
import random
import numpy as np
import tensorflow as tf
from models import *


# Task Settings --------------------------------------------------------------------------------------------------------

task = 'CBCT_Teeth'  # FBCT_Teeth, CBCT_Teeth, Chest

test_file = np.load('./data/%s_test.npy' % task)
gt_file = np.load('./data/%s_gt.npy' % task)


# Testing --------------------------------------------------------------------------------------------------------------

model_path = 'log/%s' % task
result_path = 'result/%s' % task
if not os.path.isdir(result_path):
    os.makedirs(result_path)


def compute_mse(img1, img2):
    mse_ = ((img1 - img2) ** 2).mean()
    return mse_


def compute_rmse(img1, img2):
    mse_ = compute_mse(img1, img2)
    rmse_ = np.sqrt(mse_)
    return rmse_


def compute_psnr(img1, img2):
    mse_ = compute_mse(img1, img2)
    psnr_ = 10 * np.log10((1.0 ** 2) / mse_)
    return psnr_


with tf.Session() as sess:

    input_ph = tf.placeholder(tf.float32, shape=[None, None, None, 1])
    deblur_output = network(input_ph)

    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir='%s' % model_path)

    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('* Testing model loaded: ' + ckpt.model_checkpoint_path)

        test_npy = np.zeros([test_file.shape[0], test_file.shape[1], test_file.shape[2], test_file.shape[3]])
        test_npy_name = ('%s/%s_result.npy' % (result_path, task))
        original_psnr = 0
        original_rmse = 0
        deblur_psnr = 0
        deblur_rmse = 0

        for test_index in range(test_file.shape[0]):
            test_input = test_file[test_index:test_index+1, :, :, :]
            gt_slice = gt_file[test_index:test_index+1, :, :, :]
            test_output = sess.run([deblur_output], feed_dict={input_ph: test_input})
            test_result = test_output[0][0]
            test_npy[test_index, :, :, :] = test_result
            # saving as png image
            test_img = test_result - test_result.min()
            test_img *= (255/test_img.max())
            cv2.imwrite('%s/%d.png' % (result_path, test_index+1), test_img)

            # computing psnr, rmse
            original_psnr += compute_psnr(gt_slice, test_input)
            original_rmse += compute_rmse(gt_slice, test_input)
            deblur_psnr += compute_psnr(gt_slice, test_result)
            deblur_rmse += compute_rmse(gt_slice, test_result)

        # saving as npy file
        np.save(test_npy_name, test_npy)

        original_psnr /= test_file.shape[0]
        original_rmse /= test_file.shape[0]
        deblur_psnr /= test_file.shape[0]
        deblur_rmse /= test_file.shape[0]

        print('[Original]\nPSNR avg: %f\nRMSE avg: %f' % (original_psnr, original_rmse))
        print('[Proposed]\nPSNR avg: %f\nRMSE avg: %f' % (deblur_psnr, deblur_rmse))
        print('* Testing has been finished.')

    else:

        print('* Testing model does not exist or load failed.')
