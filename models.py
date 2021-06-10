import tensorflow as tf
import tensorflow.contrib.slim as slim


def convert_tensor(tensor):
    tensor -= tf.reduce_min(tensor)
    tensor *= (1/tf.reduce_max(tensor))
    tensor_resize = tf.image.resize_images(tensor, [224, 224])  # resize image into vgg input shape
    tensor_vgg = tf.tile(tensor_resize, [1, 1, 1, 3])  # gray-scale to rgb (duplicate)
    return tensor_vgg


# ResBlock
def ResBlock(res_input):
    temp = slim.conv2d(res_input, 64, [5, 5], activation_fn=None)
    temp = tf.nn.relu(temp)
    res = slim.conv2d(temp, 64, [5, 5], activation_fn=None)
    res_output = res_input + res
    return res_output


# attention module
def global_average_pooling(GAP_input):
    avg_pool = tf.reduce_mean(GAP_input, axis=[1, 2])
    temp = tf.expand_dims(avg_pool, 1)
    temp = tf.expand_dims(temp, 1)
    channel_weight = temp
    GAP_output = tf.multiply(GAP_input, channel_weight)
    return GAP_output


# AttBlock
def AttBlock(rgap_input):
    temp = slim.conv2d(rgap_input, 64, [5, 5], activation_fn=None)
    temp = tf.nn.relu(temp)
    res = slim.conv2d(temp, 64, [5, 5], activation_fn=None)
    res_gap = global_average_pooling(res)
    rgap_output = rgap_input + res_gap
    return rgap_output


def network(input_img):
    with tf.variable_scope('deblur'):
        net = slim.conv2d(input_img, 64, [5, 5], activation_fn=None)
        for layer_ in range(10):
            net = AttBlock(net)
        deblur_img = slim.conv2d(net, 1, [5, 5], activation_fn=None)
        return deblur_img
