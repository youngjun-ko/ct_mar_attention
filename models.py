import tensorflow as tf


def convert_tensor(tensor):
    tensor -= tf.reduce_min(tensor)
    tensor *= (1/tf.reduce_max(tensor))
    tensor_resize = tf.image.resize_images(tensor, [224, 224])  # resize image into vgg input shape
    tensor_vgg = tf.tile(tensor_resize, [1, 1, 1, 3])  # gray-scale to rgb (duplicate)
    return tensor_vgg


def ResBlock(inputs, scope='resblock'):
    with tf.variable_scope(scope):
        outputs = tf.layers.conv2d(inputs, 64, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv0')
        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.conv2d(outputs, 64, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
        outputs = inputs + outputs
    return outputs


def global_average_pooling(inputs):
    avg_pool = tf.reduce_mean(inputs, axis=[1, 2])
    avg_pool = tf.expand_dims(avg_pool, 1)
    avg_pool = tf.expand_dims(avg_pool, 1)
    avg_pool = tf.sigmoid(avg_pool)
    outputs = tf.multiply(inputs, avg_pool)
    return outputs


def global_max_pooling(inputs):
    max_pool = tf.reduce_max(inputs, axis=[1, 2])
    max_pool = tf.expand_dims(max_pool, 1)
    max_pool = tf.expand_dims(max_pool, 1)
    max_pool = tf.sigmoid(max_pool)
    outputs = tf.multiply(inputs, max_pool)
    return outputs


def AttBlock(inputs, task, scope='attblock'):
    with tf.variable_scope(scope):
        outputs = tf.layers.conv2d(inputs, 64, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv0')
        outputs = tf.nn.relu(outputs)
        outputs = tf.layers.conv2d(outputs, 64, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
        if task == 'Chest':
            outputs = global_average_pooling(outputs)
        else:
            outputs = global_max_pooling(outputs)
        outputs = inputs + outputs
    return outputs


def network(inputs, task):
    with tf.variable_scope('deblur'):
        net = tf.layers.conv2d(inputs, 64, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv0')
        for i in range(10):
            net = AttBlock(net, task, scope='attblock' + str(i))
#             net = ResBlock(net, scope='resblock' + str(i))
        outputs = tf.layers.conv2d(net, 1, 5, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(), name='conv1')
    return outputs
