import tensorflow as tf
import numpy as np

class AlexNet(object):

  def __init__(self, x, num_classes):

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes

    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):

    # 1st Layer: Conv -> pooling
    conv1 = conv(self.X, 4, 4, 32, 1, 1, name = 'conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv2 = conv(pool1, 5, 5, 48, 1, 1, name = 'conv2')
    pool2 = max_pool(conv2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv3 = conv(pool2, 5, 5, 64, 1, 1, name = 'conv3')
    pool3 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv4 = conv(pool3, 5, 5, 128, 1, 1, padding = 'VALID', name = 'conv4')

    # 5th Layer: Flatten -> FC (ReLu)
    flattened = tf.reshape(conv4, [-1, 128*4*4], name = 'flattened')
    fc1 = fc(flattened, 128*4*4, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(relu1, 128, 128*self.NUM_CLASSES, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(relu2, 128*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        input_channels = int(x.get_shape()[-1])
        n = filter_height * filter_width * num_filters
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        kernel = tf.get_variable(shape = [filter_height, filter_width,
                            input_channels, num_filters], dtype=tf.float32, name='weights',
                            initializer=tf.contrib.layers.xavier_initializer(), trainable = True)
        biases = tf.get_variable(shape=[num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0),
		                    trainable = True, name='biases')
        conv = convolve(x, kernel)
        conv_ = tf.nn.bias_add(conv, biases)
        # conv_ = tf.nn.relu(bias, name=name)
        return conv_

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                            strides = [1, stride_y, stride_x, 1],
                            padding = padding, name = name)

def avg_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        return tf.nn.avg_pool(x, ksize=[1, filter_height, filter_width, 1],
                            strides = [1, stride_y, stride_x, 1],
                            padding = padding, name = name)

def fc(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable(shape = [num_in, num_out], dtype=tf.float32, name='weights',
                              initializer=tf.contrib.layers.xavier_initializer(), trainable = True)
        biases = tf.get_variable(shape=[num_out], dtype=tf.float32,
                         trainable=True, name='fc_biases', initializer=tf.constant_initializer(0))
        # Matrix multiply weights and inputs and add bias
        fc = tf.nn.xw_plus_b(x, weights, biases, name=name)
        return fc
