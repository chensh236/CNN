import tensorflow as tf
import numpy as np

class AlexNet(object):

  def __init__(self, x, keep_prob, num_classes):

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob

    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):
    conv1 = conv(self.X, 7, 7, 64, 2, 2, name = 'conv1')
    conv2 = conv(conv1, 7, 7, 96, 2, 2, name = 'conv2')
    conv3 = conv(conv2, 3, 3, 96, 1, 1, name = 'conv3')
    conv4 = conv(conv3, 3, 3, 96, 1, 1, name = 'conv4')
    pool1 = max_pool(conv4, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
    # conv4 = conv(pool1, 3, 3, 256, 2, 2, name = 'conv4')
    conv5 = conv(pool1, 3, 3, 256, 1, 1, name = 'conv5')
    conv6 = conv(conv5, 1, 1, 256, 1, 1, name = 'conv6')
    conv7 = conv(conv6, 1, 1, 256, 1, 1, name = 'conv7')
    pool2 = max_pool(conv7, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')
    conv8 = conv(pool2, 3, 3, 512, 1, 1, name = 'conv8')
    conv9 = conv(conv8, 1, 1, 512, 1, 1, name = 'conv9')
    conv10 = conv(conv9, 1, 1, 512, 1, 1, name = 'conv10')
    pool3 = max_pool(conv10, 3, 3, 2, 2, padding = 'VALID', name = 'pool3')
    conv11 = conv(pool3, 3, 3, 768, 1, 1, name = 'conv11')
    conv12 = conv(conv11, 1, 1, 768, 1, 1, name = 'conv12')
    conv13 = conv(conv12, 1, 1, 768, 1, 1, name = 'conv13')
    conv14 = conv(conv13, 1, 1, self.NUM_CLASSES, 1, 1, name = 'conv14')
    result = avg_pool(conv14, 7, 7, 1, 1,padding = 'VALID', name = 'result')
    self.fc3 = tf.reshape(result, [-1, 1*1*self.NUM_CLASSES], name = 'fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def create_(self):

    # 1st Layer: Conv (ReLu) -> pooling
    conv1 = conv(self.X, 3, 3, 64, 2, 2, name = 'conv1')

    # 2nd Layer: Conv (ReLu) -> pooling
    conv2 = conv(conv1, 3, 3, 64, 2, 2, name = 'conv2')

    # 3rd Layer: Conv (ReLu) -> pooling
    conv3 = conv(conv2, 3, 3, 32, 1, 1, name = 'conv3')
    pool1 = max_pool(conv3, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')

    # 4th Layer: Flatten -> FC (ReLu) -> Dropou4
    flattened = tf.reshape(pool1, [-1, 32*31*31], name = 'flattened')
    fc1 = fc(flattened, 32*31*31, 1024, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')
    drop1 = tf.nn.dropout(relu1, self.KEEP_PROB, name = 'drop1')

    # 5th Layer: FC (ReLu) -> Dropout
    fc2 = fc(drop1, 1024, 512, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')
    drop2 = tf.nn.dropout(relu2, self.KEEP_PROB, name = 'drop2')

    # 6th Layer: FC (ReLu) -> softmax
    self.fc3 = fc(drop2, 512, self.NUM_CLASSES, name = 'fc3')
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
        bias = tf.nn.bias_add(conv, biases)
        conv_ = tf.nn.relu(bias, name=name)
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
        weights = tf.get_variable(shape = [num_in, num_out], dtype=tf.float32, name='fc_weights',
                              initializer=tf.contrib.layers.xavier_initializer(), trainable = True)
        biases = tf.get_variable(shape=[num_out], dtype=tf.float32,
                         trainable=True, name='fc_biases', initializer=tf.constant_initializer(0))
        # Matrix multiply weights and inputs and add bias
        fc = tf.nn.xw_plus_b(x, weights, biases, name=name)
        return fc
