import tensorflow as tf
import numpy as np

class AlexNet(object):

  def __init__(self, x, keep_prob, num_classes, skip_layer):

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes
    self.KEEP_PROB = keep_prob
    self.SKIP_LAYER = skip_layer

    # Call the create function to build the computational graph of AlexNet
    self.create()

  def create(self):

    # 1st Layer: Conv (w ReLu)
    conv1 = conv(self.X, 3, 3, 64, 2, 2, padding = 'VALID', name = 'conv1')

    # 2nd Layer: Conv (w ReLu)
    conv2 = conv(conv1, 3, 3, 64, 2, 2, padding = 'VALID', name = 'conv2')

    # 3rd Layer: Conv (w ReLu)
    conv3 = conv(conv2, 3, 3, 32, 1, 1, name = 'conv3')

    # 4th Layer: Max pooling
    pool4 = max_pool(conv3, 3, 3, 2, 2, padding = 'VALID', name = 'pool4')

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    flattened = tf.reshape(pool4, [-1, 32*31*31])
    fc5 = fc(flattened, 32*31*31, 256, name='fc5', droprate = self.KEEP_PROB, dropout = True)

    # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
    fc6 = fc(fc5, 256, 4096, name='fc6', droprate = self.KEEP_PROB, dropout = True)

    # 7th Layer: FC (w ReLu) -> Dropout
    self.fc7 = fc(fc6, 4096, self.NUM_CLASSES, name = 'fc7', droprate = None, dropout = False)



def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME'):
    input_channels = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('weights', shape = [filter_height, filter_width,
                 input_channels, num_filters])
        conv = tf.nn.conv2d(x, kernel, [1, stride_y, stride_x, 1], padding = padding)
        biases = tf.get_variable('biases', shape=[num_filters])
        bias = tf.nn.bias_add(conv, biases)
    conv_ = tf.nn.relu(bias, name=name)
    return conv_

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                        strides = [1, stride_y, stride_x, 1],
                        padding = padding, name = name)

def fc(x, num_in, num_out, name, droprate, dropout = True):
  with tf.variable_scope(name) as scope:
    # Create tf variables for the weights and biases
    weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
    biases = tf.get_variable('biases', [num_out], trainable=True)

    # Matrix multiply weights and inputs and add bias
    fc = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

    if (dropout):
        fc_ = tf.nn.dropout(fc, droprate)
    else:
        fc_ = fc

    return fc_
