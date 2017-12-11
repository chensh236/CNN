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

    # 1st Layer: Conv (ReLu)
    conv1 = conv(self.X, 3, 3, 64, 2, 2, name = 'conv1')

    # 2nd Layer: Conv (ReLu)
    conv2 = conv(conv1, 3, 3, 64, 2, 2, name = 'conv2')

    # 3rd Layer: Conv (ReLu)
    conv3 = conv(conv2, 3, 3, 32, 1, 1, name = 'conv3')

    # 4th Layer: Max pooling
    pool4 = max_pool(conv3, 3, 3, 2, 2, padding = 'VALID', name = 'pool4')

    # 5th Layer: Flatten -> FC (ReLu) -> Dropout
    flattened = tf.reshape(pool4, [-1, 32*31*31], name = 'flattened')
    fc5 = fc(flattened, 32*31*31, 256, name='fc5', droprate = self.KEEP_PROB, dropout = True)

    # 6th Layer: FC (ReLu) -> Dropout
    fc6 = fc(fc5, 256, 4096, name='fc6', droprate = self.KEEP_PROB, dropout = True)

    # 7th Layer: FC (ReLu) -> softmax
    self.fc7 = fc(fc6, 4096, self.NUM_CLASSES, name = 'fc7', droprate = 0, dropout = False)



def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        input_channels = int(x.get_shape()[-1])
        n = filter_height * filter_width * num_filters
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        kernel = tf.Variable(tf.truncated_normal([filter_height, filter_width,
                          input_channels, num_filters], dtype=tf.float32,
                          stddev=1e-1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_filters], dtype=tf.float32),
                         trainable=True, name='biases')
        conv = convolve(x, kernel)
        bias = tf.nn.bias_add(conv, biases)
        # bias = batch_norm(bias, phase_train, name)
        conv_ = tf.nn.relu(bias, name=name)
        return conv_

def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                            strides = [1, stride_y, stride_x, 1],
                            padding = padding, name = name)

def fc(x, num_in, num_out, name, droprate, dropout = True):
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.Variable(tf.truncated_normal([num_in, num_out], dtype=tf.float32,
                          stddev=1e-1), name='fc_weights')
        biases = tf.Variable(tf.constant(0.0, shape=[num_out], dtype=tf.float32),
                         trainable=True, name='fc_biases')
        # Matrix multiply weights and inputs and add bias
        fc = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        fc_ = tf.nn.relu(fc, name=scope.name)
        if (dropout):
            fc_ = tf.nn.dropout(fc_, droprate)
        return fc_

# def lrn(x, radius, alpha, beta, name, bias=1.0):
#     """Create a local response normalization layer."""
#     return tf.nn.local_response_normalization(x, depth_radius=radius,
#                                               alpha=alpha, beta=beta,
#                                               bias=bias, name=name)
