import tensorflow as tf
import numpy as np

class AlexNet(object):

  def __init__(self, x, num_classes):

    # Parse input arguments into class variables
    self.X = x
    self.NUM_CLASSES = num_classes

    # Call the create function to build the computational graph of AlexNet
    self.sub_new_create()

  def sub_new_create(self):
    sub_conv1 = sub_conv(self.X, 5, 5, 8, 1, 1, name='sub_conv')
    conv1 = conv(sub_conv1, 3, 3, 16, 1, 1, name = 'conv1')
    conv2 = conv(conv1, 3, 3, 16, 1, 1, name = 'conv2')
    pool1 = max_pool(conv2, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv3 = conv(pool1, 3, 3, 32, 1, 1, name = 'conv3')
    pool2 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv4 = conv(pool2, 3, 3, 48, 1, 1, name = 'conv4')
    pool3 = max_pool(conv4, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv5 = conv(pool3, 3, 3, 64, 1, 1, padding = 'VALID', name = 'conv5')
    
    # 5th Layer: Conv
    conv6 = conv(conv5, 3, 3, 72, 1, 1, padding = 'VALID', name = 'conv6')

    # 5th Layer: Flatten -> FC (ReLu)
    flattened = tf.reshape(conv6, [-1, 72*3*3], name = 'flattened')
    fc1 = fc(flattened, 72*3*3, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(relu1, 128, 16*self.NUM_CLASSES, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(relu2, 16*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def sub_luca_create(self):
    sub_conv1 = sub_conv(self.X, 5, 5, 8, 1, 1, name='sub_conv')
    conv1 = conv(sub_conv1, 5, 5, 32, 1, 1, name = 'conv1')
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
    flattened = tf.reshape(conv4, [-1, 128*3*3], name = 'flattened')
    fc1 = fc(flattened, 128*3*3, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(relu1, 128, 128*self.NUM_CLASSES, name='fc2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(fc2, 128*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def sub_create(self):
    sub_conv1 = sub_conv(self.X, 5, 5, 8, 1, 1, name='sub_conv')
    conv1 = conv(sub_conv1, 3, 3, 32, 1, 1, name = 'conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv2 = conv(pool1, 3, 3, 32, 1, 1, name = 'conv2')
    pool2 = max_pool(conv2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv3 = conv(pool2, 3, 3, 48, 1, 1, name = 'conv3')
    pool3 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv4 = conv(pool3, 3, 3, 64, 1, 1, padding = 'VALID', name = 'conv4')
    
    # 5th Layer: Conv
    conv5 = conv(conv4, 3, 3, 72, 1, 1, padding = 'VALID', name = 'conv5')

    # 5th Layer: Flatten -> FC (ReLu)
    flattened = tf.reshape(conv5, [-1, 72*3*3], name = 'flattened')
    fc1 = fc(flattened, 72*3*3, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(relu1, 128, 64*self.NUM_CLASSES, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(relu2, 64*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def new_create(self):
    # 1st Layer: Conv -> pooling
    conv1 = conv(self.X, 5, 5, 32, 1, 1, name = 'conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv2 = conv(pool1, 5, 5, 48, 1, 1, name = 'conv2')
    pool2 = max_pool(conv2, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv3 = conv(pool2, 5, 5, 64, 1, 1, name = 'conv3')
    pool3 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv4 = conv(pool3, 5, 5, 72, 1, 1, padding = 'VALID', name = 'conv4')

    # 5th Layer: Flatten -> FC (ReLu)
    flattened = tf.reshape(conv4, [-1, 72*4*4], name = 'flattened')
    fc1 = fc(flattened, 72*4*4, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')
    drop1 = tf.nn.dropout(relu1, 0.5, name='drop1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(drop1, 128, 128*self.NUM_CLASSES, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')
    drop2 = tf.nn.dropout(relu2, 0.5, name='drop2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(drop2, 128*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

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
    # relu2 = tf.nn.relu(fc2, name = 'relu2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(fc2, 128*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def vgg_create(self):
 
    # 1st Layer: Conv -> pooling
    conv1 = conv(self.X, 3, 3, 32, 1, 1, name = 'conv1')
    pool1 = max_pool(conv1, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv2 = conv(pool1, 3, 3, 48, 1, 1, name = 'conv2')
    conv3 = conv(conv2, 3, 3, 48, 1, 1, name = 'conv3')
    pool2 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv4 = conv(pool2, 3, 3, 64, 1, 1, name = 'conv4')
    conv5 = conv(conv4, 3, 3, 64, 1, 1, name = 'conv5')
    pool3 = max_pool(conv5, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv6 = conv(pool3, 3, 3, 128, 1, 1, padding = 'VALID', name = 'conv6')
    conv7 = conv(conv6, 3, 3, 128, 1, 1, padding = 'VALID', name = 'conv7')

    # 5th Layer: Flatten -> FC (ReLu)
    flattened = tf.reshape(conv7, [-1, 128*4*4], name = 'flattened')
    fc1 = fc(flattened, 128*4*4, 128, name='fc1')
    relu1 = tf.nn.relu(fc1, name = 'relu1')

    # 6th Layer: FC (ReLu)
    fc2 = fc(relu1, 128, 128*self.NUM_CLASSES, name='fc2')
    relu2 = tf.nn.relu(fc2, name = 'relu2')

    # 7th Layer: FC -> softmax
    self.fc3 = fc(relu2, 128*self.NUM_CLASSES, self.NUM_CLASSES, name='fc3')
    self.pre = tf.nn.softmax(self.fc3)

  def nin_create(self):

    # 1st Layer: Conv -> pooling
    conv1 = conv(self.X, 3, 3, 96, 1, 1, name = 'conv1')
    conv2 = conv(conv1, 1, 1, 96, 1, 1, name = 'conv2')
    conv3 = conv(conv2, 1, 1, 96, 1, 1, name = 'conv3')
    pool1 = max_pool(conv3, 2, 2, 2, 2, padding = 'VALID', name = 'pool1')

    # 2nd Layer: Conv -> pooling
    conv4 = conv(pool1, 3, 3, 128, 1, 1, name = 'conv4')
    conv5 = conv(conv4, 1, 1, 128, 1, 1, name = 'conv5')
    conv6 = conv(conv5, 1, 1, 128, 1, 1, name = 'conv6')
    pool2 = max_pool(conv6, 2, 2, 2, 2, padding = 'VALID', name = 'pool2')

    # 3rd Layer: Conv -> pooling
    conv7 = conv(pool2, 3, 3, 256, 1, 1, name = 'conv7')
    conv8 = conv(conv7, 1, 1, 256, 1, 1, name = 'conv8')
    conv9 = conv(conv8, 1, 1, 256, 1, 1, name = 'conv9')
    pool3 = max_pool(conv9, 2, 2, 2, 2, padding = 'VALID', name = 'pool3')

    # 4th Layer: Conv
    conv10 = conv(pool3, 3, 3, 512, 1, 1, name = 'conv10')
    conv11 = conv(conv10, 1, 1, 512, 1, 1, name = 'conv11')
    conv12 = conv(conv11, 1, 1, 512, 1, 1, name = 'conv12')
    conv13 = conv(conv12, 1, 1, self.NUM_CLASSES, 1, 1, name = 'conv13')
    avg_pool1 = avg_pool(conv13, 8, 8, 1, 1, padding = 'VALID', name = 'avg_pool1')

    # 5th Layer: Flatten -> FC (ReLu)
    self.result = tf.reshape(avg_pool1, [-1, 1*1*self.NUM_CLASSES], name = 'result')
    self.pre = tf.nn.softmax(self.result)

def sub_conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='VALID'):
     with tf.variable_scope(name) as scope:
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        filters= tf.get_variable('filters', shape=[5, 5, input_channels, num_filters], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(stddev = 2), trainable = True)
        biases = tf.get_variable(shape=[num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.1), trainable = True, name='biases')
        conv = convolve(x, filters)
        conv_ = tf.nn.bias_add(conv, biases)
        return conv_

def conv1_fuc(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='VALID'):
    with tf.variable_scope(name) as scope:
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        value = [-1, 2, -2, 2, -1, 2, -6, 8, -6, 2, -2, 8, -12, 8, -2, 2, -6, 8, -6, 2, -1, 2, -2, 2, -1]
        # value = [-1, 2, -1, 2, -4, 2, -1, 2, -1]
        raw_filter= tf.get_variable('raw_filter', shape=[5, 5], dtype=tf.float32, initializer=tf.constant_initializer(value))
        #x.initializer.run()
        y = tf.div(raw_filter, 12.0)
        repeat = tf.tile(y,[input_channels * num_filters,1])
        kernel = tf.reshape(repeat, [5, 5, input_channels, num_filters])
        conv = convolve(x, kernel)
        return conv

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
    with tf.variable_scope(name) as scope:
        input_channels = int(x.get_shape()[-1])
        convolve = lambda i, k: tf.nn.conv2d(i, k,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        kernel = tf.get_variable(shape = [filter_height, filter_width,
                            input_channels, num_filters], dtype=tf.float32, name='weights',
                            initializer=tf.contrib.layers.xavier_initializer(), trainable = True)
        biases = tf.get_variable(shape=[num_filters], dtype=tf.float32, initializer=tf.constant_initializer(0.1),
                            trainable = True, name='biases')
        conv = convolve(x, kernel)
        conv_ = tf.nn.bias_add(conv, biases)
        #conv_ = tf.nn.relu(conv_, name=name)
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
                         trainable=True, name='fc_biases', initializer=tf.constant_initializer(0.1))
        # Matrix multiply weights and inputs and add bias
        fc = tf.nn.xw_plus_b(x, weights, biases, name=name)
        return fc
