import os

import numpy as np
import tensorflow as tf

from datetime import datetime
from alexnet_model import AlexNet
from read_image import DataGenerator

train_file = 'H:\\shuqian\\amel\\image\\train\\'
val_file = 'H:\\shuqian\\amel\\image\\val\\'

# Learning params
learning_rate = 1e-5
num_epochs = 4000
batch_size = 20

# Network params
dropout_rate = 0.5
num_classes = 5

# How often we want to write the tf.summary data to disk
display_step = 300

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "H:\\shuqian\\amel\\first\\graph"
checkpoint_path = "H:\\shuqian\\amel\\first\\checkpoints"

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 252, 252, 3], name = 'x_images')
y = tf.placeholder(tf.float32, [batch_size, num_classes], name = 'y_labels')
keep_prob = tf.placeholder(tf.float32, name = 'prob_rate')

# Initialize model
model = AlexNet(x, keep_prob, num_classes)

# Link variable to model output
score = model.fc3
pre = model.pre

# List of trainable variables of the layers we want to train
var_list = tf.trainable_variables()

# Op for calculating the loss
with tf.name_scope("cross_ent"):
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = score, labels = y))

# Train op
with tf.name_scope("train"):
  # Get gradients of all trainable variables
  gradients = tf.gradients(loss, var_list)
  gradients = list(zip(gradients, var_list))

  # Create optimizer and apply gradient descent to the trainable variables
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
  #train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
with tf.name_scope("accuracy"):
  correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add the accuracy to the summary
tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
train_writer = tf.summary.FileWriter("H:\\shuqian\\amel\\first\\graph\\train")
test_writer = tf.summary.FileWriter("H:\\shuqian\\amel\\first\\graph\\test")

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the training and validation set
train_generator = DataGenerator(train_file, 'train', batch_size, num_classes)
val_generator = DataGenerator(val_file, 'val', batch_size, num_classes)

# Get the number of training/validation steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size / batch_size).astype(np.int16)

with tf.device('/gpu:0'):
    # Start Tensorflow session
    with tf.Session() as sess:

        # Initialize all variables
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Add the model graph to TensorBoard
        train_writer.add_graph(sess.graph)

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                        filewriter_path))

        # Loop over number of epochs
        for epoch in range(num_epochs):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))
            if epoch % 20 == 0:
                learning_rate = learning_rate / 2
            step = 1
            train_acc = 0.
            train_count = 0
            while step < train_batches_per_epoch:
                # And run the training op
                images, labels = sess.run([train_generator.images, train_generator.labels])
                op, acc = sess.run([optimizer, accuracy], feed_dict={x: images,
                                              y: labels,
                                              keep_prob: dropout_rate})
                train_acc += acc
                train_count += 1
                # Generate summary with the current batch of data and write to file
                if step % display_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: images,
                                                        y: labels,
                                                        keep_prob: 1.})
                    train_writer.add_summary(s, epoch*train_batches_per_epoch + step)
                step += 1
            train_acc /= train_count
            print("{} Train Accuracy = {:.4f}".format(datetime.now(), train_acc))

            # Validate the model on the entire validation set
            print("{} Start validation".format(datetime.now()))
            test_acc = 0.
            test_count = 0
            step = 1
            while step < val_batches_per_epoch:
                val_images, val_labels = sess.run([val_generator.images, val_generator.labels])
                acc = sess.run(accuracy, feed_dict={x: val_images,
                                                    y: val_labels,
                                                    keep_prob: 1.})
                test_acc += acc
                test_count += 1
                # Generate summary with the current batch of data and write to file
                if step % 100 == 0:
                    s = sess.run(merged_summary, feed_dict={x: val_images,
                                                        y: val_labels,
                                                        keep_prob: 1.})
                    test_writer.add_summary(s, epoch*val_batches_per_epoch + step)
                step += 1
            test_acc /= test_count
            print("{} Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

            print("{} Saving checkpoint of model...".format(datetime.now()))

            #save checkpoint of the model
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
        train_writer.close()
        test_writer.close()
        coord.request_stop()
        coord.join(threads)
