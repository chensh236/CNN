# -*- coding: UTF-8 -*-
import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class DataGenerator:
    def __init__(self, filepath, mode, batch_size, num_classes):
        self.write_to_tfrecord(filepath, mode)
        self.read_from_tfrecord(batch_size, num_classes, mode)


    def write_to_tfrecord(self, filepath, mode):
        # filepath ='H:\\shuqian\\image\\train\\'
        # 设定类别
        classes={'Kodak_M1063':0,
                 'Casio_EX-Z150':1,
                 'Nikon_CoolPixS710':2}
        #存放图片个数
        bestnum = 1000
        #第几个图片
        num = 0
        #第几个TFRecord文件
        recordfilenum = 0
        #tfrecords格式文件名
        tf_filepath = 'H:\\shuqian\\tfrecord\\'
        if mode == 'train':
            ftrecordfilename = ('train_image.tfrecords_%.2d' % recordfilenum)
        else:
            ftrecordfilename = ('test_image.tfrecords_%.2d' % recordfilenum)
        writer= tf.python_io.TFRecordWriter(tf_filepath+ftrecordfilename)

        for index, name in enumerate(classes):
            class_path = filepath + name + '\\'
            for img_name in os.listdir(class_path):
                num = num + 1
                img_path=class_path+img_name #每一个图片的地址

                # 写入下一个文件
                if num > bestnum:
                    num = 1
                    recordfilenum = recordfilenum + 1
                    #tfrecords格式文件名
                    if mode == 'train':
                        ftrecordfilename = ('train_image.tfrecords_%.2d' % recordfilenum)
                    else:
                        ftrecordfilename = ('test_image.tfrecords_%.2d' % recordfilenum)
                    writer= tf.python_io.TFRecordWriter(tf_filepath+ftrecordfilename)

                # 加载文件
                img=Image.open(img_path)
                img_raw=img.tobytes()#将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    #value=[index]决定了图片数据的类型label
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                })) #example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  #序列化为字符串

        writer.close()
        self.data_size = recordfilenum*bestnum + num

    def read_from_tfrecord(self, batch_size, num_classes, mode):
        if mode == 'train':
            files = tf.train.match_filenames_once('H:\\shuqian\\tfrecord\\train_image.tfrecords*')
        else:
            files = tf.train.match_filenames_once('H:\\shuqian\\tfrecord\\test_image.tfrecords*')
        filename_queue = tf.train.string_input_producer(files, shuffle=True) #读入流中
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
        #tf.decode_raw可以将字符串解析成图像对应的像素数组
        image = tf.decode_raw(features['img_raw'], tf.uint8)
        image = tf.reshape(image, [252,252,3])
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        label = tf.cast(features['label'], tf.int32)
        if mode == 'train':
            example_queue = tf.RandomShuffleQueue(
                # 队列容量
                capacity = 16 * batch_size,
                # 队列数据的最小容许量
                min_after_dequeue = 8 * batch_size,
                dtypes = [tf.float32, tf.int32],
                # 图片数据尺寸，标签尺寸
                shapes = [[252, 252, 3], ()])
            # 读线程的数量
            num_threads = 16
        else:
            example_queue = tf.FIFOQueue(
                3 * batch_size,
                dtypes=[tf.float32, tf.int32],
                shapes=[[252, 252, 3], ()])
            # 读线程的数量
            num_threads = 1
        # 数据入队操作
        example_enqueue_op = example_queue.enqueue([image, label])
        # 队列执行器
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
          example_queue, [example_enqueue_op] * num_threads))

        # 数据出队操作，从队列读取Batch数据
        images, labels = example_queue.dequeue_many(batch_size)
        # 将标签数据由稀疏格式转换成稠密格式
        # [ 2,       [[0,1,0,0,0]
        #   4,        [0,0,0,1,0]
        #   3,   -->  [0,0,1,0,0]
        #   5,        [0,0,0,0,1]
        #   1 ]       [1,0,0,0,0]]
        labels = tf.reshape(labels, [batch_size, 1])
        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
        labels = tf.sparse_to_dense(
                      tf.concat(values=[indices, labels], axis=1),
                      [batch_size, num_classes], 1.0, 0.0)

        #检测数据维度
        assert len(images.get_shape()) == 4
        assert images.get_shape()[0] == batch_size
        assert images.get_shape()[-1] == 3
        assert len(labels.get_shape()) == 2
        assert labels.get_shape()[0] == batch_size
        assert labels.get_shape()[1] == num_classes

        # 添加图片总结
        tf.summary.image('images', images)
        self.images = images
        self.labels = labels
