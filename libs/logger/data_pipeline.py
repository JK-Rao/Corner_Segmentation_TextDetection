# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      data_pipline.py
# @Software  PyCharm
# @Author    JK.Rao

import numpy as np
import tensorflow as tf
import cv2
from os.path import join
import os


class TfWriter(object):
    def __init__(self, fixed_height=None, fixed_width=None, fixed_chanel=None):
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self.fixed_chanel = fixed_chanel

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def write_tfrecords(self, im, wirter):
        if not self.fixed_height is None:
            im = cv2.resize(im, (self.fixed_width, self.fixed_height))
        im_raw = im.astype(np.uint8).tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'im_raw': self._bytes_feature(im_raw)}))
        wirter.write(example.SerializeToString())


class TfReader(object):
    def __init__(self, fixed_height=None, fixed_width=None, fixed_chanel=None):
        self.fixed_height = fixed_height
        self.fixed_width = fixed_width
        self.fixed_chanel = fixed_chanel

    def read_and_decode(self, filename_queue):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example, features={
            'im_raw': tf.FixedLenFeature([], tf.string)
        })
        image = tf.decode_raw(features['im_raw'], tf.uint8)
        if not self.fixed_width is None:
            image = tf.reshape(image, [self.fixed_height, self.fixed_width, self.fixed_chanel])

        return image

    def load_sample(self, file_path, batch_size, num_epochs):
        if not num_epochs:
            num_epochs = None
        filename_queue = tf.train.string_input_producer([file_path], num_epochs=num_epochs)
        image = self.read_and_decode(filename_queue)
        images = tf.train.shuffle_batch([image],
                                        batch_size=batch_size,
                                        num_threads=64,
                                        capacity=1000 + 3 * batch_size,
                                        min_after_dequeue=1000
                                        )
        return images


class DataPipeline(object):
    def __init__(self, data_type, propose, root_path, train_dir='train', val_dir='val', test_dir='test'):
        self.data_type = data_type
        self.propose = propose
        self.root_path = root_path
        self.xxx_dir = {'train': train_dir, 'val': val_dir, 'test': test_dir}

    def set_dataType(self, data_type):
        self.data_type = data_type

    def set_propose(self, propose):
        self.propose = propose

    def imgs2tfrecords(self, object_path, object_name, recorder=TfWriter(), flag=cv2.IMREAD_COLOR):
        writer = tf.python_io.TFRecordWriter(join(object_path, object_name + '.tfrecords'))
        file_path = join(self.root_path, self.xxx_dir[self.propose])
        file_lines = os.listdir(file_path)
        for index, file_name in enumerate(file_lines):
            im = cv2.imread(join(file_path, file_name), flags=flag)
            if im is None:
                print('Error in reading %s...' % file_name)
                continue
            recorder.write_tfrecords(im, writer)
            if index % 1000 == 0:
                print('processing %d/%d...' % (index, len(file_lines)))
        print('writing end...')

    def tfrecords2imgs_tensor(self, sess, tf_data_name, batch_size, recorder=TfReader(), flag=cv2.IMREAD_COLOR):
        tfdata_path = join(self.root_path, self.xxx_dir[self.propose])
        file_path_name = join(tfdata_path, tf_data_name + self.data_type)
        if not os.path.exists(file_path_name):
            raise IOError('No such file: \'%s\'' % file_path_name)
        images_tensor = recorder.load_sample(file_path_name, batch_size, None)
        return images_tensor

    @staticmethod
    def tensor2data(sess,tensor, normal=True):
        data = sess.run(tensor)
        data = data / 255. - 0.5 if normal else data
        return data
