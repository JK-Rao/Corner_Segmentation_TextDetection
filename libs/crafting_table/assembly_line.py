# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      assembly.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
import os
import cv2


class AssemblyLine(object):
    def __init__(self, config, graph, network):
        self.sess = tf.Session(graph=graph, config=config)
        self.network = network
        self.iter_num = 0
        self.summary_writer = None

        self.sess.as_default()

    def create_summary(self, log_path):
        summ_dict = self.network.get_summary()
        merged = None
        for key in summ_dict:
            tf.summary.scalar(key, summ_dict[key])
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(log_path, self.sess.graph)
            self.summary_writer = writer
        return merged

    def write_summary(self, mg):
        self.summary_writer.add_summary(mg, self.iter_num)

    def close_summary_writer(self):
        if self.summary_writer is None:
            print('Error in close writer...')
        else:
            self.summary_writer.close()

    def structure_train_context(self):
        raise NotImplementedError('Must be subclassed.')

    def restore_test_context(self):
        raise NotImplementedError('Must be subclassed.')

    def get_saver(self, vars, max_to_keep=100):
        return tf.train.Saver(vars, max_to_keep=max_to_keep)

    def save_model(self, saver, save_path_name, write_meta_graph=True):
        self.sess.as_default()
        saver.save(self.sess, save_path_name, write_meta_graph=write_meta_graph)

    def restore_model(self, model_name, parameter_name):
        saver = tf.train.import_meta_graph(model_name)
        graph = tf.get_default_graph()
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, parameter_name)
        return graph
