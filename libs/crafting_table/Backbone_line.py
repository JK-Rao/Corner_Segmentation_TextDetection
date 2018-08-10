# _*_ coding: utf-8 _*_
# @Time      18-8-9 下午3:00
# @File      Backbone_line.py
# @Software  PyCharm
# @Author    JK.Rao

from .assembly_line import AssemblyLine
from ..logger.factory import get_sample_tensor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from os.path import join
import os


def flatten_concat(stand_data):
    cls_data = None
    reg_data = None
    seg_data = None
    process_one_scale = False
    for i in range(7):
        cls_data_scale = None
        reg_data_scale = None
        seg_data_scale = None
        for batch_data in stand_data:
            cls_data_scale = batch_data['cls_mask'][i] if cls_data_scale is None \
                else np.append(cls_data_scale, batch_data['cls_mask'][i], axis=0)
            reg_data_scale = batch_data['reg_mask'][i] if reg_data_scale is None \
                else np.append(reg_data_scale, batch_data['reg_mask'][i], axis=0)
            if not process_one_scale:
                seg_data_scale = batch_data['seg_mask'][0] if seg_data_scale is None \
                    else np.append(seg_data_scale, batch_data['seg_mask'][0], axis=0)

        cls_data = np.reshape(cls_data_scale, [-1, 2]) if cls_data is None \
            else np.append(cls_data, np.reshape(cls_data_scale, [-1, 2]), axis=0)
        reg_data = np.reshape(reg_data_scale, [-1, 4]) if reg_data is None \
            else np.append(reg_data, np.reshape(reg_data_scale, [-1, 4]), axis=0)
        if not process_one_scale:
            seg_data = np.reshape(seg_data_scale, [-1, 1]) if seg_data is None \
                else np.append(seg_data, np.reshape(seg_data_scale, [-1, 1]), axis=0)
        process_one_scale = True
    return {'cls_data': cls_data,
            'reg_data': reg_data,
            'seg_data': seg_data}


class Backbone_line(AssemblyLine):
    def __init__(self, network):
        AssemblyLine.__init__(self, self.get_config(), network.get_graph(), network)
        self.batch_size = 24
        self.val_size = 24
        self.IMG_CHANEL = self.network.IMG_CHANEL

    @staticmethod
    def get_config():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def structure_train_context(self):
        loss_dict = self.network.structure_loss()
        opti_dict = self.network.define_optimizer(loss_dict)
        self.sess.run(tf.global_variables_initializer())
        for iter in range(1000):
            Y_train_mb, X_train_mb = get_sample_tensor('CPD', batch_size=[iter * 24, iter * 24 + 24])
            Y_tain_mb_flatten = flatten_concat(Y_train_mb)

            self.sess.run(opti_dict, feed_dict={self.network.X: X_train_mb,
                                                self.network.Ycls: Y_tain_mb_flatten['cls_data'],
                                                self.network.Yreg: Y_tain_mb_flatten['reg_data'],
                                                self.network.Yseg: Y_tain_mb_flatten['seg_data'],
                                                self.network.on_train: True,
                                                self.network.batch_size: 24
                                                })
            if iter % 10 == 0:
                Y_val_mb, X_val_mb = get_sample_tensor('CPD', batch_size=[24 * 1000 + iter * 10,
                                                                          24 * 1000 + iter * 10 + 10])
                Y_val_mb_flatten = flatten_concat(Y_train_mb)
                los_cls, los_reg, los_seg \
                    = self.sess.run(list(loss_dict.values), feed_dict={self.network.X: X_val_mb,
                                                                       self.network.Ycls: Y_val_mb_flatten['cls_data'],
                                                                       self.network.Yreg: Y_val_mb_flatten['reg_data'],
                                                                       self.network.Yseg: Y_val_mb_flatten['seg_data'],
                                                                       self.network.on_train: False,
                                                                       self.network.batch_size: 10
                                                                       })
                print('iter step:%d cls loss:%f,reg loss:%f,seg loss:%f' % (iter, los_cls, los_reg, los_seg))