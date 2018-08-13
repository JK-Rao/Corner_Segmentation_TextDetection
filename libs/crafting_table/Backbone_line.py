# _*_ coding: utf-8 _*_
# @Time      18-8-9 下午3:00
# @File      Backbone_line.py
# @Software  PyCharm
# @Author    JK.Rao

from .assembly_line import AssemblyLine
from ..logger.factory import get_sample_tensor
import tensorflow as tf
import numpy as np
import time
import cv2
import copy
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
        self.batch_size = 8
        self.val_size = 1
        self.IMG_CHANEL = self.network.IM_CHANEL

    @staticmethod
    def get_config():
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        return config

    def artificial_check(self, X_mb, Y_mb, scale_table):
        # show imgs
        for img_index in range(X_mb.shape[0]):
            dyeing_X = copy.deepcopy(X_mb[img_index])
            cv2.imshow('img', X_mb[img_index])
            Y_m = Y_mb[img_index]
            seg_map = Y_m['seg_mask'][0][0]
            # segment check
            for channel in range(4):
                if channel < 3:
                    dyeing_X[:, :, channel] += ((255 - dyeing_X[:, :, channel]) * (seg_map[:, :, channel])).astype(
                        np.uint8)
                    seg_map_rgb = copy.deepcopy(seg_map[:, :, channel] * 255).astype(np.uint8)
                    seg_map_rgb = cv2.cvtColor(seg_map_rgb, cv2.COLOR_GRAY2BGR)
                else:
                    dyeing_X[:, :, channel - 1] += (
                            (255 - dyeing_X[:, :, channel - 1]) * (seg_map[:, :, channel])).astype(np.uint8)
                    dyeing_X[:, :, 0] += ((255 - dyeing_X[:, :, 0]) * (seg_map[:, :, channel])).astype(np.uint8)
                    seg_map_rgb=copy.deepcopy(seg_map[:,:,channel]*255).astype(np.uint8)
                    seg_map_rgb=cv2.cvtColor(seg_map_rgb,cv2.COLOR_GRAY2BGR)
                cv2.imshow('seg map%d' % channel, seg_map_rgb)
            cv2.imshow('img_dyeing', dyeing_X)
            # classify check
            for scale in range(7):
                cls_map = Y_m['cls_mask'][scale][0]
                reg_map = Y_m['reg_mask'][scale][0]
                point_type_len = cls_map.shape[2] / 8
                color = tuple()
                for scale_type in range(cls_map.shape[2] / 2):
                    default_box_width=scale_table[scale][scale_type % point_type_len]
                    index = np.where(cls_map[:, :, scale_type * 2 + 1] > 0.5)
                    index = np.array(index).T
                    index_img = index * int(256 / (2 ** scale))+int(256 / (2 ** scale))/2

                    if scale_type / point_type_len == 0:
                        color = (0, 0, 255)
                    elif scale_type / point_type_len == 1:
                        color = (0, 255, 0)
                    elif scale_type / point_type_len == 2:
                        color = (255, 0, 0)
                    elif scale_type / point_type_len == 3:
                        color = (255, 0, 255)
                    for orde, ind in enumerate(index_img):
                        # regression check
                        reg_val = reg_map[index[orde][0], index[orde][1], 4 * scale_type:4 * scale_type + 4]
                        Dx = reg_val[0] * default_box_width
                        Dy = reg_val[1] * default_box_width
                        Ds = np.exp(reg_val[2]) * default_box_width
                        ind[1] = ind[1] + Dx
                        ind[0] = ind[0] + Dy
                        # detection
                        print(ind[1],ind[0])
                        cv2.circle(dyeing_X, (ind[1], ind[0]), 2, color, 2)

            cv2.imshow('img_dyeing', dyeing_X)
            cv2.waitKey()

    def structure_train_context(self):
        loss_dict = self.network.structure_loss()
        opti_dict = self.network.define_optimizer(loss_dict)
        self.sess.run(tf.global_variables_initializer())
        init_vgg16 = self.network.vgg16_initializer
        init_vgg16(self.sess)
        scale_table = [[256, 232, 208, 184],
                       [124, 136, 148, 160],
                       [88, 96, 104, 112],
                       [56, 64, 72, 80],
                       [36, 40, 44, 48],
                       [20, 24, 28, 32],
                       [4, 8, 6, 10, 12, 16]]
        for iter in range(1):
            if iter % 10 == 0:
                Y_val_mb, X_val_mb = get_sample_tensor('CPD', batch_size=[self.batch_size * 1000 + iter * self.val_size,
                                                                          self.batch_size * 1000 + iter * self.val_size \
                                                                          + self.val_size])
                print('val')
                Y_val_mb_flatten = flatten_concat(Y_val_mb)
                self.artificial_check(X_val_mb, Y_val_mb,scale_table)
                los_cls, los_reg, los_seg \
                    = self.sess.run([tf.reduce_mean(loss_dict['cls loss']),
                                     tf.reduce_mean(loss_dict['reg loss']),
                                     tf.reduce_mean(loss_dict['seg loss'])],
                                    feed_dict={self.network.X: X_val_mb,
                                               self.network.Ycls: Y_val_mb_flatten['cls_data'],
                                               self.network.Yreg: Y_val_mb_flatten['reg_data'],
                                               self.network.Yseg: Y_val_mb_flatten['seg_data'],
                                               self.network.on_train: False,
                                               self.network.batch_size: self.val_size
                                               })
                print('iter step:%d cls loss:%f,reg loss:%f,seg loss:%f' % (iter, los_cls, los_reg, los_seg))

                t1 = time.time()
                self.sess.run([self.network.get_pred()[2],
                               self.network.get_pred()[0]['f11_CPD'],
                               self.network.get_pred()[1]['f11_CPD'],
                               self.network.get_pred()[0]['f10_CPD'],
                               self.network.get_pred()[1]['f10_CPD'],
                               self.network.get_pred()[0]['f9_CPD'],
                               self.network.get_pred()[1]['f9_CPD'],
                               self.network.get_pred()[0]['f8_CPD'],
                               self.network.get_pred()[1]['f8_CPD'],
                               self.network.get_pred()[0]['f7_CPD'],
                               self.network.get_pred()[1]['f7_CPD'],
                               self.network.get_pred()[0]['f4_CPD'],
                               self.network.get_pred()[1]['f4_CPD'],
                               self.network.get_pred()[0]['f3_CPD'],
                               self.network.get_pred()[1]['f3_CPD']],
                              feed_dict={self.network.X: X_val_mb,
                                         self.network.on_train: False,
                                         self.network.batch_size: self.val_size
                                         })
                print('spend %f' % (time.time() - t1))

            Y_train_mb, X_train_mb = get_sample_tensor('CPD', batch_size=[iter * self.batch_size,
                                                                          iter * self.batch_size + self.batch_size])
            Y_tain_mb_flatten = flatten_concat(Y_train_mb)

            self.sess.run(opti_dict, feed_dict={self.network.X: X_train_mb,
                                                self.network.Ycls: Y_tain_mb_flatten['cls_data'],
                                                self.network.Yreg: Y_tain_mb_flatten['reg_data'],
                                                self.network.Yseg: Y_tain_mb_flatten['seg_data'],
                                                self.network.on_train: True,
                                                self.network.batch_size: self.batch_size
                                                })
