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
from ..network.factory import get_network
from ..network.Backbone_net import Backbone_net
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
            # seg_data = seg_data_scale if seg_data is None \
            #     else np.append(seg_data, seg_data_scale)
            process_one_scale = True
    return {'cls_data': cls_data,
            'reg_data': reg_data,
            'seg_data': seg_data}


class Backbone_line(AssemblyLine):
    def __init__(self):
        AssemblyLine.__init__(self, self.get_config(), tf.get_default_graph())
        self.batch_size = 8
        self.solo_batch_size = 8
        self.val_size = 2
        self.IMG_CHANEL = 3

    @staticmethod
    def get_config():
        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.allow_growth = True
        return config

    def artificial_check(self, X_mb, Y_mb, scale_table):
        strides = [128, 85.3333, 64, 32, 16, 8, 4]
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
                    seg_map_rgb = copy.deepcopy(seg_map[:, :, channel] * 255).astype(np.uint8)
                    seg_map_rgb = cv2.cvtColor(seg_map_rgb, cv2.COLOR_GRAY2BGR)
                cv2.imshow('seg map%d' % channel, seg_map_rgb)
            cv2.imshow('img_dyeing', dyeing_X)
            # classify check
            for scale in range(7):
                cls_map = Y_m['cls_mask'][scale][0]
                reg_map = Y_m['reg_mask'][scale][0]
                point_type_len = cls_map.shape[2] / 8
                color = tuple()
                for scale_type in range(cls_map.shape[2] / 2):
                    default_box_width = scale_table[scale][scale_type % point_type_len]
                    index = np.where(cls_map[:, :, scale_type * 2 + 1] > 0.5)
                    index = np.array(index).T
                    # index_img = index * int(256 / (2 ** scale)) + int(256 / (2 ** scale)) / 2
                    index_img = index * int(strides[scale]) + int(strides[scale] / 2)

                    if scale_type / point_type_len == 0:
                        color = (255, 0, 0)
                    elif scale_type / point_type_len == 1:
                        color = (0, 255, 0)
                    elif scale_type / point_type_len == 2:
                        color = (0, 0, 255)
                    elif scale_type / point_type_len == 3:
                        color = (255, 0, 255)
                    for orde, ind in enumerate(index_img):
                        # regression check
                        reg_val = reg_map[index[orde][0], index[orde][1], 4 * scale_type:4 * scale_type + 4]
                        Dx = reg_val[0] * default_box_width
                        Dy = reg_val[1] * default_box_width
                        Ss = int(np.exp(reg_val[2]) * default_box_width)
                        # Dy=0
                        # Dx=0
                        # Ss = default_box_width
                        ind[1] = ind[1] + Dx
                        ind[0] = ind[0] + Dy
                        # detection
                        cv2.circle(dyeing_X, (ind[1], ind[0]), 2, color, 2)
                        rect_lt = (ind[1] - Ss // 2, ind[0] - Ss // 2)
                        rect_rb = (ind[1] + Ss // 2, ind[0] + Ss // 2)
                        cv2.rectangle(dyeing_X, rect_lt, rect_rb, color, 1)

            cv2.imshow('img_dyeing', dyeing_X)
            cv2.waitKey()

    def structure_train_context(self):
        opti = tf.train.AdamOptimizer(0.0001)
        tower_grads = list()
        device_num = 4
        self.solo_batch_size = self.batch_size // device_num
        nets = list()
        test_loss = None
        for i in range(device_num):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU%d' % i):
                    net = get_network('CSTR', global_reuse=False if i == 0 else True)
                    nets.append(net)
                    loss_dict = net.structure_loss()
                    loss =loss_dict['cls loss'] + 0*loss_dict['reg loss'] + 0 * loss_dict['seg loss']
                    # loss = loss_dict['cls loss']
                    if i ==0:
                        test_loss = loss
                    grads = opti.compute_gradients(loss)
                    tower_grads.append(grads)

        grads = self.average_gradients(tower_grads)
        apply_gradinet_op = opti.apply_gradients(grads)
        # test_op = tf.train.AdamOptimizer(0.0001).minimize(test_loss)

        self.sess.run(tf.global_variables_initializer())
        vgg16_initializer = nets[0].vgg16_initializer
        vgg16_initializer(self.sess)

        # loss_dict_val = nets[0].structure_loss()

        offset = 0

        scale_table = [[184, 208, 232, 256],
                       [124, 136, 148, 160],
                       [88, 96, 104, 112],
                       [56, 64, 72, 80],
                       [36, 40, 44, 48],
                       [20, 24, 28, 32],
                       [4, 8, 6, 10, 12, 16]]

        merged = self.create_summary(nets[0].get_summary(), './data/logs/log_CSTR_2_cls_lr:0.0001_zero')
        for iter in range(90000):
            feed_dict_val = None
            if iter % 10 == 0:
                print('val testing...')
                feed_dict_val = dict()
                Y_val_mb, X_val_mb = get_sample_tensor('CPD', batch_size=[self.batch_size * 1000 + iter * self.val_size,
                                                                          self.batch_size * 1000 + iter * self.val_size \
                                                                          + self.val_size])

                # self.artificial_check(X_val_mb, Y_val_mb, scale_table)
                if X_val_mb is None:
                    continue
                actually_batch_size = X_val_mb.shape[0]

                Y_val_mb_flatten = flatten_concat(Y_val_mb)
                feed_dict_val[nets[0].X] = X_val_mb
                feed_dict_val[nets[0].Ycls] = Y_val_mb_flatten['cls_data']
                feed_dict_val[nets[0].Yreg] = Y_val_mb_flatten['reg_data']
                feed_dict_val[nets[0].Yseg] = Y_val_mb_flatten['seg_data']
                feed_dict_val[nets[0].on_train] = False
                feed_dict_val[nets[0].batch_size] = actually_batch_size

                los_cls, los_reg, los_seg, mg \
                    = self.sess.run([test_loss,
                                     test_loss,
                                     test_loss,
                                     merged],
                                    feed_dict=feed_dict_val)
                self.iter_num = iter
                self.write_summary(mg)
                print('iter step:%d total loss:%f cls loss:%f,reg loss:%f,seg loss:%f'
                      % (iter, (los_cls + los_reg + los_seg), los_cls, los_reg,
                         los_seg))

            # training scope
            print('opti iter%d...' % iter)
            t_iter_start = time.time()
            stretch = self.batch_size
            while True:
                Y_train_mb, X_train_mb = get_sample_tensor('CPD', batch_size=[iter * self.batch_size + offset,
                                                                              iter * self.batch_size + stretch + offset])
                # break
                if X_train_mb is None:
                    continue
                actually_batch_size = X_train_mb.shape[0]
                if actually_batch_size < self.batch_size:
                    stretch += self.batch_size - actually_batch_size
                    print('Error!!!!!!!!!!!!!')
                    continue
                break

            feed_dict = dict()
            # self.artificial_check(X_train_mb,Y_train_mb,scale_table)
            for device_id in range(device_num):
    # test = init_template_f_in(scale_table, [128, 85.3333, 64, 32, 16, 8, 4])
                Y_tain_mb_flatten = flatten_concat(Y_train_mb[device_id * self.solo_batch_size:
                                                              (device_id + 1) * self.solo_batch_size])
                # print(np.sum(Y_tain_mb_flatten['cls_data'][:,1]))
                feed_dict[nets[device_id].X] = X_train_mb[device_id * self.solo_batch_size:
                                                          (device_id + 1) * self.solo_batch_size]
                feed_dict[nets[device_id].Ycls] = Y_tain_mb_flatten['cls_data']
                feed_dict[nets[device_id].Yreg] = Y_tain_mb_flatten['reg_data']
                feed_dict[nets[device_id].Yseg] = Y_tain_mb_flatten['seg_data']
                feed_dict[nets[device_id].on_train] = True
                feed_dict[nets[device_id].batch_size] = self.solo_batch_size

            t_iter_pre_opti = time.time()
            _, train_loss = self.sess.run([apply_gradinet_op, test_loss], feed_dict=feed_dict)
            print('optimizer update successful, total spend:%fs and opti spend:%fs this time...'
                  % ((time.time() - t_iter_start), (time.time() - t_iter_pre_opti)))
            print('optimizer update successful, iter:%d loss:%f'
                  % (iter, train_loss))
            a = 1
        self.close_summary_writer()

    # scale_table = [[256, 232, 208, 184],
    #                [124, 136, 148, 160],
    #                [88, 96, 104, 112],
    #                [56, 64, 72, 80],
    #                [36, 40, 44, 48],
    #                [20, 24, 28, 32],
    #                [4, 8, 6, 10, 12, 16]]
    #
    # saver = self.get_saver(total_vars)
    # merged = self.create_summary('./data/logs/log_CSTR')
    # for iter in range(50000):
    #     if iter % 50 == 0:
    #         Y_val_mb, X_val_mb = get_sample_tensor('CPD', batch_size=[self.batch_size * 1000 + iter * self.val_size,
    #                                                                   self.batch_size * 1000 + iter * self.val_size \
    #                                                                   + self.val_size])
    #         if X_val_mb is None:
    #             continue
    #         actually_batch_size = X_val_mb.shape[0]
    #         print('val testing...')
    #         Y_val_mb_flatten = flatten_concat(Y_val_mb)
    #         # self.artificial_check(X_val_mb, Y_val_mb, scale_table)
    #         los_cls, los_reg, los_seg, mg, OHEM_data, OHEM_data_cls \
    #             = self.sess.run([loss_dict['cls loss'],
    #                              loss_dict['reg loss'],
    #                              loss_dict['seg loss'],
    #                              merged,
    #                              OHEM,
    #                              OHEM_cls],
    #                             feed_dict={self.network.X: X_val_mb,
    #                                        self.network.Ycls: Y_val_mb_flatten['cls_data'],
    #                                        self.network.Yreg: Y_val_mb_flatten['reg_data'],
    #                                        self.network.Yseg: Y_val_mb_flatten['seg_data'],
    #                                        self.network.on_train: False,
    #                                        self.network.batch_size: actually_batch_size
    #                                        })
    #         print('iter step:%d total loss:%f cls loss:%f,reg loss:%f,seg loss:%f'
    #               % (iter, (los_cls + los_reg + los_seg * self.network.lamd), los_cls, los_reg,
    #                  los_seg * self.network.lamd))
    #         self.iter_num = iter
    #         self.write_summary(mg)
    #
    #         # t1 = time.time()
    #         # self.sess.run([self.network.get_pred()[2],
    #         #                self.network.get_pred()[0]['f11_CPD'],
    #         #                self.network.get_pred()[1]['f11_CPD'],
    #         #                self.network.get_pred()[0]['f10_CPD'],
    #         #                self.network.get_pred()[1]['f10_CPD'],
    #         #                self.network.get_pred()[0]['f9_CPD'],
    #         #                self.network.get_pred()[1]['f9_CPD'],
    #         #                self.network.get_pred()[0]['f8_CPD'],
    #         #                self.network.get_pred()[1]['f8_CPD'],
    #         #                self.network.get_pred()[0]['f7_CPD'],
    #         #                self.network.get_pred()[1]['f7_CPD'],
    #         #                self.network.get_pred()[0]['f4_CPD'],
    #         #                self.network.get_pred()[1]['f4_CPD'],
    #         #                self.network.get_pred()[0]['f3_CPD'],
    #         #                self.network.get_pred()[1]['f3_CPD']],
    #         #               feed_dict={self.network.X: X_val_mb,
    #         #                          self.network.on_train: False,
    #         #                          self.network.batch_size: self.val_size
    #         #                          })
    #         # print('spend %f' % (time.time() - t1))
    #
    #     print('opti iter%d...' % iter)
    #     Y_train_mb, X_train_mb = get_sample_tensor('CPD', batch_size=[iter * self.batch_size,
    #                                                                   iter * self.batch_size + self.batch_size])
    #     if X_train_mb is None:
    #         continue
    #     actually_batch_size = X_train_mb.shape[0]
    #     Y_tain_mb_flatten = flatten_concat(Y_train_mb)
    #
    #     self.sess.run(opti_dict, feed_dict={self.network.X: X_train_mb,
    #                                         self.network.Ycls: Y_tain_mb_flatten['cls_data'],
    #                                         self.network.Yreg: Y_tain_mb_flatten['reg_data'],
    #                                         self.network.Yseg: Y_tain_mb_flatten['seg_data'],
    #                                         self.network.on_train: True,
    #                                         self.network.batch_size: actually_batch_size
    #                                         })
