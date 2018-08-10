# _*_ coding: utf-8 _*_
# @Time      18-8-3 上午10:47
# @File      Backbone_net.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
from .network import Network
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import tensorflow.contrib.layers as layers
import os
import numpy as np


class Backbone_net(Network):
    def __init__(self):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        Network.__init__(self, 'backbone')
        self.IM_HEIGHT = 512
        self.IM_WIDTH = 512
        self.IM_CHANEL = 3
        self.global_step = tf.Variable(0, trainable=False)
        self.X = tf.placeholder(tf.float32, shape=[None, self.IM_HEIGHT, self.IM_WIDTH, self.IM_CHANEL], name='X')
        self.Ycls = tf.placeholder(tf.float32, shape=[None, 2], name='Ycls')
        self.Yreg = tf.placeholder(tf.float32, shape=[None, 4], name='Yreg')
        self.Yseg = tf.placeholder(tf.float32, shape=[None, 1], name='Yseg')

        self.on_train = tf.placeholder(tf.bool, [], name='on_train')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.detect_dict = {}
        self.off_dict = {}
        self.seg = None

        self.vgg16_variables = None
        self.vgg16_initializer = None

        self.setup(self.X, 'backbone')

    def setup(self, x, scope_name, reuse=False):
        conv5_3 = self.load_vgg_model()
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            self.feed(conv5_3, 'abandon tensor') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale5_3', 'offset5_3', 'mean5_3', 'var5_3') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 1024, 3, 3, 2, 2, 'conv6_W', 'conv6_b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale6', 'offset6', 'mean6', 'var6') \
                .relu('save tensor') \
                .conv2d('abandon tensor', 1024, 1, 1, 2, 2, 'conv7_W', 'conv7_b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale7', 'offset7', 'mean7', 'var7') \
                .relu('save tensor') \
                .conv2d('abandon tensor', 256, 1, 1, 1, 1, 'conv8_1W', 'conv8_1b') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 512, 3, 3, 2, 2, 'conv8_2W', 'conv8_2b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale8', 'offset8', 'mean8', 'var8') \
                .relu('save tensor') \
                .conv2d('abandon tensor', 128, 1, 1, 1, 1, 'conv9_1W', 'conv9_1b') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 256, 3, 3, 2, 2, 'conv9_2W', 'conv9_2b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale9', 'offset9', 'mean9', 'var9') \
                .relu('save tensor') \
                .conv2d('abandon tensor', 128, 1, 1, 1, 1, 'conv10_1W', 'conv10_1b') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 256, 3, 3, 2, 2, 'conv10_2W', 'conv10_2b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale10', 'offset10', 'mean10', 'var10') \
                .relu('save tensor')

            f11 = self.layer_tensor_demand()
            f10 = self.deconvolution_model([self.batch_size, 4, 4, 256], f11, self.layers[5], 'deconv_f10')
            f9 = self.deconvolution_model([self.batch_size, 8, 8, 256], f10, self.layers[4], 'deconv_f9')
            f8 = self.deconvolution_model([self.batch_size, 16, 16, 256], f9, self.layers[3], 'deconv_f8')
            f7 = self.deconvolution_model([self.batch_size, 32, 32, 256], f8, self.layers[2], 'deconv_f7')
            f4 = self.deconvolution_model([self.batch_size, 64, 64, 256], f7, self.layers[1], 'deconv_f4')
            f3 = self.deconvolution_model([self.batch_size, 128, 128, 256], f4, self.layers[0], 'deconv_f3')

            f_mix = list()
            f_mix.append({'tensor': f11, 'num': 4, 'name': 'f11_CPD'})
            f_mix.append({'tensor': f10, 'num': 4, 'name': 'f10_CPD'})
            f_mix.append({'tensor': f9, 'num': 4, 'name': 'f9_CPD'})
            f_mix.append({'tensor': f8, 'num': 4, 'name': 'f8_CPD'})
            f_mix.append({'tensor': f7, 'num': 4, 'name': 'f7_CPD'})
            f_mix.append({'tensor': f4, 'num': 4, 'name': 'f4_CPD'})
            f_mix.append({'tensor': f3, 'num': 6, 'name': 'f3_CPD'})
            self.setup_corner_point_dect(f_mix)
            self.setup_position_sen_seg(f_mix)

    def flatten_tensor(self, tensor):
        return tf.reshape(tensor, [-1, tensor.get_shape().as_list()[-1]])

    def structure_loss(self):
        self.feed(self.detect_dict['f11_CPD'], 'flatten tensor x2') \
            .concat_tensor(self.detect_dict['f10_CPD'], 2) \
            .concat_tensor(self.detect_dict['f9_CPD'], 2) \
            .concat_tensor(self.detect_dict['f8_CPD'], 2) \
            .concat_tensor(self.detect_dict['f7_CPD'], 2) \
            .concat_tensor(self.detect_dict['f4_CPD'], 2) \
            .concat_tensor(self.detect_dict['f3_CPD'], 2)
        flatten_pred_cls = self.pre_process_tensor
        self.feed(self.off_dict['f11_CPD'], 'flatten tensor x4') \
            .concat_tensor(self.off_dict['f10_CPD'], 4) \
            .concat_tensor(self.off_dict['f9_CPD'], 4) \
            .concat_tensor(self.off_dict['f8_CPD'], 4) \
            .concat_tensor(self.off_dict['f7_CPD'], 4) \
            .concat_tensor(self.off_dict['f4_CPD'], 4) \
            .concat_tensor(self.off_dict['f3_CPD'], 4)
        flatten_pred_reg = self.pre_process_tensor
        self.feed(self.seg, 'flatten tensor x1')
        flatten_pred_seg = self.pre_process_tensor

        OHEM_mask = self.Ycls[:, 1] > 1
        OHEM_mask = tf.logical_or(OHEM_mask, self.Ycls[:, 1] >= 1)
        OHEM_mask = tf.reshape(tf.cast(OHEM_mask, dtype=tf.int32), shape=[-1, 1])
        pos_num = tf.reduce_sum(OHEM_mask)
        neg_num = pos_num * 3
        val, index = tf.nn.top_k(flatten_pred_cls[:, 1], k=neg_num)
        cls_pos = tf.reshape(flatten_pred_cls[:, 1], shape=[-1, 1])
        OHEM_mask_cls = tf.cast(OHEM_mask, dtype=tf.bool)
        OHEM_mask_cls = tf.logical_or(OHEM_mask_cls, cls_pos >= val[-1])
        OHEM_mask_cls = tf.cast(OHEM_mask_cls, dtype=tf.float32)
        # cls loss
        epsilon=1e-8
        loss_cls = -tf.reduce_sum(self.Ycls * tf.log(flatten_pred_cls+epsilon), axis=[1], keep_dims=True) * OHEM_mask_cls
        # reg loss
        delta_reg = tf.abs(flatten_pred_reg - self.Yreg)
        OHEM_mask = tf.cast(OHEM_mask, dtype=tf.float32)
        smooth_l1_sign = tf.cast(tf.reshape(delta_reg < 1, shape=[-1, 4]), dtype=tf.float32)
        loss_reg = tf.reduce_sum(0.5 * tf.pow(delta_reg, 2) * smooth_l1_sign + \
                                 (delta_reg - 0.5) * (1 - smooth_l1_sign), axis=[1], keep_dims=True) * OHEM_mask
        # seg loss
        loss_seg = 1 - tf.reduce_sum(2 * self.Yseg * flatten_pred_seg) / tf.reduce_sum(self.Yseg + flatten_pred_seg)

        return {'cls loss': loss_cls,
                'reg loss': loss_reg,
                'seg loss': loss_seg}

    def define_optimizer(self, loss_dict):
        backbone_vars = self.get_trainable_var('backbone')
        vgg_vars = self.get_trainable_var('vgg_16')
        total_vars = backbone_vars + vgg_vars
        loss = tf.reduce_sum(loss_dict['cls loss']) + tf.reduce_sum(loss_dict['reg loss']) + loss_dict['seg loss']
        optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss,
                                                            global_step=self.global_step,
                                                            var_list=total_vars)
        return optimizer

    def setup_corner_point_dect(self, f):
        for f_in in f:
            scor, offs = self.detect_model(f_in['tensor'], f_in['num'], f_in['name'])
            self.detect_dict[f_in['name']] = scor
            self.off_dict[f_in['name']] = offs

    def setup_position_sen_seg(self, f):
        f_sum = None
        for f_index, f_in in enumerate(f):
            if f_index < 2:
                continue
            f_sum = tf.image.resize_images(f_in['tensor'], [128, 128]) if f_sum is None \
                else f_sum + tf.image.resize_images(f_in['tensor'], [128, 128])
        self.feed(f_sum, 'abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, 'PSS_conv_1_W', 'PSS_conv_1_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    'PSS_1_scale', 'PSS_1_offset', 'PSS_1_mean', 'PSS_1_var') \
            .relu('abandon tensor') \
            .deconv2d('abandon tensor', [self.batch_size, 256, 256, 256], 2, 2, 2, 2, 'PSS_deconv_1_W',
                      'PSS_deconv_1_b') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, 'PSS_conv_2_W', 'PSS_conv_2_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    'PSS_2_scale', 'PSS_2_offset', 'PSS_2_mean', 'PSS_2_var') \
            .relu('abandon tensor') \
            .deconv2d('save tensor', [self.batch_size, 512, 512, 4], 2, 2, 2, 2, 'PSS_deconv_2_W', 'PSS_deconv_2_b')
        pss_pred = self.layer_tensor_pop()
        self.seg = pss_pred

    def detect_model(self, f, default_box_num, name):
        self.feed(f, 'abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, name + '_conv1_1_W', name + '_conv1_1_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_1_1_scale', name + '_1_1_offset', name + '_1_1_mean', name + '_1_1_var') \
            .relu('abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, name + '_conv1_2_W', name + '_conv1_2_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_1_2_scale', name + '_1_2_offset', name + '_1_2_mean', name + '_1_2_var') \
            .relu('abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, name + '_conv1_3_W', name + '_conv1_3_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_1_3_scale', name + '_1_3_offset', name + '_1_3_mean', name + '_1_3_var') \
            .relu('save tensor')
        conv = self.layer_tensor_pop()
        self.feed(f, 'abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, name + '_conv2_1_W', name + '_conv2_1_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_2_1_scale', name + '_2_1_offset', name + '_2_1_mean', name + '_2_1_var') \
            .relu('save tensor')
        conv_short = self.layer_tensor_pop()

        self.feed(conv + conv_short, 'abandon tensor') \
            .conv2d('abandon tensor', 256, 1, 1, 1, 1, name + '_conv3_W', name + '_conv3_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_3_scale', name + '_3_offset', name + '_3_mean', name + '_3_var') \
            .relu('save tensor') \
            .conv2d('abandon tensor', default_box_num * 4 * 2, 1, 1, 1, 1, name + '_conv4_top_W', name + '_conv4_top_b') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_4_top_scale', name + '_4_top_offset', name + '_4_top_mean', name + '_4_top_var')\
            .softmax('save tensor')

        scor_pred = self.layer_tensor_pop()

        self.feed(self.layer_tensor_pop(), 'abandon tensor') \
            .conv2d('abandon tensor', default_box_num * 4 * 4, 1, 1, 1, 1, name + '_conv4_bottom_W',
                    name + '_conv4_bottom_b') \
            .normal('save tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_4_bottom_scale', name + '_4_bottom_offset', name + '_4_bottom_mean',
                    name + '_4_bottom_var')

        offs_pred = self.layer_tensor_pop()

        return scor_pred, offs_pred

    def deconvolution_model(self, deconv_size, deconv_layer, feature_layer, name):
        self.feed(deconv_layer, 'abandon tensor') \
            .deconv2d('abandon tensor', deconv_size, 2, 2, 2, 2, name + '_top_deconv_W', name + '_top_deconv_b') \
            .conv2d('abandon tensor', 512, 3, 3, 1, 1, name + '_top_conv_W', name + '_top_conv_b') \
            .normal('save tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_top_scale', name + '_top_offset', name + '_top_mean', name + '_top_var')
        top_tensor = self.layer_tensor_pop()

        self.feed(feature_layer, 'abandon tensor') \
            .conv2d('abandon tensor', 512, 3, 3, 1, 1, name + '_bottom_deconv_W_1', name + '_bottom_deconv_b_1') \
            .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_bottom_scale_1', name + '_bottom_offset_1', name + '_bottom_mean_1',
                    name + '_bottom_var_1') \
            .relu('abandon tensor') \
            .conv2d('abandon tensor', 512, 3, 3, 1, 1, name + '_bottom_deconv_W_2', name + '_bottom_deconv_b_2') \
            .normal('save tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_bottom_scale_2', name + '_bottom_offset_2', name + '_bottom_mean_2',
                    name + '_bottom_var_2')
        bottom_tensor = self.layer_tensor_pop()

        ES_layer = top_tensor + bottom_tensor

        self.feed(ES_layer, 'abandon tensor') \
            .relu('save tensor')
        return self.layer_tensor_demand()

    def get_graph(self):
        return self.graph

    def load_vgg_model(self):
        graph = self.graph
        with tf.variable_scope('vgg_16'):
            self.feed(self.X, 'abandon tensor') \
                .conv2d('abandon tensor', 64, 3, 3, 1, 1, 'conv1/conv1_1/weights', 'conv1/conv1_1/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 64, 3, 3, 1, 1, 'conv1/conv1_2/weights', 'conv1/conv1_2/biases') \
                .relu('abandon tensor') \
                .max_pool2d('abandon tensor', 2, 2, 2, 2) \
                .conv2d('abandon tensor', 128, 3, 3, 1, 1, 'conv2/conv2_1/weights', 'conv2/conv2_1/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 128, 3, 3, 1, 1, 'conv2/conv2_2/weights', 'conv2/conv2_2/biases') \
                .relu('abandon tensor') \
                .max_pool2d('abandon tensor', 2, 2, 2, 2) \
                .conv2d('abandon tensor', 256, 3, 3, 1, 1, 'conv3/conv3_1/weights', 'conv3/conv3_1/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 256, 3, 3, 1, 1, 'conv3/conv3_2/weights', 'conv3/conv3_2/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 256, 3, 3, 1, 1, 'conv3/conv3_3/weights', 'conv3/conv3_3/biases') \
                .relu('save tensor') \
                .max_pool2d('abandon tensor', 2, 2, 2, 2) \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv4/conv4_1/weights', 'conv4/conv4_1/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv4/conv4_2/weights', 'conv4/conv4_2/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv4/conv4_3/weights', 'conv4/conv4_3/biases') \
                .relu('save tensor') \
                .max_pool2d('abandon tensor', 2, 2, 2, 2) \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv5/conv5_1/weights', 'conv5/conv5_1/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv5/conv5_2/weights', 'conv5/conv5_2/biases') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 512, 3, 3, 1, 1, 'conv5/conv5_3/weights', 'conv5/conv5_3/biases') \
                .relu('abandon tensor')
        model_path = 'model/vgg_model/vgg_16.ckpt'
        assert (os.path.isfile(model_path))

        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        variables_to_restore.pop(0)
        self.vgg16_variables = variables_to_restore
        init_vgg = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
        self.vgg16_initializer = init_vgg

        # with tf.Session(graph=graph) as sess:
        #     sess.run(tf.global_variables_initializer())
        #     init_vgg(sess)

        return self.layer_tensor_demand()
