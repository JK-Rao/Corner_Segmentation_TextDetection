# _*_ coding: utf-8 _*_
# @Time      18-8-3 上午10:47
# @File      Backbone_net.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
from .network import Network


class Backbone_net(Network):
    def __init__(self):
        Network.__init__(self, 'backbone')
        self.IM_HEIGHT = 512
        self.IM_WIDTH = 512
        self.IM_CHANEL = 3
        self.global_step = tf.Variable(0, trainable=False)
        self.X = tf.placeholder(tf.float32, shape=[None, self.IM_HEIGHT, self.IM_WIDTH, self.IM_CHANEL], name='X')
        self.Yc = tf.placeholder(tf.float32, shape=[None, self.IM_HEIGHT, self.IM_WIDTH, 2], name='Yc')
        self.Yl = tf.placeholder(tf.float32, shape=[None, self.IM_HEIGHT, self.IM_WIDTH, 4], name='Yl')
        self.Ys = tf.placeholder(tf.float32, shape=[None, self.IM_HEIGHT, self.IM_WIDTH, 1], name='Ys')

        self.on_train = tf.placeholder(tf.bool, [], name='on_train')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        self.detect_dict = {}
        self.off_dict = {}
        self.seg = None

        self.setup(self.X, 'backbone')

    def setup(self, x, scope_name, reuse=False):
        conv5_3 = self.load_vgg_model()
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()
            self.feed(conv5_3, 'abandon tensor') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale5_3', 'offset5_3', 'mean5_3', 'var5_3') \
                .relu('abandon tensor') \
                .conv2d('abandon tensor', 1024, 3, 3, 1, 1, 'conv6_W', 'conv6_b') \
                .normal('abandon tensor', self.on_train, 0.5, [0, 1, 2], 'scale6', 'offset6', 'mean6', 'var6') \
                .relu('save tensor') \
                .conv2d('abandon tensor', 1024, 3, 3, 2, 2, 'conv7_W', 'conv7_b') \
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

    def structure_loss(self):
        pass

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
            .normal('save tensor', self.on_train, 0.5, [0, 1, 2],
                    name + '_4_top_scale', name + '_4_top_offset', name + '_4_top_mean', name + '_4_top_var')
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

    def load_vgg_model(self):
        with open("model/vgg16-20160129.tfmodel", mode='rb') as f:
            fileContent = f.read()
        images = tf.placeholder("float", [None, 512, 512, 3])
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)
        tf.import_graph_def(graph_def, input_map={"images": images})
        graph = tf.get_default_graph()

        conv3_3 = graph.get_tensor_by_name('import/conv3_3/Relu:0')
        conv4_3 = graph.get_tensor_by_name('import/conv4_3/Relu:0')
        conv5_3 = graph.get_tensor_by_name('import/conv5_3/Relu:0')
        self.layers.append(conv3_3)
        self.layers.append(conv4_3)
        return conv5_3
