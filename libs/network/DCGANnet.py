# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      DCGANnet.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
from .network import Network


class DCGANnet(Network):
    def __init__(self, net_name, IMG_SHAPE):
        Network.__init__(self, net_name)
        self.IMG_HEIGHT = IMG_SHAPE[0]
        self.IMG_WIDTH = IMG_SHAPE[1]
        self.IMG_CHANEL = IMG_SHAPE[2]
        self.Z_dim = 100
        self.X = tf.placeholder(tf.float32, shape=[None, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANEL], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
        self.on_train = tf.placeholder(tf.bool, name='on_train')
        self.batch_pattern = tf.placeholder(tf.int32, name='batch_size')
        self.global_step = tf.Variable(0, trainable=False)
        self.Min_distance = tf.Variable(0., trainable=False)
        self.Ave_distance = tf.Variable(0., trainable=False)

        self.gen_im = None
        self.dis_real = None
        self.dis_fake = None

        self.setup()

    def setup(self):
        self.gen_im = self.setup_g(self.Z, self.net_name[0])
        self.dis_real = self.setup_d(self.X, self.net_name[1])
        self.dis_fake = self.setup_d(self.gen_im, self.net_name[1], reuse=True)

    def get_summary(self):
        return {'val_loss': self.structure_loss()['val_loss'],
                'd_loss': self.structure_loss()['d_loss'],
                'g_loss': self.structure_loss()['g_loss'],
                'min_distance': self.Min_distance,
                'ave_distance': self.Ave_distance}

    def get_pred(self):
        return {'gen_im': self.gen_im,
                'dis_real': self.dis_real,
                'dis_fake': self.dis_fake}

    def get_gen_vars(self):
        return self.get_trainable_var(self.net_name[0])

    def get_dis_vars(self):
        return self.get_trainable_var(self.net_name[1])

    def structure_loss(self):
        ep = 1e-7
        Val_loss = -tf.reduce_mean(tf.log(self.dis_real + ep))
        D_loss = -tf.reduce_mean(tf.log(self.dis_real + ep) + tf.log(1. - self.dis_fake + ep))
        G_loss = -tf.reduce_mean(tf.log(self.dis_fake + ep))
        return {'val_loss': Val_loss,
                'd_loss': D_loss,
                'g_loss': G_loss}

    def define_optimizer(self, loss_dict):
        # loss_dict = self.structure_loss()
        gen_vars = self.get_gen_vars()
        dis_vars = self.get_dis_vars()
        D_optimizer = tf.train.AdamOptimizer(0.0002).minimize(loss_dict['d_loss'],
                                                              var_list=dis_vars)
        G_optimizer = tf.train.AdamOptimizer(0.001).minimize(loss_dict['g_loss'],
                                                             global_step=self.global_step,
                                                             var_list=gen_vars)
        return {'d_opti': D_optimizer, 'g_opti': G_optimizer}

    def setup_g(self, x, scope_name, reuse=False, load_net=False):
        if not load_net:
            with tf.variable_scope(scope_name) as scope:
                if reuse:
                    scope.reuse_variables()
                self.feed(x) \
                    .mulfc(128 * 5 * 8, name_W='G_W_line', name_b='G_b_line') \
                    .reshape([-1, 8, 5, 128]) \
                    .normal(self.on_train, 0.5, [0, 1, 2], 'G_sca_line', 'G_off_line', 'G_mea_line', 'G_var_line') \
                    .relu() \
                    .deconv2d([self.batch_pattern, 16, 10, 256], 3, 3, 2, 2, 'G_W_1', 'G_b_1', padding='SAME') \
                    .normal(self.on_train, 0.5, [0, 1, 2], 'G_sca_1', 'G_off_1', 'G_mea_1', 'G_var_1') \
                    .relu() \
                    .deconv2d([self.batch_pattern, 32, 20, self.IMG_CHANEL], 3, 3, 2, 2, 'G_W_2', 'G_b_2',
                              padding='SAME') \
                    .tanh(2)
                tf.add_to_collection(name='out', value=self.pre_process_tensor)
                return self.pre_process_tensor

    def setup_d(self, x, scope_name, reuse=False, load_net=False):
        if not load_net:
            with tf.variable_scope(scope_name) as scope:
                if reuse:
                    scope.reuse_variables()
                self.feed(x) \
                    .conv2d(128, 3, 3, 2, 2, 'D_W_0', 'D_b_0') \
                    .lrelu() \
                    .conv2d(256, 3, 3, 2, 2, 'D_W_1', 'D_b_1') \
                    .normal(self.on_train, 0.5, [0, 1, 2], 'D_sca_1', 'D_off_1', 'D_mea_1', 'D_var_1') \
                    .lrelu() \
                    .reshape([-1, 8 * 5 * 128]) \
                    .mulfc(1, name_W='G_W_2', name_b='G_b_2') \
                    .sigmoid()
                tf.add_to_collection(name='out', value=self.pre_process_tensor)
                return self.pre_process_tensor
