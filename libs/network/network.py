# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      network.py
# @Software  PyCharm
# @Author    JK.Rao

import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average


def layer(op):
    def layer_decorate(self, *args, **kwargs):
        self.input = self.pre_process_tensor
        self.pre_process_tensor = op(self, self.input, *args[1:], **kwargs)
        if args[0] == 'save tensor':
            self.layers.append(self.pre_process_tensor)
        return self

    return layer_decorate


def flattener(op):
    def tensor_flattener(self, *args):
        concat_tensor = tf.reshape(args[0], shape=[-1, args[1]])
        self.input = self.pre_process_tensor
        self.pre_process_tensor = op(self, self.input, concat_tensor)
        return self

    return tensor_flattener


class Network(object):
    def __init__(self, net_name):
        self.net_name = net_name
        self.pre_process_tensor = None
        self.input = None
        self.layers = []

    def setup(self, x, scope_name, reuse=False):
        raise NotImplementedError('Must be subclassed.')

    def restore(self):
        raise NotImplementedError('Must be subclassed.')

    def get_summary(self):
        raise NotImplementedError('Must be subclassed.')

    def get_pred(self):
        raise NotImplementedError('Must be subclassed.')

    def structure_loss(self):
        raise NotImplementedError('Must be subclassed.')

    def define_optimizer(self, loss_dict):
        raise NotImplementedError('Must be subclassed.')

    def get_trainable_var(self, name):
        vars = []
        for var in tf.global_variables():
            if var.name.split('/')[0] == name and var.name.split('/')[1] != name:
                vars.append(var)
        return vars

    def feed(self, x, save_tensor):
        self.pre_process_tensor = x
        if save_tensor == 'save tensor':
            self.layers.append(self.pre_process_tensor)
        elif save_tensor == 'flatten tensor x2':
            self.pre_process_tensor = tf.reshape(self.pre_process_tensor,
                                                 shape=[-1, 2])
        elif save_tensor == 'flatten tensor x4':
            self.pre_process_tensor = tf.reshape(self.pre_process_tensor,
                                                 shape=[-1, 4])
        elif save_tensor == 'flatten tensor x1':
            self.pre_process_tensor = tf.reshape(self.pre_process_tensor,
                                                 shape=[-1, 1])
        return self

    def weight_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(mean=0., stddev=0.02))

    def bias_var(self, shape, name):
        return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))

    def layer_tensor_pop(self, index=-1):
        return self.layers.pop(index)

    def layer_tensor_demand(self, index=-1):
        return self.layers[index]

    @flattener
    def concat_tensor(self, tensor_org, tensor_cc):
        return tf.concat([tensor_org, tensor_cc], axis=0)

    @layer
    def normal(self, x, on_train, decay, axes, name_scale, name_offset, name_mean, name_var):
        # batch-normalization
        shape = x.get_shape().as_list()[-1]
        scale = tf.get_variable(name_scale, shape, initializer=tf.ones_initializer(), trainable=True)
        offset = tf.get_variable(name_offset, shape, initializer=tf.zeros_initializer(), trainable=True)
        variance_epsilon = 1e-7
        mean_p = tf.get_variable(name_mean, shape, initializer=tf.zeros_initializer(), trainable=False)
        var_p = tf.get_variable(name_var, shape, initializer=tf.ones_initializer(), trainable=False)

        # moving average
        def mean_var_with_update():
            mean_ba, var_ba = tf.nn.moments(x, axes, name='moments')
            with tf.control_dependencies([assign_moving_average(mean_p, mean_ba, decay),
                                          assign_moving_average(var_p, var_ba, decay)]):
                return tf.identity(mean_ba), tf.identity(var_ba)

        # with tf.variable_scope('EMA'):
        mean, var = tf.cond(on_train, mean_var_with_update, lambda: (mean_p, var_p))

        return tf.nn.batch_normalization(x, mean, var, offset, scale, variance_epsilon)

    @layer
    def deconv2d(self, x, output_size, k_h, k_w, d_h, d_w, name_W, name_b, padding='SAME'):
        w = self.weight_var([k_h, k_w, output_size[-1], x.get_shape().as_list()[-1]], name=name_W)
        deconv = tf.nn.conv2d_transpose(x, w, output_shape=output_size, strides=[1, d_h, d_w, 1], padding=padding) + \
                 self.bias_var([output_size[-1]], name=name_b)
        return deconv

    @layer
    def conv2d(self, x, output_dim, k_h, k_w, d_h, d_w, name_W, name_b, padding='SAME'):
        w = self.weight_var([k_h, k_w, x.get_shape().as_list()[-1], output_dim], name=name_W)
        conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding) + \
               self.bias_var([output_dim], name=name_b)
        return conv

    @layer
    def lrelu(self, x, leak=0.2):
        return tf.maximum(x, leak * x)

    @layer
    def relu(self, x):
        return tf.nn.relu(x)

    @layer
    def mulfc(self, x, output_dim, name_W, name_b):
        w = self.weight_var([x.get_shape().as_list()[-1], output_dim], name=name_W)
        fc = tf.matmul(x, w) + self.bias_var([output_dim], name=name_b)
        return fc

    @layer
    def max_pool2d(self, x, k_h, k_w, d_h, d_w, padding='SAME'):
        return tf.nn.max_pool(x, [1, k_h, k_w, 1], [1, d_h, d_w, 1], padding)

    @layer
    def reshape(self, x, out_size):
        return tf.reshape(x, shape=out_size)

    @layer
    def tanh(self, x, scale):
        return tf.nn.tanh(x) / scale

    @layer
    def sigmoid(self, x):
        return tf.nn.sigmoid(x)

    @layer
    def softmax(self, x):
        return tf.nn.softmax(x)
