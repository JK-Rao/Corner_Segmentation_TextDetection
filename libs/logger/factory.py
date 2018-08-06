# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGAN_data_stream import DCGAN_get_pipeline


def get_sample_tensor(model_name, sess, propose, batch_size, filename):
    if model_name == 'DCGAN':
        return DCGAN_get_pipeline(sess, propose, batch_size, filename)
