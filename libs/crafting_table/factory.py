# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGAN_line import DCGANLine
from os.path import join


def train_model(name, inster_number, annotation):
    if name == 'DCGAN':
        info_dict = {'inster_number': inster_number, 'annotation': annotation, 'batch_size': 128, 'val_size': 64}
        line = DCGANLine('train', info_dict)
        line.structure_train_context()


def test_model(name, model_path, model_name, parameter_name):
    if name == 'DCGAN':
        info_dict = {'model_name': join(model_path, model_name),
                     'parameter_name': join(model_path, parameter_name),
                     'batch_size':128}
        line = DCGANLine('test', info_dict)
        return line.restore_test_context()
