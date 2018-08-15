# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGANnet import DCGANnet
from .Backbone_net import Backbone_net


def get_network(name,global_reuse=False):
    if name == 'DCGAN':
        return DCGANnet(['gen', 'dis'], [32, 20, 1])
    elif name == 'CSTR':
        return Backbone_net(global_reuse)


def get_network_name(name):
    if name == 'DCGAN':
        return ['gen', 'dis']
