# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGAN_data_stream import DCGAN_get_pipeline
from .CPD_stream import ground_truth2feature_map
import scipy.io as sio


def get_sample_tensor(model_name, sess=None, propose=None, batch_size=None, filename=None):
    if model_name == 'DCGAN':
        return DCGAN_get_pipeline(sess, propose, batch_size, filename)
    elif model_name=='CPD':
        CPD_mat=sio.loadmat('./model/gt_model/gt.mat')
        dicts=list()
        if not batch_size is None:
            for i in range(batch_size[0],batch_size[1]):
                gt_array = CPD_mat['wordBB'][0][i]
                dicts.append(ground_truth2feature_map(gt_array))
        else:
            gt_array=CPD_mat['wordBB'][0][0]
            dicts.append(ground_truth2feature_map(gt_array))
        return dicts
