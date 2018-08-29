# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      factory.py
# @Software  PyCharm
# @Author    JK.Rao

from .DCGAN_data_stream import DCGAN_get_pipeline
from .CPD_stream import ground_truth2feature_map
import scipy.io as sio
import cv2
from os.path import join
import numpy as np
import copy
import random
import time

CPD_mat = sio.loadmat('./data/img_data/gt.mat')
sampling_list = range(CPD_mat['imnames'].shape[1])
random.shuffle(sampling_list)
train_sampling_list = sampling_list[:-100000]
val_sampling_list = sampling_list[-100000:-1]


def random_list():
    random.shuffle(train_sampling_list)


def get_sample_tensor(model_name, sess=None, propose=None, batch_size=None, filename=None):
    if model_name == 'DCGAN':
        return DCGAN_get_pipeline(sess, propose, batch_size, filename)
    elif model_name == 'CPD':
        t0 = time.time()
        img_batch = None
        dicts = list()
        if not batch_size is None:
            for i in range(batch_size[0], batch_size[1]):
                img = cv2.imread(
                    join('/home/cj3/Downloads/im/SynthText',
                         CPD_mat['imnames'][0][train_sampling_list[i] if filename == 'train' else
                         val_sampling_list[i]][0].encode('gb18030')))
                img_height, img_width = img.shape[0:2]
                img = cv2.resize(img, (512, 512))
                img = img[np.newaxis, :]

                gt_array = copy.deepcopy(CPD_mat['wordBB'][0][train_sampling_list[i] if filename == 'train' else
                val_sampling_list[i]])
                gt_array[0] = gt_array[0] * 512. / img_width
                gt_array[1] = gt_array[1] * 512. / img_height
                gt_array.astype(np.int32)
                if len(gt_array.shape) < 3:
                    gt_array = gt_array.reshape((gt_array.shape[0], gt_array.shape[1], 1))
                    # continue
                img_batch = img if img_batch is None else np.append(img_batch, img, axis=0)
                dicts.append(ground_truth2feature_map(gt_array))

        else:
            img = cv2.imread(join('/home/cj3/Downloads/im/SynthText', CPD_mat['imnames'][0][0][0].encode('gb18030')))
            img_height, img_width = img.shape[0:2]
            img = cv2.resize(img, (512, 512))
            img = img[np.newaxis, :]
            img_batch = img

            gt_array = CPD_mat['wordBB'][0][0]
            gt_array[0] = gt_array[0] * 512. / img_width
            gt_array[1] = gt_array[1] * 512. / img_height
            gt_array.astype(np.int32)
            dicts.append(ground_truth2feature_map(gt_array))

        # print('propcess spend %fs for 8 imgs.' % (time.time() - t0))
        return dicts, img_batch
