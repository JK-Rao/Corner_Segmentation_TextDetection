# _*_ coding: utf-8 _*_
# @Time      18-8-7 上午11:07
# @File      CPD_stream.py
# @Software  PyCharm
# @Author    JK.Rao

import numpy as np
from .data_pipeline import TfReader
from .data_pipeline import TfWriter
from ..tools import gadget
import time
from multiprocessing import Process, Queue


class CPDReader(TfReader):
    def __init__(self):
        TfReader.__init__(self, 32, 20, 1)


class CPDWriter(TfWriter):
    def __init__(self):
        TfWriter.__init__(self, 32, 20, 1)


def init_CPD_mask(shape, channel, mask_type):
    # print('init mask of %s' % mask_type)
    gt_mask = None
    if mask_type == 'cls':
        for i in range(channel):
            if i % 2 == 0:
                gt_mask = np.ones(shape=shape, dtype=np.float32) if gt_mask is None \
                    else np.append(gt_mask, np.ones(shape=shape, dtype=np.float32), axis=3)
            else:
                gt_mask = np.zeros(shape=shape, dtype=np.float32) if gt_mask is None \
                    else np.append(gt_mask, np.zeros(shape=shape, dtype=np.float32), axis=3)
    elif mask_type == 'reg':
        for i in range(channel):
            gt_mask = np.zeros(shape=shape, dtype=np.float32) if gt_mask is None \
                else np.append(gt_mask, np.zeros(shape=shape, dtype=np.float32), axis=3)
    elif mask_type == 'seg':
        for i in range(channel):
            gt_mask = np.zeros(shape=shape, dtype=np.float32) if gt_mask is None \
                else np.append(gt_mask, np.zeros(shape=shape, dtype=np.float32), axis=3)
    return gt_mask


# gt_rects:[[cx,cy,ss,ss,point_type]...]
def gt_array2gt_rects(gt_array):
    gt_rects = list()
    for i in range(gt_array.shape[2]):
        if np.max(gt_array[:, :, i]) > 512:
            continue
        ssl = np.sqrt((gt_array[0, 3, i] - gt_array[0, 0, i]) ** 2 + (gt_array[1, 3, i] - gt_array[1, 0, i]) ** 2)
        sst = np.sqrt((gt_array[0, 1, i] - gt_array[0, 0, i]) ** 2 + (gt_array[1, 1, i] - gt_array[1, 0, i]) ** 2)
        ssr = np.sqrt((gt_array[0, 2, i] - gt_array[0, 1, i]) ** 2 + (gt_array[1, 2, i] - gt_array[1, 1, i]) ** 2)
        ssb = np.sqrt((gt_array[0, 3, i] - gt_array[0, 2, i]) ** 2 + (gt_array[1, 3, i] - gt_array[1, 2, i]) ** 2)
        ss = np.sort([ssl, sst, ssr, ssb])[0]
        for point_type in range(4):
            gt_rects.append([gt_array[0, point_type, i], gt_array[1, point_type, i], ss, ss, point_type])
    return gt_rects


# gt_array:A 3d tensor,[2,4,None]
def ground_truth2feature_map(gt_array):
    t0 = time.time()
    # global gt_cls_mask_f11, gt_cls_mask_f10, gt_cls_mask_f9, gt_cls_mask_f8, gt_cls_mask_f7, gt_cls_mask_f4, \
    #     gt_cls_mask_f3, gt_reg_mask_f11, gt_reg_mask_f10, gt_reg_mask_f9, gt_reg_mask_f8, gt_reg_mask_f7, gt_reg_mask_f4, \
    #     gt_reg_mask_f3, gt_seg_mask
    gt_cls_mask_f11 = init_CPD_mask([1, 2, 2, 1], 32, 'cls')
    gt_cls_mask_f10 = init_CPD_mask([1, 4, 4, 1], 32, 'cls')
    gt_cls_mask_f9 = init_CPD_mask([1, 8, 8, 1], 32, 'cls')
    gt_cls_mask_f8 = init_CPD_mask([1, 16, 16, 1], 32, 'cls')
    gt_cls_mask_f7 = init_CPD_mask([1, 32, 32, 1], 32, 'cls')
    gt_cls_mask_f4 = init_CPD_mask([1, 64, 64, 1], 32, 'cls')
    gt_cls_mask_f3 = init_CPD_mask([1, 128, 128, 1], 48, 'cls')

    gt_reg_mask_f11 = init_CPD_mask([1, 2, 2, 1], 64, 'reg')
    gt_reg_mask_f10 = init_CPD_mask([1, 4, 4, 1], 64, 'reg')
    gt_reg_mask_f9 = init_CPD_mask([1, 8, 8, 1], 64, 'reg')
    gt_reg_mask_f8 = init_CPD_mask([1, 16, 16, 1], 64, 'reg')
    gt_reg_mask_f7 = init_CPD_mask([1, 32, 32, 1], 64, 'reg')
    gt_reg_mask_f4 = init_CPD_mask([1, 64, 64, 1], 64, 'reg')
    gt_reg_mask_f3 = init_CPD_mask([1, 128, 128, 1], 96, 'reg')

    gt_seg_mask = init_CPD_mask([1, 512, 512, 1], 4, 'seg')
    gt_rects = gt_array2gt_rects(gt_array)
    for gt_rect in gt_rects:
        gt_cls_mask_f11, gt_reg_mask_f11 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f11, gt_reg_mask_f11,
                                                                      [184, 208, 232, 256], 256, gt_rect[4])
        gt_cls_mask_f10, gt_reg_mask_f10 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f10, gt_reg_mask_f10,
                                                                      [124, 136, 148, 160], 128, gt_rect[4])
        gt_cls_mask_f9, gt_reg_mask_f9 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f9, gt_reg_mask_f9,
                                                                    [88, 96, 104, 112], 64, gt_rect[4])
        gt_cls_mask_f8, gt_reg_mask_f8 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f8, gt_reg_mask_f8,
                                                                    [56, 64, 72, 80], 32, gt_rect[4])
        gt_cls_mask_f7, gt_reg_mask_f7 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f7, gt_reg_mask_f7,
                                                                    [36, 40, 44, 48], 16, gt_rect[4])
        gt_cls_mask_f4, gt_reg_mask_f4 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f4, gt_reg_mask_f4,
                                                                    [20, 24, 28, 32], 8, gt_rect[4])
        gt_cls_mask_f3, gt_reg_mask_f3 = gadget.project_feature_map(gt_rect[0:4], gt_cls_mask_f3, gt_reg_mask_f3,
                                                                    [4, 8, 6, 10, 12, 16], 4, gt_rect[4])

    gt_seg_mask = gadget.project_feature_map_seg(gt_array, gt_seg_mask)

    return {'cls_mask': [gt_cls_mask_f11, gt_cls_mask_f10, gt_cls_mask_f9,
                         gt_cls_mask_f8, gt_cls_mask_f7, gt_cls_mask_f4, gt_cls_mask_f3],
            'reg_mask': [gt_reg_mask_f11, gt_reg_mask_f10, gt_reg_mask_f9,
                         gt_reg_mask_f8, gt_reg_mask_f7, gt_reg_mask_f4, gt_reg_mask_f3],
            'seg_mask': [gt_seg_mask]}
