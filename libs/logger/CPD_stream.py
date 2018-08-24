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
import copy
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
        if np.min(gt_array[:, :, i]) > 512:
            continue
        ssl = np.sqrt((gt_array[0, 3, i] - gt_array[0, 0, i]) ** 2 + (gt_array[1, 3, i] - gt_array[1, 0, i]) ** 2)
        sst = np.sqrt((gt_array[0, 1, i] - gt_array[0, 0, i]) ** 2 + (gt_array[1, 1, i] - gt_array[1, 0, i]) ** 2)
        ssr = np.sqrt((gt_array[0, 2, i] - gt_array[0, 1, i]) ** 2 + (gt_array[1, 2, i] - gt_array[1, 1, i]) ** 2)
        ssb = np.sqrt((gt_array[0, 3, i] - gt_array[0, 2, i]) ** 2 + (gt_array[1, 3, i] - gt_array[1, 2, i]) ** 2)
        ss = np.sort([ssl, sst, ssr, ssb])[0]
        for point_type in range(4):
            gt_rects.append([gt_array[0, point_type, i], gt_array[1, point_type, i], ss, ss, point_type])
    return gt_rects


gt_cls_mask_f11 = init_CPD_mask([1, 4, 4, 1], 32, 'cls')
gt_cls_mask_f10 = init_CPD_mask([1, 6, 6, 1], 32, 'cls')
gt_cls_mask_f9 = init_CPD_mask([1, 8, 8, 1], 32, 'cls')
gt_cls_mask_f8 = init_CPD_mask([1, 16, 16, 1], 32, 'cls')
gt_cls_mask_f7 = init_CPD_mask([1, 32, 32, 1], 32, 'cls')
gt_cls_mask_f4 = init_CPD_mask([1, 64, 64, 1], 32, 'cls')
gt_cls_mask_f3 = init_CPD_mask([1, 128, 128, 1], 48, 'cls')

gt_reg_mask_f11 = init_CPD_mask([1, 4, 4, 1], 64, 'reg')
gt_reg_mask_f10 = init_CPD_mask([1, 6, 6, 1], 64, 'reg')
gt_reg_mask_f9 = init_CPD_mask([1, 8, 8, 1], 64, 'reg')
gt_reg_mask_f8 = init_CPD_mask([1, 16, 16, 1], 64, 'reg')
gt_reg_mask_f7 = init_CPD_mask([1, 32, 32, 1], 64, 'reg')
gt_reg_mask_f4 = init_CPD_mask([1, 64, 64, 1], 64, 'reg')
gt_reg_mask_f3 = init_CPD_mask([1, 128, 128, 1], 96, 'reg')

gt_seg_mask = init_CPD_mask([1, 512, 512, 1], 4, 'seg')


# gt_array:A 3d tensor,[2,4,None]
def ground_truth2feature_map(gt_array):
    global gt_cls_mask_f11, gt_cls_mask_f10, gt_cls_mask_f9, gt_cls_mask_f8, gt_cls_mask_f7, gt_cls_mask_f4, \
        gt_cls_mask_f3, gt_reg_mask_f11, gt_reg_mask_f10, gt_reg_mask_f9, gt_reg_mask_f8, gt_reg_mask_f7, gt_reg_mask_f4, \
        gt_reg_mask_f3, gt_seg_mask

    gt_rects = gt_array2gt_rects(gt_array)
    default_boxes_f = list()
    position_f = list()
    gt_rects_f = list()
    gt_boxes = list()
    scale_list_f = list()
    cls_maps = [copy.deepcopy(gt_cls_mask_f11), copy.deepcopy(gt_cls_mask_f10), copy.deepcopy(gt_cls_mask_f9),
                copy.deepcopy(gt_cls_mask_f8), copy.deepcopy(gt_cls_mask_f7), copy.deepcopy(gt_cls_mask_f4),
                copy.deepcopy(gt_cls_mask_f3)]
    reg_maps = [copy.deepcopy(gt_reg_mask_f11), copy.deepcopy(gt_reg_mask_f10), copy.deepcopy(gt_reg_mask_f9),
                copy.deepcopy(gt_reg_mask_f8), copy.deepcopy(gt_reg_mask_f7), copy.deepcopy(gt_reg_mask_f4),
                copy.deepcopy(gt_reg_mask_f3)]
    map_size_table = [4, 6, 8, 16, 32, 64, 128]
    scale_table = [[184, 208, 232, 256],
                   [124, 136, 148, 160],
                   [88, 96, 104, 112],
                   [56, 64, 72, 80],
                   [36, 40, 44, 48],
                   [20, 24, 28, 32],
                   [4, 8, 6, 10, 12, 16]]
    strides = [128, 85.3333, 64, 32, 16, 8, 4]
    for gt_rect in gt_rects:
        for map_index in range(7):  # 7 different resolving map from f11 to f3
            default_boxes, position, gt_re, gt_box, scale_list = \
                gadget.project_feature_map_simple(gt_rect[0:4],
                                                  map_size_table[map_index],
                                                  map_size_table[map_index],
                                                  scale_table[map_index],
                                                  strides[map_index],
                                                  gt_rect[4],
                                                  map_index)
            default_boxes_f += default_boxes
            position_f += position
            gt_rects_f += gt_re
            gt_boxes += gt_box
            scale_list_f += scale_list

    # test
    iou_matrix = gadget.calcul_matrix_iou(np.array(default_boxes_f), np.array(gt_boxes))
    cls_maps, reg_maps = gadget.project_feature_map_iou(cls_maps, reg_maps, np.array(default_boxes_f),
                                                        np.array(iou_matrix),
                                                        np.array(position_f), np.array(gt_rects_f),
                                                        np.array(scale_list_f))
    gt_seg_mask_counterpart = gadget.project_feature_map_seg(gt_array, copy.deepcopy(gt_seg_mask))

    return {'cls_mask': cls_maps,
            'reg_mask': reg_maps,
            'seg_mask': [gt_seg_mask_counterpart]}
