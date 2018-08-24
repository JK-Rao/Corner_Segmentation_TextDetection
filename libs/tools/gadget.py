# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      gadget.py
# @Software  PyCharm
# @Author    JK.Rao

import os
import numpy as np
import shutil
import cv2
import copy
import time
import cupy as cp


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def transform_file(source_file, obj_path, obj_name=None):
    mk_dir(obj_path)
    if obj_name is None:
        file_name = source_file.split('/')[-1]
    else:
        file_name = obj_name
    shutil.copy(source_file, os.path.join(obj_path, file_name))


# point format:[x_left,y_top,x_right,y_bottom]
def calcul_iou(ltrb_point1s, ltrb_point2s):
    x = [ltrb_point1s[0], ltrb_point1s[2], ltrb_point2s[0], ltrb_point2s[2]]
    y = [ltrb_point1s[1], ltrb_point1s[3], ltrb_point2s[1], ltrb_point2s[3]]

    if x[0] >= x[3] or x[1] <= x[2] or y[0] >= y[3] or y[1] <= y[2]:
        ins_score = 0
    else:
        x_sort = np.sort(x)
        y_sort = np.sort(y)

        ins_score = float(x_sort[2] - x_sort[1]) * (y_sort[2] - y_sort[1]) / (
                (x[1] - x[0]) * (y[1] - y[0]) + (x[3] - x[2]) * (y[3] - y[2]) -
                (x_sort[2] - x_sort[1]) * (y_sort[2] - y_sort[1]))
    return ins_score


# default_matrix:2d tensor like [[left, top, right, bottom]...]
# gt_matrix: as same as default_matrix' format
def calcul_matrix_iou(default_matrix, gt_matrix):
    area_default = (default_matrix[:, 2] - default_matrix[:, 0]) * (default_matrix[:, 3] - default_matrix[:, 1])
    area_gt = (gt_matrix[:, 2] - gt_matrix[:, 0]) * (gt_matrix[:, 3] - gt_matrix[:, 1])
    f_in_matrix = np.append(default_matrix, gt_matrix, axis=-1)
    x_in_matrix = np.delete(f_in_matrix, [1, 3, 5, 7], axis=-1)
    y_in_matrix = np.delete(f_in_matrix, [0, 2, 4, 6], axis=-1)
    interset_flag = x_in_matrix[:, 0] < x_in_matrix[:, 3]
    interset_flag = np.logical_and(interset_flag, x_in_matrix[:, 1] > x_in_matrix[:, 2])
    interset_flag = np.logical_and(interset_flag, y_in_matrix[:, 0] < y_in_matrix[:, 3])
    interset_flag = np.logical_and(interset_flag, y_in_matrix[:, 1] > y_in_matrix[:, 2]).astype(np.float32)

    x_in_matrix.sort(axis=-1)
    y_in_matrix.sort(axis=-1)
    area_i = (x_in_matrix[:, 2] - x_in_matrix[:, 1]) * (y_in_matrix[:, 2] - y_in_matrix[:, 1])
    epsilon = 1e-10
    return interset_flag * area_i / (area_default + area_gt - area_i )


# gt_rect:[cx,cy,ss,ss]
# cls_map:4d tensor
# reg_map:4d tensor
# scale:[scale of default box]
# stride:stride of map
# point_type:0-3
# threshold:the threshold of iou
def project_feature_map(gt_rect, cls_map, reg_map, scale, stride, point_type, threshold=0.5):
    test_time = 0.
    top, bottom, left, right = gt_rect[1] - gt_rect[3] // 2, \
                               gt_rect[1] + gt_rect[3] // 2, \
                               gt_rect[0] - gt_rect[2] // 2, \
                               gt_rect[0] + gt_rect[2] // 2
    max_scale = max(scale)
    height, width = cls_map.shape[1:3]
    default_boxes = list()
    position = list()
    for step_H in range(int(np.floor((top - max_scale // 2) / stride)),
                        int(np.ceil((bottom + max_scale // 2) / stride))):
        for step_W in range(int(np.floor((left - max_scale // 2) / stride)),
                            int(np.ceil((right + max_scale // 2) / stride))):
            if step_H < 0 or step_W < 0 or step_H > height - 1 or step_W > width - 1:
                continue
            for scal_index, scal in enumerate(scale):
                default_box = [int(step_W * stride + stride / 2 - scal / 2),
                               int(step_H * stride + stride / 2 - scal / 2),
                               int(step_W * stride + stride / 2 + scal / 2),
                               int(step_H * stride + stride / 2 + scal / 2)]
                default_boxes.append(default_box)
                position.append([step_H, step_W, scal_index, len(scale)])
                t0 = time.time()
                ins_score = calcul_iou(default_box, [int(left), int(top), int(right), int(bottom)])
                test_time += (time.time() - t0)
                if ins_score > threshold:
                    # print(ins_score)
                    # print(step_H, step_W, scal)
                    cent_dbox = [int((default_box[0] + default_box[2]) / 2),
                                 int((default_box[1] + default_box[3]) / 2),
                                 scal,
                                 scal]
                    cls_map[0, step_H, step_W,
                    point_type * len(scale) * 2 + scal_index * 2:point_type * len(scale) * 2 + scal_index * 2 + 2] = \
                        [0, 1]

                    reg_map[0, step_H, step_W,
                    point_type * len(scale) * 4 + scal_index * 4:point_type * len(scale) * 4 + +scal_index * 4 + 4] = \
                        [(gt_rect[0] - cent_dbox[0]) / float(scal),
                         (gt_rect[1] - cent_dbox[1]) / float(scal),
                         np.log(gt_rect[2] / float(scal)),
                         np.log(gt_rect[3] / float(scal))]
    return cls_map, reg_map, test_time


def project_feature_map_simple(gt_rect, map_height, map_width, scale, stride, point_type, map_type):
    # t_start = time.time()
    top, bottom, left, right = gt_rect[1] - gt_rect[3] // 2, \
                               gt_rect[1] + gt_rect[3] // 2, \
                               gt_rect[0] - gt_rect[2] // 2, \
                               gt_rect[0] + gt_rect[2] // 2
    # max_scale = max(scale)
    height, width = map_height, map_width
    default_boxes = list()
    position = list()
    gt_rects = list()
    gt_boxes = list()
    scale_list = list()
    valid_number = 0
    test_n = 0

    for scal_index, scal in enumerate(scale):
        if scal < gt_rect[-1] * 0.707 or 0.707 * scal > gt_rect[-1]:  # 0.707=sqrt(1^2/2)
            continue
        invalid_border_length = min(scal, gt_rect[-1]) / 3.
        # invalid_border_length0
        for step_H in range(int(np.floor((top - scal / 2 + invalid_border_length) / stride)),
                            int(np.ceil((bottom + scal / 2 - invalid_border_length) / stride))):
            for step_W in range(int(np.floor((left - scal / 2 + invalid_border_length) / stride)),
                                int(np.ceil((right + scal / 2 - invalid_border_length) / stride))):
                test_n += 1
                if step_H < 0 or step_W < 0 or step_H > height - 1 or step_W > width - 1:
                    continue

                valid_number += 1
                default_box = [int(step_W * stride + stride / 2 - scal / 2),
                               int(step_H * stride + stride / 2 - scal / 2),
                               int(step_W * stride + stride / 2 + scal / 2),
                               int(step_H * stride + stride / 2 + scal / 2)]

                default_boxes.append(default_box)
                position.append([step_H, step_W, scal_index, len(scale), point_type, map_type])
                gt_rects.append(gt_rect)
                gt_boxes.append([int(left), int(top), int(right), int(bottom)])
                scale_list.append(scal)

    # t_over = time.time()
    # if t_over - t_start > 0.1:
    #     print(t_over - t_start, stride, len(scale_list), test_n, gt_rect[-1])
    return default_boxes, position, gt_rects, gt_boxes, scale_list


# cls_map: list of feature_map include f11 to f3
# reg_map: as same as cls_map's format
# default_boxes: default box matrix :[[left, top, right, bottom]...]
# iou_matrix: a vector include iou information
# position_matrix: [[step_H, step_W, scal_index, len(scale),point_type,map_type]...]
# gt_rects: ground truth like: [[x, y, ss, ss]...]
# scale_matrix: the matrix include scale
def project_feature_map_iou(cls_maps, reg_maps, default_boxes, iou_matrix, position_matrix, gt_rects, scale_matrix,
                            threshold=0.5):
    iou_matrix = iou_matrix > threshold
    position_matrix_roi = position_matrix[iou_matrix]
    scale_matrix_roi = scale_matrix[iou_matrix]
    gt_rects_iou = gt_rects[iou_matrix]
    default_boxes_iou = default_boxes[iou_matrix]
    reg_matrix_x = gt_rects_iou[:, 0] - (default_boxes_iou[:, 0] + default_boxes_iou[:, 2]) / 2.
    reg_matrix_y = gt_rects_iou[:, 1] - (default_boxes_iou[:, 1] + default_boxes_iou[:, 3]) / 2.
    for pos_index, pos in enumerate(position_matrix_roi):
        step_H = pos[0]
        step_W = pos[1]
        scal_index = pos[2]
        scal_length = pos[3]
        point_type = pos[4]
        cls_maps[pos[-1]][0, step_H, step_W,
        point_type * scal_length * 2 + scal_index * 2:point_type * scal_length * 2 + scal_index * 2 + 2] = [0, 1]
        reg_maps[pos[-1]][0, step_H, step_W,
        point_type * scal_length * 4 + scal_index * 4:point_type * scal_length * 4 + +scal_index * 4 + 4] = \
            [reg_matrix_x[pos_index] / float(scale_matrix_roi[pos_index]),
             reg_matrix_y[pos_index] / float(scale_matrix_roi[pos_index]),
             np.log(gt_rects_iou[pos_index, 2] / float(scale_matrix_roi[pos_index])),
             np.log(gt_rects_iou[pos_index, 3] / float(scale_matrix_roi[pos_index]))]
    return cls_maps, reg_maps


def init_template_f_in(scales, strides):
    templet_f_in = np.zeros([7, 128, 128, 6, 4], dtype=np.int32)
    templet_height_index = np.array(range(128) * 128).reshape([128, 128]).T
    templet_width_index = np.array(range(128) * 128).reshape([128, 128])
    for sca_index, scale in enumerate(scales):
        for default_index, default_box in enumerate(scale):
            templet_f_in[sca_index, :, :, default_index, 0] = np.cast[np.int32]((templet_width_index + 0.5) * strides[
                sca_index] - default_box // 2)
            templet_f_in[sca_index, :, :, default_index, 1] = np.cast[np.int32]((templet_height_index + 0.5) * strides[
                sca_index] - default_box // 2)
            templet_f_in[sca_index, :, :, default_index, 2] = np.cast[np.int32]((templet_width_index + 0.5) * strides[
                sca_index] + default_box // 2)
            templet_f_in[sca_index, :, :, default_index, 3] = np.cast[np.int32]((templet_height_index + 0.5) * strides[
                sca_index] + default_box // 2)
    return templet_f_in.reshape([1, 7, 128, 128, 6, 4])


# gt_matrix: 2d tensor like:[[left, top, right, bottom],...]
# cls_f_in:6d tensor like:[gt_s, scale, height, width, default box, [left, top, right, bottom]]
# reg_map:4d tensor
# scale:[scale of default box]
# stride:stride of map
# point_type:0-3
# threshold:the threshold of iou
def project_feature_map_matrix(gt_matrix, scales, strides, threshold=0.5):
    template_f_in = init_template_f_in(scales, strides)
    gt_matrix = gt_matrix.reshape([-1, 1, 1, 1, 1, 4])
    cls_f_in = None
    gts_num = gt_matrix.shape[0]
    t_1 = time.time()
    for gt_num in range(gts_num):
        cls_f_in = template_f_in if cls_f_in is None else np.append(cls_f_in, template_f_in, axis=0)
    # init
    area_default_boxes = ((cls_f_in[:, :, :, :, :, 2] - cls_f_in[:, :, :, :, :, 0]) *
                          (cls_f_in[:, :, :, :, :, 3] - cls_f_in[:, :, :, :, :, 1])).reshape(
        [gts_num, 7, 128, 128, 6, 1])
    area_gt = ((gt_matrix[:, :, :, :, :, 2] - gt_matrix[:, :, :, :, :, 0]) *
               (gt_matrix[:, :, :, :, :, 3] - gt_matrix[:, :, :, :, :, 1])).reshape([gts_num, 1, 1, 1, 1, 1])
    valid_region = cls_f_in[:, :, :, :, :, 0] < gt_matrix[:, :, :, :, :, 2]
    valid_region = np.logical_and(valid_region, cls_f_in[:, :, :, :, :, 2] > gt_matrix[:, :, :, :, :, 0])
    valid_region = np.logical_and(valid_region, cls_f_in[:, :, :, :, :, 1] < gt_matrix[:, :, :, :, :, 3])
    valid_region = np.logical_and(valid_region, cls_f_in[:, :, :, :, :, 3] > gt_matrix[:, :, :, :, :, 1])
    valid_region = np.cast[np.float32](valid_region).reshape([gts_num, 7, 128, 128, 6, 1])
    # cls_f_in = cls_f_in * valid_region
    # combine_in_gt: [left1, right1, left2, right2, top1, bottom1, top2, bottom2]
    combine_in_gt = np.zeros([gts_num, 7, 128, 128, 6, 8], dtype=np.int32)
    t0 = time.time()
    print(t0 - t_1)
    combine_in_gt[:, :, :, :, :, 0] = cls_f_in[:, :, :, :, :, 0]
    combine_in_gt[:, :, :, :, :, 1] = cls_f_in[:, :, :, :, :, 2]
    combine_in_gt[:, :, :, :, :, 2] = gt_matrix[:, :, :, :, :, 0]
    combine_in_gt[:, :, :, :, :, 3] = gt_matrix[:, :, :, :, :, 2]
    combine_in_gt[:, :, :, :, :, 4] = cls_f_in[:, :, :, :, :, 1]
    combine_in_gt[:, :, :, :, :, 5] = cls_f_in[:, :, :, :, :, 3]
    combine_in_gt[:, :, :, :, :, 6] = gt_matrix[:, :, :, :, :, 1]
    combine_in_gt[:, :, :, :, :, 7] = gt_matrix[:, :, :, :, :, 3]
    combine_in_gt = combine_in_gt * valid_region
    t1 = time.time()
    print(t1 - t0)
    # sort rectangle points
    combine_in_gt_W = np.sort(combine_in_gt[:, :, :, :, :, 0:4], axis=-1)
    combine_in_gt_H = np.sort(combine_in_gt[:, :, :, :, :, 4:8], axis=-1)
    t2 = time.time()
    print(t2 - t1)
    area_i = ((combine_in_gt_W[:, :, :, :, :, 2] - combine_in_gt_W[:, :, :, :, :, 1]) *
              (combine_in_gt_H[:, :, :, :, :, 2] - combine_in_gt_H[:, :, :, :, :, 1])).reshape(
        [gts_num, 7, 128, 128, 6, 1])

    iou = area_i / (area_default_boxes + area_gt - area_i)

    return iou


# gt_matrix: 1d tensor like:[left, top, right, bottom]
# cls_f_in:6d tensor like:[1, scale, height, width, default box, [left, top, right, bottom]]
# reg_map:4d tensor
# scale:[scale of default box]
# stride:stride of map
# point_type:0-3
# threshold:the threshold of iou
t0_0 = time.time()
templete_one = cp.ones((7, 128, 128, 1), dtype=cp.float32)
templete_zero = cp.zeros((7, 128, 128, 1), dtype=cp.float32)
templete_cls = cp.concatenate([templete_one, templete_zero] * 24, axis=-1)
print('**********%f' % (time.time() - t0_0))


def project_feature_map_matrix_cupy(cls_f_in, gt_matrix, combine_in_gt, point_type, threshold=0.5):
    gt_matrix = gt_matrix.reshape([-1, 1, 1, 1, 1, 4])
    # init

    t_1 = time.time()
    area_default_boxes = ((cls_f_in[:, :, :, :, :, 2] - cls_f_in[:, :, :, :, :, 0]) *
                          (cls_f_in[:, :, :, :, :, 3] - cls_f_in[:, :, :, :, :, 1])).reshape([7, 128, 128, 6])
    t0 = time.time()
    print(t0 - t_1)
    area_gt = ((gt_matrix[:, :, :, :, :, 2] - gt_matrix[:, :, :, :, :, 0]) *
               (gt_matrix[:, :, :, :, :, 3] - gt_matrix[:, :, :, :, :, 1])).reshape([1, 1, 1, 1])
    valid_region = cls_f_in[:, :, :, :, :, 0] < gt_matrix[:, :, :, :, :, 2]
    valid_region = cp.logical_and(valid_region, cls_f_in[:, :, :, :, :, 2] > gt_matrix[:, :, :, :, :, 0])
    valid_region = cp.logical_and(valid_region, cls_f_in[:, :, :, :, :, 1] < gt_matrix[:, :, :, :, :, 3])
    valid_region = cp.logical_and(valid_region, cls_f_in[:, :, :, :, :, 3] > gt_matrix[:, :, :, :, :, 1])
    valid_region = valid_region.reshape([1, 7, 128, 128, 6, 1]).astype('f')
    # cls_f_in = cls_f_in * valid_region
    # combine_in_gt: [left1, right1, left2, right2, top1, bottom1, top2, bottom2]

    combine_in_gt[:, :, :, :, :, 0] = cls_f_in[:, :, :, :, :, 0]
    combine_in_gt[:, :, :, :, :, 1] = cls_f_in[:, :, :, :, :, 2]
    combine_in_gt[:, :, :, :, :, 2] = gt_matrix[:, :, :, :, :, 0]
    combine_in_gt[:, :, :, :, :, 3] = gt_matrix[:, :, :, :, :, 2]
    combine_in_gt[:, :, :, :, :, 4] = cls_f_in[:, :, :, :, :, 1]
    combine_in_gt[:, :, :, :, :, 5] = cls_f_in[:, :, :, :, :, 3]
    combine_in_gt[:, :, :, :, :, 6] = gt_matrix[:, :, :, :, :, 1]
    combine_in_gt[:, :, :, :, :, 7] = gt_matrix[:, :, :, :, :, 3]
    combine_in_gt = combine_in_gt * valid_region
    t1 = time.time()
    print(t1 - t0)
    # sort rectangle points
    combine_in_gt_W = cp.sort(combine_in_gt[:, :, :, :, :, 0:4], axis=-1)
    combine_in_gt_H = cp.sort(combine_in_gt[:, :, :, :, :, 4:8], axis=-1)
    t2 = time.time()
    print(t2 - t1)
    area_i = ((combine_in_gt_W[:, :, :, :, :, 2] - combine_in_gt_W[:, :, :, :, :, 1]) *
              (combine_in_gt_H[:, :, :, :, :, 2] - combine_in_gt_H[:, :, :, :, :, 1])).reshape([7, 128, 128, 6])

    iou = area_i / (area_default_boxes + area_gt - area_i)
    gts_feature = (iou > threshold).astype(cp.float32)

    cls_feature = cp.concatenate([1. - gts_feature, gts_feature], axis=-1)
    cls_feature = cls_feature.reshape([7, 128, 128, 12])

    t3 = time.time()
    templete_cls[0:6, :, :, 8 * point_type:8 * point_type + 8] += cls_feature[0:6, :, :, 0:8]
    templete_cls[6, :, :, 8 * point_type:8 * point_type + 12] += cls_feature[6, :, :, 0:12]
    print(time.time() - t3)

    return iou


# gt_array:3d tensor shape like [2,4,?]
# seg_map:4d tensor shape like [1,512,512,4]
def project_feature_map_seg(gt_array, seg_map):
    for point_index in range(gt_array.shape[-1]):
        point_corner = gt_array[:, :, point_index].T
        assert point_corner.shape[0] == 4, 'AmountError: incorrect number in point corner.'
        full_corner = copy.deepcopy(point_corner)
        for i in range(4):
            before_point = point_corner[i, :]
            after_point = point_corner[0, :] if i == 3 else point_corner[i + 1, :]
            center_point = np.array([(before_point[0] + after_point[0]) / 2,
                                     (before_point[1] + after_point[1]) / 2]).reshape([1, 2]).astype(np.int32)
            full_corner = np.insert(full_corner, i * 2 + 1, center_point, axis=0)
        center_point = np.array([(full_corner[1, 0] + full_corner[5, 0]) / 2,
                                 (full_corner[3, 1] + full_corner[7, 1]) / 2]).astype(np.int32)

        points_list = list()
        points_list.append((full_corner[0] - 1).tolist() + (full_corner[1] - 1).tolist() + (center_point - 1).tolist() +
                           (full_corner[-1] - 1).tolist())
        points_list.append((full_corner[1] - np.array([0, 1])).tolist() + (full_corner[2] - np.array([0, 1])).tolist() +
                           (full_corner[3] - np.array([0, 1])).tolist() + (center_point - np.array([0, 1])).tolist())
        points_list.append(center_point.tolist() + full_corner[3].tolist() + full_corner[4].tolist() +
                           full_corner[5].tolist())
        points_list.append((full_corner[-1] - np.array([1, 0])).tolist() + (center_point - np.array([1, 0])).tolist() +
                           (full_corner[5] - np.array([1, 0])).tolist() + (full_corner[6] - np.array([1, 0])).tolist())
        for i in range(4):
            seg_map[0, :, :, i] = cv2.drawContours(copy.deepcopy(seg_map[0, :, :, i]),
                                                   [np.array(points_list[i]).reshape([4, 2]).astype(np.int32)],
                                                   0, 1., cv2.FILLED)
    return seg_map


def array2list_CSTR_dict(dict):
    for batch in dict:
        cls_list = batch['cls_mask']
        for index, cls_array in enumerate(cls_list):
            cls_list[index] = cls_array.tolist()
        reg_list = batch['reg_mask']
        for index, reg_array in enumerate(reg_list):
            reg_list[index] = reg_array.tolist()
        seg_list = batch['seg_mask']
        for index, seg_array in enumerate(seg_list):
            seg_list[index] = seg_array.tolist()
    return dict


if __name__ == '__main__':
    # cls_mask = np.ones([1, 64, 64, 1], dtype=np.float32)
    # for i in range(47):
    #     if i % 2 == 0:
    #         cls_mask = np.append(cls_mask, np.zeros([1, 64, 64, 1], dtype=np.float32), axis=3)
    #     else:
    #         cls_mask = np.append(cls_mask, np.ones([1, 64, 64, 1], dtype=np.float32), axis=3)
    # reg_mask = np.zeros([1, 64, 64, 96], dtype=np.float32)
    # cls_mask, reg_mask = project_feature_map([153.57, 72.53, 23.367, 23.367], cls_mask, reg_mask,
    #                                          [20, 24, 28, 32], 8, 1)

    # seg_map = np.zeros([1, 512, 512, 4], dtype=np.float32)
    # gt_array = np.array([[10.1, 30, 30, 10], [10, 10, 30, 30]]).reshape([2, 4, 1])
    # seg_map = project_feature_map_seg(gt_array, seg_map)
    # cv2.imshow('test1', seg_map[0, :, :, 0])
    # cv2.imshow('test2', seg_map[0, :, :, 1])
    # cv2.imshow('test3', seg_map[0, :, :, 2])
    # cv2.imshow('test4', seg_map[0, :, :, 3])
    # cv2.waitKey()
    # a = 1

    scale_table = [[184, 208, 232, 256],
                   [124, 136, 148, 160],
                   [88, 96, 104, 112],
                   [56, 64, 72, 80],
                   [36, 40, 44, 48],
                   [20, 24, 28, 32],
                   [4, 8, 6, 10, 12, 16]]
    strides = [128, 85.3333, 64, 32, 16, 8, 4]
    # test = init_template_f_in(scale_table, [128, 85.3333, 64, 32, 16, 8, 4])
    gt_matrix = np.array([[10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                          [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250]],
                         dtype=np.int32)
    t0 = time.time()
    test1 = project_feature_map_matrix(gt_matrix, scale_table, strides)
    print(time.time() - t0)

    gt_matrix_cus = cp.array([[10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250],
                              [10, 10, 110, 110], [200, 200, 250, 250], [10, 10, 110, 110], [200, 200, 250, 250]],
                             dtype=cp.int32)
    cls_f_in = cp.array(init_template_f_in(scale_table, strides))
    combine_in_gt = cp.zeros((1, 7, 128, 128, 6, 8), dtype=cp.int32)
    print('#############')
    t1 = time.time()
    for gt_matrix_cu in gt_matrix_cus:
        gts_index = project_feature_map_matrix_cupy(cls_f_in, gt_matrix_cu, combine_in_gt, 0)
    #     for gt_index in gts_index:

    print(time.time() - t1)
    a = 1
