# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      gadget.py
# @Software  PyCharm
# @Author    JK.Rao

import os
import numpy as np
import shutil


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


# gt_rect:[cx,cy,ss,ss]
# cls_map:4d tensor
# reg_map:4d tensor
# scale:[scale of default box]
# stride:stride of map
# point_type:0-3
# threshold:the threshold of iou
def project_feature_map(gt_rect, cls_map, reg_map, scale, stride, point_type, threshold=0.5):
    top, bottom, left, right = gt_rect[1] - gt_rect[3] // 2, \
                               gt_rect[1] + gt_rect[3] // 2, \
                               gt_rect[0] - gt_rect[2] // 2, \
                               gt_rect[0] + gt_rect[2] // 2
    max_scale = max(scale)
    height, width = cls_map.shape[1:3]
    for step_H in range(int(np.floor((top - max_scale // 2) / stride)),
                        int(np.ceil((bottom + max_scale // 2) / stride))):
        for step_W in range(int(np.floor((left - max_scale // 2) / stride)),
                            int(np.ceil((right + max_scale // 2) / stride))):
            if step_H < 0 or step_W < 0 or step_H > height or step_W > width:
                continue
            for scal_index, scal in enumerate(scale):
                default_box = [int(step_W * stride + stride / 2 - scal / 2),
                               int(step_H * stride + stride / 2 - scal / 2),
                               int(step_W * stride + stride / 2 + scal / 2),
                               int(step_H * stride + stride / 2 + scal / 2)]
                ins_score = calcul_iou(default_box, [int(left), int(top), int(right), int(bottom)])
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
    return cls_map, reg_map


if __name__ == '__main__':
    cls_mask = np.ones([1, 64, 64, 1], dtype=np.float32)
    for i in range(47):
        if i % 2 == 0:
            cls_mask = np.append(cls_mask, np.zeros([1, 64, 64, 1], dtype=np.float32), axis=3)
        else:
            cls_mask = np.append(cls_mask, np.ones([1, 64, 64, 1], dtype=np.float32), axis=3)
    reg_mask = np.zeros([1, 64, 64, 96], dtype=np.float32)
    cls_mask, reg_mask = project_feature_map([153.57, 72.53, 23.367, 23.367], cls_mask, reg_mask, [20, 24, 28, 32], 8,
                                             1)
    a = 1
