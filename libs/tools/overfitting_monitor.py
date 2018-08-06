# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      overfitting_monitor.py
# @Software  PyCharm
# @Author    JK.Rao

from __future__ import print_function
import cv2
import numpy as np
import copy
import os


def calcu_Euclidean_distance(im1, im2):
    im1 = im1.astype(np.int32)
    im2 = im2.astype(np.int32)
    return np.linalg.norm(im1 - im2, ord=2)


def min_distance_set(obj_file, file_set):
    try:
        index = file_set.index(obj_file)
    except ValueError as e:
        print(e)
        return -1, -1
    temp_set = copy.deepcopy(file_set)
    temp_set.pop(index)
    min_distance = 999999.
    ave_distance = 0.
    min_obj_name = obj_file
    min_file_name = ''
    for file_name in temp_set:
        dis = calcu_Euclidean_distance(cv2.imread(obj_file, -1), cv2.imread(file_name, -1))
        min_distance, min_file_name = [dis, file_name] if dis < min_distance else [min_distance, min_file_name]
        ave_distance += dis
    ave_distance /= len(temp_set)
    return min_distance, ave_distance, min_obj_name, min_file_name


def random_sampling(file_set, iter_num):
    min_dis = 999999.
    ave_dis = 0.
    abs_obj_name = ''
    abs_file_name = ''
    print('calculate distance',end='')
    for i in range(iter_num):
        print(' %d/%d' % (i + 1, iter_num),end='')
        min, ave, prob_obj_name, prob_file_name = min_distance_set(file_set[np.random.randint(0, len(file_set))],
                                                                   file_set)
        min_dis, abs_obj_name, abs_file_name = [min, prob_obj_name, prob_file_name] if min < min_dis \
            else [min_dis, abs_obj_name, abs_file_name]
        ave_dis += ave
    print('')
    return min_dis, ave_dis / iter_num, abs_obj_name, abs_file_name


if __name__ == '__main__':
    inster_number = 0
    iter_num = 1
    PATH = './out_bank_CNN_num%d' % inster_number
    file_names = os.listdir(PATH)
    file_names = [os.path.join(PATH, a) for a in file_names]
    min_dis, ave_dis, obj_name, file_name = random_sampling(file_names, iter_num)
    print(min_dis, ave_dis, obj_name, file_name)
    cv2.imshow('obj', cv2.imread(obj_name))
    cv2.imshow('file', cv2.imread(file_name))
    cv2.waitKey()
