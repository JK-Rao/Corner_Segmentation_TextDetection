# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      gadget.py
# @Software  PyCharm
# @Author    JK.Rao

import os
import shutil


def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def transform_file(source_file, obj_path,obj_name=None):
    mk_dir(obj_path)
    if obj_name is None:
        file_name = source_file.split('/')[-1]
    else:
        file_name=obj_name
    shutil.copy(source_file, os.path.join(obj_path, file_name))
