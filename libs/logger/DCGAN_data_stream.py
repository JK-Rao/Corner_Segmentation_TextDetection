# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      DCGAN_data_stream.py
# @Software  PyCharm
# @Author    JK.Rao

from .data_pipeline import TfReader, TfWriter, DataPipeline
import cv2


class DCGANReader(TfReader):
    def __init__(self):
        TfReader.__init__(self, 32, 20, 1)


class DCGANWriter(TfWriter):
    def __init__(self):
        TfWriter.__init__(self, 32, 20, 1)


def DCGAN_get_pipeline(sess, propose, batch_size, filename):
    stream = DataPipeline('.tfrecords', propose, './data')
    return stream.tfrecords2imgs_tensor(sess, filename, batch_size, DCGANReader(), cv2.IMREAD_GRAYSCALE)
