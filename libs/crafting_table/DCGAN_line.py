# _*_ coding: utf-8 _*_
# @Time      18-7-27 下午5:15
# @File      DCGAN_line.py
# @Software  PyCharm
# @Author    JK.Rao

from .assembly_line import AssemblyLine
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
from os.path import join
import os
from ..network.factory import get_network
from ..network.factory import get_network_name
from ..logger.factory import get_sample_tensor
from ..logger.data_pipeline import DataPipeline
from ..tools.gadget import mk_dir
from ..tools.overfitting_monitor import random_sampling


class DCGANLine(AssemblyLine):
    def __init__(self, propose, info_dict):
        if propose == 'train':
            AssemblyLine.__init__(self, DCGANLine.get_config(), get_network('DCGAN'))
            self.inster_number = info_dict['inster_number']
            self.annotation = info_dict['annotation']
            self.batch_size = info_dict['batch_size']
            self.val_size = info_dict['val_size']
            self.IMG_CHANEL = self.network.IMG_CHANEL
        elif propose == 'test':
            AssemblyLine.__init__(self, DCGANLine.get_config(), None)
            self.model_name = info_dict['model_name']
            self.parameter_name = info_dict['parameter_name']
            self.batch_size = info_dict['batch_size']
            self.Z_dim = 100
            self.IMG_CHANEL = 1
        else:
            raise ValueError('No type like:%s of DCGANLine class...' % propose)

    @staticmethod
    def get_config():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def sample_Z(self, m, n):
        return np.random.uniform(-1., 1., size=[m, n])

    def plot(self, samples):
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):  # [i,samples[i]] imax=16
            sample = sample + 0.5
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_aspect('equal')
            if self.IMG_CHANEL == 1:
                plt.imshow(sample.reshape(32, 20), cmap='Greys_r')
            else:
                plt.imshow(sample.reshape(32, 20, self.network.IMG_CHANEL), cmap='Greys_r')
        return fig

    def structure_train_context(self):
        saver = self.get_saver(self.network.get_trainable_var(self.network.net_name[0]))
        loss_dict = self.network.structure_loss()
        opti_dict = self.network.define_optimizer(loss_dict)
        merged = self.create_summary('.logs/log_num%d' % self.iter_num)
        with self.sess:
            self.sess.run(tf.global_variables_initializer())
            self.save_model(saver,
                            './model_DCGAN_num%d%s/iter_meta.ckpt' % (self.inster_number, self.annotation))

            i = 0
            mk_dir('out_bank_CNN_num%d%s/' % (self.inster_number, self.annotation))
            X_mb_tensor = get_sample_tensor('DCGAN', self.sess, 'train', self.batch_size,
                                            'train_num%d' % self.inster_number)
            X_val_tensor = get_sample_tensor('DCGAN', self.sess, 'val', self.val_size,
                                             'val_num%d' % self.inster_number)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

            for iter in range(70000):
                if iter % 1000 == 0:
                    self.iter_num = iter
                    samples = self.sess.run(self.network.get_pred()['gen_im'], feed_dict={
                        self.network.Z: self.sample_Z(16, self.network.Z_dim), self.network.on_train: False,
                        self.network.batch_pattern: 16})  # 16*784
                    fig = self.plot(samples)
                    plt.savefig('out_bank_CNN_num%d%s/' % (self.inster_number, self.annotation) + '/{}.png'.format(
                        str(i).zfill(3)),
                                bbox_inches='tight')
                    i += 1
                    plt.close(fig)
                X_mb = DataPipeline.tensor2data(self.sess,X_mb_tensor)

                _, D_loss_curr = self.sess.run([opti_dict['d_opti'], loss_dict['d_loss']], feed_dict={
                    self.network.X: X_mb,
                    self.network.Z: self.sample_Z(self.batch_size, self.network.Z_dim),
                    self.network.on_train: True,
                    self.network.batch_pattern: self.batch_size})

                _, G_loss_curr = self.sess.run([opti_dict['g_opti'], loss_dict['g_loss']], feed_dict={
                    self.network.Z: self.sample_Z(self.batch_size, self.network.Z_dim),
                    self.network.on_train: True,
                    self.network.batch_pattern: self.batch_size})
                # if iter % 100 == 0:
                #     print('Iter:%d  G_loss:%f,D_loss:%f' % (iter, G_loss_curr, D_loss_curr))
                if iter % 1000 == 0:
                    # overfitting record
                    j = 0
                    print('Iter:%d  G_loss:%f,D_loss:%f' % (iter, G_loss_curr, D_loss_curr))
                    samples = self.sess.run(self.network.get_pred()['gen_im'], feed_dict={
                        self.network.Z: self.sample_Z(1000, self.network.Z_dim),
                        self.network.on_train: False, self.network.batch_pattern: 1000})
                    PATH = './temp_CNN_num%d' % self.inster_number
                    mk_dir(PATH)
                    for temp_file in os.listdir(PATH):
                        os.remove(join(PATH,temp_file))
                    for line in range(1000):
                        cv2.imwrite(join(PATH, '%08d.jpg' % j),
                                    np.round((samples[line, :, :, 0] + 0.5) * 255))
                        j += 1
                    iter_num = 10
                    file_names = os.listdir(PATH)
                    file_names = [os.path.join(PATH, a) for a in file_names]
                    min_dis, ave_dis, _, _ = random_sampling(file_names, iter_num)
                    self.sess.run(tf.assign(self.network.Min_distance, min_dis))
                    self.sess.run(tf.assign(self.network.Ave_distance, ave_dis))
                    # loss record
                    X_val = DataPipeline.tensor2data(self.sess,X_val_tensor)
                    mg = self.sess.run(merged, feed_dict={
                        self.network.X: X_val,
                        self.network.on_train: False,
                        self.network.Z: self.sample_Z(self.val_size, self.network.Z_dim),
                        self.network.batch_pattern: self.val_size})
                    self.write_summary(mg)

                gl_step = self.sess.run(self.network.global_step)
                if gl_step % 10000 == 0:
                    self.save_model(saver, './model_DCGAN_num%d%s/iter_%d_num%d.ckpt' \
                                    % (self.inster_number, self.annotation, gl_step, self.inster_number),
                                    write_meta_graph=False)

            coord.request_stop()
            coord.join(threads)
            self.close_summary_writer()
        self.sess.close()

    def restore_test_context(self, print_im=True):
        mk_dir('./out')
        graph = self.restore_model(self.model_name, self.parameter_name)
        Z = graph.get_operation_by_name('Z').outputs[0]
        on_train = graph.get_operation_by_name('on_train').outputs[0]
        batch_size = graph.get_operation_by_name('batch_size').outputs[0]
        gen_im = graph.get_collection('out', scope=get_network_name('DCGAN')[0])[0]
        # gen_vars = self.network.get_trainable_var(self.network.net_name[0])
        if print_im:
            for iter_num in range(8):
                print('print img %d/%d' % (iter_num+1, 8))
                samples = self.sess.run(gen_im, feed_dict={
                    Z: self.sample_Z(16, 100), on_train: False,
                    batch_size: 16})
                fig = self.plot(samples)
                plt.savefig('out/' + '%08d.png' % iter_num, bbox_inches='tight')
                plt.close(fig)
        samples = self.sess.run(gen_im, feed_dict={Z: self.sample_Z(self.batch_size, 100),
                                                   on_train: False, batch_size: self.batch_size})
        self.sess.close()
        return samples
