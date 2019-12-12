import numpy as np
import tensorflow as tf
import random
from .Neural_MINE import Neural_MINE
from .Classifier_MI import Classifier_MI


class CCMI(object):
    def __init__(self, X, Y, Z, tester, metric, num_boot_iter, h_dim, max_ep):

        self.dim_x = X.shape[1]
        self.dim_y = Y.shape[1]
        self.dim_z = Z.shape[1]
        self.data_xyz = np.hstack((X, Y, Z))
        self.data_xz = np.hstack((X, Z))
        self.threshold = 1e-4

        self.tester = tester
        self.metric = metric
        self.num_boot_iter = num_boot_iter
        self.h_dim = h_dim
        self.max_ep = max_ep

    def split_train_test(self, data):
        total_size = data.shape[0]
        train_size = int(2*total_size/3)
        data_train = data[0:train_size,:]
        data_test = data[train_size:, :]
        return data_train, data_test

    def gen_bootstrap(self, data):
        np.random.seed()
        random.seed()
        num_samp = data.shape[0]
        #I = np.random.choice(num_samp, size=num_samp, replace=True)
        I = np.random.permutation(num_samp)
        data_new = data[I, :]
        return data_new

    def get_cmi_est(self):
        print('Tester = {}, metric = {}'.format(self.tester, self.metric))
        I_xyz = self.get_mi_est(self.data_xyz)
        I_xz = self.get_mi_est(self.data_xz)

        cmi_est = I_xyz - I_xz

        return cmi_est

    def get_mi_est(self, data):
        I_list = []
        for t in range(self.num_boot_iter):
            tf.reset_default_graph()
            data_t = self.gen_bootstrap(data)
            data_train, data_eval = self.split_train_test(data_t)
            if self.tester == 'Neural':
                batch_size = 512 if self.metric == 'donsker_varadhan' else 128
                neurMINE = Neural_MINE(data_train, data_eval, self.dim_x,
                                       metric=self.metric, batch_size=batch_size)
                I_t, _ = neurMINE.train()
            elif self.tester == 'Classifier':
                classMINE = Classifier_MI(data_train, data_eval, self.dim_x,
                                          h_dim=self.h_dim, max_ep=self.max_ep)
                I_t = classMINE.train_classifier_MLP()
            else:
                raise NotImplementedError
            I_list.append(I_t)
        mi_est = np.mean(I_list)
        return mi_est

    def is_indp(self, cmi_est):
        if max(0, cmi_est) < self.threshold:
            return True
        else:
            return False
