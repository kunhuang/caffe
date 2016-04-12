import caffe
import numpy as np


class AlaskaLossLayer(caffe.Layer):
    """
    """

    def test_forward():
        n = 2
        num_L = 3
        w = h = 2

        # Except background
        L = np.asarray([[0, 0, 1], [0, 1, 1]])
        S = np.asarray([[[[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.1, 0.9]], [[0.1, 0.1], [0.9, 0.1]]], [[[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]]]])
        alpha = np.asarray([[[[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]]], [[[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]], [[0.1, 0.1], [0.9, 0.1]]]])

        max_L = np.max(S.reshape(n, num_L, w*h), axis=2)
        max_flatten_index = np.argmax(S.reshape(n, num_L, w*h), axis=2)

        # Without background
        true_L = np.argwhere(L[:,:-1]==1)
        false_L = np.argwhere(L[:,:-1]==0)
        true_sum = np.sum(np.log(max_L[[true_L[:,0],true_L[:,1]]]))/len(true_L)
        false_sum = np.sum(np.log(1.-max_L[[false_L[:,0],false_L[:,1]]]))/len(false_L)

        # Elementwise multiplication
        log_sum = np.sum(alpha*np.log(S))/(num_L*w*h)

        top0 = true_sum + false_sum + log_sum
        # self.top[0].data[...] = true_sum + false_sum + log_sum


    def test_backward():
        n = 2
        num_L = 3
        w = h = 2

        diff = np.zeros_like(S, dtype=np.float32)
        diff = diff.reshape(n, num_L, w*h)

        d0, d1 = np.argwhere(max_flatten_index)[:,0], np.argwhere(max_flatten_index)[:,1]
        true_max_diff = np.zeros_like(diff, dtype=np.float32)
        true_max_diff[true_L[:,0], true_L[:,1], max_flatten_index[true_L[:,0], true_L[:,1]]] = 1./max_L[[true_L[:,0],true_L[:,1]]]
        diff += -true_max_diff/len(true_L)

        false_max_diff = np.zeros_like(diff, dtype=np.float32)
        false_max_diff[false_L[:,0], false_L[:,1], max_flatten_index[false_L[:,0], false_L[:,1]]] = 1./(1.-max_L[[false_L[:,0],false_L[:,1]]])
        diff += false_max_diff/len(false_L)

        diff += np.sum(np.divide(alpha, S))/(num_L*w*h)

    def setup(self, bottom, top):
        '''
        Args:
            bottom[0]:S, (float[n*num_L*w*h]), 0~1
            bottom[1]:L, (bool[n*num_L*1*1]), including the background, in L[:, -1]
            bottom[2]:alpha, (float[n*num_L*w*h]), 0~1
        '''
        # check input pair

        self.n, self.num_L, self.w, self.h = self.bottom[0].data.shape

        if len(bottom) != 3:
            raise Exception("bottom[0]:S, bottom[1]:alpha, bottom[2]:L")
        if bottom[0].data.shape != bottom[1].data.shape:
            raise Exception("S and alpha should be same shape")


    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        pdb.set_trace()
        max_L = np.max(bottom[0].data, axis=[2,3])


        self.diff[...] = bottom[0].data - bottom[1].data
        

        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        pdb.set_trace()
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
