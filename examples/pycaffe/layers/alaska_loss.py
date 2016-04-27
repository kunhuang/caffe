import caffe
import numpy as np
import pdb

class AlaskaLossLayer(caffe.Layer):
    """
    """
    def gradient_checker(self, bottom, top):
        

        pdb.set_trace()
        
        delta = 0.0000001
        
        loss = self._forward(bottom, top)
        
        bottom[0].data[0, 0, 0, 0] += delta
        loss_ = self._forward(bottom, top)
        bottom[0].data[0, 0, 0, 0] -= delta
        numeric_gradient_0 = (loss_-loss)/delta
        
        bottom[0].data[0, 1, 0, 0] += delta
        loss_ = self._forward(bottom, top)
        bottom[0].data[0, 1, 0, 0] -= delta
        numeric_gradient_1 = (loss_-loss)/delta

        gradient = self._backward(top, None, bottom)
        real_gradient_0 = gradient[0,0,0,0]
        real_gradient_1 = gradient[0,1,0,0]
        
        print numeric_gradient_0, ' vs. ', real_gradient_0
        print numeric_gradient_1, ' vs. ', real_gradient_1
        
        pdb.set_trace()
        return True

    def test_forward():
        n = 2
        num_L = 3
        w = h = 2

        np.append(L, np.ones((n,1)), axis=1)
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
            bottom[1]:alpha, (float[n*num_L*w*h]), 0~1
            bottom[2]:L, (bool[n*(num_L-1)*1*1]), except the background
        '''
        # check input pair
        
        self.n, self.num_L, self.w, self.h = bottom[0].data.shape

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

    def _forward(self, bottom, top):
        # pdb.set_trace()
        S, alpha, L = bottom[0].data, bottom[1].data, bottom[2].data
        L = np.append(L, np.ones((self.n,1)), axis=1)
        # Except background
        
        self.max_L = np.max(S.reshape(self.n, self.num_L, self.w*self.h), axis=2)
        self.max_flatten_index = np.argmax(S.reshape(self.n, self.num_L, self.w*self.h), axis=2)

        # Without background
        self.true_L = np.argwhere(L[:,:-1]==1)
        self.false_L = np.argwhere(L[:,:-1]==0)
        # TODO, avoid overflow
        true_sum = np.sum(np.log(0.0000001+self.max_L[[self.true_L[:,0],self.true_L[:,1]]]))/len(self.true_L)
        false_sum = np.sum(np.log(1.-self.max_L[[self.false_L[:,0],self.false_L[:,1]]]))/len(self.false_L)

        # Elementwise multiplication
        # TODO, avoid overflow
        log_sum = np.sum(alpha*np.log(0.00000001+S))/(self.num_L*self.w*self.h)

        return -(true_sum + false_sum + log_sum)/self.n
    
    def forward(self, bottom, top):
        if self.gradient_checker(bottom, top) is False:
            raise Exception('Gradient not correct')
            
        top[0].data[...] = self._forward(bottom, top)

    def _backward(self, top, propagate_down, bottom):
        # pdb.set_trace()
        diff = np.zeros((self.n, self.num_L, self.w*self.h), dtype=np.float32)
        # diff = self.diff.reshape(self.n, self.num_L, self.w*self.h)

        d0, d1 = np.argwhere(self.max_flatten_index)[:,0], np.argwhere(self.max_flatten_index)[:,1]
        true_max_diff = np.zeros_like(diff, dtype=np.float32)
        # TODO, avoid overflow
        true_max_diff[self.true_L[:,0], self.true_L[:,1], self.max_flatten_index[self.true_L[:,0], self.true_L[:,1]]] = 1./(0.00000001+self.max_L[[self.true_L[:,0],self.true_L[:,1]]])
        diff += -true_max_diff/len(self.true_L)

        false_max_diff = np.zeros_like(diff, dtype=np.float32)
        false_max_diff[self.false_L[:,0], self.false_L[:,1], self.max_flatten_index[self.false_L[:,0], self.false_L[:,1]]] = -1./(1.-self.max_L[[self.false_L[:,0],self.false_L[:,1]]])
        diff += -false_max_diff/len(self.false_L)

        # TODO, avoid flow
        # diff += -np.sum(np.divide(bottom[1].data, 0.00000001+bottom[0].data))/(self.num_L*self.w*self.h)
        
        diff = diff.reshape(self.n, self.num_L, self.w, self.h)
        diff += -np.divide(bottom[1].data, 0.00000001+bottom[0].data)/(self.num_L*self.w*self.h)
        diff /= self.n

        return diff
    
    def backward(self, top, propagate_down, bottom):
        self.diff = self._backward(top, propagate_down, bottom)