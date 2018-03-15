# This file contains the implementation of EuclideanLoss, SigmoidCrossEntropLoss 
# and SoftmaxWithLoss layers using caffe python layer. 

import caffe
import numpy as np
import sys

class SigmoidCrossEntropyLossLayer(caffe.Layer):
    """
    This layer is the combination of sigmoid layer (activate layer) and cross entropy loss layer.
    For more details about the formulas, please see: http://blog.csdn.net/qiusuoxiaozi/article/details/73250863 
    """
    def setup(self, bottom, top):
        # set up parameters
        params = eval(self.param_str)
        if params.has_key("ignore_label"):
            self.ignore_label = params["ignore_label"]
        else:
            self.ignore_label = -1
        if params.has_key("normalization"):
            self.normalization = params["normalization"]
        else:
            self.normalization = 1
            
        # check input pairs
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute loss.")
            
    def reshape(self, bottom, top):
        # batch size
        self._outer_num = bottom[0].shape[0]
        # instance num
        self._inner_num = bottom[0].count / self._outer_num
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same count.")
        # loss output is scalar
        top[0].reshape(1)
        
    def _get_normalizer(self, valid_count):
        # full mode
        if self.normalization == 0:
            normalizer = self._outer_num * self._inner_num
        # valid mode
        elif self.normalization == 1:
            normalizer = valid_count
        # batch size mode
        elif self.normalization == 2:
            normalizer = self._outer_num
        # none mode
        else:
            normalizer = 1.0
        # Some users will have no labels for some examples in order to 'turn off' a
        # particular loss in a multi-task setup. The max prevents NaNs in that case.
        return np.max([1.0, normalizer])
    
    @staticmethod
    def _sigmoid(x):
        ## The following code is not numerical stability
        # return 1. / (1. + np.exp(-x))
        return 0.5 * np.tanh(x * 0.5) + 0.5
    
    def forward(self, bottom, top):
        ### get data
        input_data = bottom[0].data
        labels = bottom[1].data
        ### ignore label
        idx = (labels != self.ignore_label)
        input_data = input_data[idx]
        labels = labels[idx]
        ### valid count 
        valid_count = np.sum(idx)
        ### calculate loss
        ##  The following code is not numerical stability
        # scores = self._sigmoid(input_data)
        # loss = - np.sum(labels * np.log(scores) + (1 - labels) * np.log(1 - scores))
        ##  numerical stability
        loss  = - np.sum(
            input_data * (labels - (input_data >= 0)) -
            np.log(1 + np.exp(input_data - 2 * input_data * (input_data >= 0)))
        )
        ### output
        self.normalizer = self._get_normalizer(valid_count)
        top[0].data[...] = loss / self.normalizer
        
    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] == True:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0] == True:
            # First, compute the diff
            input_data = bottom[0].data
            labels = bottom[1].data
            # Zero out gradient of ignored targets.
            diff = self._sigmoid(input_data) - labels
            diff[labels == self.ignore_label] = 0
            # Scale down gradient
            # Note that: top[0].diff is the loss weight and it can be set at the train prototxt.
            # In python layer, the default value of loss weight is 0.0 while in loss layer, the default 
            # value is 1.0
            bottom[0].diff[...] = diff * top[0].diff / self.normalizer

class SoftmaxWithLossLayer(caffe.Layer):
    """
    This layer is the combination of softmax layer (activate layer) and multi nomial logistic loss layer. 
    """
    def setup(self, bottom, top):
        # set up parameters
        params = eval(self.param_str)
        if params.has_key("ignore_label"):
            self.ignore_label = params["ignore_label"]
        else:
            self.ignore_label = -1
        if params.has_key("normalization"):
            self.normalization = params["normalization"]
        else:
            self.normalization = 1
        if params.has_key("axis"):
            self.axis = params["axis"]
        else:
            self.axis = 1
            
        # check input pairs
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute loss.")
        if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
            raise Exception("The data and label should have the same first dimension.")
    
    def reshape(self, bottom, top):
        bottom0_shape = list(bottom[0].shape)
        # check axis
        if self.axis < -len(bottom[0].shape) or self.axis >= len(bottom[0].shape):
            raise Exception("Axis is out of range")
        # change to formal canonical axis
        self.axis = self.axis + len(bottom0_shape) if self.axis < 0 else self.axis
        # used for reshape
        self._outer_num = int(np.prod( bottom0_shape[0 : self.axis]))
        self._class_num = int(bottom0_shape[self.axis])
        self._inner_num = int(np.prod( bottom0_shape[self.axis + 1 : ]))
        # check input dimensions match
        if self._outer_num * self._inner_num != bottom[1].count:
            raise Exception('''Number of labels must match number of predictions.
            e.g., if softmax axis == 1 and prediction shape is (N, C, H, W),
            label count (number of labels) must be N*H*W,
            with integer values in {0, 1, ..., C-1}.''')
        # loss output is scalar
        top[0].reshape(1)
        # softmax output
        if len(top) == 2:
            top[1].reshape(*bottom0_shape)
    
    def _get_normalizer(self, valid_count):
        # full mode
        if self.normalization == 0:
            normalizer = self._outer_num * self._inner_num
        # valid mode
        elif self.normalization == 1:
            normalizer = valid_count
        # batch size mode
        elif self.normalization == 2:
            normalizer = self._outer_num
        # none mode
        else:
            normalizer = 1.0
        # Some users will have no labels for some examples in order to 'turn off' a
        # particular loss in a multi-task setup. The max prevents NaNs in that case.
        return np.max([1.0, normalizer])
    
    @staticmethod
    def _softmax(x, axis):
        # substract the max to avoid numerical issues.
        # If you have questions about axis, see: https://segmentfault.com/q/1010000010111006/a-1020000010131823
        scale_data = np.max(x, axis = axis, keepdims = True)
        x = np.exp(x - scale_data)
        return x / np.sum(x, axis = axis, keepdims = True)
    
    def forward(self, bottom, top):
        # softmax output
        prob_data = self._softmax(bottom[0].data, self.axis)
        if len(top) == 2:
            top[1].data[...] = prob_data
        # prepare data (reshape)
        prob_data = prob_data.reshape((self._outer_num, self._class_num, self._inner_num))
        labels = bottom[1].data
        labels = labels.reshape((self._outer_num, self._inner_num))
        # valid_count
        valid_count = np.sum(labels != self.ignore_label)
        # loss: (TODO) Vectorized following code
        loss = 0
        for i in range(self._outer_num):
            for j in range(self._inner_num):
                label = int(labels[i][j])
                # ignore label
                if label == self.ignore_label: continue
                score = prob_data[i][label][j]
                score = np.max([score, sys.float_info.min])
                loss -= np.log(score)
        self.normalizer = self._get_normalizer(valid_count)
        top[0].data[...] = loss / self.normalizer
    
    def backward(self, top, propagate_down, bottom):
        if propagate_down[1] == True:  raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0] == False: return
        
        diff = self._softmax(bottom[0].data, self.axis)
        labels = bottom[1].data
        diff.resize((self._outer_num, self._class_num, self._inner_num))
        labels = labels.reshape((self._outer_num, self._inner_num))
        # (TODO) Vectorized following code
        for i in range(self._outer_num):
            for j in range(self._inner_num):
                label = int(labels[i][j])
                # ignore label
                if label == self.ignore_label: diff[i][label] = 0
                diff[i][label][j] -= 1
                
        diff.resize(list(bottom[0].diff.shape))
        bottom[0].diff[...] = diff * top[0].diff / self.normalizer
            
# The following class is copied from: https://github.com/BVLC/caffe/blob/master/examples/pycaffe/layers/pyloss.py
class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * self.diff / bottom[i].num
