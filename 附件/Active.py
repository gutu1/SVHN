import numpy as np
from Layer import Layer
import gc
#active function
class Sigmoid(Layer):
    def __init__(self):
        pass
    def __call__(self,s,mode = True):
        return self.forward(s,mode)
    def forward(self,s,mode = True):
        if mode:
            self.inputs = s
        else:
            self.inputs = 0
        return self.func(s)
    def backward(self,delta2):
        if delta2.shape != self.inputs.shape:
            delta2 = delta2.reshape(self.inputs.shape)
        delta1 = delta2 * self.df_func(self.inputs)
        del self.inputs,delta2
        gc.collect()
        return delta1
    def step(self,learing_rate):
        pass
    def func(self,s):
        return 1 / (1 + np.exp(-s))
    def df_func(self,s):
        return self.func(s) * (1 - self.func(s))


class Relu(Layer):
    def __init__(self):
        pass
    def __call__(self,s,mode = True):
        return self.forward(s,mode)
    def forward(self,s,mode = True):
        if mode:
            self.inputs = s
        else:
            self.inputs = 0
        return self.func(s)
    def backward(self,delta2):
        if delta2.shape != self.inputs.shape:
            delta2 = delta2.reshape(self.inputs.shape)
        delta1 = delta2 * self.df_func(self.inputs)
        del self.inputs,delta2
        gc.collect()
        return delta1
    def step(self,learing_rate):
        pass
    def func(self,s):
        return s * (s > 0)
    def df_func(self,s):
        return (s > 0)


class SoftMax(Layer):
    def __init__(self):
        pass
    def __call__(self,s,mode = True):
        return self.forward(s,mode)
    def forward(self,s,mode=True):
        if mode:
            self.outputs = self.func(s)
        else:
            self.outputs = 0
        return self.func(s)
    def backward(self,delta2):
        delta1 = np.array([])
        batch_size = delta2.shape[0]
        for i in range(batch_size):
            delta1 = np.append(delta1,np.dot(delta2[i:i+1,:],self.df_func()[i]))
        delta1 = delta1.reshape(batch_size,delta2.shape[1])
        #这里的delta1应该是 y_pred -y
        del delta2,self.outputs
        gc.collect()
        return delta1
    def step(self,learing_rate):
        pass
    def func(self,s):
        max = np.max(s)
        return np.exp(s-max) / np.sum(np.exp(s-max),axis = 1,keepdims=True)
    def df_func(self):
        size = self.outputs.shape[1]
        batch_size = self.outputs.shape[0]
        array = np.array([])
        for i in range(batch_size):
            array = np.append(array,-np.dot(self.outputs[i:i+1,:].T,self.outputs[i:i+1,:]) +
                                                    np.identity(size) * self.outputs[i:i+1,:])
        array = array.reshape(batch_size,size,size)
        return array

