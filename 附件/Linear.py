from Layer import Layer
import numpy as np
import gc
class Linear(Layer):
    def __init__(self,num_in,num_out):
        super(Linear, self).__init__()
        assert isinstance(num_in, int) and num_in > 0
        assert isinstance(num_out, int) and num_out > 0
        self.num_in = num_in
        self.num_out = num_out
        self.weight = np.zeros((num_in,num_out))
        self.bias = np.zeros((1,num_out))
        self.grad_w = None
        self.grad_b = None
        self.reset_parameters()
    def __call__(self, inputs,mode):
        return self.forward(inputs,mode)
    def reset_parameters(self):
        bound = np.sqrt(6./(self.num_in + self.num_out))
        self.weight = np.random.uniform(-bound,bound,(self.num_in, self.num_out))
        del bound
        gc.collect()
    def forward(self, inputs,mode):
        # inputs.shape == [N,num_in]
        inputs = inputs.reshape(inputs.shape[0],-1)
        if mode:
            self.inputs = inputs
        else:
            self.inputs = 0
        #print(inputs.shape)
        assert len(inputs.shape) == 2 and inputs.shape[1] == self.num_in
        assert self.weight.shape == (self.num_in, self.num_out)
        assert self.bias.shape == (1,self.num_out)
        z = np.dot(inputs,self.weight) + self.bias
        return z
    def step(self,learning_rate):
        self.weight -= learning_rate*self.grad_w
        self.bias -= learning_rate*self.grad_b
    def backward(self,delta_2):
        delta_1 = np.dot(delta_2,self.weight.T)
        self.grad_w = np.dot(self.inputs.T,delta_2)
        self.grad_b = np.sum(delta_2,axis = 0,keepdims=True)
        del self.inputs,delta_2
        gc.collect()
        return delta_1