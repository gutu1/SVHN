import numpy as np
from Module import Module

class Loss:
    def __init__(self, value, prediction_gradient, backward_func):
        self.back_func = backward_func
        self.value = value
        self.prediction_gradient = prediction_gradient

    def backward(self):
        self.back_func(self.prediction_gradient)



class MSE():
    def __init__(self,Module):
        self.loss = None
        self.backward_func = Module.backward
    def __call__(self, predictions, targets):
        assert predictions.shape == targets.shape
        value = np.sum(np.square(targets-predictions)) / 2 / predictions.shape[0]
        prediction_gradient = (predictions - targets)/predictions.shape[0]  # 保存预测值的梯度，以后用来反向传播
        self.loss = Loss(value,prediction_gradient,self.backward_func)
        return self.loss

class CrossEntropy():
    def __init__(self,Module):
        self.loss = None
        self.backward_func = Module.backward
    def __call__(self, predictions, targets):
        assert predictions.shape == targets.shape
        value = -np.sum((targets * np.log(predictions))) / predictions.shape[0]
        prediction_gradient = -(targets / predictions)   # 保存预测值的梯度，以后用来反向传播
        self.loss = Loss(value,prediction_gradient,self.backward_func)
        return self.loss

