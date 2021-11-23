#网络层抽象类定义
from abc import ABCMeta, abstractmethod
#不可被其他文件导入
__all__ = ['Module']

class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_out):
        pass