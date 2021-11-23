#import Layer

class Sequential():
    def __init__(self,layer_list=[]):
        #super(Sequential,self).__init__()
        self.layer_list = layer_list
    
    def forward(self,x,mode):
        out = x
        for layer in self.layer_list:
	        out = layer(out,mode)
        return out
    def backward(self,output_delta):
        layer_num = len(self.layer_list)
        delta = output_delta
        for i in range(layer_num - 1, -1, -1):
	        # 反向遍历各个层, 将期望改变量反向传播
            delta = self.layer_list[i].backward(delta)
    def step(self, leaning_rate):
	    for layer in self.layer_list:
	        layer.step(leaning_rate)
    def add_layer(self,layer):
        self.layer_list.append(layer)