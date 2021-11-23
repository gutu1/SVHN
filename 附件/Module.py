
class Module():
    def __init__(self, Sequential):
        self.Sequential = Sequential
        self.mode = True
    def __call__(self, x):
        return self.forward(x)
    def forward(self, x):
        return self.Sequential.forward(x,self.mode)
    def backward(self,output_delta):
        self.Sequential.backward(output_delta)
    def step(self, lr):
        self.Sequential.step(lr)
    def add_layer(self,layer):
        self.Sequential.add_layer(layer)
    def If_train(self,mode = True):
        self.mode = mode