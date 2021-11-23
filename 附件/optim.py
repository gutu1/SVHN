

class SGD():
    def __init__(self,Module,learing_rate,lr_decay = 0):
        self.Module = Module
        self.learning_rate = learing_rate
        self.lr_decay = lr_decay
        if lr_decay:
            self.epoch = 0
    def step(self):
        if self.lr_decay:
            self.epoch += 1
            self.learning_rate = self.learning_rate *( 1 - self.lr_decay)**(self.epoch//100)
        else:
            self.learning_rate = self.learning_rate * 1
        self.Module.step(self.learning_rate)