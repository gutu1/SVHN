from Layer import Layer
import numpy as np
import gc

def im2col(img,kernel_size,stride=1,padding=0):
    #img.shape = [batch,channel,height,weight]
    N,C,H,W = img.shape
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size_w = kernel_size
    else:
        assert len(kernel_size) == 2
        kernel_size_h = kernel_size[0]
        kernel_size_w = kernel_size[1]
    if isinstance(padding, int):
        padding_h = padding_w = padding
    else:
        assert len(kernel_size) == 2
        padding_h = padding[0]
        padding_w = padding[1]
    out_h = (H + 2 * padding_h - kernel_size_h)//stride + 1
    out_w = (W + 2 * padding_w - kernel_size_w)//stride + 1 
    #填充padiing  默认为0
    img = np.pad(img,[(0,0), (0,0), (padding_h, padding_h), (padding_w, padding_w)],'constant')
    col = np.zeros((N*out_h*out_w,C * kernel_size_h * kernel_size_w))
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + kernel_size_h
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + kernel_size_w
            col[y_start+x::out_w * out_h, :] = img[:, :,y_min:y_max, x_min:x_max].reshape(N, -1)
    del img
    gc.collect()
    return col


def col2img(col,kernel_size,output_shape,stride=1):
    #col.shape = [N*out_h*out_w,kernel_size_h * kernel_size_w * C]
    N,C,H,W = output_shape
    if isinstance(kernel_size, int):
        kernel_size_h = kernel_size_w = kernel_size
    else:
        assert len(kernel_size) == 2
        kernel_size_h = kernel_size[0]
        kernel_size_w = kernel_size[1]
    assert col.shape[1] == C*kernel_size_h*kernel_size_w
    out_h = (H - kernel_size_h)//stride + 1
    out_w = (W - kernel_size_w)//stride + 1 
    assert col.shape[0] == N*out_h*out_w
    img = np.zeros(output_shape)
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + kernel_size_h
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + kernel_size_w
            img[:, :,y_min:y_max, x_min:x_max] += col[y_start+x::out_w * out_h, :].reshape(N,C,kernel_size_h,kernel_size_w)
    del col
    gc.collect()
    return img

class Conv2d(Layer):
    def __init__(self,input_channel,output_channel,kernel_size,stride = 1,padding = 0):
        super(Conv2d).__init__()
        assert isinstance(input_channel, int) and input_channel > 0
        assert isinstance(output_channel, int) and output_channel > 0
        assert isinstance(stride, int) and stride > 0
        assert isinstance(padding, int) and padding >= 0
        self.input_channel = input_channel
        self.output_channel = output_channel
        if isinstance(kernel_size, int):
            self.kernel_size_h = self.kernel_size_w = kernel_size
        else:
            assert len(kernel_size) == 2
            self.kernel_size_h = kernel_size[0]
            self.kernel_size_w = kernel_size[1]
        self.stride = stride
        if isinstance(padding, int):
            self.padding_h = self.padding_w = padding
        else:
            assert len(kernel_size) == 2
            self.padding_h = padding[0]
            self.padding_w = padding[1]
        self.kernel = np.zeros((self.output_channel,self.input_channel,self.kernel_size_h,self.kernel_size_w))
        self.bias = np.zeros((1,output_channel))
        self.grad_w = np.zeros(self.kernel.shape)
        self.grad_b = None
        self.reset_parameters()
    def reset_parameters(self):
        bound = np.sqrt(6./(self.output_channel+self.input_channel))
        #bound = 1
        #self.kernel = np.random.randn(self.output_channel, self.input_channel, self.kernel_size_h, self.kernel_size_w) / np.sqrt(self.output_channel / 2.)
        self.kernel = np.random.uniform(-bound,bound,(self.output_channel,self.input_channel,self.kernel_size_h,self.kernel_size_w))
        #self.kernel = np.arange(self.output_channel*self.input_channel*self.kernel_size_h*self.kernel_size_w).reshape((self.output_channel,self.input_channel,self.kernel_size_h,self.kernel_size_w))
    def __call__(self, inputs,mode =True):
        return self.forward(inputs,mode)
    def forward(self, inputs,mode = True):
        # inputs.shape == [batch,channel,height,weight]
        if mode:
            self.inputs = inputs
        else: 
            self.inputs = 0
        #self.inputs = np.pad(inputs,[(0,0), (0,0), (self.padding_h, self.padding_h), (self.padding_w, self.padding_w)],'constant')
        assert len(inputs.shape) == 4 and inputs.shape[1] == self.input_channel
        assert inputs.shape[2] >= self.kernel_size_h and inputs.shape[3] >= self.kernel_size_w 
        N,C,H,W = inputs.shape
        out_h = (H + 2 * self.padding_h - self.kernel_size_h)//self.stride + 1
        out_w = (W + 2 * self.padding_w - self.kernel_size_w)//self.stride + 1 
        temp_inputs = im2col(inputs,(self.kernel_size_h,self.kernel_size_w),self.stride,(self.padding_h,self.padding_w))
        temp_kernel = im2col(self.kernel,(self.kernel_size_h,self.kernel_size_w)).T
        # z.shape  [N*out_h*out_w,output_channel]
        z = np.dot(temp_inputs,temp_kernel) + self.bias
        del temp_inputs,temp_kernel
        gc.collect()
        #z = z.T
        if mode:
            self.outputs = z.reshape(N,out_h,out_w,self.output_channel).transpose(0,3,1,2)
            del z
            gc.collect()
            return self.outputs
        else:
            self.outputs = 0
            return z.reshape(N,out_h,out_w,self.output_channel).transpose(0,3,1,2)
    def step(self,learning_rate):
        self.kernel -= learning_rate*self.grad_w
        self.bias -= learning_rate*self.grad_b
        del self.inputs,self.outputs
        gc.collect()
    def backward(self, delta_2):
        N,C,H,W = self.inputs.shape
        out_h = (H + 2 * self.padding_h - self.kernel_size_h)//self.stride + 1
        out_w = (W + 2 * self.padding_w - self.kernel_size_w)//self.stride + 1 
        delta_2 = delta_2.reshape(self.outputs.shape)
        self.grad_b = np.sum(delta_2,axis = (0,2,3),keepdims=True)
        self.grad_b = self.grad_b.reshape(self.grad_b.shape[0],self.grad_b.shape[1])
        oh, ow = delta_2.shape[2:]
        inputs = np.pad(self.inputs,[(0,0), (0,0), (self.padding_h, self.padding_h), (self.padding_w, self.padding_w)],'constant')
        # inputs [N,input_channel,in_h,in_w]
        # grad_w [output_channel,input_channel,kernel_h,kernel_w]
        # delta2 [N,output_channel,out_h,out_w]
        '''
        for h in range(self.kernel_size_h):
            for w in range(self.kernel_size_w):
                self.grad_w[:,:,h,w] = np.tensordot(delta_2, inputs[:,:,h:h+oh, w:w+ow], [[0,2,3], [0,2,3]])
        '''
        tem_delta2 = delta_2.transpose(1,2,3,0).reshape(self.output_channel,-1)
        self.grad_w = tem_delta2 @ im2col(self.inputs,(self.kernel_size_h,self.kernel_size_w),self.stride,
                                        (self.padding_h,self.padding_w))\
                                        .reshape(N,out_h*out_w,C,self.kernel_size_h*self.kernel_size_w)\
                                            .transpose(2,3,1,0)\
                                                .reshape(self.kernel_size_h*self.kernel_size_w*C,out_h*out_w*N).T
        self.grad_w = self.grad_w.reshape(self.kernel.shape)
        assert self.grad_w.shape == self.kernel.shape
        assert self.grad_b.shape == self.bias.shape
        #对卷积核沿着宽高两个维度进行翻转
        flip_kernel = np.flipud(np.fliplr(np.flip(self.kernel))).transpose(1,0,2,3)
        if self.stride > 1:
            N,C,H,W = self.inputs.shape
            #假设stride为1时forward的卷积图大小
            out_h = (H + 2*self.padding_h - self.kernel_size_h)//1 + 1
            out_w = (W + 2*self.padding_w - self.kernel_size_w)//1 + 1 
            temp_delta = np.zeros((self.outputs.shape[0],self.outputs.shape[1],out_h,out_w))
            temp_delta[:,:,::self.stride,::self.stride] = delta_2
            delta_2 = temp_delta
        #delta_2 = np.pad(delta_2,[(0,0), (0,0), (self.kernel_size_h-1, self.kernel_size_h-1), (self.kernel_size_w-1, self.kernel_size_w-1)],'constant')
        delta_2 = im2col(delta_2,(self.kernel_size_h,self.kernel_size_w),1,(self.kernel_size_h-1,self.kernel_size_w-1))
        flip_kernel = im2col(flip_kernel,(self.kernel_size_h,self.kernel_size_w)).T
        delta_1 = np.dot(delta_2,flip_kernel).reshape(self.inputs.shape[0],self.inputs.shape[2]+2*self.padding_h,self.inputs.shape[3]+2*self.padding_w,self.input_channel).transpose(0,3,1,2)
        delta_1 = delta_1[:,:,self.padding_h:delta_1.shape[2]-self.padding_h,self.padding_w:delta_1.shape[3]-self.padding_w]
        assert delta_1.shape == self.inputs.shape
        del delta_2,flip_kernel,inputs
        gc.collect()
        #print(self.grad_w)
        #print(self.grad_b)

        #
        return delta_1

class Maxpool(Layer):
    def __init__(self, size, stride = 1):
        self.size = size  # maxpool框的尺寸
        self.stride = stride
        
    def __call__(self, inputs,mode = True):
        return self.forward(inputs,mode)
    
    def forward(self, inputs,mode = True):
         # inputs.shape == [batch,channel,height,weight]
        assert len(inputs.shape) == 4 and inputs.shape[2] >= self.size and inputs.shape[3] >= self.size
        N,C,H,W = inputs.shape
        self.shape = inputs.shape
        out_h = (H - self.size)//self.stride + 1
        out_w = (W - self.size)//self.stride + 1 
        tempinputs = im2col(inputs,self.size,self.stride) #[N*out_h*out_w ,C * kernel_size_h * kernel_size_w]
        tempinputs = tempinputs.reshape(N*out_h*out_w*C,self.size * self.size)
        outputs = np.max(tempinputs,axis = 1,keepdims=True)
        self.index = np.argmax(tempinputs,axis=1)      #[N,C,out_h*out_w]
        del tempinputs,inputs
        gc.collect()
        outputs=outputs.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        return outputs
        
    def backward(self, delta_2):
        #[N*out_h*out_w*C,self.size * self.size]
        N,C,H,W = self.shape
        out_h = (H - self.size)//self.stride + 1
        out_w = (W - self.size)//self.stride + 1 
        delta_2 = delta_2.reshape((N,C,out_h,out_w))
        delta_1 = np.zeros((N*out_h*out_w*C,self.size*self.size))
        delta_2 = delta_2.transpose(0,2,3,1).reshape(-1,1)    #[N,out_h,out_w,C] -> (1,-1)
        delta_1[range(delta_1.shape[0]),self.index] = delta_2.reshape(delta_2.shape[0])
        delta_1 = delta_1.reshape(N*out_h*out_w,-1)
        delta_1 = col2img(delta_1,self.size,self.shape,self.stride)
        del delta_2
        gc.collect()
        return delta_1
    def step(self,lr):
        pass
'''
a = np.arange(1,257).reshape(4,4,4,4)
conv = Conv2d(4,3,2,2,1)
b = conv(a)
c = conv.backward(b)
#print(b)
#print(c)
'''