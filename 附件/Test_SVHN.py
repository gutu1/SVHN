from scipy.io import loadmat
from Linear import Linear
import Loss
from Module import Module
from Sequential import Sequential
import optim
import Active
from Conv2d import Conv2d,Maxpool
import numpy as np
import matplotlib.pyplot as plt
import math, pickle

def accuracy(a, y):
    size = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    acc = sum(idx_a==idx_y) /size
    return acc

test = loadmat("./test_32x32.mat")
train = loadmat("./train_32x32.mat")
train_size = 73257
test_size = 26032
trainData, trainy = train['X'], train['y'].T
testData, testy = test['X'], test['y'].T
trainLabels = np.zeros((10,train_size),dtype='int')
testLabels = np.zeros((10,test_size),dtype='int')
for l in range(train_size):
    trainLabels[trainy[0][l]-1][l] = 1
for l in range(test_size):
    testLabels[testy[0][l]-1][l] = 1
trainLabels = trainLabels.T
testLabels = testLabels.T
train_x = trainData.reshape(-1,train_size).T 
train_x = train_x.reshape(train_size,3,32,32) /255.
test_x = testData.reshape(-1,test_size).T    
test_x = test_x.reshape(test_size,3,32,32) /255.
train_x = train_x[:10000,:,:,:]
test_x = test_x[:2000,:,:,:]
trainLabels = trainLabels[:10000,:]
testLabels = testLabels[:2000,:]

print(train_x.shape,test_x.shape,trainLabels.shape,testLabels.shape)
class Net(Module):
    def __init__(self):
        self.Sequential = Sequential([    
                            Linear(3072,1024),
                            Active.Relu(),        
                            Linear(1024,256),
                            Active.Sigmoid(),
                            Linear(256,64),
                            Active.Sigmoid(), 
                            Linear(64,10),
                            Active.SoftMax()])
        super(Net,self).__init__(self.Sequential)


model = Net()
optimizer = optim.SGD(model,learing_rate=0.01)
criterion = Loss.CrossEntropy(model)

E_list = []
Loss_list = []
Acc_list = []
batch_size = 100

for epoch in range(100):
    E_list.append(epoch)
    # Forward 前向传播
    sample_idxs = np.random.permutation(train_x.shape[0])
    num_batch = int(np.ceil(train_x.shape[0]/batch_size))
    train_cost = 0
    for batch_idx in range(num_batch):
            x = train_x[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),train_x.shape[0])],:,:,:]
            y = trainLabels[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),trainLabels.shape[0])],:]
            y_pred = model(x)
            loss = criterion(y_pred,y)
            train_cost += loss.value
            loss.backward()
            optimizer.step()
    train_cost /= batch_idx
    Loss_list.append(train_cost) #添加到loss列表中
    train_pre = model(train_x)
    Acc_list.append(accuracy(train_pre,trainLabels))
    print("epoch= ",epoch," train cost = ",train_cost," Acuuracy on train set:",Acc_list[-1])

plt.figure()
plt.plot(E_list,Loss_list,'r-*',label = 'train_costs')
plt.legend()
plt.grid()
plt.savefig("J_FC.png")
plt.show()
plt.close()

plt.figure()
plt.plot(E_list,Acc_list,'b-*',label = 'train_acc')
plt.legend()
plt.grid()
plt.savefig("Acc_FC.png")
plt.show()
plt.close()


acc = []
batch_size = 100
sample_idxs = np.random.permutation(test_x.shape[0])
num_batch = int(np.ceil(test_x.shape[0]/batch_size))
for batch_idx in range(num_batch):
    x = test_x[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),test_x.shape[0])],:,:,:]
    y = testLabels[sample_idxs[batch_size*batch_idx:min(batch_size*(batch_idx + 1),testLabels.shape[0])],:]
    y_pred = model(x)
    acc.append(accuracy(y_pred,y))
print(" Acuuracy on test set:",sum(acc)/len(acc))

test_pre = model(test_x)
test_acc = accuracy(test_pre,testLabels)
print("Acuuracy on test set:",test_acc)

model_name = 'FC_model.pkl'
with open(model_name, 'wb') as f:
    pickle.dump(model, f)
print("model saved to {}".format(model_name))