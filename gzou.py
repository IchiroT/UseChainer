import numpy as np
import chainer
from chainer import cuda,Function,gradient_check,\
    Variable,optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L

import os
import gzip


print(os.getcwd())
os.chdir("./chainer")
print(os.getcwd())

xtex="train-images-idx3-ubyte.gz"
ytex="train-labels-idx1-ubyte.gz"
xtes="t10k-images-idx3-ubyte.gz"
ytes="t10k-labels-idx1-ubyte.gz"


with gzip.open(xtex, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
xtrain=data.reshape(-1,784).astype(np.float32)

with gzip.open(xtes, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)
xtest=data.reshape(-1,784).astype(np.float32)

with gzip.open(ytex, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)
ytrain=data.reshape(-1,1)

ny=np.zeros(ytrain.shape[0]*10).astype(np.float32)
ny=ny.reshape(ytrain.shape[0],10)
for j in range(ny.shape[0]):
    ny[j,ytrain[j]]=1
yy=ny.reshape(-1,10)
ytrain=yy

with gzip.open(ytes, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)
yans=data.reshape(-1,1)

class IrisChain(Chain):
    def __init__(self):
        super(IrisChain,self).__init__(
            l1=L.Linear(784,70),
            l2=L.Linear(70,10),
        )
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x),y)
    def fwd(self,x):
        h1=F.sigmoid(self.l1(x))
        h2=self.l2(h1)
        return h2

model=IrisChain()
optimizer=optimizers.SGD()
optimizer.setup(model)
for i in range(5000):
    x=Variable(xtrain)
    y=Variable(ytrain)

    model.zerograds()
    loss=model(x,y)
    loss.backward()#oh!!
    optimizer.update()



xt=Variable(xtest,volatile='on')
yt=model.fwd(xt)
ans=yt.data
nrow,ncol=ans.shape
ok=0
for i in range(nrow):
    cls=np.argmax(ans[i,:])
    if cls==yans[i]:
        ok+=1


print (ok,"/",nrow, " = ",(ok*1.0)/nrow)
