import numpy as np
import chainer
from chainer import cuda,Function,gradient_check,\
    Variable,optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L


from sklearn import datasets
iris=datasets.load_iris()

X=iris.data.astype(np.float32)
Y=iris.target
N=Y.size
Y2=np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]]=1.0

index=np.arange(N)
xtrain=X[index[index%2!=0],:]
ytrain=Y2[index[index%2!=0],:]
xtest=X[index[index%2==0],:]
yans=Y[index[index%2==0]]


print(ytrain)
