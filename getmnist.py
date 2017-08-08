# coding: cp932

import io
import sys
import numpy as np
import gzip
from PIL import Image

import pickle

tex="train-images-idx3-ubyte.gz"

import os

print(os.getcwd())
os.chdir("trainD")
print(os.getcwd())

with gzip.open(tex, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=16)

data=data.reshape(-1,784)
picdat=np.zeros([10000,28,28])
for i in range(10000):
    img=data[i].reshape(28,28)
    # pic=Image.fromarray(np.uint8(img))
    # pic.show()
    picdat[i]=img
    print(i)

print(picdat[9])
f=open("picdat.pickle",mode="wb")
pickle.dump(picdat,f)

print("finished")
