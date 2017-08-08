import pickle
import os
import random

os.chdir("trainD")

f=open("picdat.pickle","rb")
obj=pickle.load(f);

for i in range(10000):
    print(obj[i])
