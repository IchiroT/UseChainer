import os
import numpy as np

os.chdir("./chainer")

tex="Sketch.png"

print(open(tex,"r"))

with open(tex, 'rb') as f:
    data = np.frombuffer(f.read(), np.uint8, offset=8)
print(data)
